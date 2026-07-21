#!/usr/bin/env python3
"""Train-resample bootstrap that supplies ONLY the ±SD error bars for the data-scaling
decoding panel (readout LOO R² vs #training-tasks; WM + Openness; IDRNN z vs dim-matched
vanilla h). EXACT port of compute_loo_r2_traintest_bootstrap_perfold.py to the 7 task
constellations. Companion compute_datascaling_canonical_bf.py produces the plotted POINT
(canonical R²), the one-sided correlation BF10, and the permutation p, and merges
everything into datascaling_panel.csv; this script writes boot_sd (the error bar).

Per (group, arch, constellation) and bootstrap b:
  T = (n, n-1) train resample (row i = bootstrap of participants \\ {i}); the SAME T is
  used across the 10 seeds AND the group's traits (shared draw, so the seed/trait
  collapse is a proper per-b statistic — the WM traits are correlated, independent
  draws would understate the SD of their mean).
  For each seed × group-trait: LOO over folds; per fold the ENTIRE readout is fit on
  that fold's bootstrap train rows only — PCA dim-match to Z_DIM (vanilla RAW h only),
  standardisation, RidgeCV α — giving that seed's per-participant predictions ŷ.
  SEEDS are collapsed in ENSEMBLE order — average the 10 seeds' ŷ, then score once —
  because that is the statistic the plotted point (canonical R² of the seed-averaged
  ŷ) estimates. The two orders differ by the seed-disagreement (bagging) term:
      R²(y, mean_s ŷ^s) = mean_s R²_s + Σ_i Var_s(ŷ_i^s)/SS_tot   (ensemble ≥ mean),
  so score-then-average would put the SD of a systematically LOWER statistic under an
  ensemble-order bar. TRAITS are collapsed by averaging the per-trait R²_b — matching
  the panel point, which is the mean over group traits of per-trait canonical R² (ŷ
  cannot be ensembled across traits; y differs).
  R²_b = mean over group-traits of the per-trait ensemble R²_b.
→ B values of R²_b; the panel error bar = SD of these (boot_sd). boot_*_sta columns
  (score-then-average over seeds, from the IDENTICAL RidgeCV fits) are written as a
  sanity check that the ordering does not move the SD. boot_mean and the 2.5/97.5 CI
  are transparency only — the plotted point is the canonical R² from the companion
  script (the bootstrap mean of a nonlinear stat is biased low).

Captures readout-fitting (training-sample) uncertainty; test set fixed at the group
cohort. joblib across the 56 (group × arch × constellation × trait) units — trait units
of the same (group, arch, constellation) share the T stream via a common child
SeedSequence — BLAS pinned to 1 thread; run via compute_datascaling_perfold.sbatch.

Outputs: final_plots/thalmann_z3_datascaling/datascaling_traintest_bootstrap_perfold.csv
         (+ _raw.npz with the per-group and per-trait R²_b arrays)
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np
import pandas as pd
import torch
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except Exception:
    pass
from sklearn.linear_model import RidgeCV
from sklearn.decomposition import PCA
from joblib import Parallel, delayed

import datascaling_reps as D
from decode_thalmann_canonical import load_targets

B = int(os.environ.get("B_BOOT", "500"))
N_JOBS = int(os.environ.get("N_JOBS", os.environ.get("SLURM_CPUS_PER_TASK", "4")))


def boot_loo_yhat(Z, y, T):
    """LOO over folds with a bootstrapped train set T (n, n-1). For each held-out
    participant i the ENTIRE readout is fit on fold i's bootstrap train rows only —
    the PCA dim-match (if Z has > Z_DIM cols, i.e. vanilla raw h), the per-fold
    standardisation, and the ridge-alpha selection — then i is predicted with those
    train-fitted transforms. Same folds/fits as the main panel's boot_loo_r2, but
    returns the per-participant ŷ (n,) instead of the scalar R², so the caller can
    collapse seeds in ensemble order (average ŷ across seeds, then score once) — the
    same statistic as the canonical bar."""
    n = len(y)
    yhat = np.empty(n)
    for i in range(n):
        Ztr, ytr = Z[T[i]], y[T[i]]                  # this fold's bootstrap train set
        z_te = Z[i]                                  # the held-out participant
        if Ztr.shape[1] > D.Z_DIM:                   # dim-match: PCA fit on TRAIN only
            pca = PCA(n_components=D.Z_DIM).fit(Ztr)
            Ztr = pca.transform(Ztr)
            z_te = pca.transform(z_te[None])[0]
        mz, sz = Ztr.mean(0), Ztr.std(0) + D.EPS     # standardise by train stats
        my, sy = ytr.mean(),  ytr.std()  + D.EPS
        model = RidgeCV(alphas=list(D.ALPHAS)).fit((Ztr - mz) / sz, (ytr - my) / sy)
        yhat[i] = model.predict(((z_te - mz) / sz)[None])[0] * sy + my
    return yhat


def _r2(y, yhat):
    return 1.0 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)


def draw_T(n, rng):
    """(n, n-1) train resample: row i = bootstrap of {0..n-1}\\{i} (with replacement)."""
    raw = rng.integers(0, n - 1, size=(n, n - 1))      # in [0, n-2]
    ii = np.arange(n)[:, None]
    return raw + (raw >= ii)                            # map to {0..n-1}\\{i}


def _run_unit(Zs, y, seed):
    """One (group, arch, constellation, trait) unit: B bootstrap replicates, each
    collapsing the 10 seed reps under a shared T in BOTH orders:
      ens = R²(y, mean_seeds ŷ)  — ensemble order, same statistic as the canonical
                                   bar; this is THE error-bar column
      sta = mean_seeds R²(y, ŷ)  — score-then-average; identical RidgeCV fits, kept
                                   as a side column to confirm the SDs agree
    Module-level so joblib can pickle it. Trait units of the same (group, arch,
    constellation) receive the SAME child seed and draw T in the same order, so their
    T streams are identical and the trait average is a per-b statistic."""
    rng = np.random.default_rng(seed)
    n = len(y)
    ens = np.empty(B)
    sta = np.empty(B)
    for b in range(B):
        T = draw_T(n, rng)                              # same T across seeds (and traits)
        YH = np.stack([boot_loo_yhat(Z, y, T) for Z in Zs])   # (seeds, n)
        ens[b] = _r2(y, YH.mean(0))
        sta[b] = float(np.mean([_r2(y, yh) for yh in YH]))
    return ens, sta


def main():
    print(f"B={B}  n_jobs={N_JOBS}")
    reps = D.load_reps_raw()
    ref = np.load(f"{D.datadir('t0')}/subids_full.npy").astype(int)
    tg = load_targets(ref)
    masks = {g: D.group_mask(tg, gt, reps) for g, gt in D.GROUPS.items()}
    for g, m in masks.items():
        print(f"[{g}] n={int(m.sum())}")

    # 28 (group, arch, constellation) units fan out into 56 trait tasks; each unit gets
    # one child SeedSequence shared by its traits (identical T streams within a unit).
    units28 = [(g, a, s) for g in D.GROUPS for a in D.ARCHES for s in D.SUBSETS]
    seeds28 = dict(zip(units28, np.random.SeedSequence(0).spawn(len(units28))))
    tasks = [(g, a, s, t) for (g, a, s) in units28 for t in D.GROUPS[g]]
    res = Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(_run_unit)([Z[masks[g]] for Z in reps[(s, a)]],
                           tg[t].values.astype(float)[masks[g]],
                           seeds28[(g, a, s)])
        for (g, a, s, t) in tasks)
    ens_trait = {k: r[0] for k, r in zip(tasks, res)}
    sta_trait = {k: r[1] for k, r in zip(tasks, res)}

    rows = []
    raw = {}
    print(f"\n{'group':<10}{'arch':<9}{'sub':<6}{'ens mean ±sd [CI]':>36}{'sta sd':>9}")
    for (g, a, s) in units28:
        arr = np.mean([ens_trait[(g, a, s, t)] for t in D.GROUPS[g]], axis=0)   # (B,)
        arr_sta = np.mean([sta_trait[(g, a, s, t)] for t in D.GROUPS[g]], axis=0)
        lo, hi = np.nanpercentile(arr, [2.5, 97.5])
        rows.append(dict(group=g, arch=a, subset=s, ntasks=D.NTASK[s],
                         n=int(masks[g].sum()),
                         boot_mean=float(arr.mean()), boot_sd=float(np.nanstd(arr)),
                         boot_ci_lo=float(lo), boot_ci_hi=float(hi),
                         boot_mean_sta=float(arr_sta.mean()),
                         boot_sd_sta=float(np.nanstd(arr_sta))))
        raw[f"ens_{g}_{a}_{s}"] = arr
        raw[f"sta_{g}_{a}_{s}"] = arr_sta
        for t in D.GROUPS[g]:
            raw[f"ens_{g}_{a}_{s}_{t}"] = ens_trait[(g, a, s, t)]
            raw[f"sta_{g}_{a}_{s}_{t}"] = sta_trait[(g, a, s, t)]
        print(f"{g:<10}{a:<9}{s:<6}{arr.mean():>+12.4f} ±{np.nanstd(arr):.4f} "
              f"[{lo:+.4f},{hi:+.4f}]{np.nanstd(arr_sta):>9.4f}")
    os.makedirs(D.OUTDIR, exist_ok=True)
    out = f"{D.OUTDIR}/datascaling_traintest_bootstrap_perfold.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    np.savez(f"{D.OUTDIR}/datascaling_traintest_bootstrap_perfold_raw.npz", **raw)
    print(f"\nSaved {out}  (+ _raw.npz)")


if __name__ == "__main__":
    main()
