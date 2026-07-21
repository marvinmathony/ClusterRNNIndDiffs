#!/usr/bin/env python3
"""Canonical (non-bootstrap) LOO decoding + tests for the data-scaling panel — the
plotted POINT per (group, arch, task-constellation), the one-sided correlation BF10,
and the permutation p. EXACT port of compute_loo_r2_canonical_bf.py to the 7 task
constellations; the ±SD error bars come from the companion
compute_datascaling_traintest_bootstrap_perfold.py.

Per (group, arch, constellation, trait):
  for each real participant i (held out once): fit the leakage-free readout on the
  other n-1 REAL participants — per-fold PCA dim-match (vanilla RAW h only), per-fold
  standardisation, per-fold RidgeCV α — predict i, then average ŷ_i over the 10 seeds
  → one honest out-of-sample prediction per participant.
  canonical_r2  = 1 − SS_res/SS_tot            'chance' = always predict the trait mean
  bf10_greater  = ONE-SIDED JZS correlation BF, bayesfactor_pearson(r(ŷ,y), n, 'greater')
  perm_p        = permutation p (permute y vs the fixed ŷ, one-sided, 10000 draws) —
                  the robust primary arbiter
  bf10_twosided = TRANSPARENCY ONLY (under LOO the null of r is strongly NEGATIVE, a
                  two-sided BF flags that artifact as evidence; and pingouin's one-sided
                  BF is unstable for |r| ≳ 0.5, so trust bf10_greater only there —
                  covers every positive claim here). Same traps as the main panel; the
                  e_i error-reduction t-BF is NOT computed (demonstrated pathological in
                  compute_loo_r2_canonical_bf.py).
Tests are computed per trait and saved, but NOT annotated in the figure.

Panel point per (group, arch, constellation) = mean over the group's traits of
canonical_r2 (wm = 3 traits, openness = 1), matching the previous panel's averaging.

Run via compute_datascaling_perfold.sbatch (cpu_p, joblib across the 56 units, BLAS=1).

Outputs (all under final_plots/thalmann_z3_datascaling/):
  datascaling_canonical_bf.csv    -- long: per (group, arch, subset, trait): n,
                                     canonical_r2, pearson_r, bf10_greater, perm_p,
                                     bf10_twosided
  datascaling_canonical_bf_yhat.npz  -- per-participant ŷ and y per unit
  datascaling_panel.csv           -- WIDE, what the figure reads: per (group, arch,
                                     subset): ntasks, n, canonical_r2 (trait-mean),
                                     boot_sd (from the companion bootstrap CSV)
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
from pingouin import bayesfactor_pearson

import datascaling_reps as D
from decode_thalmann_canonical import load_targets

N_JOBS = int(os.environ.get("N_JOBS", os.environ.get("SLURM_CPUS_PER_TASK", "4")))


def loo_yhat(Z, y):
    """Canonical LOO predictions: for each i, fit the leakage-free readout on the real
    other n-1 participants and predict i. Everything (PCA dim-match, standardisation,
    RidgeCV alpha) is fit on the train rows only. Verbatim from
    compute_loo_r2_canonical_bf.py."""
    n = len(y)
    yhat = np.empty(n)
    idx = np.arange(n)
    for i in range(n):
        tr = idx[idx != i]                            # the real other n-1 (no resampling)
        Ztr, ytr = Z[tr], y[tr]
        z_te = Z[i]
        if Ztr.shape[1] > D.Z_DIM:                    # dim-match: PCA fit on TRAIN only
            pca = PCA(n_components=D.Z_DIM).fit(Ztr)
            Ztr = pca.transform(Ztr)
            z_te = pca.transform(z_te[None])[0]
        mz, sz = Ztr.mean(0), Ztr.std(0) + D.EPS
        my, sy = ytr.mean(),  ytr.std()  + D.EPS
        model = RidgeCV(alphas=list(D.ALPHAS)).fit((Ztr - mz) / sz, (ytr - my) / sy)
        yhat[i] = model.predict(((z_te - mz) / sz)[None])[0] * sy + my
    return yhat


def _run_unit(Zs, y):
    """One (group, arch, constellation, trait) unit: seed-averaged canonical ŷ, R²,
    one-sided JZS correlation BF vs chance, and the permutation p."""
    n = len(y)
    yhat = np.mean([loo_yhat(Z, y) for Z in Zs], axis=0)                  # seed-averaged
    r2 = float(1.0 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2))
    r = float(np.corrcoef(yhat, y)[0, 1])
    bf_greater = float(bayesfactor_pearson(r, n, alternative="greater"))
    bf_twosided = float(bayesfactor_pearson(r, n))                        # transparency
    rng = np.random.default_rng(0)
    perm = np.array([np.corrcoef(rng.permutation(y), yhat)[0, 1] for _ in range(10000)])
    perm_p = float((np.sum(perm >= r) + 1) / (len(perm) + 1))
    return dict(n=n, canonical_r2=r2, pearson_r=r, bf10_greater=bf_greater,
                bf10_twosided=bf_twosided, perm_p=perm_p, yhat=yhat, y=y)


def main():
    print(f"N_JOBS={N_JOBS}")
    reps = D.load_reps_raw()
    ref = np.load(f"{D.datadir('t0')}/subids_full.npy").astype(int)
    tg = load_targets(ref)
    masks = {g: D.group_mask(tg, gt, reps) for g, gt in D.GROUPS.items()}
    for g, m in masks.items():
        print(f"[{g}] n={int(m.sum())}")

    tasks = [(g, a, s, t) for g in D.GROUPS for a in D.ARCHES for s in D.SUBSETS
             for t in D.GROUPS[g]]
    res = Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(_run_unit)([Z[masks[g]] for Z in reps[(s, a)]],
                           tg[t].values.astype(float)[masks[g]])
        for (g, a, s, t) in tasks)

    rows = []
    yh = {}
    print(f"\n{'group':<10}{'arch':<9}{'sub':<6}{'trait':<14}{'R2':>9}{'r':>8}"
          f"{'BF_gt':>13}{'perm_p':>10}")
    for (g, a, s, t), d in zip(tasks, res):
        rows.append(dict(group=g, arch=a, subset=s, ntasks=D.NTASK[s], trait=t, n=d["n"],
                         canonical_r2=d["canonical_r2"], pearson_r=d["pearson_r"],
                         bf10_greater=d["bf10_greater"], perm_p=d["perm_p"],
                         bf10_twosided=d["bf10_twosided"]))
        yh[f"{g}_{a}_{s}_{t}_yhat"] = d["yhat"]
        yh[f"{g}_{a}_{s}_{t}_y"] = d["y"]
        print(f"{g:<10}{a:<9}{s:<6}{t:<14}{d['canonical_r2']:>+9.3f}{d['pearson_r']:>+8.2f}"
              f"{d['bf10_greater']:>13.3g}{d['perm_p']:>10.4f}")
    os.makedirs(D.OUTDIR, exist_ok=True)
    long = pd.DataFrame(rows)
    out = f"{D.OUTDIR}/datascaling_canonical_bf.csv"
    long.to_csv(out, index=False)
    np.savez(f"{D.OUTDIR}/datascaling_canonical_bf_yhat.npz", **yh)
    print(f"\nSaved {out}  (+ _yhat.npz)")

    # Merged panel CSV: the figure's point (canonical R², trait-mean) + the error bar
    # (train-resample bootstrap SD from the companion). BF/perm stay in the long CSV.
    panel = (long.groupby(["group", "arch", "subset"], sort=False)
                 .agg(ntasks=("ntasks", "first"), n=("n", "first"),
                      canonical_r2=("canonical_r2", "mean"))
                 .reset_index())
    sd_path = f"{D.OUTDIR}/datascaling_traintest_bootstrap_perfold.csv"
    if os.path.exists(sd_path):
        sd = pd.read_csv(sd_path)[["group", "arch", "subset", "boot_sd"]]
        panel = panel.merge(sd, on=["group", "arch", "subset"], how="left")
    else:
        print(f"WARNING: {sd_path} missing — run the bootstrap companion; boot_sd=NaN")
        panel["boot_sd"] = np.nan
    panel_out = f"{D.OUTDIR}/datascaling_panel.csv"
    panel.to_csv(panel_out, index=False)
    print(f"Saved {panel_out}")
    print("\n=== mean-of-means per (group, arch, ntasks) — canonical ===")
    print(panel.groupby(["group", "arch", "ntasks"]).canonical_r2.mean().round(4).to_string())


if __name__ == "__main__":
    main()
