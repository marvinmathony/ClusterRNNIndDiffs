#!/usr/bin/env python3
"""Train-resample bootstrap that supplies ONLY the ±SD error bars for the LOO-R² decoding
panel. Companion to compute_loo_r2_canonical_bf.py, which produces the BAR (canonical R²),
the BF10 test, and the permutation p, and merges everything into loo_r2_panel.csv. This
script writes idrnn_sd / vanilla_sd; the panel reads those as the error bars.

Per bootstrap b:
  for each real participant i (held out as test):
      train = resample(participants \\ {i}, with replacement)   # SAME draw across seeds
      for each seed: fit the readout on train (PCA dim-match if >Z_DIM cols e.g. vanilla h,
                     standardise, RidgeCV α) -- ALL fit on train rows only -- then predict i
  R²_b = R²(y, mean_seeds ŷ)   # ENSEMBLE order: average the per-seed ŷ, THEN score once
→ B values of R²_b; the panel error bar = SD of these (idrnn_sd / vanilla_sd).
  This ordering matches the canonical bar (also average-ŷ-then-score) so the error bar's
  estimand equals the bar's. idrnn_sd_seedmean / vanilla_sd_seedmean = SD of the old
  score-then-average ordering, written only to confirm the two SDs agree.
  (mean and 2.5/97.5 CI are also written, but the panel bar is the canonical R² from the
   companion script -- the bootstrap mean of a nonlinear stat is biased low.)

Captures readout-fitting (training-sample) uncertainty; test set fixed at the full cohort.

Implementation: for each (b, seed) the n LOO refits are an explicit fold loop; PCA dim-match,
standardisation, and ridge α are ALL fit per fold on that fold's bootstrap train rows only
(no leakage of the held-out participant). ~B×n×seeds RidgeCV fits; joblib across the 28
(trait×arch) units, BLAS pinned to 1 thread -- run via compute_loo_r2_perfold.sbatch.

Outputs: {THAL_FULL}/decoding/loo_r2_traintest_bootstrap_perfold.csv  (+ _raw.npz)
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
os.environ.setdefault("THAL_FULL", "final_plots/thalmann_z3_3task_full")
os.environ.setdefault("THAL_DATA", "data_thalmann_3task_full")
os.environ.setdefault("N_SEEDS", "10")

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

import seed_averaged_representation as S
from decode_thalmann_canonical import load_targets, ALL_KEYS

FULL = os.environ["THAL_FULL"]; DATA = os.environ["THAL_DATA"]
Z_DIM = 3
B = int(os.environ.get("B_BOOT", "500"))
ALPHAS = (0.001, 0.01, 0.1, 1, 10, 100, 1000)
EPS = 1e-8
N_JOBS = int(os.environ.get("N_JOBS", os.environ.get("SLURM_CPUS_PER_TASK", "4")))


def full_data_alpha(Z, y):
    Zs = (Z - Z.mean(0)) / (Z.std(0) + EPS)
    ys = (y - y.mean()) / (y.std() + EPS)
    return float(RidgeCV(alphas=list(ALPHAS)).fit(Zs, ys).alpha_)


def boot_loo_yhat(Z, y, T):
    """LOO predictions ŷ (n,) for ONE seed's representation under bootstrap train set T.
    For each held-out participant i the ENTIRE readout is fit on fold i's bootstrap train
    rows only — the PCA dim-match (if Z has > Z_DIM cols, e.g. vanilla h), the per-fold
    standardisation, and the ridge-alpha selection — then i is projected/scored with those
    train-fitted transforms. No feature or hyperparameter leakage from the held-out
    participant. Row i of T = train indices for held-out i. Returns the per-participant ŷ
    (NOT R²) so the caller can average ŷ across seeds before scoring (ensemble ordering)."""
    n = len(y)
    yhat = np.empty(n)
    for i in range(n):
        Ztr, ytr = Z[T[i]], y[T[i]]                  # this fold's bootstrap train set
        z_te = Z[i]                                  # the held-out participant
        if Ztr.shape[1] > Z_DIM:                     # dim-match: PCA fit on TRAIN only
            pca = PCA(n_components=Z_DIM).fit(Ztr)
            Ztr = pca.transform(Ztr)
            z_te = pca.transform(z_te[None])[0]
        mz, sz = Ztr.mean(0), Ztr.std(0) + EPS       # standardise by train stats
        my, sy = ytr.mean(),  ytr.std()  + EPS
        model = RidgeCV(alphas=list(ALPHAS)).fit((Ztr - mz) / sz, (ytr - my) / sy)
        yhat[i] = model.predict(((z_te - mz) / sz)[None])[0] * sy + my
    return yhat


def _r2(y, yhat):
    return 1.0 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)


def draw_T(n, rng):
    """(n, n-1) train resample: row i = bootstrap of {0..n-1}\\{i} (with replacement)."""
    raw = rng.integers(0, n - 1, size=(n, n - 1))      # in [0, n-2]
    ii = np.arange(n)[:, None]
    return raw + (raw >= ii)                            # map to {0..n-1}\\{i}


def run_arch(reps_arch, y_full, rng):
    m = np.isfinite(y_full)
    for rep in reps_arch:
        m &= np.all(np.isfinite(rep), axis=1)
    if m.sum() < 30:
        return dict(n=int(m.sum()), mean=np.nan, sd=np.nan, sd_seedmean=np.nan,
                    ci_lo=np.nan, ci_hi=np.nan, R2b=np.array([]), R2b_seedmean=np.array([]))
    yv = y_full[m].astype(float); n = len(yv)
    Zs = [rep[m] for rep in reps_arch]
    R2b = np.empty(B)            # ENSEMBLE order: average per-seed ŷ, then score once
    R2b_seedmean = np.empty(B)   # score-per-seed then average (kept only to compare SDs)
    for b in range(B):
        T = draw_T(n, rng)                              # same train resample across seeds
        yhats = np.stack([boot_loo_yhat(Z, yv, T) for Z in Zs])   # (n_seeds, n)
        R2b[b] = _r2(yv, yhats.mean(0))                          # matches the canonical bar's ordering
        R2b_seedmean[b] = float(np.mean([_r2(yv, yh) for yh in yhats]))
    lo, hi = np.nanpercentile(R2b, [2.5, 97.5])
    return dict(n=n, mean=float(R2b.mean()), sd=float(np.nanstd(R2b)),
                sd_seedmean=float(np.nanstd(R2b_seedmean)),
                ci_lo=float(lo), ci_hi=float(hi), R2b=R2b, R2b_seedmean=R2b_seedmean)


def _run_unit(reps_arch, y, seed):
    """One (target, arch) LOO-bootstrap unit with its own RNG. Module-level so joblib
    can pickle it across worker processes."""
    return run_arch(reps_arch, y, np.random.default_rng(seed))


def main():
    print(f"B={B}  n_jobs={N_JOBS}")
    reps = {"idrnn": [S._idrnn_z(d) for d in S._seed_dirs("idrnn")],
            "vanilla": [S._vanilla_h(d) for d in S._seed_dirs("vanilla")]}   # raw h; PCA per-fold in boot_loo_yhat
    print(f"loaded seeds: idrnn={len(reps['idrnn'])} vanilla={len(reps['vanilla'])}")
    subids = np.load(os.path.join(DATA, "subids_full.npy")).astype(int)
    targets = load_targets(subids)

    # 28 independent (target, arch) units run in parallel; each gets its own child RNG
    # (reproducible, but the exact bootstrap draws differ from the old single-stream
    # serial version — the estimand is unchanged).
    units = [(key, arch) for key in ALL_KEYS for arch in ("idrnn", "vanilla")]
    seeds = np.random.SeedSequence(0).spawn(len(units))
    res = Parallel(n_jobs=N_JOBS)(
        delayed(_run_unit)(reps[arch], targets[key].values.astype(float), sq)
        for (key, arch), sq in zip(units, seeds))
    R = {(key, arch): r for (key, arch), r in zip(units, res)}

    rows = []; raw = {}
    print(f"{'target':<14}{'IDRNN mean ±sd(ens) [sd_seedmn]':>38}{'Vanilla mean ±sd':>22}")
    for key in ALL_KEYS:
        i = R[(key, "idrnn")]; v = R[(key, "vanilla")]
        rows.append(dict(target=key, n_idrnn=i["n"], n_vanilla=v["n"],
                         idrnn_mean=i["mean"], idrnn_sd=i["sd"], idrnn_sd_seedmean=i["sd_seedmean"],
                         idrnn_ci_lo=i["ci_lo"], idrnn_ci_hi=i["ci_hi"],
                         vanilla_mean=v["mean"], vanilla_sd=v["sd"], vanilla_sd_seedmean=v["sd_seedmean"],
                         vanilla_ci_lo=v["ci_lo"], vanilla_ci_hi=v["ci_hi"]))
        raw[f"idrnn_{key}"] = i["R2b"]; raw[f"vanilla_{key}"] = v["R2b"]
        raw[f"idrnn_{key}_seedmean"] = i["R2b_seedmean"]; raw[f"vanilla_{key}_seedmean"] = v["R2b_seedmean"]
        print(f"{key:<14}{i['mean']:>+8.3f} ±{i['sd']:.3f} [{i['sd_seedmean']:.3f}]"
              f"{v['mean']:>+13.3f} ±{v['sd']:.3f}")
    out = f"{FULL}/decoding/loo_r2_traintest_bootstrap_perfold.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    np.savez(f"{FULL}/decoding/loo_r2_traintest_bootstrap_perfold_raw.npz", **raw)   # raw R²_b for any future summary
    print(f"\nSaved {out}  (+ raw .npz)")


if __name__ == "__main__":
    main()
