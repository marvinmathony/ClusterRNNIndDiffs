#!/usr/bin/env python3
"""Supervisor's bootstrap for the LOO-R² decoding panel: bootstrap the TRAINING
fold (refit the readout), keep LOO test = each real participant, no leakage.

Per bootstrap b:
  for each real participant i (held out as test):
      train = resample(participants \\ {i}, with replacement)   # SAME draw across seeds
      for each seed: fit ridge on train (seed latent), predict i
  R²_seed = R²(y, yhat_seed)  over the n held-out predictions
  R²_b    = mean over seeds
→ B values of R²_b; bar = mean, error bar = 2.5/97.5 percentile CI.

This captures readout-fitting (training-sample) uncertainty (test set fixed at the
full cohort), distinct from the nested participant×seed bootstrap (which fixed the
readout and resampled the evaluation set).

Implementation: for each (b, seed) the n LOO refits are an explicit fold loop; ridge α
is refit per fold via RidgeCV on that fold's own bootstrap train set (classic nested
selection, no leakage of the held-out participant). Slower than a fixed-α batched solve
(≈ B×n×seeds RidgeCV fits) but α is honestly chosen per fold.

Outputs: {THAL_FULL}/decoding/loo_r2_traintest_bootstrap.csv
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

import seed_averaged_representation as S
from decode_thalmann_canonical import load_targets, ALL_KEYS

FULL = os.environ["THAL_FULL"]; DATA = os.environ["THAL_DATA"]
Z_DIM = 3
B = int(os.environ.get("B_BOOT", "500"))
ALPHAS = (0.001, 0.01, 0.1, 1, 10, 100, 1000)
EPS = 1e-8


def full_data_alpha(Z, y):
    Zs = (Z - Z.mean(0)) / (Z.std(0) + EPS)
    ys = (y - y.mean()) / (y.std() + EPS)
    return float(RidgeCV(alphas=list(ALPHAS)).fit(Zs, ys).alpha_)


def boot_loo_r2(Z, y, T):
    """LOO over folds with a bootstrapped train set T (n, n-1). For each held-out
    participant i the ENTIRE readout is fit on fold i's bootstrap train rows only —
    the PCA dim-match (if Z has > Z_DIM cols, e.g. vanilla h), the per-fold
    standardisation, and the ridge-alpha selection — then i is projected/scored with
    those train-fitted transforms. No feature or hyperparameter leakage from the
    held-out participant. Row i of T = train indices for held-out i; returns LOO R²."""
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
        return np.nan, np.nan, np.nan, np.nan, int(m.sum()), np.array([])
    yv = y_full[m].astype(float); n = len(yv)
    Zs = [rep[m] for rep in reps_arch]
    R2b = np.empty(B)
    for b in range(B):
        T = draw_T(n, rng)                              # same train resample across seeds
        R2b[b] = np.mean([boot_loo_r2(Z, yv, T) for Z in Zs])
    lo, hi = np.nanpercentile(R2b, [2.5, 97.5])
    return float(R2b.mean()), float(lo), float(hi), float(np.nanstd(R2b)), n, R2b


def main():
    print(f"B={B}")
    reps = {"idrnn": [S._idrnn_z(d) for d in S._seed_dirs("idrnn")],
            "vanilla": [S._vanilla_h(d) for d in S._seed_dirs("vanilla")]}   # raw h; PCA per-fold in boot_loo_r2
    print(f"loaded seeds: idrnn={len(reps['idrnn'])} vanilla={len(reps['vanilla'])}")
    subids = np.load(os.path.join(DATA, "subids_full.npy")).astype(int)
    targets = load_targets(subids)
    rng = np.random.default_rng(0)
    rows = []; raw = {}
    print(f"{'target':<14}{'IDRNN mean ±sd [CI]':>34}{'Vanilla mean ±sd':>22}")
    for key in ALL_KEYS:
        y = targets[key].values.astype(float)
        im, ilo, ihi, isd, ni, iarr = run_arch(reps["idrnn"], y, rng)
        vm, vlo, vhi, vsd, nv, varr = run_arch(reps["vanilla"], y, rng)
        rows.append(dict(target=key, n_idrnn=ni, n_vanilla=nv,
                         idrnn_mean=im, idrnn_sd=isd, idrnn_ci_lo=ilo, idrnn_ci_hi=ihi,
                         vanilla_mean=vm, vanilla_sd=vsd, vanilla_ci_lo=vlo, vanilla_ci_hi=vhi))
        raw[f"idrnn_{key}"] = iarr; raw[f"vanilla_{key}"] = varr
        print(f"{key:<14}{im:>+8.3f} ±{isd:.3f} [{ilo:+.3f},{ihi:+.3f}]{vm:>+12.3f} ±{vsd:.3f}")
    out = f"{FULL}/decoding/loo_r2_traintest_bootstrap.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    np.savez(f"{FULL}/decoding/loo_r2_traintest_bootstrap_raw.npz", **raw)   # raw R²_b for any future summary
    print(f"\nSaved {out}  (+ raw .npz)")


if __name__ == "__main__":
    main()
