#!/usr/bin/env python3
"""Nested participant×seed bootstrap CIs for the seed-averaged LOO R² decoding
(IDRNN z vs dim-matched Vanilla h), for the poster panel-c bootstrap variant.

Version A (no CV-leakage): out-of-fold LOO predictions are computed ONCE per seed
on the real sample (exactly as seed_averaged_representation.py does). The bootstrap
only resamples the *evaluation*: for each iteration we draw seeds (from the 10) and
participants (with replacement), recompute each resampled seed's R² on the resampled
participants, and average over the resampled seeds → one bootstrap value of the
seed-averaged LOO R². Percentiles give the CI. This propagates BOTH participant-
sampling and seed (retraining) variability, matching panel b's participant bootstrap.

Outputs: {THAL_FULL}/decoding/loo_r2_nested_bootstrap.csv
"""
import os
# Hard single-thread every BLAS/threading layer BEFORE numpy/torch import, so the
# cluster's 200% (2-core) CPU cap is never tripped. Pair with `taskset -c 0,1` at
# launch, which physically pins the whole process tree to 2 logical CPUs.
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
from sklearn.model_selection import LeaveOneOut

# reuse the production helpers (identical latents + LOO method that wrote the CSV)
import seed_averaged_representation as S
from decode_thalmann_canonical import load_targets, ALL_KEYS, pc_match

FULL = os.environ["THAL_FULL"]; DATA = os.environ["THAL_DATA"]
N_SEEDS = int(os.environ["N_SEEDS"]); Z_DIM = 3
B = 2000
ALPHAS = (0.001, 0.01, 0.1, 1, 10, 100, 1000)


def loo_preds(Z, y):
    """Out-of-fold LOO predictions — same standardise+RidgeCV recipe as
    seed_averaged_representation.loo_r2_and_r, but returns the prediction vector."""
    loo = LeaveOneOut(); preds = np.zeros(len(y))
    for tr, te in loo.split(Z):
        mz, sz = Z[tr].mean(0), Z[tr].std(0) + 1e-8
        my, sy = y[tr].mean(), y[tr].std() + 1e-8
        preds[te] = RidgeCV(alphas=list(ALPHAS)).fit((Z[tr]-mz)/sz, (y[tr]-my)/sy
                            ).predict((Z[te]-mz)/sz) * sy + my
    return preds


def reps_for(arch):
    dirs = S._seed_dirs(arch)
    print(f"[{arch}] {len(dirs)} seeds: {[int(d.split('seed_')[1]) for d in dirs]}")
    return [S._idrnn_z(d) if arch == "idrnn" else pc_match(S._vanilla_h(d), Z_DIM) for d in dirs]


def nested_ci(reps, y_full, rng):
    """Returns (seed_avg_r2_point, per_seed_r2_list, ci_lo, ci_hi, n)."""
    # common participant set: finite trait + finite rep across every seed
    m = np.isfinite(y_full)
    for rep in reps:
        m &= np.all(np.isfinite(rep), axis=1)
    if m.sum() < 30:
        return np.nan, [], np.nan, np.nan, int(m.sum())
    yv = y_full[m].astype(float); n = len(yv)
    # out-of-fold predictions per seed (computed once on the real sample → no leakage)
    P = np.vstack([loo_preds(rep[m], yv) for rep in reps])          # (n_seeds, n)
    sstot_full = np.sum((yv - yv.mean())**2)
    per_seed_r2 = [1.0 - np.sum((yv - P[s])**2)/sstot_full for s in range(len(reps))]
    point = float(np.mean(per_seed_r2))
    # nested bootstrap: resample seeds AND participants
    boots = np.empty(B)
    nseed = P.shape[0]
    for b in range(B):
        s_idx = rng.integers(0, nseed, nseed)
        p_idx = rng.integers(0, n, n)
        yb = yv[p_idx]; sstot = np.sum((yb - yb.mean())**2) + 1e-12
        Pb = P[s_idx][:, p_idx]                                     # (nseed, n)
        r2_s = 1.0 - np.sum((yb[None, :] - Pb)**2, axis=1) / sstot  # per resampled seed
        boots[b] = r2_s.mean()
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return point, per_seed_r2, float(lo), float(hi), n


def main():
    subids = np.load(os.path.join(DATA, "subids_full.npy")).astype(int)
    targets = load_targets(subids)
    idr = reps_for("idrnn"); van = reps_for("vanilla")
    ref = pd.read_csv(f"{FULL}/decoding/seed_averaged_decoding.csv").set_index("target")
    rng = np.random.default_rng(0)
    rows = []
    print(f"\n{'target':<14}{'IDRNN R2 (CI)':>30}{'csv mean':>10}{'Vanilla R2 (CI)':>30}{'csv mean':>10}")
    for key in ALL_KEYS:
        y = targets[key].values.astype(float)
        ip, _, ilo, ihi, ni = nested_ci(idr, y, rng)
        vp, _, vlo, vhi, nv = nested_ci(van, y, rng)
        icsv = float(ref.loc[key, "idrnn_loo_r2_mean"]) if key in ref.index else np.nan
        vcsv = float(ref.loc[key, "vanilla_loo_r2_mean"]) if key in ref.index else np.nan
        rows.append(dict(target=key, n_idrnn=ni, n_vanilla=nv,
                         idrnn_r2_mean=ip, idrnn_ci_lo=ilo, idrnn_ci_hi=ihi,
                         vanilla_r2_mean=vp, vanilla_ci_lo=vlo, vanilla_ci_hi=vhi))
        print(f"{key:<14}{ip:>+8.3f} [{ilo:+.3f},{ihi:+.3f}]{icsv:>+10.3f}"
              f"{vp:>+8.3f} [{vlo:+.3f},{vhi:+.3f}]{vcsv:>+10.3f}")
    out = f"{FULL}/decoding/loo_r2_nested_bootstrap.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
