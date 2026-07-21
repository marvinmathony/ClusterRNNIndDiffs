#!/usr/bin/env python3
"""Diagnostic: order of seed aggregation for the canonical LOO R2.

Compares, per (trait, arch):
  r2_ens      = R2(y, mean over seeds of yhat)        # average-then-score (the panel bar)
  r2_seedmean = mean over seeds of R2(y, yhat_seed)   # score-then-average
  gap         = r2_ens - r2_seedmean                  # == sum_i Var_seeds(yhat_i) / SS_tot >= 0
  r2_seedsd   = SD over seeds of the per-seed R2      # seed variability, for reference
Identity check column verifies gap == seed-prediction-variance / SS_tot to numerical precision.

Reuses loo_yhat / masking from compute_loo_r2_canonical_bf (same leakage-free readout).
Outputs: {FULL}/decoding/loo_r2_seed_order_diag.csv
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import compute_loo_r2_canonical_bf as C          # sets THAL_FULL/THAL_DATA defaults, torch threads
import seed_averaged_representation as S
from decode_thalmann_canonical import load_targets, ALL_KEYS

FULL = os.environ["THAL_FULL"]; DATA = os.environ["THAL_DATA"]
N_JOBS = int(os.environ.get("N_JOBS", os.environ.get("SLURM_CPUS_PER_TASK", "4")))


def run_unit(reps_arch, y_full):
    m = np.isfinite(y_full)
    for rep in reps_arch:
        m &= np.all(np.isfinite(rep), axis=1)
    if m.sum() < 30:
        return None
    yv = y_full[m].astype(float)
    yhats = np.stack([C.loo_yhat(rep[m], yv) for rep in reps_arch])      # (S, n) per-seed
    ss_tot = float(np.sum((yv - yv.mean()) ** 2))
    r2_seeds = np.array([1.0 - np.sum((yv - yh) ** 2) / ss_tot for yh in yhats])
    r2_ens = float(1.0 - np.sum((yv - yhats.mean(0)) ** 2) / ss_tot)
    gap_identity = float(yhats.var(0, ddof=0).sum() / ss_tot)            # analytic gap
    return dict(n=int(m.sum()), r2_ens=r2_ens, r2_seedmean=float(r2_seeds.mean()),
                gap=float(r2_ens - r2_seeds.mean()), gap_identity=gap_identity,
                r2_seedsd=float(r2_seeds.std(ddof=1)))


def main():
    print(f"N_JOBS={N_JOBS}")
    reps = {"idrnn": [S._idrnn_z(d) for d in S._seed_dirs("idrnn")],
            "vanilla": [S._vanilla_h(d) for d in S._seed_dirs("vanilla")]}
    print(f"loaded seeds: idrnn={len(reps['idrnn'])} vanilla={len(reps['vanilla'])}")
    subids = np.load(os.path.join(DATA, "subids_full.npy")).astype(int)
    targets = load_targets(subids)

    units = [(k, a) for k in ALL_KEYS for a in ("idrnn", "vanilla")]
    res = Parallel(n_jobs=N_JOBS)(
        delayed(run_unit)(reps[a], targets[k].values.astype(float)) for (k, a) in units)

    rows = []
    print(f"{'target':<14}{'arch':<9}{'R2_ens':>9}{'R2_seedmean':>13}{'gap':>8}{'seedSD':>8}")
    for (k, a), d in zip(units, res):
        if d is None:
            continue
        rows.append(dict(target=k, arch=a, **d))
        print(f"{k:<14}{a:<9}{d['r2_ens']:>+9.3f}{d['r2_seedmean']:>+13.3f}"
              f"{d['gap']:>8.3f}{d['r2_seedsd']:>8.3f}")
    df = pd.DataFrame(rows)
    ok = np.allclose(df["gap"], df["gap_identity"], atol=1e-10)
    print(f"\nidentity gap == seed-pred-variance/SS_tot: {'OK' if ok else 'MISMATCH'}")
    out = f"{FULL}/decoding/loo_r2_seed_order_diag.csv"
    df.to_csv(out, index=False)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
