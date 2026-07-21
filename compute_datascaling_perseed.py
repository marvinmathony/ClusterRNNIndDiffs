#!/usr/bin/env python3
"""Per-constellation, per-seed decoding LOO R² for the data-scaling climbs (WM + Openness,
IDRNN z vs dim-matched vanilla h). For each task constellation (subset) and each of the 10
seeds, compute the LOO R² (per-fold RidgeCV, exactly matching seed_averaged_representation.py);
for WM the per-seed readout is the mean over the 3 WM traits. Output per-constellation
seed_mean ± seed_sd (over the 10 seeds). Reuses the cached reps (no forward passes).

Output: final_plots/thalmann_z3_datascaling/datascaling_perconstellation.csv
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import LeaveOneOut
from decode_thalmann_canonical import load_targets

SUBSETS = ["t0", "t1", "t2", "t01", "t02", "t12", "t012"]
NTASK = {"t0": 1, "t1": 1, "t2": 1, "t01": 2, "t02": 2, "t12": 2, "t012": 3}
GROUPS = {"openness": ["BIG5_open"], "wm": ["WM_composite", "WM_WMU", "WM_SS"]}
ARCHES = ["idrnn", "vanilla"]
ALPHAS = (0.001, 0.01, 0.1, 1, 10, 100, 1000)
OUTDIR = "final_plots/thalmann_z3_datascaling"


def reps(sub, arch):
    return np.load(f"final_plots/thalmann_z3_ds_{sub}/decoding/_bootreps_{arch}.npy")  # (10,238,3)


def loo_r2(Z, y):
    """Per-fold RidgeCV LOO R² — identical recipe to seed_averaged_representation.loo_r2_and_r."""
    loo = LeaveOneOut(); preds = np.zeros(len(y))
    for tr, te in loo.split(Z):
        mz, sz = Z[tr].mean(0), Z[tr].std(0) + 1e-8
        my, sy = y[tr].mean(), y[tr].std() + 1e-8
        preds[te] = RidgeCV(alphas=list(ALPHAS)).fit((Z[tr]-mz)/sz, (y[tr]-my)/sy
                            ).predict((Z[te]-mz)/sz) * sy + my
    return 1.0 - np.sum((y - preds)**2) / np.sum((y - y.mean())**2)


def main():
    ref = np.load("data_sub_t0_full/subids_full.npy").astype(int)
    tg = load_targets(ref)
    rows = []
    for gname, gtraits in GROUPS.items():
        m = np.ones(len(ref), bool)
        for t in gtraits:
            m &= np.isfinite(tg[t].values.astype(float))
        REP = {(sub, arch): reps(sub, arch) for sub in SUBSETS for arch in ARCHES}
        for (sub, arch), arr in REP.items():
            for s in range(arr.shape[0]):
                m &= np.all(np.isfinite(arr[s]), axis=1)
        n = int(m.sum())
        ys = {t: tg[t].values.astype(float)[m] for t in gtraits}
        for arch in ARCHES:
            for sub in SUBSETS:
                per_seed = []
                for s in range(10):
                    Z = REP[(sub, arch)][s][m]
                    per_seed.append(float(np.mean([loo_r2(Z, ys[t]) for t in gtraits])))
                per_seed = np.array(per_seed)
                rows.append(dict(group=gname, arch=arch, subset=sub, ntasks=NTASK[sub], n=n,
                                 seed_mean=float(per_seed.mean()), seed_sd=float(per_seed.std())))
    df = pd.DataFrame(rows)
    os.makedirs(OUTDIR, exist_ok=True)
    df.to_csv(f"{OUTDIR}/datascaling_perconstellation.csv", index=False)
    print(df.round(4).to_string(index=False))
    print("\n=== mean-of-means per (group, arch, ntasks) ===")
    mom = df.groupby(["group", "arch", "ntasks"]).seed_mean.mean().round(4)
    print(mom.to_string())
    print(f"\nSaved {OUTDIR}/datascaling_perconstellation.csv")


if __name__ == "__main__":
    main()
