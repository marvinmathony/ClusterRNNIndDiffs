#!/usr/bin/env python3
"""Per-constellation participant bootstrap for the data-scaling climbs — STABLE version.

Refitting a readout inside every bootstrap is intrinsically unstable for the fragile
dim-matched vanilla readout (LOO-train-resample overfits → catastrophic numerator; OOB → tiny
denominator → explosion). Instead: compute honest out-of-fold LOO predictions ONCE per
(constellation, arch, seed, trait) with per-fold RidgeCV (alpha fitted per fold), then bootstrap
only the EVALUATION — resample participants (with replacement) and recompute R² on the fixed
(y, ŷ) pairs. Stable denominator (variance over all n), bounded predictions, no artifact; the
point ≈ the standard LOO value (no refit-on-fewer-participants bias).

Per (bootstrap b, seed): value = mean over group-traits of R²(y[idx], ŷ[idx]). Pool all
B×10 values per constellation → point = mean of pool, SD = SD of pool. Reuses cached reps.

Merges boot_mean/boot_sd into datascaling_perconstellation.csv.
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
GROUPS = {"openness": ["BIG5_open"], "wm": ["WM_composite", "WM_WMU", "WM_SS"]}
ARCHES = ["idrnn", "vanilla"]
ALPHAS = (0.001, 0.01, 0.1, 1, 10, 100, 1000)
B = int(os.environ.get("B_BOOT", "500"))
EPS = 1e-8
CSV = "final_plots/thalmann_z3_datascaling/datascaling_perconstellation.csv"


def reps(sub, arch):
    return np.load(f"final_plots/thalmann_z3_ds_{sub}/decoding/_bootreps_{arch}.npy")


def loo_preds(Z, y):
    """Per-fold RidgeCV out-of-fold predictions (alpha fitted per fold) — the stable pipeline recipe."""
    loo = LeaveOneOut(); preds = np.zeros(len(y))
    for tr, te in loo.split(Z):
        mz, sz = Z[tr].mean(0), Z[tr].std(0) + EPS
        my, sy = y[tr].mean(), y[tr].std() + EPS
        preds[te] = RidgeCV(alphas=list(ALPHAS)).fit((Z[tr]-mz)/sz, (y[tr]-my)/sy
                            ).predict((Z[te]-mz)/sz) * sy + my
    return preds


def main():
    print(f"B={B}  (evaluation bootstrap on fixed per-fold-RidgeCV OOF predictions)")
    ref = np.load("data_sub_t0_full/subids_full.npy").astype(int)
    tg = load_targets(ref)
    out = {}
    for gname, gtraits in GROUPS.items():
        m = np.ones(len(ref), bool)
        for t in gtraits:
            m &= np.isfinite(tg[t].values.astype(float))
        REP = {(sub, arch): reps(sub, arch) for sub in SUBSETS for arch in ARCHES}
        for arr in REP.values():
            for s in range(arr.shape[0]):
                m &= np.all(np.isfinite(arr[s]), axis=1)
        n = int(m.sum())
        ys = {t: tg[t].values.astype(float)[m] for t in gtraits}
        # honest OOF predictions once per (arch, sub, seed, trait)
        PRED = {}
        for arch in ARCHES:
            for sub in SUBSETS:
                for s in range(10):
                    Z = REP[(sub, arch)][s][m]
                    for t in gtraits:
                        PRED[(arch, sub, s, t)] = loo_preds(Z, ys[t])
        # evaluation bootstrap: resample participants, recompute R² on the fixed (y, ŷ) pairs
        pool = {(arch, sub): np.empty(B * 10) for arch in ARCHES for sub in SUBSETS}
        rng = np.random.default_rng(0)
        for b in range(B):
            idx = rng.integers(0, n, n)                       # shared participant resample
            for arch in ARCHES:
                for sub in SUBSETS:
                    for s in range(10):
                        r2t = []
                        for t in gtraits:
                            yv = ys[t][idx]; yh = PRED[(arch, sub, s, t)][idx]
                            r2t.append(1.0 - np.sum((yv - yh)**2) / (np.sum((yv - yv.mean())**2) + EPS))
                        pool[(arch, sub)][b * 10 + s] = np.mean(r2t)
        for arch in ARCHES:
            for sub in SUBSETS:
                arr = pool[(arch, sub)]
                out[(gname, arch, sub)] = (float(arr.mean()), float(arr.std()))
        lo = min(v[1] for k, v in out.items() if k[0] == gname)
        hi = max(v[1] for k, v in out.items() if k[0] == gname)
        print(f"[{gname}] n={n}  boot_sd range: {lo:.4f}..{hi:.4f}")
    df = pd.read_csv(CSV)
    df["boot_mean"] = [out[(r.group, r.arch, r.subset)][0] for r in df.itertuples()]
    df["boot_sd"] = [out[(r.group, r.arch, r.subset)][1] for r in df.itertuples()]
    df.to_csv(CSV, index=False)
    print(df[["group", "arch", "subset", "ntasks", "seed_mean", "boot_mean", "boot_sd"]].round(4).to_string(index=False))
    print(f"\nUpdated {CSV}")


if __name__ == "__main__":
    main()
