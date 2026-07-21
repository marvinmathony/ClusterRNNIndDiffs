#!/usr/bin/env python3
"""Panel-c data for the MIXED-COHORT, EXCLUDE-ONLY-AVERAGE condition ("all available
data"): every has_s1 participant (n=236), vanilla h = trial-average over only the
trials that exist (single-session participants contribute their true S1-only
average; no zero-block artifact — see pooled retention confound memo).

Vanilla: canonical LOO R² (seed-averaged ŷ) + one-sided JZS BF + perm p + B=500
ensemble-order train-resample bootstrap SD, all 14 traits, per-trait cohort =
has_s1 & finite target. IDRNN columns are CONDITION-IDENTICAL to the S1-extraction
panel (z is extraction-independent; same cohort rule) and are copied from
loo_r2_panel_s1extract.csv rather than recomputed.

Output: final_plots/thalmann_z3_s2_3task_full/decoding/loo_r2_panel_mixedcorr.csv
(same columns as loo_r2_panel_s1extract.csv — drop-in for panel-c-style plotting).
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from compute_s1extract_panel import canonical_unit, boot_unit, ALL_KEYS, FULL, B, N_JOBS

def main():
    print(f"B={B} N_JOBS={N_JOBS}")
    Hall = np.load("final_plots/thalmann_z3_ds_t012/decoding/_bootreps_vanilla_rawh.npy")
    Hs1  = np.load(f"{FULL}/decoding/_bootreps_vanilla_rawh_s1extract.npy")
    tg   = pd.read_csv(f"{FULL}/decoding/trait_targets.csv")
    meta = pd.read_csv("data_thalmann_s2/df_all.csv").set_index("subid").loc[tg["subid"]]
    has_s1 = (meta["has_s1"] == 1).values
    s1only = has_s1 & (meta["has_s2"] == 0).values
    Hcorr = Hall.copy(); Hcorr[:, s1only] = Hs1[:, s1only]
    print(f"cohort has_s1 n={has_s1.sum()} (s1-only {s1only.sum()})")

    def unit_sel(k):
        return has_s1 & np.isfinite(tg[k].values.astype(float))

    canon = Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(canonical_unit)([Z[unit_sel(k)] for Z in Hcorr],
                                tg[k].values.astype(float)[unit_sel(k)])
        for k in ALL_KEYS)
    seeds = np.random.SeedSequence(7).spawn(len(ALL_KEYS))
    sds = Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(boot_unit)([Z[unit_sel(k)] for Z in Hcorr],
                           tg[k].values.astype(float)[unit_sel(k)], sq)
        for k, sq in zip(ALL_KEYS, seeds))

    idr = pd.read_csv(f"{FULL}/decoding/loo_r2_panel_s1extract.csv").set_index("target")
    rows = []
    for k, c, s in zip(ALL_KEYS, canon, sds):
        rows.append(dict(target=k, n=int(unit_sel(k).sum()),
                         idrnn_r2=idr.loc[k, "idrnn_r2"], vanilla_r2=c["r2"],
                         idrnn_sd=idr.loc[k, "idrnn_sd"], vanilla_sd=s,
                         idrnn_bf=idr.loc[k, "idrnn_bf"], vanilla_bf=c["bf"],
                         idrnn_perm_p=idr.loc[k, "idrnn_perm_p"], vanilla_perm_p=c["perm_p"]))
        print(f"{k:<14} vanilla {c['r2']:+.4f}±{s:.4f} (BF {c['bf']:.3g})")
    out = f"{FULL}/decoding/loo_r2_panel_mixedcorr.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Saved {out}")

if __name__ == "__main__":
    main()
