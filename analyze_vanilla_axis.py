#!/usr/bin/env python3
"""Unsupervised axis-discovery control: can the ABLATED model's representation
organize the exploration/exploitation axis the way the IDRNN's does?

Identical pipeline to the IDRNN consensus axis (analyze_z2_seed_matched.py):
standardize each seed's latent dims, concatenate the 10 seeds, pooled PCA.
PRIMARY definition (2026-07-12 convention): the consensus axis is PC1 — fixed ex
ante by the representation's own variance structure, no component selection; the
horizon-switch anchor fixes only the arbitrary PCA sign. The switch-anchored
argmax component is reported as a transparency row (definition='anchored').
Run for both architectures in each analysis regime:
  idrnn/has_s1 (236)  idrnn/both (175)      — reference (expect PC1, joint open+/WM−)
  vanilla S1-extracted/has_s1 (236)         — the adopted panel-c convention
  vanilla full-extraction/both (175)        — vanilla's most favourable regime
(The mixed-cohort regime is omitted: vanilla's PC1 there is the coverage flag —
see pooled retention confound.)

Output: final_plots/thalmann_z3_s2_3task_full/decoding/vanilla_axis_test.csv
  one row per (arch, regime): anchor PC index, its variance ratio, r+BF with
  openness / WM_composite / CEI, r with horizon and overall switch.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
from pingouin import bayesfactor_pearson

FULL = "final_plots/thalmann_z3_s2_3task_full"
tg   = pd.read_csv(f"{FULL}/decoding/trait_targets.csv")
meta = pd.read_csv("data_thalmann_s2/df_all.csv").set_index("subid").loc[tg["subid"]]
has_s1 = (meta["has_s1"] == 1).values
both   = has_s1 & (meta["has_s2"] == 1).values

Zi   = np.load("final_plots/thalmann_z3_ds_t012/decoding/_bootreps_idrnn.npy")
Hall = np.load("final_plots/thalmann_z3_ds_t012/decoding/_bootreps_vanilla_rawh.npy")
Hs1  = np.load(f"{FULL}/decoding/_bootreps_vanilla_rawh_s1extract.npy")
# corrected exclude-only all-session average (mixed cohort): S1-only participants'
# rows are their true S1 average — no zero-block contamination (see confound memo)
_s1only = has_s1 & (meta["has_s2"] == 0).values
Hcorr = Hall.copy(); Hcorr[:, _s1only] = Hs1[:, _s1only]

c   = np.load("data_sub_t012_full/c_train.npy")
tid = np.load("data_sub_t012_full/task_ids_per_block.npy")


def switch_rate(bl):
    s = p = 0
    for b in range(bl.shape[0]):
        ch = bl[b]; ch = ch[ch >= 0]
        if len(ch) < 2: continue
        s += int((ch[1:] != ch[:-1]).sum()); p += len(ch) - 1
    return s / p if p else np.nan


N = c.shape[0]
sw_hor = np.array([switch_rate(c[i][tid == 2]) for i in range(N)])
sw_all = np.array([switch_rate(c[i]) for i in range(N)])


def rr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return (float(pearsonr(a[m], b[m])[0]), int(m.sum())) if m.sum() > 2 else (np.nan, 0)


def axis_test(reps, sel):
    """PRIMARY definition (per 2026-07-12 convention): the consensus axis is PC1 —
    fixed ex ante by the representation's own variance structure, no component
    selection. The horizon-switch anchor is used ONLY to fix the arbitrary PCA
    sign (cannot affect magnitudes/BFs). A second row reports the switch-anchored
    argmax component for transparency (the earlier, selection-based variant)."""
    cols = [(Z[sel] - Z[sel].mean(0)) / (Z[sel].std(0) + 1e-9) for Z in reps]
    X = np.hstack(cols)
    pca = PCA(n_components=min(10, X.shape[1])).fit(X)
    sc = pca.transform(X)
    sh = sw_hor[sel]
    j_anchor = int(np.argmax([abs(rr(sc[:, k], sh)[0]) for k in range(sc.shape[1])]))
    rows = []
    for definition, j in (("pc1", 0), ("anchored", j_anchor)):
        v = sc[:, j]
        if rr(v, sh)[0] < 0:
            v = -v
        row = dict(definition=definition, pc=j + 1,
                   var_ratio=float(pca.explained_variance_ratio_[j]),
                   r_sw_hor=rr(v, sh)[0], r_sw_all=rr(v, sw_all[sel])[0], n=int(sel.sum()))
        for k in ("BIG5_open", "WM_composite", "CEI"):
            r, n = rr(v, tg[k].values.astype(float)[sel])
            row[f"r_{k}"] = r
            row[f"bf_{k}"] = float(bayesfactor_pearson(r, n))
        rows.append(row)
    return rows


REGIMES = [
    ("idrnn",   "has_s1_236",         Zi,    has_s1),
    ("idrnn",   "both_175",           Zi,    both),
    ("vanilla", "mixed_corr_236",     Hcorr, has_s1),
    ("vanilla", "s1_extract_236",     Hs1,   has_s1),
    ("vanilla", "both_full_175",      Hall,  both),
]
rows = []
for arch, regime, R, sel in REGIMES:
    for row in axis_test(R, sel):
        row = dict(arch=arch, regime=regime, **row)
        rows.append(row)
        print(f"{arch:<8}{regime:<18}{row['definition']:<9} PC{row['pc']} (var {row['var_ratio']:.3f})  "
              f"open r={row['r_BIG5_open']:+.3f} BF={row['bf_BIG5_open']:.3g}  "
              f"WM r={row['r_WM_composite']:+.3f} BF={row['bf_WM_composite']:.3g}  "
              f"sw_hor r={row['r_sw_hor']:+.3f}")
out = f"{FULL}/decoding/vanilla_axis_test.csv"
pd.DataFrame(rows).to_csv(out, index=False)
print(f"Saved {out}")
