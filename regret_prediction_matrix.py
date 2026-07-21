#!/usr/bin/env python3
"""Regret -> held-out-target prediction matrix, for BOTH source tasks.

Addresses the fixed-schedule inflation concern: every subject saw the SAME
reward schedule, so per-subject regret is a pure-policy signal (no env noise) —
cleaner than usual designs.  Task 1 (one 200-trial block) is the most "isolated"
signal; task 0 (30 short 10-trial blocks) has more sampling variance / a less
isolated subject signal, so it's the conservative predictor.

For each SOURCE task ∈ {task0, task1} and each regret quantity
  R1 = human (observed),  R2 = exact-env single seed,
  R3 = exact-env marg-RNG, R4 = marg-env+RNG (counterfactual)
we correlate (Pearson + Spearman) with held-out targets:
  • horizon (task 3) regret: all / h5 / h10
  • working memory: composite, updating, OS, SS
  • questionnaires: PANAS_PA/NA, STICSA, PHQ, CEI, BIG5_open, CFA AxDep/posMood/negMood/Exp

Env THAL_FULL.  Outputs {THAL_FULL}/regret/regret_prediction_matrix.csv + heatmap.
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, ".")
from decode_thalmann_canonical import load_targets, label_for

FULL = os.environ.get("THAL_FULL", "final_plots/thalmann_z3_full")
d = np.load(f"{FULL}/regret/step1_cross_task_regret.npz")
subids = d["subids"].astype(int)
tg = load_targets(subids)

TARGETS = {
    "horizon (all)":  d["hum_regret_task3"],
    "horizon h5":     d["hum_regret_task3_h5"],
    "horizon h10":    d["hum_regret_task3_h10"],
    "WM composite":   tg["WM_composite"].values.astype(float),
    "WM updating":    tg["WM_WMU"].values.astype(float),
    "WM OS":          tg["WM_OS"].values.astype(float),
    "WM SS":          tg["WM_SS"].values.astype(float),
    "PANAS PA":       tg["PANAS_PA"].values.astype(float),
    "PANAS NA":       tg["PANAS_NA"].values.astype(float),
    "STICSA":         tg["STICSA"].values.astype(float),
    "PHQ":            tg["PHQ"].values.astype(float),
    "CEI":            tg["CEI"].values.astype(float),
    "BIG5 open":      tg["BIG5_open"].values.astype(float),
    "CFA AxDep":      tg["AxDep"].values.astype(float),
    "CFA Exp":        tg["Exp"].values.astype(float),
}

REGRETS = {
    ("task0", "R1 human"):        d["hum_regret_task0"],
    ("task0", "R2 exact-1seed"):  d["sim_regret_task0_r2_exact_single"],
    ("task0", "R3 exact-margRNG"):d["sim_regret_task0_r3_exact_margrng"],
    ("task0", "R4 marg-env"):     d["sim_regret_task0_r4_marg"],
    ("task1", "R1 human"):        d["hum_regret_task1"],
    ("task1", "R2 exact-1seed"):  d["sim_regret_task1_r2_exact_single"],
    ("task1", "R3 exact-margRNG"):d["sim_regret_task1_r3_exact_margrng"],
    ("task1", "R4 marg-env"):     d["sim_regret_task1_r4_marg"],
}

rows = []
for (src, reg), xv in REGRETS.items():
    for tname, yv in TARGETS.items():
        m = np.isfinite(xv) & np.isfinite(yv)
        if m.sum() < 30:
            continue
        r, p = stats.pearsonr(xv[m], yv[m]); rho, ps = stats.spearmanr(xv[m], yv[m])
        rows.append(dict(source=src, regret=reg, target=tname, n=int(m.sum()),
                         pearson_r=r, pearson_p=p, spearman_rho=rho, spearman_p=ps))
df = pd.DataFrame(rows)
out_csv = f"{FULL}/regret/regret_prediction_matrix.csv"
df.to_csv(out_csv, index=False)
print(f"Saved {out_csv}")

# ── Heatmap: rows = source×regret, cols = targets, cell = Pearson r (* if p<.05) ──
row_keys = list(REGRETS.keys()); col_keys = list(TARGETS.keys())
M = np.full((len(row_keys), len(col_keys)), np.nan)
P = np.full_like(M, 1.0)
for i, (src, reg) in enumerate(row_keys):
    for j, t in enumerate(col_keys):
        sel = df[(df.source == src) & (df.regret == reg) & (df.target == t)]
        if len(sel):
            M[i, j] = sel.pearson_r.iloc[0]; P[i, j] = sel.pearson_p.iloc[0]

fig, ax = plt.subplots(figsize=(0.55*len(col_keys)+3, 0.5*len(row_keys)+2))
vmax = np.nanmax(np.abs(M))
im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
ax.set_xticks(range(len(col_keys))); ax.set_xticklabels(col_keys, rotation=45, ha="right", fontsize=7)
ax.set_yticks(range(len(row_keys))); ax.set_yticklabels([f"{s}·{r}" for s, r in row_keys], fontsize=7)
for i in range(len(row_keys)):
    for j in range(len(col_keys)):
        if np.isfinite(M[i, j]):
            star = "*" if P[i, j] < .05 else ""
            ax.text(j, i, f"{M[i,j]:+.2f}{star}", ha="center", va="center", fontsize=5.5,
                    color="white" if abs(M[i, j]) > vmax*0.6 else "black")
ax.axhline(3.5, color="k", lw=1.2)   # separate task0 (top) from task1 (bottom)
ax.set_title(f"Regret → held-out target (Pearson r)  [{os.path.basename(FULL)}]\n"
             "top 4 rows = task0 predictor (noisier), bottom 4 = task1", fontsize=9)
fig.colorbar(im, ax=ax, fraction=0.025, label="Pearson r")
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"{FULL}/regret/regret_prediction_matrix.{ext}", dpi=300, bbox_inches="tight")
print(f"Saved {FULL}/regret/regret_prediction_matrix.png")

# console summary: task0 vs task1, R1 (human) vs R4 (counterfactual)
print("\n=== horizon (all) prediction: task0 vs task1 ===")
for src in ("task0", "task1"):
    for reg in ("R1 human", "R4 marg-env"):
        s = df[(df.source == src) & (df.regret == reg) & (df.target == "horizon (all)")]
        if len(s):
            print(f"  {src} {reg:<14} r={s.pearson_r.iloc[0]:+.3f} (p={s.pearson_p.iloc[0]:.3g})")
