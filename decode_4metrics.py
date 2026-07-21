#!/usr/bin/env python3
"""Plot four SEED-AVERAGED decodability metrics (IDRNN vs dim-matched Vanilla),
all targets, from seed_averaged_decoding.csv (run seed_averaged_representation.py
first so the {arch}_{pearson,spearman,loo_r,loo_r2}_mean/sd columns exist).

Seed-averaged (not single-seed) — otherwise a lucky seed shows spurious signal
(e.g. vanilla openness) that vanishes under averaging.

Env THAL_FULL. Outputs {THAL_FULL}/decoding/four_metrics.{png,pdf}.
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")
from decode_thalmann_canonical import label_for, ALL_KEYS, PERSONALITY, COMPOSITE

FULL = os.environ.get("THAL_FULL", "final_plots/thalmann_z3_3task_full")
COL_I, COL_V = "#0272b2", "#ec6f00"
df = pd.read_csv(f"{FULL}/decoding/seed_averaged_decoding.csv").set_index("target").reindex(ALL_KEYS)
n_seeds = "10"

labels = [label_for(k).replace("\n", " ") for k in df.index]
x = np.arange(len(labels)); w = .38
METRICS = [("pearson", "Pearson r (in-sample multiple-R)"),
           ("spearman", "Spearman r (in-sample)"),
           ("loo_r", "LOO r (pred vs target; neg = no-info artifact)"),
           ("loo_r2", "LOO R² (out-of-sample)")]
fig, axes = plt.subplots(2, 2, figsize=(15, 9))
for ax, (mkey, title) in zip(axes.ravel(), METRICS):
    im, isd = df[f"idrnn_{mkey}_mean"], df[f"idrnn_{mkey}_sd"]
    vm, vsd = df[f"vanilla_{mkey}_mean"], df[f"vanilla_{mkey}_sd"]
    ax.bar(x - w/2, im, w, yerr=isd, color=COL_I, edgecolor="k", linewidth=.4, capsize=1.5, label="IDRNN z")
    ax.bar(x + w/2, vm, w, yerr=vsd, color=COL_V, edgecolor="k", linewidth=.4, capsize=1.5,
           label="Vanilla h (dim-matched)")
    ax.axhline(0, color="k", lw=.8)
    for xb in (len(PERSONALITY)-.5, len(PERSONALITY)+len(COMPOSITE)-.5):
        ax.axvline(xb, color="grey", lw=.5, ls=":", alpha=.6)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_title(title, fontsize=10, fontweight="bold"); ax.set_ylabel(f"{mkey} (mean±SD)")
    ax.legend(fontsize=8); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
fig.suptitle(f"Seed-averaged decodability — 4 metrics, IDRNN vs dim-matched Vanilla  "
             f"[{os.path.basename(FULL)}, mean±SD over seeds]\n"
             "(personality | composite | working-memory groups)", fontweight="bold")
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"{FULL}/decoding/four_metrics.{ext}", dpi=250, bbox_inches="tight")
print(f"Saved {FULL}/decoding/four_metrics.png (seed-averaged)")
print("IDRNN LOO R²>0:", [t for t in df.index if df.loc[t, 'idrnn_loo_r2_mean'] > 0])
print("Vanilla LOO R²>0:", [t for t in df.index if df.loc[t, 'vanilla_loo_r2_mean'] > 0])
