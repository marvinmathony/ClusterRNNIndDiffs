#!/usr/bin/env python3
"""
Summarise the Sloutsky hyperparameter search.

Reads all metrics.json files from hp_search_results_sloutsky/,
produces:
  - hp_search_results_sloutsky/ranked_results.csv
  - plots_sloutsky/hp_search_summary.png

Usage:
    python summarize_hp_search_sloutsky.py
"""

import json
import glob
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = "hp_search_results_sloutsky"
PLOT_DIR    = "plots_sloutsky"
os.makedirs(PLOT_DIR, exist_ok=True)

# ── Load all metrics.json ─────────────────────────────────────────────────────
rows = []
for path in glob.glob(f"{RESULTS_DIR}/**/metrics.json", recursive=True):
    with open(path) as f:
        d = json.load(f)
    agg = d.get("aggregate", {})
    if "mean_cv_val_loss" not in agg:
        print(f"Skipping (old format, rerun hyperparam_eval_sloutsky.py): {path}")
        continue
    rows.append({
        "lmbd":             d["lmbd"],
        "z_dim":            d["z_dim"],
        "hidden":           d["hidden"],
        "enc_hidden":       d["enc_hidden"],
        "mean_cv_val_loss": agg["mean_cv_val_loss"],
        "std_cv_val_loss":  agg["std_cv_val_loss"],
        "mean_cv_epoch":    agg["mean_cv_selected_epoch"],
        "std_cv_epoch":     agg["std_cv_selected_epoch"],
        "n_seeds":          agg["n_seeds_evaluated"],
    })

if not rows:
    raise RuntimeError("No valid metrics.json files found in " + RESULTS_DIR)

df = (pd.DataFrame(rows)
        .sort_values("mean_cv_val_loss")
        .reset_index(drop=True))
df.index += 1
df.index.name = "rank"

csv_path = os.path.join(RESULTS_DIR, "ranked_results.csv")
df.to_csv(csv_path)
print(f"Saved CSV → {csv_path}\n")
print(df[["lmbd", "z_dim", "enc_hidden",
          "mean_cv_val_loss", "std_cv_val_loss", "mean_cv_epoch"]].to_string())

best = df.iloc[0]
print(f"\n★  Best: lmbd={best.lmbd}, z={best.z_dim}, "
      f"hidden={best.hidden}, enc_hidden={best.enc_hidden}")
print(f"   mean_cv_val_loss = {best.mean_cv_val_loss:.4f} ± {best.std_cv_val_loss:.4f}  "
      f"(mean epoch {best.mean_cv_epoch:.0f})")

# ── Plot ──────────────────────────────────────────────────────────────────────
enc_hidden_vals = sorted(df["enc_hidden"].unique())
z_vals          = sorted(df["z_dim"].unique())
lmbd_vals       = sorted(df["lmbd"].unique())

# Colour by enc_hidden, marker by z_dim
palette  = ["#4477AA", "#EE6677", "#228833", "#CCBB44"]
eh_colors = {eh: palette[i] for i, eh in enumerate(enc_hidden_vals)}
z_markers = {z: m for z, m in zip(z_vals, ["o", "s", "^", "D"])}

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

# ── Panel 1: CV val loss vs lambda, one line per (enc_hidden × z_dim) ────────
ax = axes[0]
for eh in enc_hidden_vals:
    for z in z_vals:
        sub = df[(df["enc_hidden"] == eh) & (df["z_dim"] == z)].sort_values("lmbd")
        if sub.empty:
            continue
        ax.errorbar(
            sub["lmbd"], sub["mean_cv_val_loss"],
            yerr=sub["std_cv_val_loss"],
            color=eh_colors[eh], marker=z_markers[z],
            linestyle="-", linewidth=1.2, markersize=6, capsize=3,
            label=f"eh={eh}, z={z}",
        )
ax.set_xlabel("Lambda")
ax.set_ylabel("Mean CV val loss  (↓ better)")
ax.set_title("CV val loss vs Lambda")
ax.set_xscale("symlog", linthresh=0.001)
ax.legend(fontsize=7, ncol=2, loc="upper left")

# ── Panel 2: heatmap — best loss per (enc_hidden × z_dim), min over lmbd ─────
ax = axes[1]
pivot_best = df.groupby(["enc_hidden", "z_dim"])["mean_cv_val_loss"].min().unstack()
vmin = df["mean_cv_val_loss"].min()
vmax = df["mean_cv_val_loss"].quantile(0.80)
im = ax.imshow(pivot_best.values, aspect="auto", cmap="RdYlGn_r",
               vmin=vmin, vmax=vmax)
ax.set_xticks(range(len(pivot_best.columns)))
ax.set_xticklabels([f"z={z}" for z in pivot_best.columns])
ax.set_yticks(range(len(pivot_best.index)))
ax.set_yticklabels([f"eh={eh}" for eh in pivot_best.index])
ax.set_title("Best CV val loss per cell\n(min over lambda)")
plt.colorbar(im, ax=ax, label="mean CV val loss")
for i, eh in enumerate(pivot_best.index):
    for j, z in enumerate(pivot_best.columns):
        val = pivot_best.loc[eh, z]
        ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=9,
                color="white" if val > np.nanmean(pivot_best.values) else "black")
best_i, best_j = np.unravel_index(np.nanargmin(pivot_best.values), pivot_best.shape)
ax.add_patch(plt.Rectangle((best_j - 0.5, best_i - 0.5), 1, 1,
                             fill=False, edgecolor="gold", linewidth=3))

# ── Panel 3: bar chart — top 15 combos ───────────────────────────────────────
ax = axes[2]
top = df.head(15)
labels = [f"λ={r.lmbd}, z={r.z_dim}, eh={r.enc_hidden}"
          for _, r in top.iterrows()]
colors = [eh_colors[r.enc_hidden] for _, r in top.iterrows()]
bars = ax.barh(range(len(top)), top["mean_cv_val_loss"],
               xerr=top["std_cv_val_loss"],
               color=colors, edgecolor="white", capsize=3)
ax.set_yticks(range(len(top)))
ax.set_yticklabels(labels, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("Mean CV val loss  (↓ better)")
ax.set_title("Top 15 configurations")
# Legend for enc_hidden colours
from matplotlib.patches import Patch
legend_handles = [Patch(color=eh_colors[eh], label=f"enc_hidden={eh}")
                  for eh in enc_hidden_vals]
ax.legend(handles=legend_handles, fontsize=8, loc="lower right")

fig.suptitle("Sloutsky IDRNN Hyperparameter Search", fontsize=13, fontweight="bold")
fig.tight_layout()

plot_path = os.path.join(PLOT_DIR, "hp_search_summary.png")
fig.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"\nSaved plot → {plot_path}")
