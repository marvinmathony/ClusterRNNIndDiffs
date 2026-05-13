#!/usr/bin/env python3
"""
Summarise the Dezfouli hyperparameter search.

Reads all metrics.json files from hp_search_results_dezfouli/,
produces:
  - hp_search_results_dezfouli/ranked_results.csv
  - plots_dezfouli/hp_search_summary.png

Usage:
    python summarize_hp_search_dezfouli.py
"""

import json
import glob
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

RESULTS_DIR = "hp_search_results_dezfouli"
PLOT_DIR    = "plots_dezfouli"
os.makedirs(PLOT_DIR, exist_ok=True)

rows = []
for path in glob.glob(f"{RESULTS_DIR}/**/metrics.json", recursive=True):
    with open(path) as f:
        d = json.load(f)
    agg = d.get("aggregate", {})
    if "mean_cv_val_loss" not in agg:
        print(f"Skipping (missing aggregate): {path}")
        continue
    rows.append({
        "lmbd":             d["lmbd"],
        "z_dim":            d["z_dim"],
        "hidden":           d["hidden"],
        "enc_hidden":       d["enc_hidden"],
        "step1_epochs":     d.get("step1_epochs", 3000),  # default for legacy results
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
print(df[["lmbd", "z_dim", "enc_hidden", "step1_epochs",
          "mean_cv_val_loss", "std_cv_val_loss", "mean_cv_epoch"]].to_string())

best = df.iloc[0]
print(f"\n★  Best: lmbd={best.lmbd}, z={best.z_dim}, "
      f"hidden={best.hidden}, enc_hidden={best.enc_hidden}, step1_epochs={best.step1_epochs:.0f}")
print(f"   mean_cv_val_loss = {best.mean_cv_val_loss:.4f} ± {best.std_cv_val_loss:.4f}  "
      f"(mean epoch {best.mean_cv_epoch:.0f})")

enc_hidden_vals  = sorted(df["enc_hidden"].unique())
z_vals           = sorted(df["z_dim"].unique())
step1_vals       = sorted(df["step1_epochs"].unique())

palette   = ["#4477AA", "#EE6677", "#228833", "#CCBB44"]
eh_colors = {eh: palette[i] for i, eh in enumerate(enc_hidden_vals)}
z_markers = {z: m for z, m in zip(z_vals, ["o", "s", "^", "D"])}
s1_lines  = {s: ls for s, ls in zip(step1_vals, ["-", "--", ":"])}

fig, axes = plt.subplots(1, 3, figsize=(19, 5))

# Panel 1: CV val loss vs lambda — best over step1 per (eh, z) combo
ax = axes[0]
df_best_s1 = df.groupby(["enc_hidden", "z_dim", "lmbd"], as_index=False)["mean_cv_val_loss"].min()
for eh in enc_hidden_vals:
    for z in z_vals:
        sub = df_best_s1[(df_best_s1["enc_hidden"] == eh) & (df_best_s1["z_dim"] == z)].sort_values("lmbd")
        if sub.empty:
            continue
        ax.plot(
            sub["lmbd"], sub["mean_cv_val_loss"],
            color=eh_colors[eh], marker=z_markers[z],
            linestyle="-", linewidth=1.2, markersize=6,
            label=f"eh={eh}, z={z}",
        )
ax.set_xlabel("Lambda")
ax.set_ylabel("Min CV val loss over step1  (↓ better)")
ax.set_title("CV val loss vs Lambda\n(best step1_epochs per combo)")
ax.set_xscale("symlog", linthresh=0.001)
ax.legend(fontsize=7, ncol=2, loc="upper left")

# Panel 2: heatmap — best loss per (enc_hidden × z_dim), min over lambda & step1
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
ax.set_title("Best CV val loss per (enc_hidden × z)\n(min over λ and step1_epochs)")
plt.colorbar(im, ax=ax, label="mean CV val loss")
for i, eh in enumerate(pivot_best.index):
    for j, z in enumerate(pivot_best.columns):
        val = pivot_best.loc[eh, z]
        ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=9,
                color="white" if val > np.nanmean(pivot_best.values) else "black")
best_i, best_j = np.unravel_index(np.nanargmin(pivot_best.values), pivot_best.shape)
ax.add_patch(plt.Rectangle((best_j - 0.5, best_i - 0.5), 1, 1,
                             fill=False, edgecolor="gold", linewidth=3))

# Panel 3: top 15 combos (includes step1_epochs in label)
ax = axes[2]
top = df.head(15)
labels = [f"λ={r.lmbd}, z={r.z_dim}, eh={r.enc_hidden}, s1={r.step1_epochs:.0f}"
          for _, r in top.iterrows()]
colors = [eh_colors[r.enc_hidden] for _, r in top.iterrows()]
ax.barh(range(len(top)), top["mean_cv_val_loss"],
        xerr=top["std_cv_val_loss"],
        color=colors, edgecolor="white", capsize=3)
ax.set_yticks(range(len(top)))
ax.set_yticklabels(labels, fontsize=7)
ax.invert_yaxis()
ax.set_xlabel("Mean CV val loss  (↓ better)")
ax.set_title("Top 15 configurations")
legend_handles = [Patch(color=eh_colors[eh], label=f"enc_hidden={eh}")
                  for eh in enc_hidden_vals]
ax.legend(handles=legend_handles, fontsize=8, loc="lower right")

fig.suptitle("Dezfouli IDRNN Hyperparameter Search (v2: includes step1_epochs)",
             fontsize=13, fontweight="bold")
fig.tight_layout()

plot_path = os.path.join(PLOT_DIR, "hp_search_summary.png")
fig.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"\nSaved plot → {plot_path}")

# Extra: step1_epochs effect — best loss per step1, collapsing over other HPs
print("\n--- Effect of step1_epochs (best combo per value) ---")
for s1 in step1_vals:
    sub = df[df["step1_epochs"] == s1]
    if sub.empty:
        continue
    best_s1 = sub.iloc[0]
    print(f"  step1={s1:5.0f}:  best val_loss={best_s1.mean_cv_val_loss:.4f}  "
          f"(λ={best_s1.lmbd}, z={best_s1.z_dim:.0f}, eh={best_s1.enc_hidden:.0f})")
