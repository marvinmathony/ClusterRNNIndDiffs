#!/usr/bin/env python3
"""
Summarise the dezfouli vanilla AblatedRNN HP sweep over `hidden`.

Reads all metrics.json files under hp_search_results_dezfouli_vanilla/,
prints a ranked table, writes ranked_results.csv, and emits a small bar plot.

Usage:
    python summarize_hp_search_dezfouli_vanilla.py
"""

import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = "hp_search_results_dezfouli_vanilla"
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
        "hidden":           d["hidden"],
        "mean_cv_val_loss": agg["mean_cv_val_loss"],
        "std_cv_val_loss":  agg["std_cv_val_loss"],
        "mean_cv_epoch":    agg["mean_cv_selected_epoch"],
        "std_cv_epoch":     agg["std_cv_selected_epoch"],
        "n_seeds":          agg["n_seeds_evaluated"],
    })

if not rows:
    raise RuntimeError(f"No valid metrics.json files found in {RESULTS_DIR}/")

df = (pd.DataFrame(rows)
        .sort_values("mean_cv_val_loss")
        .reset_index(drop=True))
df.index += 1
df.index.name = "rank"

csv_path = os.path.join(RESULTS_DIR, "ranked_results.csv")
df.to_csv(csv_path)
print(f"Saved CSV → {csv_path}\n")
print(df.to_string())

best = df.iloc[0]
print(f"\n★  Best: hidden={int(best.hidden)}")
print(f"   mean_cv_val_loss = {best.mean_cv_val_loss:.4f} ± {best.std_cv_val_loss:.4f}  "
      f"(mean epoch {best.mean_cv_epoch:.0f})")

# ── Plot: cv_val_loss vs hidden ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.5, 4.5))
df_plot = df.sort_values("hidden")
ax.errorbar(df_plot["hidden"], df_plot["mean_cv_val_loss"],
            yerr=df_plot["std_cv_val_loss"],
            marker="o", color="#E67E22", capsize=4, linewidth=1.4,
            markersize=8, markeredgecolor="k", markeredgewidth=0.5)
best_h = int(best.hidden)
ax.scatter([best_h], [best.mean_cv_val_loss],
           s=220, marker="*", color="red", zorder=5,
           edgecolor="k", linewidth=0.6, label=f"best (h={best_h})")
ax.set_xlabel("Vanilla hidden size")
ax.set_ylabel("Mean CV val loss  (↓ better)")
ax.set_title("Dezfouli vanilla HP sweep — single-axis (hidden)")
ax.set_xticks(sorted(df_plot["hidden"].astype(int).unique()))
ax.legend(loc="best", fontsize=9)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
fig.tight_layout()
out_png = os.path.join(PLOT_DIR, "hp_search_summary_vanilla.png")
fig.savefig(out_png, dpi=150)
plt.close(fig)
print(f"Saved plot → {out_png}")
