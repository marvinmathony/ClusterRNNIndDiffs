#!/usr/bin/env python3
"""
Summarise the Thalmann hyperparameter search.

Reads all metrics.json files from hp_search_results_thalmann/,
produces:
  - hp_search_results_thalmann/ranked_results.csv
  - plots_thalmann/hp_search_summary.png

Usage:
    python summarize_hp_search_thalmann.py
"""
import json, glob, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

RESULTS_DIR = "hp_search_results_thalmann"
PLOT_DIR    = "plots_thalmann"
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
        "step1_epochs":     d.get("step1_epochs", 1000),
        "mean_cv_val_loss": agg["mean_cv_val_loss"],
        "std_cv_val_loss":  agg["std_cv_val_loss"],
        "mean_cv_epoch":    agg["mean_cv_selected_epoch"],
        "std_cv_epoch":     agg["std_cv_selected_epoch"],
        "n_seeds":          agg["n_seeds_evaluated"],
    })

if not rows:
    raise RuntimeError("No valid metrics.json found in " + RESULTS_DIR)

df = (pd.DataFrame(rows)
        .sort_values("mean_cv_val_loss")
        .reset_index(drop=True))
df.index += 1
df.index.name = "rank"

csv_path = os.path.join(RESULTS_DIR, "ranked_results.csv")
df.to_csv(csv_path)
print(f"Saved → {csv_path}\n")
print(df[["lmbd", "z_dim", "enc_hidden", "step1_epochs",
          "mean_cv_val_loss", "std_cv_val_loss", "mean_cv_epoch"]].head(20).to_string())

best = df.iloc[0]
print(f"\n★  Best: λ={best.lmbd}, z={best.z_dim}, "
      f"h={best.hidden}, eh={best.enc_hidden}, s1={best.step1_epochs:.0f}")
print(f"   mean_cv_val_loss = {best.mean_cv_val_loss:.4f} ± {best.std_cv_val_loss:.4f}"
      f"  (mean epoch {best.mean_cv_epoch:.0f})")

enc_hidden_vals = sorted(df["enc_hidden"].unique())
z_vals          = sorted(df["z_dim"].unique())
step1_vals      = sorted(df["step1_epochs"].unique())
lmbd_vals       = sorted(df["lmbd"].unique())

palette   = ["#4477AA", "#EE6677", "#228833", "#CCBB44"]
eh_colors = {eh: palette[i % len(palette)] for i, eh in enumerate(enc_hidden_vals)}
z_markers = {z: m for z, m in zip(z_vals, ["o", "s", "^", "D"])}

fig, axes = plt.subplots(1, 3, figsize=(19, 5))

# Panel 1: CV val loss vs lambda — best over step1 per (eh, z) combo
ax = axes[0]
df_best = df.groupby(["enc_hidden", "z_dim", "lmbd"], as_index=False)["mean_cv_val_loss"].min()
for eh in enc_hidden_vals:
    for z in z_vals:
        sub = df_best[(df_best["enc_hidden"] == eh) & (df_best["z_dim"] == z)].sort_values("lmbd")
        if sub.empty:
            continue
        ax.plot(sub["lmbd"], sub["mean_cv_val_loss"],
                color=eh_colors[eh], marker=z_markers[z],
                linestyle="-", linewidth=1.2, markersize=6,
                label=f"eh={eh}, z={z}")
ax.set_xlabel("Lambda")
ax.set_ylabel("Min CV val loss over step1  (↓ better)")
ax.set_title("CV val loss vs Lambda\n(best step1_epochs per combo)")
ax.set_xscale("log")
ax.legend(fontsize=7, ncol=2, loc="upper left")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# Panel 2: heatmap — best loss per (enc_hidden × z_dim)
ax = axes[1]
pivot = df.groupby(["enc_hidden", "z_dim"])["mean_cv_val_loss"].min().unstack()
vmin = df["mean_cv_val_loss"].min()
vmax = df["mean_cv_val_loss"].quantile(0.8)
im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn_r", vmin=vmin, vmax=vmax)
ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels([f"z={z}" for z in pivot.columns])
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels([f"eh={eh}" for eh in pivot.index])
ax.set_title("Best CV val loss per (enc_hidden × z)\n(min over λ and step1_epochs)")
plt.colorbar(im, ax=ax, label="mean CV val loss")
for i, eh in enumerate(pivot.index):
    for j, z in enumerate(pivot.columns):
        val = pivot.loc[eh, z]
        if np.isfinite(val):
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=9,
                    color="white" if val > np.nanmean(pivot.values) else "black")
best_i, best_j = np.unravel_index(np.nanargmin(pivot.values), pivot.shape)
ax.add_patch(plt.Rectangle((best_j - 0.5, best_i - 0.5), 1, 1,
                             fill=False, edgecolor="gold", linewidth=3))

# Panel 3: top 15 configurations
ax = axes[2]
top    = df.head(15)
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
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

fig.suptitle("Thalmann IDRNN Hyperparameter Search\n"
             "(block_weights: restless-only NLL in step-2)",
             fontsize=12, fontweight="bold")
fig.tight_layout()
out = os.path.join(PLOT_DIR, "hp_search_summary.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved plot → {out}")

print("\n--- Effect of lambda (best combo per value) ---")
for l in lmbd_vals:
    sub = df[df["lmbd"] == l]
    if sub.empty: continue
    b = sub.iloc[0]
    print(f"  λ={l:.3f}:  best val_loss={b.mean_cv_val_loss:.4f}  "
          f"(z={b.z_dim}, eh={b.enc_hidden}, s1={b.step1_epochs:.0f})")

print("\n--- Effect of step1_epochs (best combo per value) ---")
for s1 in step1_vals:
    sub = df[df["step1_epochs"] == s1]
    if sub.empty: continue
    b = sub.iloc[0]
    print(f"  s1={s1:5.0f}:  best val_loss={b.mean_cv_val_loss:.4f}  "
          f"(λ={b.lmbd}, z={b.z_dim}, eh={b.enc_hidden})")
