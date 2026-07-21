"""
Regenerate plots_thalmann_s2/step1_vs_vanilla/step1_vs_vanilla_decoding.png
from decoding_results.csv with signed LOO bars.

LOO r can go negative when there is no real signal and the cross-validated
fit overfits — that's diagnostic information that gets erased by plotting
|r|. The in-sample panel still uses |r| because it comes from sqrt(R²)
and is non-negative by construction.

Reads:
  plots_thalmann_s2/step1_vs_vanilla/decoding_results.csv
  plots_thalmann_s2/step1_vs_vanilla/seed_summary.csv
Writes:
  plots_thalmann_s2/step1_vs_vanilla/step1_vs_vanilla_decoding.png  (overwrite)
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "plots_thalmann_s2/step1_vs_vanilla"
csv_path = os.path.join(OUT_DIR, "decoding_results.csv")
seed_path = os.path.join(OUT_DIR, "seed_summary.csv")
out_png   = os.path.join(OUT_DIR, "step1_vs_vanilla_decoding.png")

decode_df = pd.read_csv(csv_path)

# Recover best seeds from seed_summary.csv (is_best column)
seeds = pd.read_csv(seed_path)
best_idrnn_seed = int(seeds.query("model == 'IDRNN'   & is_best").iloc[0]["seed"])
best_vanilla_seed = int(seeds.query("model == 'Vanilla' & is_best").iloc[0]["seed"])

SCALE_LABELS = {
    "PANAS_PA":  "PANAS\nPos. Affect",
    "PANAS_NA":  "PANAS\nNeg. Affect",
    "STICSA":    "STICSA\nAnxiety",
    "PHQ":       "PHQ-9\nDepression",
    "CEI":       "CEI\nCuriosity",
    "BIG5_open": "BIG5\nOpenness",
    "AxDep":     "CFA\nAxDep",
    "posMood":   "CFA\nposMood",
    "negMood":   "CFA\nnegMood",
    "Exp":       "CFA\nExploration",
}


def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "n.s."


labels = [SCALE_LABELS.get(s, s) for s in decode_df["scale"]]
x = np.arange(len(labels))
w = 0.35
c1, c2 = "#4C72B0", "#DD8452"

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
panels = [
    (axes[0], "r_idrnn",     "r_vanilla",     "p_idrnn",     "p_vanilla",
     "p_steiger",     "In-sample OLS (R²-based, k-biased)",     True),
    (axes[1], "r_idrnn_loo", "r_vanilla_loo", "p_idrnn_loo", "p_vanilla_loo",
     "p_steiger_loo", "LOO RidgeCV (unbiased — true signal test)", False),
]
for ax, ci, cv, pi_, pv_, ps_, title, use_abs in panels:
    vi = decode_df[ci].abs() if use_abs else decode_df[ci]
    vv = decode_df[cv].abs() if use_abs else decode_df[cv]
    ax.bar(x - w/2, vi, w,
           color=c1, alpha=0.85, edgecolor="k", linewidth=0.5,
           label=f"IDRNN step-1 lookup (seed {best_idrnn_seed})")
    ax.bar(x + w/2, vv, w,
           color=c2, alpha=0.85, edgecolor="k", linewidth=0.5,
           label=f"Vanilla h (seed {best_vanilla_seed})")

    for i, row in decode_df.iterrows():
        ri = abs(row[ci]) if use_abs else row[ci]
        rv = abs(row[cv]) if use_abs else row[cv]
        yi = ri + 0.01 if ri >= 0 else ri - 0.01
        yv = rv + 0.01 if rv >= 0 else rv - 0.01
        ax.text(i - w/2, yi, sig_stars(row[pi_]),
                ha="center", va="bottom" if ri >= 0 else "top",
                fontsize=8, fontweight="bold", color=c1)
        ax.text(i + w/2, yv, sig_stars(row[pv_]),
                ha="center", va="bottom" if rv >= 0 else "top",
                fontsize=8, fontweight="bold", color=c2)
        y_top = max(ri, rv) + 0.07
        ax.plot([i - w/2, i - w/2, i + w/2, i + w/2],
                [y_top - 0.01, y_top, y_top, y_top - 0.01],
                color="k", linewidth=0.8)
        stars = sig_stars(row[ps_])
        ax.text(i, y_top + 0.005, stars, ha="center", va="bottom",
                fontsize=8, color=("k" if stars != "n.s." else "grey"))

    if ci == "r_idrnn":
        ch_i = decode_df["chance_r_idrnn"].mean()
        ch_v = decode_df["chance_r_vanilla"].mean()
        z_dim_i = int(round((ch_i ** 2) * (decode_df["n"].mean() - 1)))
        z_dim_v = int(round((ch_v ** 2) * (decode_df["n"].mean() - 1)))
        ax.axhline(ch_i, color=c1, ls="--", lw=1.2, alpha=0.7,
                   label=f"chance r (k={z_dim_i}) ≈ {ch_i:.3f}")
        ax.axhline(ch_v, color=c2, ls="--", lw=1.2, alpha=0.7,
                   label=f"chance r (k={z_dim_v}) ≈ {ch_v:.3f}")
    else:
        ax.axhline(0, color="grey", lw=1, alpha=0.5)

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9, rotation=20, ha="right")
    ax.set_ylabel("|r|  (decodability)" if use_abs else "r  (signed)",
                  fontsize=11)
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

axes[0].set_ylim(0, min(1.0, axes[0].get_ylim()[1] + 0.1))
loo_vals = pd.concat([decode_df["r_idrnn_loo"], decode_df["r_vanilla_loo"]])
lo, hi = float(loo_vals.min()), float(loo_vals.max())
pad = max(0.10, 0.15 * (hi - lo))
axes[1].set_ylim(min(0, lo) - pad, max(0, hi) + pad)

fig.suptitle(
    "Questionnaire decoding — step-1 lookup z vs vanilla h  "
    "(S1+S2 pooled, 238 subjects)",
    fontweight="bold",
)
fig.tight_layout()
fig.savefig(out_png, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out_png}")
