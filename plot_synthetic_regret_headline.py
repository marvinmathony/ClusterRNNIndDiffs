"""
plot_synthetic_regret_headline.py — Two minimal plots for the synthetic
three-regression study (uniform-α dataset, discrete env mixture).

Plot 1: scatter  — R3 marg IDRNN simulated regret vs true α.
Plot 2: bar      — Pearson r across 5 predictor variants, with Bayes factors
                   (JZS BF_10) annotating each bar. Same minimalism as
                   plots_thalmann/step1_cross_task_regret_bars.png.

Vanilla+h is intentionally dropped — its `v_h_aligned` is a non-principled
post-hoc heuristic.

Reads:
  plots_dataset{ID}/step1_three_regressions.npz
  data_dataset{ID}/true_parameter_values.csv
Saves:
  plots_dataset{ID}/step1_three_regressions_scatter.png
  plots_dataset{ID}/step1_three_regressions_bars.png
"""
import os, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats
from pingouin import bayesfactor_pearson


ap = argparse.ArgumentParser()
ap.add_argument("--dataset_id", type=int, default=0)
args = ap.parse_args()

DATA_DIR = f"data_dataset{args.dataset_id}"
PLOT_DIR = f"plots_dataset{args.dataset_id}"
NPZ      = os.path.join(PLOT_DIR, "step1_three_regressions.npz")

d = np.load(NPZ)
y = pd.read_csv(f"{DATA_DIR}/true_parameter_values.csv")["alphaP_list"].values

BARS = [
    ("ground truth model", "R1", d["R1_raw_mean_reward"]),  # observed regret per session
    ("IDRNN",              "R2", d["R2_idrnn"]),
    ("IDRNN",              "R3", d["R3_idrnn"]),
    ("Vanilla",            "R2", d["R2_van0"]),
    ("Vanilla",            "R3", d["R3_van0"]),
]

MODEL_COLORS = {
    "ground truth model": "#666666",
    "IDRNN":              "#4C72B0",
    "Vanilla":            "#DD8452",
}
REG_ALPHA = {"R1": 0.45, "R2": 0.70, "R3": 1.00}


def r_with_ci(x, y, n_boot=2000, seed=0):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 5:
        return np.nan, (np.nan, np.nan), 0
    xv, yv = x[m], y[m]
    r0 = float(stats.pearsonr(xv, yv)[0])
    rng = np.random.default_rng(seed)
    n = len(xv)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        a, b = xv[idx], yv[idx]
        boots[i] = np.corrcoef(a, b)[0, 1] if a.std() and b.std() else np.nan
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return r0, (float(lo), float(hi)), int(m.sum())


# ── Plot 1: scatter (R3 marg IDRNN → true α) ────────────────────────────────
fig, ax = plt.subplots(figsize=(5.5, 5))
x_hl = d["R3_idrnn"]
m = np.isfinite(x_hl) & np.isfinite(y)
ax.scatter(x_hl[m], y[m], s=22, alpha=0.55,
           color=MODEL_COLORS["IDRNN"], edgecolors="black", linewidths=0.3)
slope, intercept = np.polyfit(x_hl[m], y[m], 1)
xs = np.linspace(x_hl[m].min(), x_hl[m].max(), 100)
ax.plot(xs, slope * xs + intercept, color="#C44E52", lw=1.8, ls="--")
r, ci, n = r_with_ci(x_hl, y, seed=1)
bf = bayesfactor_pearson(r, n)
ax.text(0.04, 0.96, f"r = {r:+.3f}\nBF₁₀ = {bf:.2g}\nn = {n}",
        transform=ax.transAxes, va="top", ha="left",
        fontsize=10,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85))
ax.set_xlabel("simulated regret (R3 marg IDRNN)")
ax.set_ylabel("true α")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
out = os.path.join(PLOT_DIR, "step1_three_regressions_scatter.png")
fig.savefig(out, dpi=170, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out}")


# ── Plot 2: bar chart ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.5, 5))
records = []
for model, reg, x in BARS:
    r, ci, n = r_with_ci(x, y, seed=1)
    bf = bayesfactor_pearson(r, n)
    records.append({"model": model, "reg": reg, "r": r, "ci": ci, "n": n, "bf": bf})

xs = np.arange(len(records))
heights = [rec["r"] for rec in records]
lo_err  = [abs(rec["r"] - rec["ci"][0]) for rec in records]
hi_err  = [abs(rec["ci"][1] - rec["r"]) for rec in records]
colors  = [MODEL_COLORS[rec["model"]] for rec in records]
alphas  = [REG_ALPHA[rec["reg"]]      for rec in records]

for xi, h, c, a in zip(xs, heights, colors, alphas):
    ax.bar(xi, h, color=c, alpha=a,
           edgecolor="black", linewidth=0.6, zorder=2)
# Subtle error bars
ax.errorbar(xs, heights, yerr=[lo_err, hi_err],
            fmt="none", ecolor="#888888", elinewidth=0.7,
            capsize=0, zorder=3)

# BF annotation
for xi, rec in zip(xs, records):
    if rec["bf"] >= 100:
        txt = f"{rec['bf']:.2g}"
    else:
        txt = f"{rec['bf']:.2f}"
    yloc = rec["r"] + (hi_err[xi] if rec["r"] >= 0 else -lo_err[xi]) \
           + (0.012 if rec["r"] >= 0 else -0.012)
    va = "bottom" if rec["r"] >= 0 else "top"
    ax.text(xi, yloc, txt, ha="center", va=va, fontsize=8)

ax.axhline(0, color="grey", lw=0.7, ls=":")
ax.set_xticks([])
ax.set_ylabel("Pearson r")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Two legends side-by-side at bottom right: Model on the left, Regression on the right
model_handles = [Patch(facecolor=MODEL_COLORS[m], edgecolor="black", label=m)
                 for m in ["ground truth model", "IDRNN", "Vanilla"]]
reg_handles = [Patch(facecolor="#444444", edgecolor="black",
                     alpha=REG_ALPHA[r], label=r)
               for r in ["R1", "R2", "R3"]]
leg1 = ax.legend(handles=model_handles, loc="lower right",
                 frameon=False, fontsize=9,
                 bbox_to_anchor=(0.85, 0.0),
                 title="Model", title_fontsize=9)
ax.add_artist(leg1)
ax.legend(handles=reg_handles, loc="lower right",
          frameon=False, fontsize=9,
          bbox_to_anchor=(1.0, 0.0),
          title="Regression", title_fontsize=9)

fig.tight_layout()
out = os.path.join(PLOT_DIR, "step1_three_regressions_bars.png")
fig.savefig(out, dpi=170, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out}")

# Reference table
print("\nStats (predictor → true α, JZS BF_10):")
print(f"{'model':<10}{'reg':<5}{'r [95% CI]':>26}{'BF_10':>12}")
for rec in records:
    bf_str = f"{rec['bf']:.2g}" if rec["bf"] >= 100 else f"{rec['bf']:.2f}"
    print(f"  {rec['model']:<10}{rec['reg']:<5}"
          f"{rec['r']:+.3f} [{rec['ci'][0]:+.2f}, {rec['ci'][1]:+.2f}]   "
          f"{bf_str:>10}")
