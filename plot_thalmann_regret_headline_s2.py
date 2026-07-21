"""
S2 variant of plot_thalmann_regret_headline.py.

Reads:
  plots_thalmann_s2/step1_cross_task_regret.npz
Saves:
  plots_thalmann_s2/step1_cross_task_regret_scatter.png
  plots_thalmann_s2/step1_cross_task_regret_bars.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats
from pingouin import bayesfactor_pearson

PLOT_DIR = "plots_thalmann_s2" + os.environ.get("S2_RUN_SUFFIX", "")
NPZ = os.path.join(PLOT_DIR, "step1_cross_task_regret.npz")
d = np.load(NPZ)
y = d["hum_regret_task3_h10"]

BARS = [
    ("Human",     "R1", d["hum_regret_task1"]),
    ("IDRNN",     "R2", d["sim_regret_task1_exact"]),
    ("IDRNN",     "R3", d["sim_regret_task1_marg"]),
    ("Vanilla",   "R2", d["sim_regret_task1_van0_exact"]),
    ("Vanilla",   "R3", d["sim_regret_task1_van0_marg"]),
    ("Vanilla+h", "R2", d["sim_regret_task1_vanH_exact"]),
    ("Vanilla+h", "R3", d["sim_regret_task1_vanH_marg"]),
]

MODEL_COLORS = {
    "Human":     "#666666",
    "IDRNN":     "#4C72B0",
    "Vanilla":   "#DD8452",
    "Vanilla+h": "#8C564B",
}
REG_ALPHA = {"R1": 0.45, "R2": 0.70, "R3": 1.00}


def r_with_sd(x, y, n_boot=2000, seed=0):
    """Pearson r with subject-bootstrap SD (B=2000, error bar = ±SD of bootstrap-r)."""
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 5:
        return np.nan, np.nan, 0
    xv, yv = x[m], y[m]
    r0 = float(stats.pearsonr(xv, yv)[0])
    rng = np.random.default_rng(seed)
    n = len(xv)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        a, b = xv[idx], yv[idx]
        boots[i] = np.corrcoef(a, b)[0, 1] if a.std() and b.std() else np.nan
    return r0, float(np.nanstd(boots)), int(m.sum())


# ── Plot 1: scatter ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.5, 5))
x_hl = d["sim_regret_task1_marg"]
m = np.isfinite(x_hl) & np.isfinite(y)
ax.scatter(x_hl[m], y[m], s=22, alpha=0.55,
           color=MODEL_COLORS["IDRNN"], edgecolors="black", linewidths=0.3)
if m.sum() >= 2 and x_hl[m].std() > 0:
    slope, intercept = np.polyfit(x_hl[m], y[m], 1)
    xs = np.linspace(x_hl[m].min(), x_hl[m].max(), 100)
    ax.plot(xs, slope * xs + intercept, color="#C44E52", lw=1.8, ls="--")
r, sd, n = r_with_sd(x_hl, y, seed=1)
bf = bayesfactor_pearson(r, n)
ax.text(0.04, 0.96, f"r = {r:+.3f}\nBF₁₀ = {bf:.2f}\nn = {n}",
        transform=ax.transAxes, va="top", ha="left",
        fontsize=10,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85))
ax.set_xlabel("simulated regret — task 1")
ax.set_ylabel("human regret — task 3 (h = 10)")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
out = os.path.join(PLOT_DIR, "step1_cross_task_regret_scatter.png")
fig.savefig(out, dpi=170, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out}")


# ── Plot 2: bar chart ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.5, 5))
records = []
for model, reg, x in BARS:
    r, sd, n = r_with_sd(x, y, seed=1)
    bf = bayesfactor_pearson(r, n)
    records.append({"model": model, "reg": reg, "r": r, "sd": sd, "n": n, "bf": bf})

xs = np.arange(len(records))
heights = [rec["r"] for rec in records]
errs    = [rec["sd"] for rec in records]
colors  = [MODEL_COLORS[rec["model"]] for rec in records]
alphas  = [REG_ALPHA[rec["reg"]]    for rec in records]

for xi, h, c, a in zip(xs, heights, colors, alphas):
    ax.bar(xi, h, color=c, alpha=a,
           edgecolor="black", linewidth=0.6, zorder=2)
ax.errorbar(xs, heights, yerr=errs,
            fmt="none", ecolor="#888888", elinewidth=0.7,
            capsize=0, zorder=3)

for xi, rec in zip(xs, records):
    txt = f"{rec['bf']:.2f}"
    yloc = rec["r"] + (errs[xi] if rec["r"] >= 0 else -errs[xi]) \
           + (0.012 if rec["r"] >= 0 else -0.012)
    va = "bottom" if rec["r"] >= 0 else "top"
    ax.text(xi, yloc, txt, ha="center", va=va, fontsize=8)

ax.axhline(0, color="grey", lw=0.7, ls=":")
ax.set_xticks([])
ax.set_ylabel("Pearson r")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

model_handles = [Patch(facecolor=MODEL_COLORS[m], edgecolor="black", label=m)
                 for m in ["Human", "IDRNN", "Vanilla", "Vanilla+h"]]
reg_handles = [Patch(facecolor="#444444", edgecolor="black",
                     alpha=REG_ALPHA[r], label=r)
               for r in ["R1", "R2", "R3"]]
leg1 = ax.legend(handles=model_handles, loc="upper right",
                 frameon=False, fontsize=9,
                 bbox_to_anchor=(1.0, 1.0),
                 title="Model", title_fontsize=9)
ax.add_artist(leg1)
ax.legend(handles=reg_handles, loc="upper right",
          frameon=False, fontsize=9,
          bbox_to_anchor=(0.82, 1.0),
          title="Regression", title_fontsize=9)

fig.tight_layout()
out = os.path.join(PLOT_DIR, "step1_cross_task_regret_bars.png")
fig.savefig(out, dpi=170, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out}")

# Reference table
print("\nStats (task1 predictor → task3 h=10, JZS BF_10):")
print(f"{'model':<11}{'reg':<5}{'r ± boot SD':>20}{'BF_10':>10}")
for rec in records:
    print(f"  {rec['model']:<11}{rec['reg']:<5}"
          f"{rec['r']:+.3f} ± {rec['sd']:.3f}   "
          f"{rec['bf']:>8.2f}")
