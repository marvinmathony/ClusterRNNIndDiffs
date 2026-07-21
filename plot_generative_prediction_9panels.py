#!/usr/bin/env python3
"""Generative-simulation prediction: 9 plots = {horizon regret, working memory,
openness} x {task0+task1 combined, task0 only, task1 only}, each with all four
regressions R1/R2/R3/R4.  POOLED S1+S2 model.

Predictor per subject:
  R1 = human regret (observed)            R2 = exact-env single seed
  R3 = exact-env marginalised over RNG    R4 = marginalised over env + RNG
Source "combined" = mean of z-scored(task0 regret) and z-scored(task1 regret)
(scale-fair; Pearson r is scale-invariant, so single-task plots are raw regret).
Same subject mask per target across the three source plots (comparable N).

Stat per bar: Pearson r + 95% bootstrap CI + JZS BF10.
Outputs: {THAL_FULL}/regret/generative_prediction/{target}__{source}.{png,pdf}
         + a 3x3 composite.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
try:
    from pingouin import bayesfactor_pearson
except Exception:
    bayesfactor_pearson = lambda r, n: np.nan

sys.path.insert(0, ".")
from decode_thalmann_canonical import load_targets

FULL = os.environ.get("THAL_FULL", "final_plots/thalmann_z3_s2_full")
OUT = f"{FULL}/regret/generative_prediction"; os.makedirs(OUT, exist_ok=True)
d = np.load(f"{FULL}/regret/step1_cross_task_regret.npz")
subids = d["subids"].astype(int)
tg = load_targets(subids)

COL_I = "#0272b2"
REG_ALPHA = {"R1": .40, "R2": .60, "R3": .80, "R4": 1.0}
REG_LABEL = {"R1": "R1\nhuman", "R2": "R2\nexact·1seed",
             "R3": "R3\nexact·margRNG", "R4": "R4\nmarg env+RNG"}
# per-task regret arrays for each regression
TASK = {
    "task0": {"R1": d["hum_regret_task0"], "R2": d["sim_regret_task0_r2_exact_single"],
              "R3": d["sim_regret_task0_r3_exact_margrng"], "R4": d["sim_regret_task0_r4_marg"]},
    "task1": {"R1": d["hum_regret_task1"], "R2": d["sim_regret_task1_r2_exact_single"],
              "R3": d["sim_regret_task1_r3_exact_margrng"], "R4": d["sim_regret_task1_r4_marg"]},
}
TARGETS = {
    "horizon_regret": ("Held-out horizon regret", d["hum_regret_task3"]),
    "working_memory": ("Working memory (composite)", tg["WM_composite"].values.astype(float)),
    "openness":       ("BIG5 openness", tg["BIG5_open"].values.astype(float)),
}
SOURCES = ["both", "task0", "task1"]
SRC_TITLE = {"both": "task0 + task1", "task0": "task0 only", "task1": "task1 only"}
REGS = ["R1", "R2", "R3", "R4"]


def _z(x, m):
    z = np.full_like(x, np.nan, dtype=float)
    z[m] = (x[m] - x[m].mean()) / (x[m].std() + 1e-12)
    return z


def predictor(source, reg, mask):
    if source == "both":
        return 0.5 * (_z(TASK["task0"][reg], mask) + _z(TASK["task1"][reg], mask))
    return TASK[source][reg]


def r_ci_bf(x, y, n_boot=3000, seed=1):
    m = np.isfinite(x) & np.isfinite(y)
    xv, yv = x[m], y[m]; n = len(xv)
    r = float(stats.pearsonr(xv, yv)[0]); p = float(stats.pearsonr(xv, yv)[1])
    rng = np.random.default_rng(seed); bs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n); bs[i] = np.corrcoef(xv[idx], yv[idx])[0, 1]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return r, p, (lo, hi), n, float(bayesfactor_pearson(r, n))


def target_mask(yv):
    # subjects with finite target AND finite task0+task1 regret (all regressions)
    m = np.isfinite(yv)
    for t in ("task0", "task1"):
        for reg in REGS:
            m &= np.isfinite(TASK[t][reg])
    return m


def draw(ax, target_key, source, title=True):
    tname, yv = TARGETS[target_key]
    mask = target_mask(yv)
    xs = np.arange(len(REGS)); recs = []
    for reg in REGS:
        x = predictor(source, reg, mask)
        r, p, ci, n, bf = r_ci_bf(x, yv)
        recs.append((reg, r, ci, bf, p))
        ax.bar(xs[REGS.index(reg)], r, color=COL_I, alpha=REG_ALPHA[reg],
               edgecolor="k", linewidth=.6, zorder=2)
        ax.errorbar(xs[REGS.index(reg)], r, yerr=[[abs(r-ci[0])], [abs(ci[1]-r)]],
                    color="#888", lw=.8, capsize=2, zorder=4)
        star = "*" if p < .05 else ""
        ax.text(xs[REGS.index(reg)], r + (.015 if r >= 0 else -.015),
                f"{r:+.2f}{star}\nBF{bf:.1f}", ha="center",
                va="bottom" if r >= 0 else "top", fontsize=6)
    ax.axhline(0, color="grey", lw=.7, ls=":")
    ax.set_xticks(xs); ax.set_xticklabels([REG_LABEL[r] for r in REGS], fontsize=6)
    ax.set_ylabel("Pearson r")
    if title:
        ax.set_title(f"{tname}\nfrom {SRC_TITLE[source]} (n={int(mask.sum())})", fontsize=8)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    return recs


def main():
    plt.rcParams.update({"font.size": 8, "axes.titleweight": "bold"})
    summary = []
    for tk in TARGETS:
        for src in SOURCES:
            fig, ax = plt.subplots(figsize=(3.6, 3.2))
            recs = draw(ax, tk, src)
            fig.tight_layout()
            for ext in ("png", "pdf"):
                fig.savefig(f"{OUT}/{tk}__{src}.{ext}", dpi=300, bbox_inches="tight")
            plt.close(fig)
            for reg, r, ci, bf, p in recs:
                summary.append(dict(target=tk, source=src, regression=reg,
                                    r=round(r, 3), p=round(p, 4), bf=round(bf, 2),
                                    ci_lo=round(ci[0], 3), ci_hi=round(ci[1], 3)))
            print(f"  {tk:<16} {src:<6} " +
                  "  ".join(f"{reg}={r:+.2f}{'*' if p<.05 else ''}" for reg, r, _, _, p in recs))

    # 3x3 composite (rows = targets, cols = sources)
    fig, axes = plt.subplots(3, 3, figsize=(11, 9))
    for i, tk in enumerate(TARGETS):
        for j, src in enumerate(SOURCES):
            draw(axes[i, j], tk, src)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/composite_3x3.{ext}", dpi=250, bbox_inches="tight")
    plt.close(fig)
    import pandas as pd
    pd.DataFrame(summary).to_csv(f"{OUT}/generative_prediction_summary.csv", index=False)
    print(f"\nSaved 9 panels + composite_3x3 + summary -> {OUT}/")


if __name__ == "__main__":
    main()
