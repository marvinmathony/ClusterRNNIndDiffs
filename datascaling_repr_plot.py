#!/usr/bin/env python3
"""Representational data-scaling: IDRNN trait/WM readout climbs as #training-tasks
grows; dim-matched vanilla stays flat.  Self-contained (loads decoding CSVs inline)
so it can be dropped straight into thalmann_results.ipynb as one cell.

Reads each subset's full-cohort seed-averaged decoding (LOO R^2):
  final_plots/thalmann_z3_ds_{sub}/decoding/seed_averaged_decoding.csv   (pooled S1+S2, all 7 subsets)
Falls back to the existing S1 chain for the 3 chain subsets until the pooled
all-7 models finish training, so the panel renders immediately.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nature_plot_style import nature_colors

plt.rcParams.update({
    "pdf.fonttype": 42, "font.family": "sans-serif",
    "font.sans-serif": ["Nimbus Sans", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 8, "axes.titleweight": "bold",
    "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "legend.frameon": False, "axes.linewidth": 1,
    "axes.spines.top": False, "axes.spines.right": False,
})
COL_IDRNN = nature_colors['Blue'][3]
COL_VANILLA = nature_colors['Orange'][3]
COL_TRUE = nature_colors['Grey'][5]

# subset -> (#training-tasks, pretty label)
SUBSETS = {"t0": (1, "2-armed"), "t1": (1, "restless"), "t2": (1, "horizon"),
           "t01": (2, "2-arm+rest"), "t02": (2, "2-arm+horiz"), "t12": (2, "rest+horiz"),
           "t012": (3, "all 3")}
# S1-chain fallback (used only until the pooled ds_{sub} models finish)
FALLBACK = {"t1": "thalmann_z3_1task_full", "t01": "thalmann_z3_full", "t012": "thalmann_z3_3task_full"}
WM = ["WM_composite", "WM_WMU", "WM_SS"]
PANELS = [("Working-memory readout", "wm", WM), ("Openness readout", "openness", ["BIG5_open"])]


def load_decode(sub):
    p = f"final_plots/thalmann_z3_ds_{sub}/decoding/seed_averaged_decoding.csv"
    src = "pooled"
    if not os.path.exists(p) and sub in FALLBACK:
        p = f"final_plots/{FALLBACK[sub]}/decoding/seed_averaged_decoding.csv"; src = "S1-chain"
    if not os.path.exists(p):
        return None, None
    return pd.read_csv(p).set_index("target"), src


def collect(targets):
    """Return per-subset {n, idrnn_mean, idrnn_sd, vanilla_mean, vanilla_sd, src}."""
    rows = {}
    for sub, (n, lab) in SUBSETS.items():
        df, src = load_decode(sub)
        if df is None:
            continue
        tk = [t for t in targets if t in df.index]
        if not tk:
            continue
        rows[sub] = dict(n=n, lab=lab, src=src,
                         im=np.mean([df.loc[t, "idrnn_loo_r2_mean"] for t in tk]),
                         isd=np.mean([df.loc[t, "idrnn_loo_r2_sd"] for t in tk]) / np.sqrt(len(tk)),
                         vm=np.mean([df.loc[t, "vanilla_loo_r2_mean"] for t in tk]),
                         vsd=np.mean([df.loc[t, "vanilla_loo_r2_sd"] for t in tk]) / np.sqrt(len(tk)))
    return rows


def draw(ax, targets):
    rows = collect(targets)
    for arch, col, off, ls, mk, lab in [("i", COL_IDRNN, -0.05, "-", "o", "IDRNN z"),
                                        ("v", COL_VANILLA, +0.05, "--", "s", "Vanilla h (dim-matched)")]:
        ns = sorted({r["n"] for r in rows.values()})
        means, errs = [], []
        for n in ns:
            vals = np.array([r[f"{arch}m"] for r in rows.values() if r["n"] == n])
            means.append(float(vals.mean()))
            errs.append(float(vals.std(ddof=1)/np.sqrt(len(vals))) if len(vals) > 1
                        else float([r[f"{arch}sd"] for r in rows.values() if r["n"] == n][0]))
            ax.scatter([n+off]*len(vals), vals, s=9, color=col, alpha=0.22, zorder=1, edgecolors="none")
        ax.errorbar([n+off for n in ns], means, yerr=errs, fmt=mk, ls=ls, color=col, lw=1.7,
                    ms=5, capsize=2.5, elinewidth=0.9, zorder=3, label=lab)
    ax.axhline(0, color=COL_TRUE, lw=.6, ls=":")
    ax.set_xticks([1, 2, 3]); ax.set_xlabel("# training tasks")
    ax.set_ylabel("decoding LOO $R^2$"); ax.legend(loc="upper left")


def main():
    os.makedirs("final_plots/thalmann_z3_datascaling", exist_ok=True)
    for title, fname, targets in PANELS:
        fig, ax = plt.subplots(figsize=(3.6, 3.0))
        draw(ax, targets); ax.set_title(title)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(f"final_plots/thalmann_z3_datascaling/repr_climb_{fname}.{ext}",
                        dpi=250, bbox_inches="tight")
        print(f"saved final_plots/thalmann_z3_datascaling/repr_climb_{fname}.png")
    for title, fname, targets in PANELS:
        print(f"\n{title}:")
        for sub, r in collect(targets).items():
            print(f"  {sub:5s} n={r['n']} [{r['src']:8s}] IDRNN={r['im']:+.3f}  Vanilla={r['vm']:+.3f}")


if __name__ == "__main__":
    main()
