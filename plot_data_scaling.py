#!/usr/bin/env python3
"""Data-scaling: does the representational readout improve with MORE tasks?
Clean nested S1 ladder — 1 task (restless) -> 2 tasks (+2-armed) -> 3 tasks (+horizon).
Hypothesis: IDRNN's individual-difference bottleneck exploits more data (readout
climbs); the dim-matched Vanilla hidden state has no such mechanism (stays flat).

Uses seed-averaged LOO R² (10 seeds) for IDRNN and dim-matched Vanilla.
Outputs final_plots/thalmann_data_scaling.{png,pdf} + data_scaling_summary.csv
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LADDER = [(1, "final_plots/thalmann_z3_1task_full", "restless"),
          (2, "final_plots/thalmann_z3_full",       "+2-armed"),
          (3, "final_plots/thalmann_z3_3task_full",  "+horizon")]
HILITE = ["WM_WMU", "WM_composite", "WM_SS", "BIG5_open", "CEI"]
WM = ["WM_WMU", "WM_composite", "WM_SS"]
COL_I, COL_V = "#0272b2", "#ec6f00"

# load seed-averaged LOO R² per model
data = {}
for nt, base, _ in LADDER:
    f = f"{base}/decoding/seed_averaged_decoding.csv"
    if not os.path.exists(f):
        raise FileNotFoundError(f"missing {f} (run its downstream first)")
    data[nt] = pd.read_csv(f).set_index("target")
ntasks = [nt for nt, _, _ in LADDER]
labels = {nt: f"{nt} task{'s' if nt>1 else ''}\n({lab})" for nt, _, lab in LADDER}

rows = []
for arch in ("idrnn", "vanilla"):
    for tgt in data[1].index:
        for nt in ntasks:
            rows.append(dict(arch=arch, target=tgt, ntasks=nt,
                             loo_r2=float(data[nt].loc[tgt, f"{arch}_loo_r2_mean"]),
                             loo_r2_sd=float(data[nt].loc[tgt, f"{arch}_loo_r2_sd"])))
df = pd.DataFrame(rows)
df.to_csv("final_plots/thalmann_data_scaling_summary.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
# Panel 1: mean LOO R² over WM targets vs #tasks
ax = axes[0]
for arch, col, mk in [("idrnn", COL_I, "o"), ("vanilla", COL_V, "s")]:
    ys = [np.mean([data[nt].loc[t, f"{arch}_loo_r2_mean"] for t in WM]) for nt in ntasks]
    es = [np.mean([data[nt].loc[t, f"{arch}_loo_r2_sd"] for t in WM])/np.sqrt(len(WM)) for nt in ntasks]
    ax.errorbar(ntasks, ys, yerr=es, marker=mk, color=col, lw=2, capsize=3,
                label=("IDRNN z" if arch == "idrnn" else "Vanilla h (dim-matched)"))
ax.axhline(0, color="grey", lw=.8, ls=":")
ax.set_xticks(ntasks); ax.set_xticklabels([labels[nt] for nt in ntasks], fontsize=8)
ax.set_ylabel("seed-avg LOO R²  (mean over WM tasks)")
ax.set_title("Working-memory readout vs training tasks", fontweight="bold")
ax.legend(); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# Panel 2: per-target lines (IDRNN solid, Vanilla dashed)
ax = axes[1]
cmap = plt.cm.viridis(np.linspace(0, .85, len(HILITE)))
for t, c in zip(HILITE, cmap):
    ax.plot(ntasks, [data[nt].loc[t, "idrnn_loo_r2_mean"] for nt in ntasks], "-o", color=c, label=f"{t} (IDRNN)")
    ax.plot(ntasks, [data[nt].loc[t, "vanilla_loo_r2_mean"] for nt in ntasks], "--x", color=c, alpha=.6)
ax.axhline(0, color="grey", lw=.8, ls=":")
ax.set_xticks(ntasks); ax.set_xticklabels([labels[nt] for nt in ntasks], fontsize=8)
ax.set_ylabel("seed-avg LOO R²"); ax.set_title("Per-target (solid=IDRNN, dashed=Vanilla)", fontweight="bold")
ax.legend(fontsize=7, ncol=2); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

fig.suptitle("Data scaling: IDRNN readout climbs with more tasks; dim-matched Vanilla does not",
             fontweight="bold")
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"final_plots/thalmann_data_scaling.{ext}", dpi=250, bbox_inches="tight")
print("Saved final_plots/thalmann_data_scaling.png + summary")

print("\nMean WM LOO R² by #tasks:")
for arch in ("idrnn", "vanilla"):
    ys = [np.mean([data[nt].loc[t, f"{arch}_loo_r2_mean"] for t in WM]) for nt in ntasks]
    print(f"  {arch:<8} 1t={ys[0]:+.3f}  2t={ys[1]:+.3f}  3t={ys[2]:+.3f}")
