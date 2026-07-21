#!/usr/bin/env python3
"""Append (or idempotently refresh) the representational data-scaling panels in
thalmann_results.ipynb as ONE self-contained cell that draws TWO separate figures
(WM readout, openness readout), data loaded inline, no suptitle.  Embeds the
rendered PNGs as outputs so they show without re-running the whole notebook."""
import base64, nbformat

NB = "thalmann_results.ipynb"
PNGS = ["final_plots/thalmann_z3_datascaling/repr_climb_wm.png",
        "final_plots/thalmann_z3_datascaling/repr_climb_openness.png"]
MARKER = "Representational data-scaling: IDRNN trait/WM readout climbs"

CELL = r'''# ── Representational data-scaling: IDRNN trait/WM readout climbs with more tasks ──
# Self-contained: loads each task-subset's full-cohort seed-averaged decoding (LOO R^2)
# and draws two separate panels (WM, openness): readout vs #training-tasks
# (IDRNN z vs dim-matched vanilla h). Pooled S1+S2 all-7 models supersede the S1 fallback.
import os
import numpy as np, pandas as pd
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
COL_IDRNN, COL_VANILLA, COL_TRUE = nature_colors['Blue'][3], nature_colors['Orange'][3], nature_colors['Grey'][5]

SUBSETS = {"t0": (1, "2-armed"), "t1": (1, "restless"), "t2": (1, "horizon"),
           "t01": (2, "2-arm+rest"), "t02": (2, "2-arm+horiz"), "t12": (2, "rest+horiz"),
           "t012": (3, "all 3")}
FALLBACK = {"t1": "thalmann_z3_1task_full", "t01": "thalmann_z3_full", "t012": "thalmann_z3_3task_full"}
WM = ["WM_composite", "WM_WMU", "WM_SS"]
PANELS = [("Working-memory readout", WM), ("Openness readout", ["BIG5_open"])]

def load_decode(sub):
    p = f"final_plots/thalmann_z3_ds_{sub}/decoding/seed_averaged_decoding.csv"; src = "pooled"
    if not os.path.exists(p) and sub in FALLBACK:
        p = f"final_plots/{FALLBACK[sub]}/decoding/seed_averaged_decoding.csv"; src = "S1-chain"
    return (pd.read_csv(p).set_index("target"), src) if os.path.exists(p) else (None, None)

def collect(targets):
    rows = {}
    for sub, (n, lab) in SUBSETS.items():
        df, src = load_decode(sub)
        if df is None: continue
        tk = [t for t in targets if t in df.index]
        if not tk: continue
        rows[sub] = dict(n=n, src=src,
            im=np.mean([df.loc[t,"idrnn_loo_r2_mean"] for t in tk]),
            isd=np.mean([df.loc[t,"idrnn_loo_r2_sd"] for t in tk])/np.sqrt(len(tk)),
            vm=np.mean([df.loc[t,"vanilla_loo_r2_mean"] for t in tk]),
            vsd=np.mean([df.loc[t,"vanilla_loo_r2_sd"] for t in tk])/np.sqrt(len(tk)))
    return rows

def draw(ax, targets, title):
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
    ax.set_xticks([1,2,3]); ax.set_xlabel("# training tasks")
    ax.set_ylabel("decoding LOO $R^2$"); ax.set_title(title); ax.legend(loc="upper left")

for title, targets in PANELS:
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    draw(ax, targets, title)
    fig.tight_layout(); plt.show()'''


def main():
    nb = nbformat.read(NB, as_version=4)
    outs = []
    for png in PNGS:
        try:
            outs.append(nbformat.v4.new_output(
                "display_data", data={"image/png": base64.b64encode(open(png, "rb").read()).decode()}, metadata={}))
        except Exception as e:
            print("warn: could not embed", png, e)
    # Idempotent refresh if the cell already exists.
    for c in nb.cells:
        if c.cell_type == "code" and MARKER in "".join(c.source):
            c.source = CELL; c.outputs = outs; c.execution_count = None
            nbformat.write(nb, NB); print(f"refreshed existing repr cell in {NB}"); return
    md = nbformat.v4.new_markdown_cell(
        "## Representational data-scaling — IDRNN readout climbs with #tasks\n"
        "Decoding LOO $R^2$ (leave-one-out, out-of-sample) of working-memory and "
        "openness from the IDRNN latent vs a dim-matched vanilla hidden state, as #training tasks grows "
        "(all 7 subsets of {2-armed, restless, horizon}, pooled S1+S2). The latent's "
        "value is **representational** — it climbs with more tasks while the dim-matched "
        "vanilla stays flat — even though the same latent does *not* improve predictive NLL.")
    code = nbformat.v4.new_code_cell(CELL); code.outputs = outs; code.execution_count = None
    nb.cells.extend([md, code]); nbformat.write(nb, NB)
    print(f"appended 2 cells to {NB} (now {len(nb.cells)} cells)")


if __name__ == "__main__":
    main()
