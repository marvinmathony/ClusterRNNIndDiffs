#!/usr/bin/env python3
"""Insert a shared SETUP cell + a self-contained PANEL-A cell into
thalmann_results.ipynb, mirroring synthetic_publication_panels.ipynb:
each plot loads its own data and renders in its own cell.

Panel a uses the PROPER per-fold held-out NLL (seed-averaged over 15 seeds,
final_plots/thalmann_z3_s2_pooled/per_participant_nll.csv) — not the in-sample
full-cohort composite.  Per-participant bars + strip + SEM + paired Bayes
factors (IDRNN vs CP-RNN tests whether the held-out latent helps predictive fit).
"""
import base64, os, nbformat

NB = "thalmann_results.ipynb"
SETUP_MARKER = "# === thalmann results — shared setup"
PANELA_MARKER = "# === panel a — held-out NLL per participant"

SETUP = r'''# === thalmann results — shared setup (palette / fonts / helpers) ===
import os, json, glob, logging
import numpy as np, pandas as pd
import matplotlib, matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr
try:
    from pingouin import bayesfactor_ttest, bayesfactor_pearson
except Exception:
    bayesfactor_ttest = bayesfactor_pearson = None
from nature_plot_style import nature_colors
matplotlib.set_loglevel("error")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
plt.rcParams.update({
    "pdf.fonttype": 42, "font.family": "sans-serif",
    "font.sans-serif": ["Nimbus Sans", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 8, "axes.titleweight": "bold",
    "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "legend.frameon": False, "axes.linewidth": 1,
    "axes.spines.top": False, "axes.spines.right": False,
})
COL_Q_CP   = nature_colors['Grey'][2]
COL_Q_EM   = nature_colors['Grey'][4]
COL_FQ_CP  = nature_colors['Green'][1]
COL_FQ_EM  = nature_colors['Green'][3]
COL_RNN_CP = nature_colors['Blue'][1]
COL_IDRNN  = nature_colors['Blue'][3]
COL_VANILLA= nature_colors['Orange'][3]
COL_TRUE   = nature_colors['Grey'][5]

def fmt_bf(bf):
    if bf is None or not np.isfinite(bf): return "—"
    return f"BF={bf:.1e}" if bf >= 1000 else (f"BF={bf:.1f}" if bf >= 10 else f"BF={bf:.2f}")

def paired_bf(a, b):
    """JZS BF for a paired difference (a vs b), aligned by row, NaNs dropped."""
    a = np.asarray(a, float); b = np.asarray(b, float); m = np.isfinite(a) & np.isfinite(b)
    if bayesfactor_ttest is None or m.sum() < 3: return float("nan")
    t = stats.ttest_rel(a[m], b[m]).statistic
    return float(bayesfactor_ttest(t, int(m.sum()), paired=True))

def bracket(ax, x1, x2, y, h, txt):
    ax.plot([x1, x2], [y + h, y + h], lw=0.5, c="black")
    ax.text((x1 + x2) / 2, y + h * 1.1, txt, ha="center", va="bottom", fontsize=6)
'''

PANELA = r'''# === panel a — held-out NLL per participant (marginalized+causal eval; matches synthetic panel a) ===
# Held-out NLL via compute_rnn_likelihoods_torch (causal: z from data up to t; marginalized over
# q(z|x)), per-trial. Chance = trial-weighted uniform: 2-armed -> log2, restless -> log4, weighted
# by each task's valid-trial share (tasks differ in #actions). Paired Bayes factors over participants.
NLL_CSV = "final_plots/thalmann_z3_s2_pooled/per_participant_nll_marginal.csv"
df = pd.read_csv(NLL_CSV)

# trial-weighted chance from the held-out valid-trial counts
tid = np.load("data_thalmann_s2/task_ids_per_block.npy")
nt0 = nt1 = 0.0
for _f in range(3):
    _c = np.load(f"data_thalmann_s2/fold{_f}/c_test.npy"); _v = (_c >= 0)
    nt0 += _v[:, tid == 0].sum(); nt1 += _v[:, tid == 1].sum()
chance = float((nt0 * np.log(2) + nt1 * np.log(4)) / (nt0 + nt1))

cog_keys = [("nll_ill_cp", "Ill-spec. CP", COL_Q_CP), ("nll_ill_em", "Ill-spec. EM", COL_Q_EM),
            ("nll_cog_cp", "Cog model CP", COL_FQ_CP), ("nll_cog_em", "Cog model EM", COL_FQ_EM),
            ("nll_cp_rnn_all", "CP RNN", COL_RNN_CP), ("nll_idrnn_all", "IDRNN", COL_IDRNN),
            ("nll_vanilla_all", "Vanilla RNN", COL_VANILLA)]
keys   = [k for k, _, _ in cog_keys]
labels = [l for _, l, _ in cog_keys]
cols   = [c for _, _, c in cog_keys]
means  = [df[k].mean() for k in keys]
sems   = [stats.sem(df[k].dropna()) for k in keys]
xs     = np.arange(len(labels))

bf_Q  = paired_bf(df["nll_ill_cp"],  df["nll_ill_em"])
bf_FQ = paired_bf(df["nll_cog_cp"],  df["nll_cog_em"])
bf_RNNCP_IDRNN = paired_bf(df["nll_cp_rnn_all"],  df["nll_idrnn_all"])
bf_Van_IDRNN   = paired_bf(df["nll_vanilla_all"], df["nll_idrnn_all"])
bf_IDRNN_FQ_EM = paired_bf(df["nll_cog_em"],      df["nll_idrnn_all"])

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.bar(xs, means, color=cols, alpha=0.92, edgecolor="black", linewidth=0.5, zorder=2)
ax.errorbar(xs, means, yerr=sems, fmt="none", ecolor="#444444", elinewidth=0.7, capsize=0, zorder=3)
rng = np.random.default_rng(0)
for i, k in enumerate(keys):
    v = df[k].dropna().values
    ax.scatter(xs[i] + rng.normal(0, 0.05, len(v)), v, s=12, c="black", alpha=0.12, zorder=4, edgecolors="none")
ax.axhline(chance, ls=":", color=COL_TRUE, lw=1.2, zorder=1)
ax.text(len(labels) - 0.45, chance, " chance", va="center", ha="left", fontsize=6, color=COL_TRUE)
ymax = max(means) + max(sems); h_step = 0.05 * (chance - min(means)); yb = ymax + h_step
bracket(ax, 0, 1, yb,                 h_step * 0.3, fmt_bf(bf_Q))
bracket(ax, 2, 3, yb,                 h_step * 0.3, fmt_bf(bf_FQ))
bracket(ax, 4, 5, yb,                 h_step * 0.3, fmt_bf(bf_RNNCP_IDRNN))
bracket(ax, 5, 6, yb + h_step * 1.3,  h_step * 0.3, fmt_bf(bf_Van_IDRNN))
bracket(ax, 3, 5, yb + 0.85 * h_step, h_step * 0.3, fmt_bf(bf_IDRNN_FQ_EM))
ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=35, ha="right")
ax.set_ylabel("NLL per trial")
ax.set_ylim(bottom=min(means) - 0.02, top=chance + h_step * 1.7)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("final_plots/thalmann_z3_s2_pooled/panel_a_heldout_nll.png", dpi=600, bbox_inches="tight")
plt.show()
print("NLL per trial:", {k: round(df[k].mean(), 4) for k in keys}, "| chance", round(chance, 4))
print(f"BF  ill-CP/EM={fmt_bf(bf_Q)}  cog-CP/EM={fmt_bf(bf_FQ)}  CP-RNN/IDRNN={fmt_bf(bf_RNNCP_IDRNN)}  "
      f"Van/IDRNN={fmt_bf(bf_Van_IDRNN)}  cogEM/IDRNN={fmt_bf(bf_IDRNN_FQ_EM)}")'''


def upsert(nb, marker, source, embed_png=None):
    out = []
    if embed_png and os.path.exists(embed_png):
        out = [nbformat.v4.new_output("display_data",
               data={"image/png": base64.b64encode(open(embed_png, "rb").read()).decode()}, metadata={})]
    for c in nb.cells:
        if c.cell_type == "code" and marker in "".join(c.source):
            c.source = source; c.outputs = out; c.execution_count = None
            return False        # refreshed in place
    nb.cells.append(nbformat.v4.new_code_cell(source)); nb.cells[-1].outputs = out
    return True                 # appended


if __name__ == "__main__":
    nb = nbformat.read(NB, as_version=4)
    a1 = upsert(nb, SETUP_MARKER, SETUP)
    a2 = upsert(nb, PANELA_MARKER, PANELA, embed_png="final_plots/thalmann_z3_s2_pooled/panel_a_heldout_nll.png")
    nbformat.write(nb, NB)
    print(f"setup {'appended' if a1 else 'refreshed'}; panel-a {'appended' if a2 else 'refreshed'}; {len(nb.cells)} cells")
