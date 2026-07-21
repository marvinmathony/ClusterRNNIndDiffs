"""plot_publication_figure_dezfouli.py — 4-panel publication figure (a/b/c/e)
for the Dezfouli replication. Panel d (env decoding) is skipped per the
replication doc since env-structure decoding doesn't apply to a single-task
human-data fit.

Aesthetics mirror synthetic_publication_panels.ipynb (Nature column widths,
nature_colors palette, paired BF brackets stacked with h_step=0.085).
Conceptual differences from synthetic:
  - Data points are PARTICIPANTS (101), not datasets (10).
  - Panel a: cog models + RNNs aggregated across all 101 participants (jittered
            dots show per-participant NLL).
  - Panel b: target is the 3-way `diag` categorical, not continuous α. The
            "RSA" subpanel uses categorical-RDM Spearman; the "decoding"
            subpanel uses LOO logistic AUC (macro) instead of ridge R².
  - Panel c: PC1(IDRNN z) vs jittered diag-int with per-group violins.
  - Panel e: per-participant R1/R2/R3 grouped by diag. Omnibus = ANOVA F per
            (model, R-type); pairwise BFs (H-D, H-B, D-B) shown as stacked
            brackets above each bar.

Inputs:
  final_plots/dezfouli_z2_spec/nested_cv_summary.json        (panel a/b/c data)
  final_plots/dezfouli_z2_spec/per_participant_nll.csv       (panel a dots)
  final_plots/dezfouli_z2_spec/latents_idrnn_train.npy       (panel c scatter)
  final_plots/dezfouli_z2_spec/three_regressions_dezfouli.npz (panel e data)
  data_dezfouli/df_train.csv                                 (diag mapping)

Outputs:
  final_plots/dezfouli_z2_spec/publication_figure/{composite,panel_*}.{png,pdf}
"""
from __future__ import annotations
import argparse, json, os, sys, logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr, f_oneway, kruskal
try:
    from pingouin import bayesfactor_ttest, bayesfactor_pearson
except Exception:
    bayesfactor_ttest = bayesfactor_pearson = None

sys.path.insert(0, ".")
from nature_plot_style import nature_colors

matplotlib.set_loglevel("error")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
logging.getLogger("matplotlib").setLevel(logging.ERROR)

plt.rcParams.update({
    "pdf.fonttype":       42,
    "font.family":        "sans-serif",
    "font.sans-serif":    ["Nimbus Sans", "DejaVu Sans"],
    "font.size":          8,
    "axes.titlesize":     8,
    "axes.titleweight":   "bold",
    "axes.labelsize":     8,
    "xtick.labelsize":    7,
    "ytick.labelsize":    7,
    "legend.fontsize":    7,
    "legend.frameon":     False,
    "axes.linewidth":     1,
    "xtick.major.width":  1,
    "ytick.major.width":  1,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.autolayout":  False,
})

MM_PER_IN     = 25.4
SINGLE_COL_IN = 89  / MM_PER_IN
DOUBLE_COL_IN = 183 / MM_PER_IN

COL_Q_CP    = nature_colors['Grey'][2]
COL_Q_EM    = nature_colors['Grey'][4]
COL_FQ_CP   = nature_colors['Green'][1]
COL_FQ_EM   = nature_colors['Green'][3]
COL_RNN_CP  = nature_colors['Blue'][1]
COL_IDRNN   = nature_colors['Blue'][3]
COL_IDRNN_H = nature_colors['Blue'][2]
COL_VANILLA = nature_colors['Orange'][3]
COL_VAN_H   = nature_colors['Skin tones'][3]
COL_VAN_H_LAST = nature_colors['Skin tones'][1]
COL_TRUE    = nature_colors['Grey'][5]

# Diagnosis group colors
COL_DIAG = {
    "Healthy":     nature_colors['Green'][3],
    "Depression":  nature_colors['Blue'][3],
    "Bipolar":     nature_colors['Orange'][3],
}
DIAG_ORDER = ["Healthy", "Depression", "Bipolar"]


def fmt_bf(bf):
    if bf is None or not np.isfinite(bf): return "—"
    return f"{bf:.1e}" if bf >= 1000 else (f"{bf:.1f}" if bf >= 10 else f"{bf:.2f}")


def bracket(ax, x1, x2, y, h, txt):
    ax.plot([x1, x2], [y + h, y + h], lw=0.5, c="black")
    ax.text((x1 + x2) / 2, y + h * 1.1, txt, ha="center", va="bottom", fontsize=6)


def paired_bf(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if bayesfactor_ttest is None or m.sum() < 3:
        return float("nan")
    t = stats.ttest_rel(a[m], b[m]).statistic
    return float(bayesfactor_ttest(t, m.sum(), paired=True))


def two_group_bf(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if bayesfactor_ttest is None or len(a) < 3 or len(b) < 3:
        return float("nan")
    t = stats.ttest_ind(a, b, equal_var=False).statistic
    nx = (len(a) + len(b)) // 2
    return float(bayesfactor_ttest(t, nx, paired=False))


# ── Panel a — held-out NLL per participant ─────────────────────────────────
def panel_a(ax, df, summary):
    """One bar per model; jittered dots per participant; reference line at
    log(2) chance."""
    keys_cols = [
        ("nll_ill_cp",  "Ill-spec. CP",  COL_Q_CP),
        ("nll_ill_em",  "Ill-spec. EM",  COL_Q_EM),
        ("nll_cog_cp",  "Cog model CP",  COL_FQ_CP),
        ("nll_cog_em",  "Cog model EM",  COL_FQ_EM),
        ("nll_idrnn",   "IDRNN",         COL_IDRNN),
        ("nll_vanilla", "Vanilla RNN",   COL_VANILLA),
    ]
    xs    = np.arange(len(keys_cols))
    means = [df[k].mean() for k, _, _ in keys_cols]
    sems  = [stats.sem(df[k].dropna()) for k, _, _ in keys_cols]
    cols  = [c for _, _, c in keys_cols]
    ax.bar(xs, means, color=cols, alpha=0.92, edgecolor="black",
           linewidth=0.5, zorder=2)
    ax.errorbar(xs, means, yerr=sems, fmt="none", ecolor="#444444",
                elinewidth=0.7, capsize=0, zorder=3)
    rng = np.random.default_rng(0)
    for i, (k, _, _) in enumerate(keys_cols):
        v = df[k].dropna().values
        if len(v) == 0: continue
        ax.scatter(xs[i] + rng.normal(0, 0.05, len(v)), v, s=8,
                   c="black", alpha=0.35, zorder=4, edgecolors="none")
    ax.axhline(np.log(2), ls=":", color=COL_TRUE, lw=1.2, zorder=1)
    ax.text(len(keys_cols) - 0.5, np.log(2) + 0.005, "log 2 (chance)",
            ha="right", va="bottom", fontsize=6, color=COL_TRUE)

    # Paired BF brackets — IDRNN vs Vanilla; Cog EM vs IDRNN
    bf_idr_van     = paired_bf(df["nll_idrnn"], df["nll_vanilla"])
    bf_cog_idr     = paired_bf(df["nll_cog_em"], df["nll_idrnn"])
    ymax = max(means) + max(sems); h_step = 0.025
    bracket(ax, 4, 5, ymax + h_step,         h_step * 0.3, fmt_bf(bf_idr_van))
    bracket(ax, 3, 4, ymax + 2.4 * h_step,   h_step * 0.3, fmt_bf(bf_cog_idr))

    ax.set_xticks(xs)
    ax.set_xticklabels([lab for _, lab, _ in keys_cols], rotation=30, ha="right")
    ax.set_ylabel("Held-out NLL per trial (101 participants)")
    ax.set_title("a", loc="left", fontweight="bold")


# ── Panel b — RSA r + LOO logistic AUC vs diag ─────────────────────────────
def panel_b(ax, summary, metric="auc"):
    """metric='auc' for LOO macro-AUC, 'rsa' for categorical-RDM Spearman r."""
    if metric == "auc":
        idr = summary["diag_decoding"]["idrnn"]["auc_macro"]
        van = summary["diag_decoding"].get("vanilla_h_pc", {}).get("auc_macro", np.nan)
        ylab = "Diag decoding (LOO macro-AUC)"
        chance = 0.5
    else:
        idr = summary["rsa"]["idrnn"]["spearman_r"]
        van = summary["rsa"].get("vanilla_h_pc", {}).get("spearman_r", np.nan)
        ylab = "RSA Spearman r  (latent dist vs diag mismatch)"
        chance = 0.0
    xs    = [0, 1]
    means = [idr, van]
    ax.bar(xs, means, color=[COL_IDRNN, COL_VANILLA], alpha=0.92,
           edgecolor="black", linewidth=0.5, zorder=2)
    ax.axhline(chance, ls=":", color=COL_TRUE, lw=1.0, zorder=1)
    ax.set_xticks(xs); ax.set_xticklabels(["IDRNN", "Vanilla h (PC1-matched)"])
    ax.set_ylabel(ylab)
    ax.set_title("b", loc="left", fontweight="bold")


# ── Panel c — PC1(IDRNN z) per diag group ──────────────────────────────────
def panel_c(ax, latents, diag):
    """If z_dim > 1, project to PC1. Plot strip+jittered dots per diag group."""
    from sklearn.decomposition import PCA
    if latents.shape[1] > 1:
        pca = PCA(n_components=1)
        z = pca.fit_transform(latents).reshape(-1)
    else:
        z = latents.reshape(-1)
    rng = np.random.default_rng(7)
    xs_by_grp = []
    means     = []
    for i, g in enumerate(DIAG_ORDER):
        m = (diag == g)
        if m.sum() == 0: continue
        ax.scatter(i + rng.normal(0, 0.08, m.sum()), z[m], s=12,
                   c=[COL_DIAG[g]], alpha=0.7, edgecolors="black", linewidth=0.3)
        ax.scatter([i], [z[m].mean()], marker="_", s=400,
                   c="black", linewidths=1.2)
        means.append(z[m].mean())
    # Pairwise BF brackets H-D, H-B, D-B (one-sample t-test pairs):
    g_h = z[diag == "Healthy"]; g_d = z[diag == "Depression"]; g_b = z[diag == "Bipolar"]
    bf_hd = two_group_bf(g_h, g_d)
    bf_hb = two_group_bf(g_h, g_b)
    bf_db = two_group_bf(g_d, g_b)
    ymax = z.max(); h_step = 0.085 * (z.max() - z.min() + 1e-3)
    bracket(ax, 0, 1, ymax,           h_step * 0.3, f"H–D {fmt_bf(bf_hd)}")
    bracket(ax, 0, 2, ymax + 1.3*h_step, h_step * 0.3, f"H–B {fmt_bf(bf_hb)}")
    bracket(ax, 1, 2, ymax + 2.6*h_step, h_step * 0.3, f"D–B {fmt_bf(bf_db)}")
    ax.set_xticks(range(len(DIAG_ORDER)))
    ax.set_xticklabels(DIAG_ORDER, rotation=0)
    ax.set_ylabel("IDRNN z (PC1)" if latents.shape[1] > 1 else "IDRNN z")
    ax.set_title("c", loc="left", fontweight="bold")


# ── Panel e — three regressions: ANOVA F + pairwise BFs by diag ────────────
def panel_e(ax, npz, anchor="R3"):
    """Per (model variant, R-type) compute the omnibus ANOVA F over diag groups
    and overlay the three pairwise BF brackets (H-D, H-B, D-B)."""
    diag = npz["diag"]
    masks = {g: (diag == g) for g in DIAG_ORDER}

    # Bars: rows = (R-type), columns = (model variant)
    R_TYPES = [
        ("R1",     "R1",               COL_TRUE,       1.00),
        ("R2 IDR", "R2_idrnn",         COL_IDRNN,      0.55),
        ("R3 IDR", "R3_idrnn",         COL_IDRNN,      1.00),
        ("R2 V+h", "R2_vanH",          COL_VAN_H,      0.55),
        ("R3 V+h", "R3_vanH",          COL_VAN_H,      1.00),
        ("R2 V0",  "R2_van0",          COL_VANILLA,    0.55),
        ("R3 V0",  "R3_van0",          COL_VANILLA,    1.00),
    ]
    # Optional Vanilla+h_lastT if present
    if "R3_vanH_lastT" in npz.files:
        R_TYPES += [
            ("R2 V+hL", "R2_vanH_lastT", COL_VAN_H_LAST, 0.55),
            ("R3 V+hL", "R3_vanH_lastT", COL_VAN_H_LAST, 1.00),
        ]

    F_stats = []
    Fpvals  = []
    bars    = []
    cols    = []
    alphas  = []
    labels  = []
    for lab, key, c, alpha in R_TYPES:
        x = npz[key].astype(float)
        groups = [x[m] for m in masks.values() if m.sum() > 0]
        groups = [g[np.isfinite(g)] for g in groups]
        if any(len(g) < 2 for g in groups):
            F_stats.append(np.nan); Fpvals.append(np.nan)
        else:
            f, p = f_oneway(*groups)
            F_stats.append(float(f)); Fpvals.append(float(p))
        bars.append(float(np.nanmean(x))); cols.append(c); alphas.append(alpha)
        labels.append(lab)
    xs = np.arange(len(bars))
    # Bar = group mean of the R value
    for xi, h, c, a in zip(xs, bars, cols, alphas):
        ax.bar(xi, h, color=c, alpha=a, edgecolor="black", linewidth=0.5, zorder=2)
    # ANOVA F as text above each bar
    ymax = max(bars) * 1.05 if bars else 1
    for xi, F in zip(xs, F_stats):
        ax.text(xi, ymax * 1.05, f"F={F:.2f}" if np.isfinite(F) else "—",
                ha="center", va="bottom", fontsize=6)
    # Pairwise BF brackets — only for the anchor variant (default R3)
    # Find indices for the R3 variants and draw 3 brackets per
    h_step = 0.085 * (max(bars) + 1e-3)
    for xi, (lab, key, c, alpha) in enumerate(R_TYPES):
        if not lab.startswith("R3"): continue
        x = npz[key].astype(float)
        bf_hd = two_group_bf(x[masks["Healthy"]],    x[masks["Depression"]])
        bf_hb = two_group_bf(x[masks["Healthy"]],    x[masks["Bipolar"]])
        bf_db = two_group_bf(x[masks["Depression"]], x[masks["Bipolar"]])
        y0 = bars[xi] * 0.65 + ymax * 0.35
        ax.text(xi, y0 + 2.0*h_step,
                f"H-D {fmt_bf(bf_hd)}\nH-B {fmt_bf(bf_hb)}\nD-B {fmt_bf(bf_db)}",
                ha="center", va="bottom", fontsize=5, color="#333333")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Mean rollout regret per participant")
    ax.set_title("e", loc="left", fontweight="bold")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hypothesis_z", type=int, default=2)
    ap.add_argument("--metric", default="step1_specificity")
    args = ap.parse_args()

    # path_tag(z=2, metric=step1_specificity) -> "_z2_spec"
    from nested_cv.config import path_tag
    tag = path_tag(args.hypothesis_z, args.metric)
    base_dir = f"final_plots/dezfouli{tag}"
    out_dir  = os.path.join(base_dir, "publication_figure")
    os.makedirs(out_dir, exist_ok=True)

    # ── Load all panel inputs ──────────────────────────────────────────────
    summary_path = os.path.join(base_dir, "nested_cv_summary.json")
    per_sub_csv  = os.path.join(base_dir, "per_participant_nll.csv")
    lat_path     = os.path.join(base_dir, "latents_idrnn_train.npy")
    npz_path     = os.path.join(base_dir, "three_regressions_dezfouli.npz")

    summary = json.load(open(summary_path)) if os.path.exists(summary_path) else None
    df      = pd.read_csv(per_sub_csv)      if os.path.exists(per_sub_csv) else None
    latents = np.load(lat_path)             if os.path.exists(lat_path) else None
    npz     = np.load(npz_path, allow_pickle=True) if os.path.exists(npz_path) else None

    diag_mapping = pd.read_csv("data_dezfouli/df_train.csv")
    diag = diag_mapping["diag"].astype(str).values

    # ── Composite (2 × 2 grid: a top-left, b top-right, c bottom-left, e bottom-right) ──
    fig = plt.figure(figsize=(DOUBLE_COL_IN, DOUBLE_COL_IN * 0.85))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.4, 1], hspace=0.55, wspace=0.4)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_e = fig.add_subplot(gs[1, 1])

    if df is not None:
        panel_a(ax_a, df, summary)
    else:
        ax_a.text(0.5, 0.5, "panel a: nested_cv_summary missing",
                  ha="center", va="center", transform=ax_a.transAxes)

    if summary is not None and "diag_decoding" in summary:
        panel_b(ax_b, summary, metric="auc")
    else:
        ax_b.text(0.5, 0.5, "panel b: diag_decoding missing",
                  ha="center", va="center", transform=ax_b.transAxes)

    if latents is not None:
        panel_c(ax_c, latents, diag[:latents.shape[0]])
    else:
        ax_c.text(0.5, 0.5, "panel c: canonical latents missing",
                  ha="center", va="center", transform=ax_c.transAxes)

    if npz is not None:
        panel_e(ax_e, npz, anchor="R3")
    else:
        ax_e.text(0.5, 0.5, "panel e: three_regressions npz missing",
                  ha="center", va="center", transform=ax_e.transAxes)

    composite_png = os.path.join(out_dir, "composite_dezfouli.png")
    composite_pdf = os.path.join(out_dir, "composite_dezfouli.pdf")
    fig.savefig(composite_png, dpi=600, bbox_inches="tight")
    fig.savefig(composite_pdf,            bbox_inches="tight")
    plt.close(fig)
    print(f"Composite -> {composite_png}\n            {composite_pdf}")

    # ── Standalone panels (one PNG+PDF each) ──────────────────────────────
    for name, fn, kwargs in [
        ("panel_a", lambda ax: panel_a(ax, df, summary)             if df is not None else None,        {}),
        ("panel_b", lambda ax: panel_b(ax, summary, metric="auc")   if summary is not None else None,   {}),
        ("panel_b_rsa", lambda ax: panel_b(ax, summary, metric="rsa") if summary is not None else None, {}),
        ("panel_c", lambda ax: panel_c(ax, latents, diag[:latents.shape[0]]) if latents is not None else None, {}),
        ("panel_e", lambda ax: panel_e(ax, npz, anchor="R3")        if npz is not None else None,       {}),
    ]:
        if fn is None: continue
        fig, ax = plt.subplots(figsize=(SINGLE_COL_IN, SINGLE_COL_IN))
        try:
            fn(ax)
            png = os.path.join(out_dir, f"{name}.png")
            pdf = os.path.join(out_dir, f"{name}.pdf")
            plt.tight_layout()
            fig.savefig(png, dpi=600, bbox_inches="tight")
            fig.savefig(pdf,            bbox_inches="tight")
            print(f"  {name} -> {png}")
        except Exception as e:
            print(f"  [skip] {name}: {type(e).__name__}: {e}")
        finally:
            plt.close(fig)


if __name__ == "__main__":
    main()
