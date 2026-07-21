#!/usr/bin/env python3
"""Assemble the Thalmann z=3 publication figure (4 panels a/b/c/e; panel d
skipped for human data).  Nature style, mirroring dezfouli/synthetic notebooks.

Inputs:
  a  final_plots/thalmann_z3/per_participant_nll.csv
  b  final_plots/thalmann_z3_full/decoding/decoding_results.csv
  c  final_plots/thalmann_z3_full/canonical/idrnn/latents_train_step1.npy (+ targets)
  e  final_plots/thalmann_z3_full/regret/step1_cross_task_regret.npz

Outputs (final_plots/thalmann_z3_full/publication_figure/):
  panel_{a,b,c,e}.{png,pdf}  +  composite.{png,pdf}
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.decomposition import PCA
from scipy import stats

sys.path.insert(0, ".")
from nature_plot_style import nature_colors
from decode_thalmann_canonical import load_targets, label_for, PERSONALITY, COMPOSITE, WM_KEYS, ALL_KEYS
try:
    from pingouin import bayesfactor_pearson
except Exception:
    bayesfactor_pearson = lambda r, n: np.nan

plt.rcParams.update({
    'pdf.fonttype': 42, 'font.family': 'sans-serif',
    'font.sans-serif': ['Nimbus Sans', 'DejaVu Sans'],
    'font.size': 8, 'axes.titlesize': 8, 'axes.titleweight': 'bold',
    'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'legend.fontsize': 7, 'legend.frameon': False, 'axes.linewidth': 1,
    'axes.spines.top': False, 'axes.spines.right': False,
})
COL_ILL_CP = nature_colors['Grey'][1]; COL_ILL_EM = nature_colors['Grey'][3]
COL_COG_CP = nature_colors['Green'][1]; COL_COG_EM = nature_colors['Green'][3]
COL_RNN_CP = nature_colors['Blue'][1]; COL_IDRNN = nature_colors['Blue'][3]
COL_VANILLA = nature_colors['Orange'][3]

# Panel a (held-out NLL) stays on the S1 nested-CV (BASE); panels b/c/e use the
# canonical full-cohort dir FULL (THAL_FULL -> pooled S1+S2 when set).
BASE = "final_plots/thalmann_z3"
FULL = os.environ.get("THAL_FULL", "final_plots/thalmann_z3_full")
OUT = f"{FULL}/publication_figure"; os.makedirs(OUT, exist_ok=True)
SINGLE = 89 / 25.4


def sig(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "n.s."


# ── Panel a — held-out NLL (all-trial) ─────────────────────────────────────────
def draw_a(ax):
    df = pd.read_csv(f"{BASE}/per_participant_nll.csv")
    bars = [("Ill-spec\nQ-CP", "nll_ill_cp", COL_ILL_CP), ("Ill-spec\nQ-EM", "nll_ill_em", COL_ILL_EM),
            ("Cog\nCP", "nll_cog_cp", COL_COG_CP), ("Cog\nEM", "nll_cog_em", COL_COG_EM),
            ("CP-RNN", "nll_cp_rnn_all", COL_RNN_CP), ("IDRNN", "nll_idrnn_all", COL_IDRNN),
            ("Vanilla", "nll_vanilla_all", COL_VANILLA)]
    xs = np.arange(len(bars))
    for x, (lab, col, c) in zip(xs, bars):
        v = df[col].dropna().values
        ax.bar(x, v.mean(), color=c, edgecolor="k", linewidth=.5, zorder=2)
        ax.errorbar(x, v.mean(), yerr=v.std()/np.sqrt(len(v)), color="k", lw=.8, capsize=2, zorder=4)
    ax.axhline(np.log(2), color="grey", ls=":", lw=.8, alpha=.7)
    ax.text(len(bars)-1, np.log(2)+.005, "log 2", fontsize=6, color="grey", ha="right", va="bottom")
    ax.set_xticks(xs); ax.set_xticklabels([b[0] for b in bars], fontsize=6, rotation=0)
    ax.set_ylabel("held-out NLL per trial"); ax.set_title("a  Held-out predictive fit", loc="left")
    ax.set_ylim(0.45, max(df["nll_ill_cp"].mean(), np.log(2)) + 0.06)


# ── Panel b — decodability.  metric="r2": LOO R^2 (>0 = signal);
#    metric="loor": LOO pred-vs-target r (negative = no-information artifact). ───
def draw_b(ax, metric="r2"):
    df = pd.read_csv(f"{FULL}/decoding/decoding_results.csv").set_index("scale").reindex(ALL_KEYS).reset_index()
    labels = [label_for(k) for k in df["scale"]]
    x = np.arange(len(labels)); w = .38
    if metric == "r2":
        ci, cv = "r2_idrnn_loo", "r2_vanilla_loo"
        ylab = "LOO $R^2$  (out-of-sample)"; title = "b  Trait / WM decodability (LOO $R^2$ > 0 = signal)"
    else:
        ci, cv = "r_idrnn_loo", "r_vanilla_loo"
        ylab = "LOO pred–target r"; title = "b  LOO prediction–target r (neg = no-info artifact)"
    ax.bar(x - w/2, df[ci], w, color=COL_IDRNN, edgecolor="k", linewidth=.4, label="IDRNN z (k=3)")
    ax.bar(x + w/2, df[cv], w, color=COL_VANILLA, edgecolor="k", linewidth=.4,
           label="Vanilla h (dim-matched, k=3)")
    ax.axhline(0, color="k", lw=.8)
    for xb in (len(PERSONALITY)-.5, len(PERSONALITY)+len(COMPOSITE)-.5):
        ax.axvline(xb, color="grey", lw=.5, ls=":", alpha=.6)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=5.5, rotation=40, ha="right")
    ax.set_ylabel(ylab); ax.set_title(title, loc="left")
    ax.legend(loc="lower left", fontsize=6)
    lo = min(df[ci].min(), df[cv].min()); hi = max(df[ci].max(), df[cv].max())
    ax.set_ylim(lo - 0.03, hi + 0.03)


# ── Panel c — IDRNN latent (best dim) vs a genuine-signal target ───────────────
def draw_c(ax, target="WM_WMU"):
    import torch
    z = np.load(f"{FULL}/canonical/idrnn/latents_train_step1.npy")
    sub = np.asarray(torch.load(f"{FULL}/canonical/idrnn/latents_idrnn_canonical.pt",
                                map_location="cpu", weights_only=False)["subids"]).astype(int)
    y = load_targets(sub)[target].values.astype(float)
    m = np.isfinite(y)
    # strongest single latent dimension for this target
    best = int(np.argmax([abs(stats.pearsonr(z[m, d], y[m])[0]) for d in range(z.shape[1])]))
    zx = z[m, best]; ym = y[m]
    flip = np.corrcoef(zx, ym)[0, 1] < 0
    zxs = -zx if flip else zx
    ax.scatter(zxs, ym, s=10, alpha=.55, color=COL_IDRNN, edgecolors="k", linewidths=.2)
    b, a = np.polyfit(zxs, ym, 1)
    xs = np.linspace(zxs.min(), zxs.max(), 50); ax.plot(xs, b*xs+a, color="#C44E52", lw=1.5, ls="--")
    r, p = stats.pearsonr(zxs, ym); rho, ps = stats.spearmanr(zxs, ym)
    ax.text(.04, .96, f"Pearson r = {r:+.2f} {sig(p)}\nSpearman ρ = {rho:+.2f} {sig(ps)}\nn = {m.sum()}",
            transform=ax.transAxes, va="top", fontsize=6.5,
            bbox=dict(facecolor="white", edgecolor="none", alpha=.8))
    ax.set_xlabel(f"IDRNN z dim {best}{' (flipped)' if flip else ''}")
    ax.set_ylabel(label_for(target).replace("\n", " "))
    ax.set_title(f"c  Latent vs {label_for(target).replace(chr(10),' ')}", loc="left")


# ── Panel e — three-regression cross-task regret (IDRNN) ───────────────────────
def _rci(x, y, seed=1, nb=2000):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 5:
        return np.nan, (np.nan, np.nan), 0
    xv, yv = x[m], y[m]; r0 = stats.pearsonr(xv, yv)[0]
    rng = np.random.default_rng(seed); n = len(xv); bs = np.empty(nb)
    for i in range(nb):
        idx = rng.integers(0, n, n); bs[i] = np.corrcoef(xv[idx], yv[idx])[0, 1]
    return float(r0), tuple(np.nanpercentile(bs, [2.5, 97.5])), int(m.sum())


# Four regret quantities (task-1 restless), in increasing marginalization:
REG_BARS = [("R1\nhuman",            "hum_regret_task1"),
            ("R2\nexact env",        "sim_regret_task1_r2_exact_single"),
            ("R3\nexact + marg RNG", "sim_regret_task1_r3_exact_margrng"),
            ("R4\nmarg env + RNG",   "sim_regret_task1_r4_marg")]
REG_ALPHA = [.4, .6, .8, 1.0]


def draw_e(ax):
    d = np.load(f"{FULL}/regret/step1_cross_task_regret.npz")
    y = d["hum_regret_task3_h10"]
    xs = np.arange(len(REG_BARS))
    for x, (lab, key), al in zip(xs, REG_BARS, REG_ALPHA):
        r, ci, n = _rci(d[key], y)
        bf = bayesfactor_pearson(r, n)
        ax.bar(x, r, color=COL_IDRNN, alpha=al, edgecolor="k", linewidth=.6, zorder=2)
        ax.errorbar(x, r, yerr=[[abs(r-ci[0])], [abs(ci[1]-r)]], color="#888", lw=.8, capsize=2, zorder=4)
        ax.text(x, r + (.02 if r >= 0 else -.02), f"BF={bf:.1f}", ha="center",
                va="bottom" if r >= 0 else "top", fontsize=6)
    ax.axhline(0, color="grey", lw=.7, ls=":")
    ax.set_xticks(xs); ax.set_xticklabels([b[0] for b in REG_BARS], fontsize=5.5)
    ax.set_ylabel("r  (sim task-1 regret → horizon regret)")
    ax.set_title("e  Cross-task transfer (IDRNN)", loc="left")


def regret_trait_table():
    """Correlate each regret quantity (task1) with held-out horizon regret AND
    the promising trait/WM candidates — does marginalization isolate signal?"""
    import torch
    d = np.load(f"{FULL}/regret/step1_cross_task_regret.npz")
    sub = d["subids"].astype(int)
    tg = load_targets(sub)
    targets = {"horizon regret (task3,h10)": d["hum_regret_task3_h10"],
               "WM updating": tg["WM_WMU"].values.astype(float),
               "WM composite": tg["WM_composite"].values.astype(float),
               "BIG5 openness": tg["BIG5_open"].values.astype(float)}
    rows = []
    print("\n=== Regret x trait correlations (Pearson r [Spearman]) ===")
    hdr = f"{'regret':<22}" + "".join(f"{t[:18]:>22}" for t in targets)
    print(hdr)
    for lab, key in [("R1 human", "hum_regret_task1"),
                     ("R2 exact single", "sim_regret_task1_r2_exact_single"),
                     ("R3 exact margRNG", "sim_regret_task1_r3_exact_margrng"),
                     ("R4 marg env+RNG", "sim_regret_task1_r4_marg")]:
        xv = d[key]; cells = []; row = {"regret": lab}
        for tname, yv in targets.items():
            m = np.isfinite(xv) & np.isfinite(yv)
            r = stats.pearsonr(xv[m], yv[m])[0]; rho = stats.spearmanr(xv[m], yv[m])[0]
            p = stats.pearsonr(xv[m], yv[m])[1]
            cells.append(f"{r:+.2f}{'*' if p<.05 else ''}[{rho:+.2f}]")
            row[tname] = r; row[tname + "_spearman"] = rho; row[tname + "_p"] = p
        rows.append(row)
        print(f"{lab:<22}" + "".join(f"{c:>22}" for c in cells))
    pd.DataFrame(rows).to_csv(f"{FULL}/regret/regret_trait_correlations.csv", index=False)
    print(f"Saved {FULL}/regret/regret_trait_correlations.csv")


def render_panel(name, fn, size):
    fig, ax = plt.subplots(figsize=size)
    fn(ax); fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/panel_{name}.{ext}", dpi=400, bbox_inches="tight")
    plt.close(fig); print(f"  panel {name} -> {OUT}/panel_{name}.png")


def main():
    have_e = os.path.exists(f"{FULL}/regret/step1_cross_task_regret.npz")
    render_panel("a", draw_a, (SINGLE*1.4, SINGLE))
    # Two panel-b variants per request: honest LOO R^2, and LOO pred-target r.
    render_panel("b_r2", lambda ax: draw_b(ax, "r2"), (SINGLE*1.8, SINGLE))
    render_panel("b_loor", lambda ax: draw_b(ax, "loor"), (SINGLE*1.8, SINGLE))
    # Panel c uses GENUINE-signal targets (LOO R^2 > 0), not the AxDep artifact.
    render_panel("c", lambda ax: draw_c(ax, "WM_WMU"), (SINGLE, SINGLE))
    render_panel("c_open", lambda ax: draw_c(ax, "BIG5_open"), (SINGLE, SINGLE))
    if have_e:
        render_panel("e", draw_e, (SINGLE, SINGLE))
        regret_trait_table()

    # composite a/b/c/e  (panel b = LOO R^2 variant)
    fig = plt.figure(figsize=(SINGLE*3.4, SINGLE*2.1))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.5, 1, 1], hspace=.55, wspace=.45,
                          left=.07, right=.985, top=.92, bottom=.12)
    draw_a(fig.add_subplot(gs[0, 0]))
    draw_b(fig.add_subplot(gs[0, 1:]), "r2")
    draw_c(fig.add_subplot(gs[1, 0]), "WM_WMU")
    draw_c(fig.add_subplot(gs[1, 1]), "BIG5_open")
    if have_e:
        draw_e(fig.add_subplot(gs[1, 2]))
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/composite.{ext}", dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"Composite -> {OUT}/composite.png")


if __name__ == "__main__":
    main()
