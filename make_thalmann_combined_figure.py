#!/usr/bin/env python3
"""Standalone builder for the combined Thalmann journal figure (NHB). Run: python make_thalmann_combined_figure.py"""
import matplotlib
matplotlib.use("Agg")


# === thalmann results — shared setup (palette / fonts / helpers) ===
import os, json, glob, logging
import numpy as np, pandas as pd
import matplotlib, matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr
from cycler import cycler
from matplotlib.colors import to_rgb, to_hex
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
    "font.size": 7, "axes.titlesize": 8, "axes.titleweight": "bold",
    "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "legend.frameon": False, "axes.linewidth": 1,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.prop_cycle": cycler("color", [
        "#dd9e9e", "#D04C4C", "#f59e57", "#6f6484", "#101e41", "#fccc9b", "#666666",
    ]),
})
MM = 1 / 25.4
W_FULL   = 180 * MM   # 7.087 in — full page width (hard max)
W_SINGLE = 89  * MM   # 3.504 in — single column

# constrained_layout fits everything WITHIN the figsize instead of overflowing it
plt.rcParams["figure.constrained_layout.use"] = True

def save_fig(fig, path, dpi=600, max_mm=180):
    """Save at the fig's exact size (no tight-crop) and assert the width cap."""
    w_in, h_in = fig.get_size_inches()
    assert w_in * 25.4 <= max_mm + 0.5, f"{w_in*25.4:.1f} mm > {max_mm} mm: {path}"
    fig.savefig(path, dpi=dpi)          # NO bbox_inches="tight"

def lighten(color, amount=0.5):
    c = np.array(to_rgb(color)); white = np.array([1, 1, 1])
    return to_hex(c + (white - c) * amount)

# Palette mirrors synthetic_publication_panels.ipynb (Nature style, manual hexes).
# Light/dark pairings (CP=light, EM=dark) preserved so panel-a pairs read.
COL_Q_EM       = "#6f64842c"                       # dark grey-blue  (ill-spec EM)
COL_Q_CP       = lighten(COL_Q_EM, 0.4)            # light grey-blue (ill-spec CP)
COL_FQ_CP      = "#dd9e9e"                          # light red       (cog model CP)
COL_FQ_EM      = "#D04C4C"                          # mid-dark red    (cog model EM)
COL_IDRNN      = "#132551"                          # nature navy     (IDRNN)
COL_RNN_CP     = lighten(COL_IDRNN, 0.2)            # light blue      (CP RNN)
COL_IDRNN_H    = nature_colors['Blue'][2]           # mid blue        (IDRNN h)
COL_VANILLA    = "#f59e57"                          # nature orange   (Vanilla)
COL_VAN_H      = "#fccc9b"                           # light orange    (Vanilla+h)
COL_VAN_H_LAST = nature_colors['Skin tones'][1]     # light brown     (Vanilla+h last)
COL_TRUE       = nature_colors['Grey'][5]           # near-black

def fmt_bf(bf):
    if bf is None or not np.isfinite(bf): return "—"
    return f"{bf:.1e}" if bf >= 1000 else (f"{bf:.1f}" if bf >= 10 else f"{bf:.2f}")

def paired_bf(a, b):
    """JZS BF for a paired difference (a vs b), aligned by row, NaNs dropped."""
    a = np.asarray(a, float); b = np.asarray(b, float); m = np.isfinite(a) & np.isfinite(b)
    if bayesfactor_ttest is None or m.sum() < 3: return float("nan")
    t = stats.ttest_rel(a[m], b[m]).statistic
    return float(bayesfactor_ttest(t, int(m.sum()), paired=True))

def bracket(ax, x1, x2, y, h, txt):
    ax.plot([x1, x2], [y + h, y + h], lw=0.5, c="black")
    ax.text((x1 + x2) / 2, y + h * 1.1, txt, ha="center", va="bottom", fontsize=6)



# === COMBINED JOURNAL FIGURE (Nature Human Behaviour) ===
#   a  reward structures      b  LOO R²        c  NLL          d  cross-task regret
#   e  trait axis (PC1)       f  PC1↔switch    g  openness scaling   h  trait signal beyond WM
# All panels are NATIVE subplots in ONE figure at W_FULL (180 mm) with uniform journal typography
# (5–7 pt). Saved WITHOUT bbox_inches="tight" so figsize stays authoritative under the globally
# enabled constrained_layout. Reads only light data files (CSV/NPZ/NPY); never runs seed-walking cells.
import matplotlib.gridspec as gridspec
import pingouin as pg
MM = 1 / 25.4
W_FULL = 180 * MM     # NHB double-column hard max

FS_BASE, FS_LABEL, FS_TICK, FS_ANNOT, FS_LETTER = 6, 6, 6, 5, 8
FS_BFNUM = 5          # dense per-bar BF labels (trait axis, 14 bars)
JOURNAL_RC = {
    "font.size": FS_BASE, "axes.titlesize": FS_BASE, "axes.labelsize": FS_LABEL,
    "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK, "legend.fontsize": FS_ANNOT,
    "mathtext.default": "regular",
}

def _bracket(ax, x1, x2, y, h, txt, fs=FS_ANNOT):
    ax.plot([x1, x2], [y + h, y + h], lw=0.6, c="black")
    ax.text((x1 + x2) / 2, y + h * 1.1, txt, ha="center", va="bottom", fontsize=fs)

# ---------- Panel a helpers: reward-structure schematic (native redraw of the TikZ figure) ----------
_ARM = {"red": (237/255, 83/255, 80/255), "cyan": (38/255, 198/255, 218/255),
        "green": (102/255, 205/255, 109/255), "purple": (171/255, 99/255, 205/255)}

def _rw_walk(n, seed):
    r = np.random.default_rng(seed)
    mu = np.zeros(n); mu[0] = r.uniform(35, 65)
    for t in range(1, n):
        mu[t] = 0.9836 * mu[t - 1] + (1 - 0.9836) * 50 + r.normal(0, 2.8)
    return mu

def _reward_frame(ax, title, xlim, xticks, subtitle):
    ax.set_xlim(*xlim); ax.set_xticks(xticks)
    ax.set_ylim(25, 75); ax.set_yticks([30, 50, 70])
    ax.set_xlabel("trial", labelpad=2)
    ax.set_title(title, loc="center", fontweight="normal", fontsize=FS_TICK, pad=4)
    ax.tick_params(length=3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.text(0.5, -0.22, subtitle, transform=ax.transAxes, ha="center", va="top",
            fontsize=FS_ANNOT, color="black")

def _draw_reward_structures(ax1, ax2, ax3):
    ax1.axvspan(1, 4.5, color=(0.93, 0.93, 0.93), lw=0, zorder=0)
    ax1.axvline(4.5, ls=(0, (4, 3)), color=(0, 0, 0, 0.5), lw=0.8, zorder=1)
    ax1.plot([1, 10], [56, 56], color=_ARM["red"],  lw=1.0, solid_capstyle="round", zorder=2)
    ax1.plot([1, 10], [44, 44], color=_ARM["cyan"], lw=1.0, solid_capstyle="round", zorder=2)
    ax1.annotate("", xy=(9.28, 55.2), xytext=(9.28, 44.8),
                 arrowprops=dict(arrowstyle="<->", color=(0, 0, 0, 0.6), lw=0.8))
    ax1.text(9.05, 50, r"$\Delta$", ha="right", va="center", fontsize=FS_ANNOT, color="black")
    ax1.text(2.75, 33, "forced", rotation=90, ha="center", va="center", fontsize=FS_ANNOT, color="black")
    ax1.text(5.1, 32.5, "1 or 6 free\nchoices", ha="left", va="center", fontsize=FS_ANNOT, color="black")
    _reward_frame(ax1, "Horizon task", (1, 10), [1, 5, 10], "stable, $\\Delta$ = 4–30\n80 rounds")
    ax1.set_ylabel("mean reward", labelpad=2)
    _rng = np.random.default_rng(0)
    stable = 46.0 * np.ones(10); drift = np.zeros(10); drift[0] = 52
    for t in range(1, 10): drift[t] = drift[t - 1] + _rng.normal(0, 3.0)
    tt = np.arange(1, 11)
    ax2.plot(tt, stable, color=_ARM["cyan"], lw=1.0, solid_capstyle="round", zorder=2)
    ax2.plot(tt, drift,  color=_ARM["red"],  lw=1.0, solid_capstyle="round", zorder=2)
    ax2.text(1.2, 45, "stable",   ha="left", va="top",    fontsize=FS_ANNOT, color=tuple(0.6 * c for c in _ARM["cyan"]))
    ax2.text(1.2, 54, "drifting", ha="left", va="bottom", fontsize=FS_ANNOT, color=tuple(0.75 * c for c in _ARM["red"]))
    _reward_frame(ax2, "Two-armed bandit", (1, 10), [1, 5, 10], "stable and/or drifting\n30 rounds")
    tr = np.arange(1, 201)
    for k, col in zip(range(4), ["red", "cyan", "green", "purple"]):
        ax3.plot(tr, _rw_walk(200, 9 * 4 + k), color=_ARM[col], lw=0.6, solid_capstyle="round", zorder=2)
    _reward_frame(ax3, "Restless bandit", (1, 200), [1, 100, 200], "4 drifting arms\n1 round")

# ---------- data shared by e–h (from poster panel 2) ----------
DEC = "final_plots/thalmann_z3_3task_full/decoding"
zc = np.load(f"{DEC}/z2_consensus_pca.npy")
tg = pd.read_csv(f"{DEC}/trait_targets.csv")
PERSONALITY = ["PANAS_PA", "PANAS_NA", "STICSA", "PHQ", "CEI", "BIG5_open"]
COMPOSITE   = ["AxDep", "posMood", "negMood", "Exp"]
WM_KEYS     = ["WM_composite", "WM_OS", "WM_SS", "WM_WMU"]
ALL_KEYS = PERSONALITY + COMPOSITE + WM_KEYS
LBL = {"PANAS_PA": "PANAS +", "PANAS_NA": "PANAS −", "STICSA": "STICSA", "PHQ": "PHQ", "CEI": "CEI",
       "BIG5_open": "Openness", "AxDep": "Anx/Dep", "posMood": "pos mood", "negMood": "neg mood",
       "Exp": "Exploration", "WM_composite": "WM comp", "WM_OS": "WM OS", "WM_SS": "WM SS", "WM_WMU": "WM upd"}
def corr_n(y):
    m = np.isfinite(zc) & np.isfinite(y); return float(pearsonr(zc[m], y[m])[0]), int(m.sum())
_c3 = np.load("data_thalmann_3task_full/c_train.npy")
_tid3 = np.load("data_thalmann_3task_full/task_ids_per_block.npy")
_N3 = _c3.shape[0]
def _switch_rate(bl):
    s = p = 0
    for b in range(bl.shape[0]):
        ch = bl[b]; ch = ch[ch >= 0]
        if len(ch) < 2: continue
        s += int((ch[1:] != ch[:-1]).sum()); p += len(ch) - 1
    return s / p if p else np.nan
sw_all = np.array([_switch_rate(_c3[i]) for i in range(_N3)])

with plt.rc_context(JOURNAL_RC):
    fig = plt.figure(figsize=(W_FULL, 7.4))
    outer = fig.add_gridspec(4, 1, height_ratios=[1.0, 0.95, 0.95, 0.95], hspace=0.55)

    # ===== Row 1: a — reward structures (3 sub-panels, narrower) + c — NLL (wider) =====
    gs0 = outer[0].subgridspec(1, 100, wspace=0.0)
    ax_r1  = fig.add_subplot(gs0[0, 0:16])
    ax_r2  = fig.add_subplot(gs0[0, 18:34])
    ax_r3  = fig.add_subplot(gs0[0, 36:52])
    ax_nll = fig.add_subplot(gs0[0, 64:100])
    _draw_reward_structures(ax_r1, ax_r2, ax_r3)

    # ===== Row 2: b — LOO R² (wider) + d — regret (wider) =====
    gs1 = outer[1].subgridspec(1, 100, wspace=0.0)
    ax_loo = fig.add_subplot(gs1[0, 0:64])
    ax_reg = fig.add_subplot(gs1[0, 73:100])

    # ===== Row 3: e — trait axis (narrower) + f — PC1↔switch (wider) =====
    gs2 = outer[2].subgridspec(1, 100, wspace=0.0)
    ax_trait = fig.add_subplot(gs2[0, 0:60])
    ax_sw    = fig.add_subplot(gs2[0, 69:100])

    # ===== Row 4: g — openness scaling (wider) + h — beyond WM (wider) =====
    gs3 = outer[3].subgridspec(1, 100, wspace=0.0)
    ax_scale = fig.add_subplot(gs3[0, 0:47])
    ax_wm    = fig.add_subplot(gs3[0, 56:100])

    # -------- Panel c: held-out NLL per participant --------
    df = pd.read_csv("final_plots/thalmann_z3_s2_pooled/per_participant_nll_marginal.csv")
    tid = np.load("data_thalmann_s2/task_ids_per_block.npy")
    nt0 = nt1 = 0.0
    for _f in range(3):
        _cc = np.load(f"data_thalmann_s2/fold{_f}/c_test.npy"); _v = (_cc >= 0)
        nt0 += _v[:, tid == 0].sum(); nt1 += _v[:, tid == 1].sum()
    chance = float((nt0 * np.log(2) + nt1 * np.log(4)) / (nt0 + nt1))
    cog_keys = [("nll_ill_cp", "Ill-spec. CP", COL_Q_CP), ("nll_ill_em", "Ill-spec. EM", COL_Q_EM),
                ("nll_cog_cp", "Cog model CP", COL_FQ_CP), ("nll_cog_em", "Cog model EM", COL_FQ_EM),
                ("nll_cp_rnn_all", "CP RNN", COL_RNN_CP), ("nll_idrnn_all", "IDRNN", COL_IDRNN),
                ("nll_vanilla_all", "Vanilla RNN", COL_VANILLA)]
    keys = [k for k, _, _ in cog_keys]; labels = [l for _, l, _ in cog_keys]; cols = [c for _, _, c in cog_keys]
    means = [df[k].mean() for k in keys]; sems = [stats.sem(df[k].dropna()) for k in keys]; xs = np.arange(len(labels))
    bf_Q = paired_bf(df["nll_ill_cp"], df["nll_ill_em"]); bf_FQ = paired_bf(df["nll_cog_cp"], df["nll_cog_em"])
    bf_RNNCP_IDRNN = paired_bf(df["nll_cp_rnn_all"], df["nll_idrnn_all"])
    bf_Van_IDRNN = paired_bf(df["nll_vanilla_all"], df["nll_idrnn_all"])
    bf_IDRNN_FQ_EM = paired_bf(df["nll_cog_em"], df["nll_idrnn_all"])
    ax_nll.bar(xs, means, color=cols, alpha=0.92, edgecolor="black", linewidth=0.5, zorder=2)
    ax_nll.errorbar(xs, means, yerr=sems, fmt="none", ecolor="#444444", elinewidth=0.7, capsize=0, zorder=3)
    ax_nll.axhline(chance, ls=":", color=COL_TRUE, lw=1.2, zorder=1)
    ymax = max(means) + max(sems); h_step = 0.05 * (chance - min(means)); yb = ymax + h_step
    _bracket(ax_nll, 0, 1, yb - 0.02, h_step * 0.3, fmt_bf(bf_Q))
    _bracket(ax_nll, 2, 3, yb - 0.02, h_step * 0.3, fmt_bf(bf_FQ))
    _bracket(ax_nll, 4, 5, yb - 3 * h_step - 0.02, h_step * 0.3, fmt_bf(bf_RNNCP_IDRNN))
    _bracket(ax_nll, 5, 6, yb - h_step * 1.8 - 0.02, h_step * 0.3, fmt_bf(bf_Van_IDRNN))
    _bracket(ax_nll, 3, 5, yb - 0.85 * h_step - 0.02, h_step * 0.3, fmt_bf(bf_IDRNN_FQ_EM))
    ax_nll.set_xticks(xs); ax_nll.set_xticklabels(labels, rotation=35, ha="right")
    ax_nll.set_ylabel("Avg. Trial NLL per Participant"); ax_nll.set_ylim(bottom=0.5, top=0.76)

    # -------- Panel d: generative cross-task regret (R1-R4) --------
    dd = np.load("final_plots/thalmann_z3_s2_full/regret/step1_cross_task_regret.npz")
    subids = dd["subids"].astype(int); yv_ = dd["hum_regret_task3"]
    ab = pd.read_csv("data/final2armedBanditSession1.csv"); o = {}
    for sid, grp in ab.groupby("ID"):
        arr = grp[["reward1", "reward2"]].values.astype(float); o[int(sid)] = float((arr.max(1) - grp["reward"].values).mean())
    R1 = np.array([o.get(int(sid), np.nan) for sid in subids])
    BARS = [("R1", R1, COL_TRUE, 1.00), ("R2", dd["sim_regret_task0_r2_exact_single"], COL_IDRNN, 0.45),
            ("R3", dd["sim_regret_task0_r3_exact_margrng"], COL_IDRNN, 0.70), ("R4", dd["sim_regret_task0_r4_marg"], COL_IDRNN, 1.00)]
    recs = []; rng = np.random.default_rng(1)
    for lab, xx, col, al in BARS:
        m = np.isfinite(xx) & np.isfinite(yv_); xv, yv = xx[m], yv_[m]; nbt = len(xv)
        r = float(pearsonr(xv, yv)[0]); boots = np.empty(2000)
        for i in range(2000):
            idx = rng.integers(0, nbt, nbt); boots[i] = np.corrcoef(xv[idx], yv[idx])[0, 1]
        recs.append(dict(color=col, alpha=al, r=r, boots=boots, bf=float(bayesfactor_pearson(r, nbt))))
    xs = np.arange(4); heights = [r["r"] for r in recs]; sd_err = [float(np.std(r["boots"])) for r in recs]
    for xi, r in zip(xs, recs):
        ax_reg.bar(xi, r["r"], color=r["color"], alpha=r["alpha"], edgecolor="black", linewidth=0.5, zorder=2)
    ax_reg.errorbar(xs, heights, yerr=sd_err, fmt="none", ecolor="#444444", elinewidth=0.7, capsize=0, zorder=3)
    allb = np.concatenate([r["boots"] for r in recs]); y_top = float(np.nanpercentile(allb, 99)) + 0.04
    for xi, r in zip(xs, recs):
        ax_reg.text(xi, y_top, fmt_bf(r["bf"]), ha="center", fontsize=FS_ANNOT, color="#333333")
    mm = np.isfinite(R1) & np.isfinite(yv_) & np.isfinite(dd["sim_regret_task0_r4_marg"])
    _zc = lambda a: (np.asarray(a, float) - np.asarray(a, float).mean()) / (np.asarray(a, float).std() + 1e-12)
    bf_cmp = paired_bf(_zc(R1[mm]) * _zc(yv_[mm]), _zc(dd["sim_regret_task0_r4_marg"][mm]) * _zc(yv_[mm]))
    _bracket(ax_reg, 0, 3, y_top + 0.06, 0.03, fmt_bf(bf_cmp))
    ax_reg.axhline(0, color="grey", lw=0.5, zorder=1); ax_reg.set_xticks(xs)
    ax_reg.set_xticklabels(["Observed", "No Marg.", "Noise Marg.", "Env.+Noise Marg."], rotation=25, ha="right")
    ax_reg.set_ylabel("Pearson r (bandit 1 → horizon)", labelpad=2)
    ax_reg.set_ylim(min(0, min(heights)) - 0.05, y_top + 0.26)

    # -------- Panel b: out-of-sample LOO R² --------
    FULL = "final_plots/thalmann_z3_3task_full"
    lbls = [LBL[k] for k in ALL_KEYS]; x = np.arange(len(lbls)); w = 0.38
    bt = pd.read_csv(f"{FULL}/decoding/loo_r2_traintest_bootstrap_perfold.csv").set_index("target").reindex(ALL_KEYS)
    ax_loo.bar(x - w / 2, bt["idrnn_mean"].values, w, yerr=bt["idrnn_sd"].values, color=COL_IDRNN,
               edgecolor="black", linewidth=0.4, capsize=1.5, error_kw=dict(elinewidth=0.7, ecolor="#444444"),
               label="IDRNN z (3-dim)", zorder=2)
    ax_loo.bar(x + w / 2, bt["vanilla_mean"].values, w, yerr=bt["vanilla_sd"].values, color=COL_VANILLA,
               edgecolor="black", linewidth=0.4, capsize=1.5, error_kw=dict(elinewidth=0.7, ecolor="#444444"),
               label="Vanilla h (dim-matched)", zorder=2)
    ax_loo.axhline(0, color=COL_TRUE, lw=0.8, zorder=1)
    for xb in (len(PERSONALITY) - .5, len(PERSONALITY) + len(COMPOSITE) - .5):
        ax_loo.axvline(xb, color="grey", lw=.5, ls=":", alpha=.5)
    ax_loo.set_xticks(x); ax_loo.set_xticklabels(lbls, rotation=45, ha="right")
    ax_loo.set_ylabel("LOO $R^2$"); ax_loo.legend(loc="upper left")

    # -------- Panel e: individual-difference axis (14 traits) --------
    labels_e = [LBL[k] for k in ALL_KEYS]; xe = np.arange(len(labels_e))
    r_e = np.array([corr_n(tg[k].values)[0] for k in ALL_KEYS])
    bf_e = [float(bayesfactor_pearson(*corr_n(tg[k].values))) for k in ALL_KEYS]
    ax_trait.bar(xe, r_e, 0.6, color=COL_IDRNN, edgecolor="black", linewidth=0.4, zorder=2)
    ax_trait.axhline(0, color=COL_TRUE, lw=0.8)
    for i, (v, b) in enumerate(zip(r_e, bf_e)):
        ax_trait.text(xe[i], v + (0.02 if v >= 0 else -0.02), fmt_bf(b).replace("BF=", ""), ha="center",
                      va="bottom" if v >= 0 else "top", fontsize=FS_BFNUM, color="#222222")
    for xb in (len(PERSONALITY) - .5, len(PERSONALITY) + len(COMPOSITE) - .5):
        ax_trait.axvline(xb, color="grey", lw=.5, ls=":", alpha=.5)
    ax_trait.set_xticks(xe); ax_trait.set_xticklabels(labels_e, rotation=45, ha="right")
    ax_trait.set_ylim(min(r_e) - 0.22, max(r_e) + 0.22); ax_trait.set_ylabel("Pearson r with PC1")

    # -------- Panel f: consensus latent vs overall switch rate --------
    rr, _ = pearsonr(zc, sw_all); bf_b = float(bayesfactor_pearson(rr, _N3))
    ax_sw.scatter(zc, sw_all, s=8, color=COL_IDRNN, alpha=0.5, edgecolors="none", zorder=2)
    bb = np.polyfit(zc, sw_all, 1); xs2 = np.linspace(zc.min(), zc.max(), 50)
    ax_sw.plot(xs2, bb[0] * xs2 + bb[1], "r:", lw=1.2, zorder=3)
    ax_sw.text(0.04, 0.96, f"r = {rr:.2f}\n{fmt_bf(bf_b)}", transform=ax_sw.transAxes, va="top", ha="left", fontsize=FS_ANNOT)
    ax_sw.set_xlabel("PC1"); ax_sw.set_ylabel("switch rate (all tasks)")

    # -------- Panel g: openness readout climbs with #training tasks --------
    _SUBSETS = {"t0": (1, "2-armed"), "t1": (1, "restless"), "t2": (1, "horizon"),
                "t01": (2, "2-arm+rest"), "t02": (2, "2-arm+horiz"), "t12": (2, "rest+horiz"), "t012": (3, "all 3")}
    _FALLBACK = {"t1": "thalmann_z3_1task_full", "t01": "thalmann_z3_full", "t012": "thalmann_z3_3task_full"}
    def _load_decode(sub):
        p = f"final_plots/thalmann_z3_ds_{sub}/decoding/seed_averaged_decoding.csv"
        if not os.path.exists(p) and sub in _FALLBACK:
            p = f"final_plots/{_FALLBACK[sub]}/decoding/seed_averaged_decoding.csv"
        return pd.read_csv(p).set_index("target") if os.path.exists(p) else None
    def _collect(tgts):
        rr_ = {}
        for sub, (nn, lab) in _SUBSETS.items():
            ddc = _load_decode(sub)
            if ddc is None: continue
            tk = [t for t in tgts if t in ddc.index]
            if not tk: continue
            rr_[sub] = dict(n=nn, im=np.mean([ddc.loc[t, "idrnn_loo_r2_mean"] for t in tk]),
                isd=np.mean([ddc.loc[t, "idrnn_loo_r2_sd"] for t in tk]) / np.sqrt(len(tk)),
                vm=np.mean([ddc.loc[t, "vanilla_loo_r2_mean"] for t in tk]),
                vsd=np.mean([ddc.loc[t, "vanilla_loo_r2_sd"] for t in tk]) / np.sqrt(len(tk)))
        return rr_
    rows = _collect(["BIG5_open"])
    for arch, col, off, ls, mk, lab in [("i", COL_IDRNN, -0.05, "-", "o", "IDRNN z"),
                                        ("v", COL_VANILLA, +0.05, "--", "s", "Vanilla h (dim-matched)")]:
        ns = sorted({r["n"] for r in rows.values()}); means_, errs_ = [], []
        for nn in ns:
            vv = np.array([r[f"{arch}m"] for r in rows.values() if r["n"] == nn])
            means_.append(float(vv.mean()))
            errs_.append(float(vv.std(ddof=1) / np.sqrt(len(vv))) if len(vv) > 1
                         else float([r[f"{arch}sd"] for r in rows.values() if r["n"] == nn][0]))
            ax_scale.scatter([nn + off] * len(vv), vv, s=8, color=col, alpha=0.22, zorder=1, edgecolors="none")
        ax_scale.errorbar([nn + off for nn in ns], means_, yerr=errs_, fmt=mk, ls=ls, color=col, lw=1.3,
                          ms=3.5, capsize=2, elinewidth=0.8, zorder=3, label=lab)
    ax_scale.axhline(0, color=COL_TRUE, lw=.6, ls=":")
    ax_scale.set_xticks([1, 2, 3]); ax_scale.set_xlabel("# training tasks")
    ax_scale.set_ylabel("Openness readout LOO $R^2$"); ax_scale.legend(loc="upper left", fontsize=FS_ANNOT - 1)

    # -------- Panel h: trait signal beyond working memory --------
    TRAITS = [("BIG5_open", "Openness"), ("CEI", "Curiosity")]
    dfw = pd.DataFrame({"z": zc, "WM": tg["WM_composite"].values, **{k: tg[k].values for k, _ in TRAITS}}).dropna()
    n_wm = len(dfw); raw, par, bfs_d = [], [], []
    for k, _ in TRAITS:
        raw.append(float(pg.corr(dfw["z"], dfw[k])["r"].values[0]))
        rp = float(pg.partial_corr(dfw, x="z", y=k, covar="WM")["r"].values[0])
        par.append(rp); bfs_d.append(float(bayesfactor_pearson(rp, n_wm - 1)))
    xd = np.arange(len(TRAITS)); wd = 0.38; GREY = nature_colors["Grey"][3]
    ax_wm.bar(xd - wd / 2, raw, wd, color=GREY, edgecolor="black", lw=0.4, label="raw  r", zorder=2)
    ax_wm.bar(xd + wd / 2, par, wd, color=COL_IDRNN, edgecolor="black", lw=0.4, label="partial  r | WM", zorder=2)
    ax_wm.axhline(0, color=COL_TRUE, lw=0.8)
    for i, (rp, bff) in enumerate(zip(par, bfs_d)):
        ax_wm.text(xd[i] + wd / 2, rp + (0.012 if rp >= 0 else -0.012), fmt_bf(bff).replace("BF=", ""),
                   ha="center", va="bottom" if rp >= 0 else "top", fontsize=FS_ANNOT, color=COL_TRUE)
    ax_wm.set_xticks(xd); ax_wm.set_xticklabels([l for _, l in TRAITS]); ax_wm.set_ylabel("Pearson r  with PC1")
    ax_wm.set_ylim(top=max(max(raw), max(par)) + 0.10); ax_wm.legend(loc="upper right")

    # -------- Panel letters --------
    ax_r1.set_title("a", loc="left", fontweight="bold", fontsize=FS_LETTER)
    ax_trait.set_title("e", loc="left", fontweight="bold", fontsize=FS_LETTER)
    for ax, lett in [(ax_loo, "b"), (ax_nll, "c"), (ax_reg, "d"), (ax_sw, "f"), (ax_scale, "g"), (ax_wm, "h")]:
        ax.set_title(lett, loc="left", fontweight="bold", fontsize=FS_LETTER)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    fig.savefig("final_plots/thalmann_combined_figure.png", dpi=600)
    fig.savefig("final_plots/thalmann_combined_figure.pdf")
    fig.savefig("final_plots/thalmann_combined_figure.svg")
    plt.show()
print("Saved final_plots/thalmann_combined_figure.png / .pdf / .svg")
