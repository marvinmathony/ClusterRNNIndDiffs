#!/usr/bin/env python3
"""Append SEED-AVERAGED, structure-matched versions of the four exploration-axis
trait panels + a pooled-PCA fixed-dimension confirmation to thalmann_results.ipynb.

These are NEW cells (distinct markers) — the existing single-seed (canonical 999)
cells are left intact. All numbers come from analyze_z2_seed_matched.py outputs:
  decoding/z2_seed_matched.csv            (per-seed, structure-matched dim)
  decoding/z2_seed_matched_summary.json   (seed-avg means/SE + PCA summary)
  decoding/z2_consensus_pca.npy           (fixed-dim consensus exploration latent)

Dimension-selection rule (per seed, trait-INDEPENDENT, documented in MD):
  the exploration axis = the latent dim most correlated with HORIZON-task switch
  rate (directed exploration's behavioural signature), sign-flipped to be positive;
  d* = argmax_d |corr(z_d, switch_horizon)|.  Cross-checked vs a fingerprint-cosine
  match and a pooled-PCA fixed dimension (PC1), all converging.
"""
import base64, os, nbformat
import build_panela_cell as A   # reuse SETUP + upsert

NB = "thalmann_results.ipynb"
DEC = "final_plots/thalmann_z3_3task_full/decoding"

# ---------------------------------------------------------------- panel cells
P1_PNG = f"{DEC}/panel_seedavg_traits.png"
P1 = r'''# === panel — seed-averaged exploration-axis trait correlations ===
import numpy as np, pandas as pd
from scipy.stats import ttest_1samp
DEC = "final_plots/thalmann_z3_3task_full/decoding"
R = pd.read_csv(f"{DEC}/z2_seed_matched.csv")          # per-seed, structure-matched dim
ITEMS = [("r_wm", "Working\nmemory"), ("r_open", "Openness"), ("r_cei", "Curiosity")]
rng = np.random.default_rng(0)
fig, ax = plt.subplots(figsize=(3.7, 3.5))
for i, (k, lab) in enumerate(ITEMS):
    v = R[k].dropna().values; m = v.mean(); se = v.std(ddof=1) / np.sqrt(len(v))
    t, _ = ttest_1samp(v, 0.0); bf = float(bayesfactor_ttest(t, len(v)))
    ax.bar(i, m, 0.62, color=COL_IDRNN, edgecolor="black", lw=0.4, zorder=2)
    ax.errorbar(i, m, yerr=se, fmt="none", ecolor="#333333", elinewidth=1.0, capsize=3, zorder=4)
    ax.scatter(np.full(len(v), i) + rng.normal(0, 0.055, len(v)), v, s=11,
               color=COL_TRUE, alpha=0.4, zorder=3, edgecolors="none")
    ax.text(i, m + (0.018 if m >= 0 else -0.018), fmt_bf(bf).replace("BF=", ""),
            ha="center", va="bottom" if m >= 0 else "top", fontsize=6.5, color=COL_TRUE)
ax.axhline(0, color=COL_TRUE, lw=0.8)
ax.set_xticks(range(len(ITEMS))); ax.set_xticklabels([l for _, l in ITEMS])
ax.set_ylabel("Pearson r  with exploration axis")
ax.set_title(f"Trait correlates (seed-averaged, n=10 seeds)")
ax.set_ylim(-0.42, 0.33)
fig.tight_layout(); fig.savefig(f"{DEC}/panel_seedavg_traits.png", dpi=600, bbox_inches="tight"); plt.show()
for k, lab in ITEMS:
    v = R[k].dropna().values; t, _ = ttest_1samp(v, 0.0)
    print(f"{lab.replace(chr(10),' '):16s} r={v.mean():+.3f} +/- {v.std(ddof=1)/np.sqrt(len(v)):.3f}  t9={t:+.2f}")'''

P2_PNG = f"{DEC}/panel_seedavg_switching.png"
P2 = r'''# === panel — seed-averaged exploration axis is directed switching ===
import numpy as np, pandas as pd
from scipy.stats import ttest_1samp, pearsonr
DEC = "final_plots/thalmann_z3_3task_full/decoding"
R = pd.read_csv(f"{DEC}/z2_seed_matched.csv")
zc = np.load(f"{DEC}/z2_seedavg_mean.npy")              # PCA-FREE: sign-aligned MEAN of the 10 per-seed matched dims
c = np.load("data_thalmann_3task_full/c_train.npy"); N = c.shape[0]
def sw(bl):
    s = p = 0
    for b in range(bl.shape[0]):
        ch = bl[b]; ch = ch[ch >= 0]
        if len(ch) < 2: continue
        s += int((ch[1:] != ch[:-1]).sum()); p += len(ch) - 1
    return s / p if p else np.nan
sw_all = np.array([sw(c[i]) for i in range(N)])
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.4))
# (left) seed-averaged exploration latent (PCA-free mean of matched dims) vs overall switch
m = np.isfinite(zc) & np.isfinite(sw_all); r, _ = pearsonr(zc[m], sw_all[m]); bf = float(bayesfactor_pearson(r, m.sum()))
ax1.scatter(zc, sw_all, s=12, color=COL_IDRNN, alpha=0.5, edgecolors="none", zorder=2)
b = np.polyfit(zc[m], sw_all[m], 1); xs = np.linspace(zc[m].min(), zc[m].max(), 50)
ax1.plot(xs, b[0]*xs + b[1], color=COL_TRUE, lw=1.3, zorder=3)
ax1.text(0.04, 0.96, f"r={r:.2f}\n{fmt_bf(bf)}", transform=ax1.transAxes, va="top", ha="left", fontsize=7)
ax1.set_xlabel("seed-averaged exploration latent\n(mean of per-seed matched dims, no PCA)"); ax1.set_ylabel("switch rate (all tasks)")
ax1.set_title("exploration axis = switching")
# (right) per-task partial r(z, switch | entropy), seed-averaged with across-seed SE
TASKS = [("pr_sw2a", "2-armed"), ("pr_swre", "restless"), ("pr_swho", "horizon")]
xt = np.arange(len(TASKS))
for i, (k, lab) in enumerate(TASKS):
    v = R[k].dropna().values; mm = v.mean(); se = v.std(ddof=1) / np.sqrt(len(v))
    t, _ = ttest_1samp(v, 0.0); bf = float(bayesfactor_ttest(t, len(v)))
    ax2.bar(i, mm, 0.6, color=COL_IDRNN, edgecolor="black", lw=0.4, zorder=2)
    ax2.errorbar(i, mm, yerr=se, fmt="none", ecolor="#333333", elinewidth=1.0, capsize=3, zorder=4)
    ax2.text(i, mm + se + 0.02, fmt_bf(bf).replace("BF=", ""), ha="center", va="bottom", fontsize=6.5, color=COL_TRUE)
ax2.axhline(0, color=COL_TRUE, lw=0.8)
ax2.set_xticks(xt); ax2.set_xticklabels([l for _, l in TASKS])
ax2.set_ylabel("partial r(z, switch | entropy)"); ax2.set_ylim(top=1.0)
ax2.set_title("directed switching (seed-averaged)")
fig.tight_layout(); fig.savefig(f"{DEC}/panel_seedavg_switching.png", dpi=600, bbox_inches="tight"); plt.show()
print("seed-avg partial r(switch|entropy):", {l: round(R[k].mean(), 3) for k, l in TASKS})'''

P3_PNG = f"{DEC}/panel_seedavg_beyond_wm.png"
P3 = r'''# === panel — seed-averaged trait signal beyond working memory ===
import numpy as np, pandas as pd
from scipy.stats import ttest_1samp
DEC = "final_plots/thalmann_z3_3task_full/decoding"
R = pd.read_csv(f"{DEC}/z2_seed_matched.csv")
TRAITS = [("r_open", "pr_open_wm", "Openness"), ("r_cei", "pr_cei_wm", "Curiosity")]
x = np.arange(len(TRAITS)); w = 0.38; GREY = nature_colors['Grey'][3]
fig, ax = plt.subplots(figsize=(3.9, 3.5))
for i, (rk, pk, lab) in enumerate(TRAITS):
    rv = R[rk].dropna().values; pv = R[pk].dropna().values
    rm, rse = rv.mean(), rv.std(ddof=1)/np.sqrt(len(rv)); pm, pse = pv.mean(), pv.std(ddof=1)/np.sqrt(len(pv))
    t, _ = ttest_1samp(pv, 0.0); bf = float(bayesfactor_ttest(t, len(pv)))
    ax.bar(i - w/2, rm, w, color=GREY, edgecolor="black", lw=0.4, zorder=2, label="raw r" if i == 0 else None)
    ax.bar(i + w/2, pm, w, color=COL_IDRNN, edgecolor="black", lw=0.4, zorder=2, label="partial r | WM" if i == 0 else None)
    ax.errorbar(i - w/2, rm, yerr=rse, fmt="none", ecolor="#333333", elinewidth=0.9, capsize=2.5, zorder=4)
    ax.errorbar(i + w/2, pm, yerr=pse, fmt="none", ecolor="#333333", elinewidth=0.9, capsize=2.5, zorder=4)
    ax.text(i + w/2, pm + pse + 0.012, fmt_bf(bf).replace("BF=", ""), ha="center", va="bottom", fontsize=6.5, color=COL_TRUE)
ax.axhline(0, color=COL_TRUE, lw=0.8)
ax.set_xticks(x); ax.set_xticklabels([l for _, _, l in TRAITS])
ax.set_ylabel("Pearson r  with exploration axis"); ax.legend(loc="upper right", fontsize=6.5)
ax.set_title("Trait signal beyond WM (seed-averaged)")
fig.tight_layout(); fig.savefig(f"{DEC}/panel_seedavg_beyond_wm.png", dpi=600, bbox_inches="tight"); plt.show()
print("raw :", {l: round(R[rk].mean(), 3) for rk, _, l in TRAITS})
print("|WM :", {l: round(R[pk].mean(), 3) for _, pk, l in TRAITS})'''

P4_PNG = f"{DEC}/panel_seedavg_latent_vs_summary.png"
P4 = r'''# === panel — seed-averaged latent extracts openness summaries miss ===
import json, numpy as np, pandas as pd
from scipy.stats import ttest_1samp
DEC = "final_plots/thalmann_z3_3task_full/decoding"
R = pd.read_csv(f"{DEC}/z2_seed_matched.csv")
S = json.load(open(f"{DEC}/z2_seed_matched_summary.json"))
base = S["open_summary_baselines"]    # seed-independent behavioural summaries
GREY2, GREY4 = nature_colors['Grey'][2], nature_colors['Grey'][4]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.4), gridspec_kw={"width_ratios": [1.5, 1]})
# (left) openness signal: overall-switch vs horizon-switch vs the model latent
ro = R["r_open"].dropna().values; rom, rose = ro.mean(), ro.std(ddof=1)/np.sqrt(len(ro))
items = [("overall\nswitch rate", base["overall_switch"], None, GREY2),
         ("horizon\nswitch rate", base["horizon_switch"], None, GREY4),
         ("exploration\nlatent", rom, rose, COL_IDRNN)]
for i, (lab, val, se, col) in enumerate(items):
    ax1.bar(i, val, 0.62, color=col, edgecolor="black", lw=0.4, zorder=2)
    if se is not None:
        ax1.errorbar(i, val, yerr=se, fmt="none", ecolor="#333333", elinewidth=1.0, capsize=3, zorder=4)
    ax1.text(i, val + 0.006, f"r={val:.2f}", ha="center", va="bottom", fontsize=6.8)
ax1.axhline(0, color=COL_TRUE, lw=0.8)
ax1.set_xticks(range(3)); ax1.set_xticklabels([l for l, *_ in items], fontsize=7)
ax1.set_ylabel("Pearson r  with openness"); ax1.set_ylim(top=0.30)
ax1.set_title("openness signal by measure")
# (right) unique contributions (seed-averaged partial correlations)
pz = R["pr_open_sw"].dropna().values; ps = R["pr_sw_open"].dropna().values
vals = [pz.mean(), ps.mean()]; ses = [pz.std(ddof=1)/np.sqrt(len(pz)), ps.std(ddof=1)/np.sqrt(len(ps))]
ax2.bar([0, 1], vals, 0.6, color=[COL_IDRNN, GREY4], edgecolor="black", lw=0.4, zorder=2)
ax2.errorbar([0, 1], vals, yerr=ses, fmt="none", ecolor="#333333", elinewidth=1.0, capsize=3, zorder=4)
ax2.axhline(0, color=COL_TRUE, lw=0.8)
for i, v in enumerate(vals):
    ax2.text(i, v + (0.01 if v >= 0 else -0.01), f"{v:+.2f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=6.8)
ax2.set_xticks([0, 1]); ax2.set_xticklabels(["latent | switch", "switch | latent"], fontsize=7)
ax2.set_ylabel("partial r  with openness"); ax2.set_title("unique contribution")
fig.tight_layout(); fig.savefig(f"{DEC}/panel_seedavg_latent_vs_summary.png", dpi=600, bbox_inches="tight"); plt.show()
print(f"openness: overall-switch={base['overall_switch']:+.3f}  horizon-switch={base['horizon_switch']:+.3f}  latent={rom:+.3f}")
print(f"partial r(latent,open|switch)={vals[0]:+.3f}   r(switch,open|latent)={vals[1]:+.3f}")'''

P5_PNG = f"{DEC}/panel_pca_fixed_dim.png"
P5 = r'''# === panel — pooled-PCA fixed-dimension confirmation ===
import json, numpy as np, pandas as pd
from scipy.stats import pearsonr
DEC = "final_plots/thalmann_z3_3task_full/decoding"
S = json.load(open(f"{DEC}/z2_seed_matched_summary.json"))["pca"]
zc = np.load(f"{DEC}/z2_consensus_pca.npy")            # PC1 scores (fixed, basis-free)
op = pd.read_csv(f"{DEC}/trait_targets.csv")["BIG5_open"].values
fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.1))
# (a) variance explained — top 3 PCs = the shared individual-difference subspace
ve = S["var_explained"]; xs = np.arange(len(ve))
cols = [COL_IDRNN if j < 3 else nature_colors['Grey'][3] for j in range(len(ve))]
axes[0].bar(xs, ve, 0.7, color=cols, edgecolor="black", lw=0.4)
axes[0].set_xticks(xs); axes[0].set_xticklabels([f"PC{j+1}" for j in xs], fontsize=6.5)
axes[0].set_ylabel("variance explained")
axes[0].set_title(f"shared subspace ({sum(ve[:3])*100:.0f}% in top 3)")
# (b) the fixed exploration PC's trait correlations
TR = [("r_wm", "WM"), ("r_open", "Openness"), ("r_cei", "Curiosity")]
for i, (k, lab) in enumerate(TR):
    axes[1].bar(i, S[k], 0.62, color=COL_IDRNN, edgecolor="black", lw=0.4)
    axes[1].text(i, S[k] + (0.012 if S[k] >= 0 else -0.012), f"{S[k]:+.2f}", ha="center",
                 va="bottom" if S[k] >= 0 else "top", fontsize=6.8)
axes[1].axhline(0, color=COL_TRUE, lw=0.8)
axes[1].set_xticks(range(len(TR))); axes[1].set_xticklabels([l for _, l in TR], fontsize=7)
axes[1].set_ylabel("Pearson r"); axes[1].set_ylim(-0.40, 0.40)
axes[1].set_title(f"PC{S['exp_pc']} (fixed exploration axis)")
# (c) scatter: fixed PC vs openness
m = np.isfinite(zc) & np.isfinite(op); r, _ = pearsonr(zc[m], op[m]); bf = float(bayesfactor_pearson(r, m.sum()))
axes[2].scatter(zc, op, s=12, color=COL_IDRNN, alpha=0.5, edgecolors="none", zorder=2)
b = np.polyfit(zc[m], op[m], 1); xx = np.linspace(zc[m].min(), zc[m].max(), 50)
axes[2].plot(xx, b[0]*xx + b[1], color=COL_TRUE, lw=1.3, zorder=3)
axes[2].text(0.04, 0.96, f"r={r:.2f}\n{fmt_bf(bf)}", transform=axes[2].transAxes, va="top", ha="left", fontsize=7)
axes[2].set_xlabel(f"fixed exploration PC{S['exp_pc']}"); axes[2].set_ylabel("openness")
axes[2].set_title("openness on the fixed dim")
V = S["validation"]
cap = ("Construction: each seed's 3 standardized latent dims are concatenated (236 subjects × 30) and ONE PCA is run; "
       f"the component is picked by horizon-switch correlation — blind to the traits.  Not overfitting: leave-one-"
       f"SUBJECT-out (refit on 235, project the held-out subject) gives r(open)={V['pooled_pca_loo']['r_open']:+.2f} "
       f"≈ in-sample {S['r_open']:+.2f}.\nThis PC1 is the MAX-VARIANCE consensus; a transparent sign-aligned mean of the "
       f"per-seed axes gives r(open)={V['simple_mean']['r_open']:+.2f}, matching de-attenuation "
       f"(single-seed {V['single_seed_open']:+.2f} ÷ √reliability {V['cross_seed_reliability']:.2f} = {V['deattenuated_open']:+.2f}).")
fig.text(0.5, -0.13, cap, ha="center", va="top", fontsize=5.6, color="#333333", wrap=True)
fig.tight_layout(); fig.savefig(f"{DEC}/panel_pca_fixed_dim.png", dpi=600, bbox_inches="tight"); plt.show()
print(f"PC{S['exp_pc']} (fixed): r(WM)={S['r_wm']:+.3f}  r(open)={S['r_open']:+.3f}  r(CEI)={S['r_cei']:+.3f}  var(top3)={sum(ve[:3]):.2f}")
print(f"validation: LOO r(open)={V['pooled_pca_loo']['r_open']}  simple-mean r(open)={V['simple_mean']['r_open']}  "
      f"reliability={V['cross_seed_reliability']}  de-attenuated={V['deattenuated_open']}")'''

P6_PNG = f"{DEC}/panel_three_method_convergence.png"
P6 = r'''# === panel — three ways to align the exploration axis converge ===
import json, numpy as np, pandas as pd
from scipy.stats import ttest_1samp
DEC = "final_plots/thalmann_z3_3task_full/decoding"
R = pd.read_csv(f"{DEC}/z2_seed_matched.csv")
P = json.load(open(f"{DEC}/z2_seed_matched_summary.json"))["pca"]
TR = [("Working\nmemory", "r_wm", "ws_r_wm", "r_wm"), ("Openness", "r_open", "ws_r_open", "r_open"),
      ("Curiosity", "r_cei", "ws_r_cei", "r_cei")]
methods = ["raw-dim\nstructure-matched", "within-seed PCA\n(per-seed, combined)", "pooled PCA\n(fixed dim)"]
cols = [COL_IDRNN, nature_colors['Blue'][1], nature_colors['Orange'][3]]
x = np.arange(len(TR)); w = 0.26
fig, ax = plt.subplots(figsize=(5.6, 3.7))
for mi, mlab in enumerate(methods):
    means, ses = [], []
    for lab, rk, wk, pk in TR:
        if mi == 0:
            v = R[rk].dropna().values; means.append(v.mean()); ses.append(v.std(ddof=1)/np.sqrt(len(v)))
        elif mi == 1:
            v = R[wk].dropna().values; means.append(v.mean()); ses.append(v.std(ddof=1)/np.sqrt(len(v)))
        else:
            means.append(P[pk]); ses.append(0.0)
    off = (mi - 1) * w
    ax.bar(x + off, means, w, color=cols[mi], edgecolor="black", lw=0.4, zorder=2, label=mlab)
    ax.errorbar(x + off, means, yerr=ses, fmt="none", ecolor="#333333", elinewidth=0.8, capsize=2, zorder=4)
ax.axhline(0, color=COL_TRUE, lw=0.8)
ax.set_xticks(x); ax.set_xticklabels([t[0] for t in TR])
ax.set_ylabel("Pearson r  with exploration axis")
ax.legend(loc="upper left", fontsize=6, ncol=1, handlelength=1.2)
ax.set_title("Three ways to align the axis across seeds converge")
fig.tight_layout(); fig.savefig(f"{DEC}/panel_three_method_convergence.png", dpi=600, bbox_inches="tight"); plt.show()
print("raw-dim    :", {t[0].replace(chr(10),' '): round(R[t[1]].mean(), 3) for t in TR})
print("within-PCA :", {t[0].replace(chr(10),' '): round(R[t[2]].mean(), 3) for t in TR})
print("pooled-PCA :", {t[0].replace(chr(10),' '): P[t[3]] for t in TR})'''

P7_PNG = f"{DEC}/panel_seedavg_correlations.png"
P7 = r'''# === panel — seed-averaged individual-difference axis (every trait) ===
import numpy as np, pandas as pd
DEC = "final_plots/thalmann_z3_3task_full/decoding"
A = pd.read_csv(f"{DEC}/z2_seed_matched_alltraits.csv").set_index("seed")   # per-seed matched-dim r, every trait
tg = pd.read_csv(f"{DEC}/trait_targets.csv")
PERSONALITY = ["PANAS_PA","PANAS_NA","STICSA","PHQ","CEI","BIG5_open"]
COMPOSITE   = ["AxDep","posMood","negMood","Exp"]
WM_KEYS     = ["WM_composite","WM_OS","WM_SS","WM_WMU"]
ALL_KEYS = PERSONALITY + COMPOSITE + WM_KEYS
LBL = {"PANAS_PA":"PANAS +","PANAS_NA":"PANAS −","STICSA":"STICSA","PHQ":"PHQ","CEI":"CEI",
       "BIG5_open":"Openness","AxDep":"Anx/Dep","posMood":"pos mood","negMood":"neg mood",
       "Exp":"Exploration","WM_composite":"WM comp","WM_OS":"WM OS","WM_SS":"WM SS","WM_WMU":"WM upd"}
labels = [LBL[k] for k in ALL_KEYS]; x = np.arange(len(labels))
means = np.array([A[k].mean() for k in ALL_KEYS])
ses   = np.array([A[k].std(ddof=1)/np.sqrt(len(A)) for k in ALL_KEYS])
ns    = [int(np.isfinite(tg[k].values).sum()) for k in ALL_KEYS]
bf    = [float(bayesfactor_pearson(means[i], ns[i])) for i in range(len(ALL_KEYS))]   # effect-size BF on the mean r
rng = np.random.default_rng(0)
fig, ax = plt.subplots(figsize=(7.4, 3.6))
ax.bar(x, means, 0.6, color=COL_IDRNN, edgecolor="black", linewidth=0.4, zorder=2)
ax.errorbar(x, means, yerr=ses, fmt="none", ecolor="#333333", elinewidth=0.8, capsize=2, zorder=4)
for i, k in enumerate(ALL_KEYS):
    v = A[k].values
    ax.scatter(x[i] + rng.normal(0, 0.05, len(v)), v, s=7, color=COL_TRUE, alpha=0.35, zorder=3, edgecolors="none")
ax.axhline(0, color=COL_TRUE, lw=0.8)
for i, (m, b) in enumerate(zip(means, bf)):
    ax.text(x[i], m + (0.02 if m >= 0 else -0.02), fmt_bf(b).replace("BF=", ""), ha="center",
            va="bottom" if m >= 0 else "top", fontsize=5.5, rotation=90, color="#222222")
for xb in (len(PERSONALITY)-.5, len(PERSONALITY)+len(COMPOSITE)-.5):
    ax.axvline(xb, color="grey", lw=.5, ls=":", alpha=.5)
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
ax.set_ylim(means.min()-0.17, means.max()+0.15)
ax.set_ylabel("Pearson r  (seed-averaged exploration axis)")
ax.set_title("Individual-difference axis, seed-averaged  (n=10 seeds; numbers = BF$_{10}$ on mean r)")
fig.tight_layout(); fig.savefig(f"{DEC}/panel_seedavg_correlations.png", dpi=600, bbox_inches="tight"); plt.show()
print({LBL[k]: round(A[k].mean(), 3) for k in ALL_KEYS})'''

P8_PNG = f"{DEC}/panel_seedavg_openness_evidence.png"
P8 = r'''# === panel — seed-averaged openness evidence vs original study ===
import numpy as np, pandas as pd
DEC = "final_plots/thalmann_z3_3task_full/decoding"
A = pd.read_csv(f"{DEC}/z2_seed_matched_alltraits.csv")
tg = pd.read_csv(f"{DEC}/trait_targets.csv")
n = int(np.isfinite(tg["BIG5_open"].values).sum())
r_ours = float(A["BIG5_open"].mean())                # mean of per-seed openness correlations
R_ORIG = 0.11                                        # original study's strongest Openness x task-measure
bf_orig = float(bayesfactor_pearson(R_ORIG, n)); bf_ours = float(bayesfactor_pearson(r_ours, n))
ratio = bf_ours / bf_orig
labels = [f"Original\n(best task measure)\nr={R_ORIG:.2f}", f"Seed-avg axis\nr={r_ours:.2f}"]
vals = [bf_orig, bf_ours]; cols = [COL_VANILLA, COL_IDRNN]
fig, ax = plt.subplots(figsize=(3.6, 3.6))
ax.bar([0, 1], vals, 0.6, color=cols, edgecolor="black", linewidth=0.5, zorder=3)
for thr, lab in [(1, "BF=1 (none)"), (3, "3 (subst.)"), (10, "10 (strong)")]:
    ax.axhline(thr, ls=":", color=COL_TRUE, lw=0.8, zorder=1)
    ax.text(1.55, thr, lab, va="center", ha="left", fontsize=6, color=COL_TRUE)
for i, v in enumerate(vals):
    ax.text(i, v*1.15, fmt_bf(v), ha="center", va="bottom", fontsize=8, fontweight="bold")
ax.annotate(f"×{ratio:.0f} evidence", xy=(0.5, bf_ours), ha="center", va="bottom", fontsize=8,
            color=COL_IDRNN, fontweight="bold", xytext=(0.5, bf_ours*2.2), arrowprops=None)
ax.set_yscale("log"); ax.set_ylim(0.1, max(vals)*4)
ax.set_xticks([0, 1]); ax.set_xticklabels(labels, fontsize=7)
ax.set_ylabel("Bayes factor BF$_{10}$ (Openness link)")
ax.set_title("Openness: undetectable → strong\n(seed-averaged exploration axis)")
fig.tight_layout(); fig.savefig(f"{DEC}/panel_seedavg_openness_evidence.png", dpi=600, bbox_inches="tight"); plt.show()
print(f"openness seed-avg r={r_ours:.3f} BF={bf_ours:.2f} | original r={R_ORIG} BF={bf_orig:.2f} | ratio {ratio:.0f}x")'''

MD_HEAD = nbformat.v4.new_markdown_cell(
    "# Seed-averaged exploration axis (structure-matched + PCA)\n"
    "Individual latent dimensions are basis-arbitrary across independent trainings, so the "
    "single-seed (canonical 999) panels above can't establish robustness on their own. "
    "Here we recompute the exploration-axis trait analysis **across all 10 seeds**. In each seed we "
    "select the exploration axis by a **trait-independent** rule — the latent dimension most "
    "correlated with **horizon-task switch rate** (the behavioural signature of directed exploration), "
    "sign-aligned positive — so the trait read-outs are not circular. The matched dimension is "
    "dim 0/1/2 depending on the seed. We cross-check against a fingerprint-cosine match and a "
    "**pooled-PCA fixed dimension** (PC1 of the standardized latents pooled over seeds), which "
    "recovers the same axis without any per-seed choice. Bars are seed means ± across-seed SE; "
    "numbers are Bayes factors from a one-sample test over the 10 seeds.")

MDS = {
    "panel_seedavg_traits": "**Seed-averaged trait correlates.** WM −0.32, openness +0.20, curiosity +0.16 — "
        "all reproduce across every seed (one-sample BF over seeds shown). The canonical-999 numbers above "
        "sit right on these means (it is representative, not an outlier).",
    "panel_seedavg_switching": "**The axis is directed switching, seed-averaged.** Left: a **PCA-free** "
        "seed-averaged latent — the sign-aligned **mean of the 10 per-seed matched exploration dimensions** "
        "(no PCA; just averaging the structure-matched axes) — tracks each subject's switch rate. Right: "
        "per-task partial r(z, switch | choice entropy) — switching beyond randomness, strongest in the "
        "horizon exploration task — averaged over seeds with across-seed error bars. (The pooled-PCA version "
        "of this latent lives only in the PCA fixed-dim panel.)",
    "panel_seedavg_beyond_wm": "**Trait signal beyond working memory, seed-averaged.** Openness and curiosity "
        "survive partialling out WM in every seed (partial r ≈ +0.19 / +0.16) — not a WM confound.",
    "panel_seedavg_latent_vs_summary": "**The latent recovers openness that summaries miss, seed-averaged.** "
        "Openness ≈ 0 with an overall switch-rate summary, modest with horizon switching, strongest with the "
        "model latent; and the latent predicts openness beyond raw switching while switching adds nothing "
        "beyond the latent.",
    "panel_pca_fixed_dim": "**PCA generalizes the finding to a fixed dimension.** *Construction:* each seed's 3 "
        "latent dims are z-scored and concatenated into one 236×30 matrix, and a **single PCA** is run; the "
        "exploration component is picked by its **horizon-switch** correlation — selection is blind to the "
        "questionnaire traits. The top 3 PCs span the shared individual-difference subspace (91% variance); "
        "PC1 is the exploration axis and reproduces WM −0.28 / openness +0.29 / curiosity +0.19, with openness "
        "loading on PC1 *only* (PC2/PC3 ≈ 0).\n\n"
        "*Why this is defensible, not a fishing artifact:* PCA is **unsupervised** (never sees the traits); "
        "and it is **not overfitting** — leave-one-**subject**-out (refit PCA on 235 subjects, project the "
        "held-out subject) gives r(open)=+0.29, identical to in-sample. *What the gain over a single seed means:* "
        "this PC1 is the **max-variance** consensus; a transparent sign-aligned **mean** of the per-seed axes "
        "(zero cross-subject fitting) gives r(open)=+0.24 — exactly the **de-attenuation** expected from "
        "averaging (single-seed +0.20 ÷ √[cross-seed reliability 0.72] = +0.24). So the consensus number is a "
        "*denoised* estimate; the conservative single-model number is the per-seed ~+0.20 reported above.",
    "panel_seedavg_correlations": "**Individual-difference axis, seed-averaged (every trait).** *What was done "
        "to the latents:* in each of the 10 seeds I take that model's per-subject latent (236×3), pick the one "
        "dimension that is the exploration axis (largest |corr| with horizon-task switch rate, sign-aligned — "
        "trait-blind), and correlate **that single seed's** axis with each trait. *What the average means:* the "
        "bar is the **mean across the 10 seeds of those per-seed correlations** (correlate-then-average), the "
        "error bar is the across-seed SE, and dots are the individual seeds — so the bar is the **typical "
        "single-model** correlation and the spread shows training-to-training reproducibility. The latents are "
        "**never combined**; each seed is analysed on its own (contrast the pooled-PCA panel, which averages "
        "the latents first → de-attenuated, larger). Numbers are BF₁₀ on the mean r (effect-size evidence, "
        "comparable to the single-seed and ensemble panels).",
    "panel_seedavg_openness_evidence": "**Openness — evidence increase, seed-averaged axis.** The original "
        "study's strongest Openness↔task-measure correlation was r=0.11 (BF₁₀≈0.33, favouring the null). The "
        "seed-averaged exploration axis (mean of per-seed openness correlations, r≈0.20) gives BF₁₀≈11 — a "
        "~32× increase, flipping no-evidence → strong. This is the **conservative single-model** estimate "
        "(each per-seed correlation is attenuated by that seed's noise; averaging the correlations does not "
        "remove that). The denoised ensemble axis (average-the-latents-first) is larger (r≈0.29, ×7757).",
    "panel_three_method_convergence": "**Three ways to identify the axis across seeds agree.** "
        "(i) **raw-dim, per-seed → combine:** in each seed pick the raw latent dim whose behavioural fingerprint "
        "matches the exploration axis, correlate with traits, average the per-seed correlations (this is the "
        "single-model estimate, ~+0.20 openness). "
        "(ii) **within-seed PCA → combine:** run PCA *inside each seed*, pick the exploration component, then "
        "average the per-seed correlations — each independent training analysed on its own footing. "
        "(iii) **pooled PCA (fixed dim):** concatenate all seeds and read one fixed component — a *denoised "
        "consensus* (no across-seed error bar; its higher openness +0.29 is the de-attenuation gain, see the "
        "PCA panel). All three give WM −0.27 to −0.32, openness +0.18 to +0.29, curiosity +0.13 to +0.19 — the "
        "conclusion does not depend on how the dimension is identified.",
}

PANELS = [("panel_seedavg_traits", P1, P1_PNG), ("panel_seedavg_switching", P2, P2_PNG),
          ("panel_seedavg_beyond_wm", P3, P3_PNG),
          ("panel_seedavg_latent_vs_summary", P4, P4_PNG), ("panel_pca_fixed_dim", P5, P5_PNG),
          ("panel_three_method_convergence", P6, P6_PNG),
          ("panel_seedavg_correlations", P7, P7_PNG), ("panel_seedavg_openness_evidence", P8, P8_PNG)]


def render(cell_src, png):
    """Render a panel PNG standalone under the shared SETUP namespace."""
    ns = {}
    exec(A.SETUP, ns)
    exec(cell_src, ns)
    assert os.path.exists(png), f"render produced no {png}"


def md_title(s):
    """Leading bold title between the first pair of ** ** — a stable idempotency key."""
    import re
    m = re.search(r"\*\*(.+?)\*\*", s)
    return m.group(1) if m else s[:40]


def upsert_md(nb, text):
    """Refresh an existing markdown cell with the same bold title, else append."""
    key = md_title(text)
    for c in nb.cells:
        if c.cell_type == "markdown" and key in "".join(c.source):
            c.source = text
            return False
    nb.cells.append(nbformat.v4.new_markdown_cell(text))
    return True


def main():
    for _, src, png in PANELS:
        render(src, png)
        print("rendered", png)
    nb = nbformat.read(NB, as_version=4)
    head_marker = "# Seed-averaged exploration axis (structure-matched + PCA)"
    if not any(c.cell_type == "markdown" and head_marker in "".join(c.source) for c in nb.cells):
        nb.cells.append(MD_HEAD)
    appended = 0
    for marker, src, png in PANELS:
        upsert_md(nb, MDS[marker])                 # refresh blurb in place (or append if new)
        cell_marker = src.splitlines()[0]          # exact marker embedded in each cell source
        if A.upsert(nb, cell_marker, src, embed_png=png):
            appended += 1
    nbformat.write(nb, NB)
    print(f"done: appended {appended} new panel cells; notebook now {len(nb.cells)} cells")


if __name__ == "__main__":
    main()
