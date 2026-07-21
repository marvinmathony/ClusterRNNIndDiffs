#!/usr/bin/env python3
"""Replicate the single-seed z2 analysis plots for the STRONGEST ENSEMBLE latent —
the pooled-PCA-across-all-concatenated-seed-latents consensus axis
(final_plots/thalmann_z3_3task_full/decoding/z2_consensus_pca.npy).

Each plot is its own self-contained cell (synthetic aesthetic, shared SETUP). New
markers => new cells; the single-seed (canonical 999) cells are left intact.
Replicated panels:
  E1  consensus latent — the individual-difference axis (r with every trait + BF)
  E2  openness — Nx evidence increase over the original study's best task measure
  E3  consensus latent = explore/exploit axis: directed switching, not randomness (2 panels)
  E4  trait signal beyond working memory (raw vs partial r | WM)
The consensus latent is a single 236-vector, so it slots in exactly where single-seed z2 was used.
"""
import os, nbformat
import build_panela_cell as A   # reuse SETUP + upsert

NB = "thalmann_results.ipynb"
DEC = "final_plots/thalmann_z3_3task_full/decoding"

_HEAD = r'''
import numpy as np, pandas as pd
from scipy.stats import pearsonr
DEC = "final_plots/thalmann_z3_3task_full/decoding"
zc = np.load(f"{DEC}/z2_consensus_pca.npy")           # ENSEMBLE: pooled PCA over all seeds' latents
tg = pd.read_csv(f"{DEC}/trait_targets.csv")          # torch-free trait scores (subids_full order)
def corr_n(y):
    m = np.isfinite(zc) & np.isfinite(y); return float(pearsonr(zc[m], y[m])[0]), int(m.sum())
'''

# ---- E1: individual-difference axis (every trait) -------------------------
E1 = "# === ensemble panel — consensus latent correlations with every trait (individual-difference axis) ===" + _HEAD + r'''
PERSONALITY = ["PANAS_PA","PANAS_NA","STICSA","PHQ","CEI","BIG5_open"]
COMPOSITE   = ["AxDep","posMood","negMood","Exp"]
WM_KEYS     = ["WM_composite","WM_OS","WM_SS","WM_WMU"]
ALL_KEYS = PERSONALITY + COMPOSITE + WM_KEYS
LBL = {"PANAS_PA":"PANAS +","PANAS_NA":"PANAS −","STICSA":"STICSA","PHQ":"PHQ","CEI":"CEI",
       "BIG5_open":"Openness","AxDep":"Anx/Dep","posMood":"pos mood","negMood":"neg mood",
       "Exp":"Exploration","WM_composite":"WM comp","WM_OS":"WM OS","WM_SS":"WM SS","WM_WMU":"WM upd"}
labels = [LBL[k] for k in ALL_KEYS]; x = np.arange(len(labels))
r = np.array([corr_n(tg[k].values)[0] for k in ALL_KEYS])
bf = [float(bayesfactor_pearson(*( (corr_n(tg[k].values)) ))) for k in ALL_KEYS]
fig, ax = plt.subplots(figsize=(7.2, 3.4))
ax.bar(x, r, 0.6, color=COL_IDRNN, edgecolor="black", linewidth=0.4, zorder=2)   # all blue = consensus latent
ax.axhline(0, color=COL_TRUE, lw=0.8)
for i, (v, b) in enumerate(zip(r, bf)):
    ax.text(x[i], v + (0.015 if v >= 0 else -0.015), fmt_bf(b).replace("BF=", ""), ha="center",
            va="bottom" if v >= 0 else "top", fontsize=5.5, rotation=90, color="#222222")
for xb in (len(PERSONALITY)-.5, len(PERSONALITY)+len(COMPOSITE)-.5):
    ax.axvline(xb, color="grey", lw=.5, ls=":", alpha=.5)
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
ax.set_ylim(min(r)-0.14, max(r)+0.12)
ax.set_ylabel("Pearson r  (ensemble consensus latent)")
ax.set_title("Ensemble consensus latent — the individual-difference axis  (numbers = BF$_{10}$)")
fig.tight_layout(); fig.savefig(f"{DEC}/panel_ens_correlations.png", dpi=600, bbox_inches="tight"); plt.show()
print({LBL[k]: round(corr_n(tg[k].values)[0], 3) for k in ALL_KEYS})'''

# ---- E2: openness Nx evidence ---------------------------------------------
E2 = "# === ensemble panel — openness evidence: consensus latent vs original study's best task measure ===" + _HEAD + r'''
R_ORIG = 0.11                                  # original study: strongest Openness x task-measure (of 13)
r_ours, n = corr_n(tg["BIG5_open"].values)
bf_orig = float(bayesfactor_pearson(R_ORIG, n)); bf_ours = float(bayesfactor_pearson(r_ours, n))
ratio = bf_ours / bf_orig
labels = [f"Original\n(best task measure)\nr={R_ORIG:.2f}", f"Ensemble latent\nr={r_ours:.2f}"]
vals = [bf_orig, bf_ours]; cols = [COL_VANILLA, COL_IDRNN]
fig, ax = plt.subplots(figsize=(3.6, 3.6))
ax.bar([0, 1], vals, 0.6, color=cols, edgecolor="black", linewidth=0.5, zorder=3)
for thr, lab in [(1, "BF=1 (none)"), (3, "3 (subst.)"), (10, "10 (strong)")]:
    ax.axhline(thr, ls=":", color=COL_TRUE, lw=0.8, zorder=1)
    ax.text(1.55, thr, lab, va="center", ha="left", fontsize=6, color=COL_TRUE)
for i, v in enumerate(vals):
    ax.text(i, v*1.15, fmt_bf(v), ha="center", va="bottom", fontsize=8, fontweight="bold")
ax.annotate(f"×{ratio:.0f} evidence", xy=(0.5, bf_ours), ha="center", va="bottom",
            fontsize=8, color=COL_IDRNN, fontweight="bold", xytext=(0.5, bf_ours*2.2), arrowprops=None)
ax.set_yscale("log"); ax.set_ylim(0.1, max(vals)*4)
ax.set_xticks([0, 1]); ax.set_xticklabels(labels, fontsize=7)
ax.set_ylabel("Bayes factor BF$_{10}$ (Openness link)")
ax.set_title("Openness: undetectable → strong\n(ensemble consensus latent)")
fig.tight_layout(); fig.savefig(f"{DEC}/panel_ens_openness_evidence.png", dpi=600, bbox_inches="tight"); plt.show()
print(f"openness: original r={R_ORIG} BF={bf_orig:.2f} | ensemble r={r_ours:.3f} BF={bf_ours:.2f} | ratio {ratio:.0f}x")'''

# ---- E3: explore/exploit axis (two panels) --------------------------------
E3 = "# === ensemble panel — consensus latent is the explore/exploit (switching) axis, not randomness ===" + r'''
import numpy as np, pandas as pd, pingouin as pg
from scipy.stats import pearsonr
DEC = "final_plots/thalmann_z3_3task_full/decoding"
zc = np.load(f"{DEC}/z2_consensus_pca.npy")
c   = np.load("data_thalmann_3task_full/c_train.npy")
tid = np.load("data_thalmann_3task_full/task_ids_per_block.npy")
N = c.shape[0]
def switch_rate(bl):
    s = p = 0
    for b in range(bl.shape[0]):
        ch = bl[b]; ch = ch[ch >= 0]
        if len(ch) < 2: continue
        s += int((ch[1:] != ch[:-1]).sum()); p += len(ch) - 1
    return s / p if p else np.nan
def entropy(bl, k):
    ch = bl[bl >= 0].astype(int)
    if len(ch) == 0: return np.nan
    pr = np.bincount(ch, minlength=k) / len(ch); pr = pr[pr > 0]
    return float(-(pr * np.log(pr)).sum())
TASKS = [("2-armed", tid == 0, 2), ("restless", tid == 1, 4), ("horizon", tid == 2, 2)]
SW = {t: np.array([switch_rate(c[i][m]) for i in range(N)]) for t, m, k in TASKS}
EN = {t: np.array([entropy(c[i][m], k) for i in range(N)]) for t, m, k in TASKS}
sw_all = np.array([switch_rate(c[i]) for i in range(N)])
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.4))
# (left) consensus latent vs overall switch rate
r, _ = pearsonr(zc, sw_all); bf = float(bayesfactor_pearson(r, N))
ax1.scatter(zc, sw_all, s=12, color=COL_IDRNN, alpha=0.5, edgecolors="none", zorder=2)
b = np.polyfit(zc, sw_all, 1); xs = np.linspace(zc.min(), zc.max(), 50)
ax1.plot(xs, b[0]*xs + b[1], color=COL_TRUE, lw=1.3, zorder=3)
ax1.text(0.04, 0.96, f"r={r:.2f}\n{fmt_bf(bf)}", transform=ax1.transAxes, va="top", ha="left", fontsize=7)
ax1.set_xlabel("ensemble consensus latent (PC1)"); ax1.set_ylabel("switch rate (all tasks)")
ax1.set_title("consensus axis (PC1) = switching")
# (right) per-task partial r(latent, switch | entropy)
xt = np.arange(len(TASKS)); prs, bfs = [], []
for t, m, k in TASKS:
    d = pd.DataFrame({"z": zc, "switch": SW[t], "entropy": EN[t]}).dropna()
    rr = float(pg.partial_corr(d, x="z", y="switch", covar="entropy")["r"].values[0])
    prs.append(rr); bfs.append(float(bayesfactor_pearson(rr, len(d) - 1)))
ax2.bar(xt, prs, 0.6, color=COL_IDRNN, edgecolor="black", lw=0.4, zorder=2)
ax2.axhline(0, color=COL_TRUE, lw=0.8)
for i, (rr, bf) in enumerate(zip(prs, bfs)):
    ax2.text(xt[i], rr + 0.02, fmt_bf(bf).replace("BF=", ""), ha="center", va="bottom", fontsize=6, color=COL_TRUE)
ax2.set_xticks(xt); ax2.set_xticklabels([t for t, _, _ in TASKS])
ax2.set_ylabel("partial r(latent, switch | entropy)"); ax2.set_ylim(top=1.0)
ax2.set_title("switching, controlling for randomness")
fig.tight_layout(); fig.savefig(f"{DEC}/panel_ens_switching.png", dpi=600, bbox_inches="tight"); plt.show()
print("r(latent,switch):", {t: round(float(pearsonr(zc, SW[t])[0]), 3) for t, _, _ in TASKS}, "| all", round(r, 3))
print("partial r(switch|entropy):", {t: round(p, 3) for t, p in zip([x[0] for x in TASKS], prs)})'''

# ---- E4: trait signal beyond working memory -------------------------------
E4 = "# === ensemble panel — trait signal beyond working memory (raw vs partial | WM) ===" + r'''
import numpy as np, pandas as pd, pingouin as pg
DEC = "final_plots/thalmann_z3_3task_full/decoding"
zc = np.load(f"{DEC}/z2_consensus_pca.npy")
tg = pd.read_csv(f"{DEC}/trait_targets.csv")
TRAITS = [("BIG5_open", "Openness"), ("CEI", "Curiosity")]
df = pd.DataFrame({"z": zc, "WM": tg["WM_composite"].values, **{k: tg[k].values for k, _ in TRAITS}}).dropna()
n = len(df); raw, par, bfs = [], [], []
for k, _ in TRAITS:
    raw.append(float(pg.corr(df["z"], df[k])["r"].values[0]))
    rp = float(pg.partial_corr(df, x="z", y=k, covar="WM")["r"].values[0])
    par.append(rp); bfs.append(float(bayesfactor_pearson(rp, n - 1)))
x = np.arange(len(TRAITS)); w = 0.38; GREY = nature_colors['Grey'][3]
fig, ax = plt.subplots(figsize=(3.8, 3.5))
ax.bar(x - w/2, raw, w, color=GREY, edgecolor="black", lw=0.4, label="raw  r(latent, trait)", zorder=2)
ax.bar(x + w/2, par, w, color=COL_IDRNN, edgecolor="black", lw=0.4, label="partial  r | WM", zorder=2)
ax.axhline(0, color=COL_TRUE, lw=0.8)
for i, (rp, bf) in enumerate(zip(par, bfs)):
    ax.text(x[i] + w/2, rp + (0.012 if rp >= 0 else -0.012), fmt_bf(bf).replace("BF=", ""),
            ha="center", va="bottom" if rp >= 0 else "top", fontsize=6.5, color=COL_TRUE)
ax.set_xticks(x); ax.set_xticklabels([l for _, l in TRAITS])
ax.set_ylabel("Pearson r  with consensus latent")
ax.legend(loc="lower left", fontsize=6.5)
ax.set_title(f"Trait signal beyond WM  (ensemble; n={n}; number = BF$_{{10}}$ on partial r)")
fig.tight_layout(); fig.savefig(f"{DEC}/panel_ens_beyond_wm.png", dpi=600, bbox_inches="tight"); plt.show()
print("raw :", {l: round(r, 3) for (_, l), r in zip(TRAITS, raw)})
print("|WM :", {l: round(r, 3) for (_, l), r in zip(TRAITS, par)}, " BF:", [round(b, 2) for b in bfs])'''

MD_HEAD = nbformat.v4.new_markdown_cell(
    "# Ensemble (pooled-PCA consensus) replication of the z2 analysis\n"
    "The single-seed (canonical 999) z2 panels above are replicated here using the **strongest ensemble "
    "latent** — the consensus axis from one PCA over **all 10 seeds' concatenated latents** "
    "(`z2_consensus_pca.npy`; this is **PC1**, the top shared-variance direction). It is a single fixed, "
    "basis-free 236-vector, so it slots in exactly where "
    "single-seed z2 was used. As a denoised consensus it carries a somewhat stronger trait signal than any "
    "single seed (de-attenuation; see the pooled-PCA validation panel). Each plot is its own cell.")

PANELS = [
    ("# === ensemble panel — consensus latent correlations with every trait", E1, f"{DEC}/panel_ens_correlations.png",
     "## Ensemble consensus latent — the individual-difference axis\n"
     "Pearson r of the consensus ensemble latent with every questionnaire/WM score (all bars = the same "
     "latent; numbers = BF₁₀ per correlation). Same axis as single-seed z2: strong negative for WM, "
     "positive for openness/curiosity — sharper because the latent is denoised across seeds."),
    ("# === ensemble panel — openness evidence", E2, f"{DEC}/panel_ens_openness_evidence.png",
     "## Openness — evidence increase over the original study (ensemble latent)\n"
     "Original study's strongest Openness↔task-measure correlation was r=0.11 (BF₁₀≈0.34, favouring the "
     "null). The ensemble consensus latent gives r≈0.29 (BF₁₀ in the thousands) — flipping from no "
     "evidence to strong evidence. The ratio is larger than the single-seed (~41×) because the consensus "
     "latent is de-attenuated; treat it as the upper-reliability estimate, single-seed as the conservative one."),
    ("# === ensemble panel — consensus latent is the explore/exploit", E3, f"{DEC}/panel_ens_switching.png",
     "## Consensus latent = the explore/exploit axis — directed switching, not randomness\n"
     "The consensus latent is **PC1** — the top shared-variance direction (≈36%); PCA is blind to switch "
     "rates and traits, so its switch correlation is **emergent, not definitional**. **Left:** PC1 tracks "
     "each subject's overall switch rate (r≈0.28). **Right:** partial r(latent, switch | choice entropy) per "
     "task — directed switching beyond randomness, **concentrated in the horizon task** (2-armed/restless ≈ 0). "
     "Note the ensemble axis is *more horizon-specific* than single-seed z2 (which also loaded on 2-armed/"
     "restless switching): the axis **shared across seeds** is specifically horizon-directed exploration, "
     "while the broader switching in z2 was partly seed-specific."),
    ("# === ensemble panel — trait signal beyond working memory", E4, f"{DEC}/panel_ens_beyond_wm.png",
     "## Trait signal beyond working memory (ensemble latent)\n"
     "Raw vs WM-partialled correlation of openness and curiosity with the ensemble consensus latent. Both "
     "survive controlling for working memory — genuine individual-difference variance beyond cognitive "
     "ability, now on the denoised ensemble axis."),
]


def render(src, png):
    ns = {}
    exec(A.SETUP, ns)
    exec(src, ns)
    assert os.path.exists(png), f"render produced no {png}"


def upsert_md(nb, text):
    """Refresh the markdown cell with the same '## title' in place, else append."""
    import re
    m = re.search(r"##\s+(.+)", text); key = m.group(1)[:40] if m else text[:40]
    for c in nb.cells:
        if c.cell_type == "markdown" and key in "".join(c.source):
            c.source = text; return
    nb.cells.append(nbformat.v4.new_markdown_cell(text))


def main():
    for _, src, png, _ in PANELS:
        render(src, png); print("rendered", png)
    nb = nbformat.read(NB, as_version=4)
    if not any(c.cell_type == "markdown" and "Ensemble (pooled-PCA consensus) replication" in "".join(c.source) for c in nb.cells):
        nb.cells.append(MD_HEAD)
    appended = 0
    for marker, src, png, md in PANELS:
        upsert_md(nb, md)
        if A.upsert(nb, marker, src, embed_png=png):
            appended += 1
    nbformat.write(nb, NB)
    print(f"done: appended {appended} ensemble panel cells; notebook now {len(nb.cells)} cells")


if __name__ == "__main__":
    main()
