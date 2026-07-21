#!/usr/bin/env python3
"""Insert 4 self-contained synthetic-aesthetic cells into thalmann_results.ipynb
(each loads its own data + plots). Rely on the shared SETUP cell for palette/helpers.
  A: in-sample multiple-R (IDRNN z vs dim-matched vanilla h)   [4-metric grid top-left]
  B: LOO pred-vs-target r (IDRNN vs vanilla)                   [4-metric grid bottom-left]
  C: latent dimension z2 correlations with every trait
  D: openness — 41x evidence increase over the original study's best task measure
All on the 3-task fit (final_plots/thalmann_z3_3task_full)."""
import base64, os, nbformat
import build_panela_cell as A           # reuse the exact SETUP cell for PNG generation

NB = "thalmann_results.ipynb"
FULL = "final_plots/thalmann_z3_3task_full"

_HEAD = r'''
import os, numpy as np, pandas as pd
FULL = "final_plots/thalmann_z3_3task_full"   # 3-task fit (cell-8b data)
PERSONALITY = ["PANAS_PA","PANAS_NA","STICSA","PHQ","CEI","BIG5_open"]
COMPOSITE   = ["AxDep","posMood","negMood","Exp"]
WM_KEYS     = ["WM_composite","WM_OS","WM_SS","WM_WMU"]
ALL_KEYS    = PERSONALITY + COMPOSITE + WM_KEYS
LBL = {"PANAS_PA":"PANAS +","PANAS_NA":"PANAS −","STICSA":"STICSA","PHQ":"PHQ","CEI":"CEI",
       "BIG5_open":"Openness","AxDep":"Anx/Dep","posMood":"pos mood","negMood":"neg mood",
       "Exp":"Exploration","WM_composite":"WM comp","WM_OS":"WM OS","WM_SS":"WM SS","WM_WMU":"WM upd"}
labels = [LBL[k] for k in ALL_KEYS]
x = np.arange(len(labels)); w = 0.38
'''

CELL_A = "# === panel — in-sample multiple-R (IDRNN z vs dim-matched vanilla h) ===" + _HEAD + r'''
df = pd.read_csv(f"{FULL}/decoding/seed_averaged_decoding.csv").set_index("target").reindex(ALL_KEYS)
fig, ax = plt.subplots(figsize=(7.2, 3.3))
ax.bar(x - w/2, df["idrnn_pearson_mean"], w, yerr=df["idrnn_pearson_sd"], color=COL_IDRNN,
       edgecolor="black", linewidth=0.4, capsize=1.5, error_kw=dict(elinewidth=0.7, ecolor="#444444"),
       label="IDRNN z (3-dim)", zorder=2)
ax.bar(x + w/2, df["vanilla_pearson_mean"], w, yerr=df["vanilla_pearson_sd"], color=COL_VANILLA,
       edgecolor="black", linewidth=0.4, capsize=1.5, error_kw=dict(elinewidth=0.7, ecolor="#444444"),
       label="Vanilla h (dim-matched)", zorder=2)
# in-sample multiple-R null E[R]~sqrt(k/(n-1)), k=3; n=236 (personality) .. 175 (WM)
ax.axhspan(np.sqrt(3/235), np.sqrt(3/174), color=COL_TRUE, alpha=0.12, zorder=0)
ax.text(len(labels)-0.5, np.sqrt(3/174), " chance (k=3)", va="bottom", ha="right", fontsize=6, color=COL_TRUE)
for xb in (len(PERSONALITY)-.5, len(PERSONALITY)+len(COMPOSITE)-.5):
    ax.axvline(xb, color="grey", lw=.5, ls=":", alpha=.5)
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
ax.set_ylabel("in-sample multiple R"); ax.legend(loc="upper left")
ax.set_title("In-sample decodability — multiple R (3 latent dims)")
fig.tight_layout(); fig.savefig(f"{FULL}/decoding/panel_multiple_R.png", dpi=600, bbox_inches="tight"); plt.show()'''

CELL_B = "# === panel — LOO pred-vs-target r (IDRNN vs dim-matched vanilla) ===" + _HEAD + r'''
df = pd.read_csv(f"{FULL}/decoding/seed_averaged_decoding.csv").set_index("target").reindex(ALL_KEYS)
fig, ax = plt.subplots(figsize=(7.2, 3.3))
ax.bar(x - w/2, df["idrnn_loo_r_mean"], w, yerr=df["idrnn_loo_r_sd"], color=COL_IDRNN,
       edgecolor="black", linewidth=0.4, capsize=1.5, error_kw=dict(elinewidth=0.7, ecolor="#444444"),
       label="IDRNN z (3-dim)", zorder=2)
ax.bar(x + w/2, df["vanilla_loo_r_mean"], w, yerr=df["vanilla_loo_r_sd"], color=COL_VANILLA,
       edgecolor="black", linewidth=0.4, capsize=1.5, error_kw=dict(elinewidth=0.7, ecolor="#444444"),
       label="Vanilla h (dim-matched)", zorder=2)
ax.axhline(0, color=COL_TRUE, lw=0.8, zorder=1)
for xb in (len(PERSONALITY)-.5, len(PERSONALITY)+len(COMPOSITE)-.5):
    ax.axvline(xb, color="grey", lw=.5, ls=":", alpha=.5)
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
ax.set_ylabel("LOO pred-vs-target r"); ax.legend(loc="lower left")
ax.set_title("Out-of-sample readout — LOO r  (negative = no generalizable signal)")
fig.tight_layout(); fig.savefig(f"{FULL}/decoding/panel_loo_r.png", dpi=600, bbox_inches="tight"); plt.show()'''

CELL_C = "# === panel — latent dimension z2 correlations with every trait ===" + _HEAD + r'''
df = pd.read_csv(f"{FULL}/decoding/idrnn_latent_correlations.csv").set_index("target").reindex(ALL_KEYS)
r = df["z2_pearson_r"].values; n = df["n"].values.astype(int)
bf = [float(bayesfactor_pearson(rr, int(nn))) for rr, nn in zip(r, n)]   # JZS BF per correlation
fig, ax = plt.subplots(figsize=(7.2, 3.4))
ax.bar(x, r, 0.6, color=COL_IDRNN, edgecolor="black", linewidth=0.4, zorder=2)   # all blue = IDRNN z2
ax.axhline(0, color=COL_TRUE, lw=0.8)
for i, (v, b) in enumerate(zip(r, bf)):                                  # report Bayes factors, not stars
    ax.text(x[i], v + (0.015 if v >= 0 else -0.015), fmt_bf(b).replace("BF=", ""), ha="center",
            va="bottom" if v >= 0 else "top", fontsize=5.5, rotation=90, color="#222222")
for xb in (len(PERSONALITY)-.5, len(PERSONALITY)+len(COMPOSITE)-.5):
    ax.axvline(xb, color="grey", lw=.5, ls=":", alpha=.5)
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
ax.set_ylim(min(r)-0.14, max(r)+0.12)
ax.set_ylabel("Pearson r  (latent dim z2)")
ax.set_title("Latent dimension z2 — the individual-difference axis  (numbers = BF$_{10}$)")
fig.tight_layout(); fig.savefig(f"{FULL}/decoding/panel_z2_correlations.png", dpi=600, bbox_inches="tight"); plt.show()'''

CELL_D = "# === panel — openness evidence: IDRNN latent vs original study's best task measure ===" + r'''
import os, numpy as np, pandas as pd
FULL = "final_plots/thalmann_z3_3task_full"
N_OPEN = 236
R_ORIG = 0.11   # original study: strongest Openness x task-measure across their 13 measures
r_ours = float(pd.read_csv(f"{FULL}/decoding/idrnn_latent_correlations.csv")
               .set_index("target").loc["BIG5_open", "z2_pearson_r"])      # IDRNN latent z2
bf_orig = float(bayesfactor_pearson(R_ORIG, N_OPEN))
bf_ours = float(bayesfactor_pearson(r_ours, N_OPEN))
labels = [f"Original\n(best task measure)\nr={R_ORIG:.2f}", f"IDRNN latent z2\nr={r_ours:.2f}"]
vals = [bf_orig, bf_ours]; cols = [COL_VANILLA, COL_IDRNN]
fig, ax = plt.subplots(figsize=(3.6, 3.6))
ax.bar([0, 1], vals, 0.6, color=cols, edgecolor="black", linewidth=0.5, zorder=3)
for thr, lab in [(1, "BF=1 (none)"), (3, "3 (subst.)"), (10, "10 (strong)")]:
    ax.axhline(thr, ls=":", color=COL_TRUE, lw=0.8, zorder=1)
    ax.text(1.55, thr, lab, va="center", ha="left", fontsize=6, color=COL_TRUE)
for i, v in enumerate(vals):
    ax.text(i, v*1.15, fmt_bf(v), ha="center", va="bottom", fontsize=8, fontweight="bold")
ax.annotate(f"×{bf_ours/bf_orig:.0f} evidence", xy=(0.5, bf_ours), ha="center", va="bottom",
            fontsize=8, color=COL_IDRNN, fontweight="bold", xytext=(0.5, bf_ours*2.2),
            arrowprops=None)
ax.set_yscale("log"); ax.set_ylim(0.1, max(vals)*4)
ax.set_xticks([0, 1]); ax.set_xticklabels(labels, fontsize=7)
ax.set_ylabel("Bayes factor BF$_{10}$ (Openness link)")
ax.set_title("Openness: undetectable → strong\n(BF favoring null → strong evidence)")
fig.tight_layout(); fig.savefig(f"{FULL}/decoding/panel_openness_evidence.png", dpi=600, bbox_inches="tight"); plt.show()
print(f"openness: original r={R_ORIG} BF={bf_orig:.2f} | ours r={r_ours:.3f} BF={bf_ours:.2f} | ratio {bf_ours/bf_orig:.0f}x")'''

PANELS = [
    ("# === panel — in-sample multiple-R", CELL_A, f"{FULL}/decoding/panel_multiple_R.png",
     "## In-sample decodability — multiple R (IDRNN z vs dim-matched vanilla)\n"
     "In-sample multiple R = √(R²) regressing each trait on the model's 3 latent dims (IDRNN z, or "
     "PCA-dim-matched vanilla h). **Error bars = ±1 SD across the 10 retrain seeds.** **Grey band = "
     "E[R] under the null** ≈ √(k/(n−1)), k=3 predictors, n=175 (WM)→236 (personality) ≈ 0.11–0.13 "
     "— the in-sample R expected from 3 *random* predictors (not the p<.05 threshold, which is ≈0.18–0.21)."),
    ("# === panel — LOO pred-vs-target r", CELL_B, f"{FULL}/decoding/panel_loo_r.png",
     "## Out-of-sample readout — LOO r (IDRNN vs dim-matched vanilla)\n"
     "Bottom-left of the 4-metric grid. Leave-one-out pred-vs-target correlation (negative = no "
     "generalizable structure). IDRNN is positive for WM + openness; dim-matched vanilla is ~0/negative."),
    ("# === panel — latent dimension z2 correlations", CELL_C, f"{FULL}/decoding/panel_z2_correlations.png",
     "## Latent dimension z2 — the individual-difference axis\n"
     "Pearson r of the single latent dim z2 with every questionnaire/WM score (all bars = IDRNN z2). "
     "Numbers above/below bars are **Bayes factors (BF₁₀)** per correlation. z2 is consistently the "
     "trait axis: strong for WM (≈−0.35, BF≈10³–10⁴) and openness (+0.21, BF≈14)."),
    ("# === panel — openness evidence", CELL_D, f"{FULL}/decoding/panel_openness_evidence.png",
     "## Openness — 41× increase in evidence over the original study\n"
     "The original study's strongest Openness↔task-measure correlation was r=0.11 (BF₁₀=0.34, "
     "favoring the null). The IDRNN latent z2 gives r=0.21 (BF₁₀≈14) — a ~41× increase, "
     "flipping from no evidence to strong evidence (n=236)."),
]


def main():
    nb = nbformat.read(NB, as_version=4)

    def upsert(marker, source, md, png):
        out = []
        if os.path.exists(png):
            out = [nbformat.v4.new_output("display_data",
                   data={"image/png": base64.b64encode(open(png, "rb").read()).decode()}, metadata={})]
        for c in nb.cells:
            if c.cell_type == "code" and marker in "".join(c.source):
                c.source = source; c.outputs = out; c.execution_count = None
                return "refreshed"
        nb.cells.append(nbformat.v4.new_markdown_cell(md))
        code = nbformat.v4.new_code_cell(source); code.outputs = out
        nb.cells.append(code)
        return "appended"

    for marker, source, png, md in PANELS:
        print(f"  {marker[:40]:42s} {upsert(marker, source, md, png)}")
    nbformat.write(nb, NB)
    print(f"{len(nb.cells)} cells")


if __name__ == "__main__":
    main()
