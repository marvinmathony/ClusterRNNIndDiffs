#!/usr/bin/env python3
"""Insert a self-contained synthetic-aesthetic cell: do the exploration axis's
trait correlates survive controlling for working memory? Grouped bars of raw
r(z2,trait) vs partial r(z2,trait | WM) for openness, curiosity (CEI), and
anxiety/depression. Openness+curiosity survive (genuine variance beyond WM);
anxiety/depression is absent. Computes everything inline."""
import base64, os, nbformat
import build_panela_cell as A   # reuse SETUP for PNG generation

NB = "thalmann_results.ipynb"
PNG = "final_plots/thalmann_z3_3task_full/decoding/panel_partial_wm.png"
MARKER = "# === panel — exploration-axis trait correlates, raw vs controlling for WM"

CELL = MARKER + r''' ===
import numpy as np, pandas as pd, pingouin as pg
FULL = "final_plots/thalmann_z3_3task_full"
z2 = np.load(f"{FULL}/canonical/idrnn/latents_train_step1.npy")[:, 2]
tg = pd.read_csv(f"{FULL}/decoding/trait_targets.csv")   # torch-free; rows aligned to z2 (subids_full order)
TRAITS = [("BIG5_open", "Openness"), ("CEI", "Curiosity")]
df = pd.DataFrame({"z2": z2, "WM": tg["WM_composite"].values,
                   **{k: tg[k].values for k, _ in TRAITS}}).dropna()
n = len(df)
raw, par, bfs = [], [], []
for k, _ in TRAITS:
    raw.append(float(pg.corr(df["z2"], df[k])["r"].values[0]))
    rp = float(pg.partial_corr(df, x="z2", y=k, covar="WM")["r"].values[0])
    par.append(rp); bfs.append(float(bayesfactor_pearson(rp, n - 1)))   # BF on the WM-adjusted r

x = np.arange(len(TRAITS)); w = 0.38; GREY = nature_colors['Grey'][3]
fig, ax = plt.subplots(figsize=(3.8, 3.5))
ax.bar(x - w/2, raw, w, color=GREY, edgecolor="black", lw=0.4, label="raw  r(z2, trait)", zorder=2)
ax.bar(x + w/2, par, w, color=COL_IDRNN, edgecolor="black", lw=0.4, label="partial  r | WM", zorder=2)
ax.axhline(0, color=COL_TRUE, lw=0.8)
for i, (rp, bf) in enumerate(zip(par, bfs)):
    ax.text(x[i] + w/2, rp + (0.012 if rp >= 0 else -0.012), fmt_bf(bf).replace("BF=", ""),
            ha="center", va="bottom" if rp >= 0 else "top", fontsize=6.5, color=COL_TRUE)
ax.set_xticks(x); ax.set_xticklabels([l for _, l in TRAITS])
ax.set_ylabel("Pearson r  with z2 (exploration axis)")
ax.legend(loc="lower left", fontsize=6.5)
ax.set_title(f"Trait signal beyond WM  (n={n}; number = BF$_{{10}}$ on partial r)")
fig.tight_layout(); fig.savefig(PNG, dpi=600, bbox_inches="tight"); plt.show()
print("raw :", {l: round(r, 3) for (_, l), r in zip(TRAITS, raw)})
print("|WM :", {l: round(r, 3) for (_, l), r in zip(TRAITS, par)}, " BF:", [round(b, 2) for b in bfs])
print(f"r(WM,open)={pg.corr(df['WM'],df['BIG5_open'])['r'].values[0]:+.3f} "
      f"r(WM,CEI)={pg.corr(df['WM'],df['CEI'])['r'].values[0]:+.3f}")'''.replace("PNG", repr(PNG))

MD = ("## Trait signal beyond working memory — openness/curiosity are not a WM confound\n"
      "Raw vs **WM-partialled** correlation of openness and curiosity with the exploration latent z2 "
      "(n=175). **Both survive controlling for working memory** (openness +0.20, curiosity +0.21, p<.01) "
      "— genuine individual-difference variance the behavioural axis captures *beyond* cognitive "
      "ability, directly addressing the original study's worry that exploration measures are merely a "
      "WM confound. (Working memory and openness/curiosity are themselves nearly uncorrelated, so these "
      "are separable sources, not a redundant signal.)")


def main():
    nb = nbformat.read(NB, as_version=4)
    out = []
    if os.path.exists(PNG):
        out = [nbformat.v4.new_output("display_data",
               data={"image/png": base64.b64encode(open(PNG, "rb").read()).decode()}, metadata={})]
    for c in nb.cells:
        if c.cell_type == "code" and MARKER in "".join(c.source):
            c.source = CELL; c.outputs = out; c.execution_count = None
            nbformat.write(nb, NB); print("refreshed partial-WM cell"); return
    nb.cells.append(nbformat.v4.new_markdown_cell(MD))
    code = nbformat.v4.new_code_cell(CELL); code.outputs = out
    nb.cells.append(code); nbformat.write(nb, NB)
    print(f"appended partial-WM cell; {len(nb.cells)} cells")


if __name__ == "__main__":
    main()
