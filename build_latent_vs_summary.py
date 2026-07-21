#!/usr/bin/env python3
"""Self-contained synthetic-aesthetic cell: the latent recovers an openness signal
that behavioural summary statistics miss. Left: openness correlation with overall
switch rate vs horizon switch rate vs the IDRNN latent z2 (escalating). Right:
partial correlations showing z2 carries openness beyond raw switching, but raw
switching adds nothing beyond z2. Computes everything inline."""
import base64, os, nbformat
import build_panela_cell as A

NB = "thalmann_results.ipynb"
PNG = "final_plots/thalmann_z3_3task_full/decoding/panel_latent_vs_summary.png"
MARKER = "# === panel — the latent extracts openness that behavioural summaries miss"

CELL = MARKER + r''' ===
import numpy as np, pandas as pd, pingouin as pg
FULL = "final_plots/thalmann_z3_3task_full"
z2 = np.load(f"{FULL}/canonical/idrnn/latents_train_step1.npy")[:, 2]
c = np.load("data_thalmann_3task_full/c_train.npy"); tid = np.load("data_thalmann_3task_full/task_ids_per_block.npy")
N = c.shape[0]
def sw(bl):
    s = p = 0
    for b in range(bl.shape[0]):
        ch = bl[b]; ch = ch[ch >= 0]
        if len(ch) < 2: continue
        s += int((ch[1:] != ch[:-1]).sum()); p += len(ch) - 1
    return s / p if p else np.nan
sw_all = np.array([sw(c[i]) for i in range(N)])
sw_h   = np.array([sw(c[i][tid == 2]) for i in range(N)])
op = pd.read_csv(f"{FULL}/decoding/trait_targets.csv")["BIG5_open"].values   # torch-free
df = pd.DataFrame({"z2": z2, "sw": sw_all, "swh": sw_h, "open": op}).dropna(); n = len(df)
def C(a, b): return float(pg.corr(df[a], df[b])["r"].values[0])

GREY2, GREY4 = nature_colors['Grey'][2], nature_colors['Grey'][4]
ITEMS = [("overall\nswitch rate", "sw", GREY2), ("horizon\nswitch rate", "swh", GREY4),
         ("IDRNN\nlatent z2", "z2", COL_IDRNN)]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.4), gridspec_kw={"width_ratios": [1.55, 1]})
for i, (lab, k, col) in enumerate(ITEMS):
    r = C(k, "open"); bf = float(bayesfactor_pearson(r, n))
    ax1.bar(i, r, 0.62, color=col, edgecolor="black", lw=0.4, zorder=2)
    ax1.text(i, r + 0.004, f"r={r:.2f}\n{fmt_bf(bf)}", ha="center", va="bottom", fontsize=6.5)
ax1.axhline(0, color=COL_TRUE, lw=0.8)
ax1.set_xticks(range(len(ITEMS))); ax1.set_xticklabels([l for l, _, _ in ITEMS], fontsize=7)
ax1.set_ylabel("Pearson r  with openness"); ax1.set_ylim(top=0.30)
ax1.set_title(f"openness signal by measure (n={n})")
# right: unique contributions (partial r)
pz = pg.partial_corr(df, x="z2", y="open", covar="sw")
ps = pg.partial_corr(df, x="sw", y="open", covar="z2")
vals = [float(pz["r"].values[0]), float(ps["r"].values[0])]
pvs  = [float(pz["p-val"].values[0]), float(ps["p-val"].values[0])]
ax2.bar([0, 1], vals, 0.6, color=[COL_IDRNN, GREY4], edgecolor="black", lw=0.4, zorder=2)
ax2.axhline(0, color=COL_TRUE, lw=0.8)
for i, (v, pp) in enumerate(zip(vals, pvs)):
    ax2.text(i, v + (0.008 if v >= 0 else -0.008), f"{v:+.2f}\np={pp:.0e}", ha="center",
             va="bottom" if v >= 0 else "top", fontsize=6.5)
ax2.set_xticks([0, 1]); ax2.set_xticklabels(["z2 | switch", "switch | z2"], fontsize=7)
ax2.set_ylabel("partial r  with openness"); ax2.set_title("unique contribution")
fig.tight_layout(); fig.savefig(PNG, dpi=600, bbox_inches="tight"); plt.show()
print(f"n={n}  r(open): overall-switch={C('sw','open'):+.3f}  horizon-switch={C('swh','open'):+.3f}  z2={C('z2','open'):+.3f}")
print(f"partial r(z2,open|switch)={vals[0]:+.3f} (p={pvs[0]:.1e})   r(switch,open|z2)={vals[1]:+.3f} (p={pvs[1]:.1e})")'''.replace("PNG", repr(PNG))

MD = ("## The latent extracts an openness signal that behavioural summaries miss\n"
      "**Left:** openness correlates ~0 with an overall switch-rate summary, modestly with horizon "
      "switching, and most strongly with the IDRNN latent z2 — the model learns *which* switching "
      "carries the trait signal (it weights directed/horizon exploration). **Right:** z2 predicts "
      "openness even after removing raw switching (partial r≈+0.26), whereas raw switching adds nothing "
      "beyond z2 (≈−0.13, n.s.). So the openness link was *accessible* in the behaviour but was "
      "discarded by conventional summary statistics — the model-based latent recovers it.")


def main():
    nb = nbformat.read(NB, as_version=4)
    out = []
    if os.path.exists(PNG):
        out = [nbformat.v4.new_output("display_data",
               data={"image/png": base64.b64encode(open(PNG, "rb").read()).decode()}, metadata={})]
    for c in nb.cells:
        if c.cell_type == "code" and MARKER in "".join(c.source):
            c.source = CELL; c.outputs = out; c.execution_count = None
            nbformat.write(nb, NB); print("refreshed latent-vs-summary cell"); return
    nb.cells.append(nbformat.v4.new_markdown_cell(MD))
    code = nbformat.v4.new_code_cell(CELL); code.outputs = out
    nb.cells.append(code); nbformat.write(nb, NB)
    print(f"appended latent-vs-summary cell; {len(nb.cells)} cells")


if __name__ == "__main__":
    main()
