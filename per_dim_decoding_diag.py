#!/usr/bin/env python3
"""
Per-dimension univariate correlation diagnostic for the step-1 vs vanilla
decoding run.  For each scale, correlate every single dimension of the
best-seed latents with the questionnaire score and report the strongest hit.

Uses the latents already saved in plots_thalmann/step1_vs_vanilla/.
"""
import os, glob
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "plots_thalmann/step1_vs_vanilla"

SCALES = {
    "PANAS_PA":  [f"PANAS_{i}"  for i in [0,2,4,6,8,14,16,18]],
    "PANAS_NA":  [f"PANAS_{i}"  for i in [1,3,5,7,9,11,13,15]],
    "STICSA":    [f"STICSA_{i}" for i in range(22)],
    "PHQ":       [f"PHQ_9_{i}"  for i in range(10)],
    "CEI":       [f"CEI_{i}"    for i in range(4)],
    "BIG5_open": [f"BIG_5_{i}"  for i in range(6)],
}

def bonf_stars(p, k):
    """Bonferroni-adjusted stars for k tests within a scale."""
    pa = min(p * k, 1.0)
    if pa < 0.001: return "***"
    if pa < 0.01:  return "**"
    if pa < 0.05:  return "*"
    return ""

def load_latents(pattern):
    f = glob.glob(os.path.join(OUT_DIR, pattern))
    assert len(f) == 1, f"Expected one file for {pattern}, got {f}"
    d = torch.load(f[0], weights_only=False)
    return d, f[0]

def main():
    idrnn, f_i = load_latents("latents_idrnn_step1_bestseed*.pt")
    van,   f_v = load_latents("latents_vanilla_bestseed*.pt")
    print(f"IDRNN   : {f_i}  z shape={idrnn['z'].shape}  seed={idrnn['seed']}")
    print(f"Vanilla : {f_v}  h shape={van['h'].shape}  seed={van['seed']}")

    subids = np.asarray(idrnn["subids"])
    assert np.array_equal(subids, np.asarray(van["subids"])), "subid order mismatch"

    quest = pd.read_csv("data/finalQuestionnaireDataSession1.csv").set_index("ID")
    for k, items in SCALES.items():
        quest[k] = quest[items].mean(axis=1)
    y_df = quest.reindex(subids)[list(SCALES.keys())]

    Z = np.asarray(idrnn["z"])    # (236, 10)
    H = np.asarray(van["h"])      # (236, 5)

    rows = []
    print(f"\n{'scale':<10}  {'model':<8}  {'dim':<4}  {'best_r':>8}  {'p':>10}  {'p_Bonf':>10}  stars")
    print("-" * 75)
    for sc in SCALES:
        y = y_df[sc].values.astype(float)
        mask = ~np.isnan(y)
        yy = y[mask]
        for name, M in [("IDRNN", Z), ("Vanilla", H)]:
            Ms = M[mask]
            rs  = np.array([pearsonr(Ms[:, k], yy)[0] for k in range(Ms.shape[1])])
            ps  = np.array([pearsonr(Ms[:, k], yy)[1] for k in range(Ms.shape[1])])
            k_best = int(np.argmax(np.abs(rs)))
            r, p = rs[k_best], ps[k_best]
            K = Ms.shape[1]
            p_bonf = min(p * K, 1.0)
            stars = bonf_stars(p, K)
            print(f"{sc:<10}  {name:<8}  {k_best:<4}  {r:+.3f}    {p:.3g}    {p_bonf:.3g}    {stars}")
            for k in range(K):
                rows.append({"scale": sc, "model": name, "dim": k,
                             "r": rs[k], "p": ps[k], "K": K})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "per_dim_correlations.csv"), index=False)
    print(f"\nSaved {os.path.join(OUT_DIR, 'per_dim_correlations.csv')}")

    # ── Heatmap: |r| per (scale, dim) for each model ──────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5),
                             gridspec_kw={"width_ratios": [10, 5]})
    for ax, name in zip(axes, ["IDRNN", "Vanilla"]):
        sub = df[df["model"] == name]
        piv = sub.pivot(index="scale", columns="dim", values="r")
        piv = piv.reindex(list(SCALES.keys()))
        im = ax.imshow(piv.values, cmap="RdBu_r", vmin=-0.25, vmax=0.25,
                        aspect="auto")
        ax.set_xticks(range(piv.shape[1]));
        ax.set_xticklabels(piv.columns.astype(int))
        ax.set_yticks(range(piv.shape[0]))
        ax.set_yticklabels(piv.index)
        ax.set_xlabel(f"{name} dim")
        # Bonferroni-adjusted significance annotation
        K = piv.shape[1]
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                pval = sub[(sub["scale"] == piv.index[i])
                          & (sub["dim"] == piv.columns[j])]["p"].values[0]
                stars = bonf_stars(pval, K)
                if stars:
                    ax.text(j, i, stars, ha="center", va="center",
                            fontsize=10, color="k", fontweight="bold")
        ax.set_title(f"{name}  —  Pearson r  (Bonf ** over {K} dims)")
        plt.colorbar(im, ax=ax, fraction=0.035)
    fig.suptitle("Per-dim correlation of latent with questionnaire score",
                 fontweight="bold")
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "per_dim_correlations.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")

if __name__ == "__main__":
    main()
