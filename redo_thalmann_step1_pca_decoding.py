#!/usr/bin/env python3
"""
PCA-reduced decoding for the cached step-1 latents.

Loads:
  plots_thalmann/step1_vs_vanilla/latents_idrnn_step1_bestseed{S}.pt  (236 × z_dim)
  plots_thalmann/step1_vs_vanilla/latents_vanilla_bestseed{S}.pt      (236 × hidden)

For n_pc ∈ {1, 2}:
  - PCA each latent set independently, keep the first n_pc components
  - In-sample OLS → r per questionnaire scale (item-aggregate + CFA factor)
  - Steiger test for IDRNN-vs-Vanilla difference

Since both sides feed n_pc predictors into the OLS, the Steiger comparison is
bias-free with respect to predictor count.

Outputs (plots_thalmann/step1_vs_vanilla/):
  decoding_results_pca.csv
  step1_vs_vanilla_pca_decoding.png
"""
import os, glob
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr, norm, f as f_dist

OUT_DIR = "plots_thalmann/step1_vs_vanilla"

SCALES = {
    "PANAS_PA":  ([f"PANAS_{i}"  for i in [0,2,4,6,8,14,16,18]], "PANAS\nPos. Affect"),
    "PANAS_NA":  ([f"PANAS_{i}"  for i in [1,3,5,7,9,11,13,15]], "PANAS\nNeg. Affect"),
    "STICSA":    ([f"STICSA_{i}" for i in range(22)],             "STICSA\nAnxiety"),
    "PHQ":       ([f"PHQ_9_{i}"  for i in range(10)],             "PHQ-9\nDepression"),
    "CEI":       ([f"CEI_{i}"    for i in range(4)],              "CEI\nCuriosity"),
    "BIG5_open": ([f"BIG_5_{i}"  for i in range(6)],              "BIG5\nOpenness"),
    "AxDep":     (["AxDep"],                                      "CFA\nAxDep"),
    "posMood":   (["posMood"],                                    "CFA\nposMood"),
    "negMood":   (["negMood"],                                    "CFA\nnegMood"),
    "Exp":       (["Exp"],                                        "CFA\nExploration"),
}
SCALE_KEYS = list(SCALES.keys())


def insample_ols(Z, y):
    n, p = Z.shape
    clf = LinearRegression().fit(Z, y)
    preds = clf.predict(Z)
    ss_res = np.sum((y - preds) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot
    dfn, dfd = p, n - p - 1
    F = (r2 / dfn) / ((1 - r2) / dfd) if r2 < 1 else np.inf
    p_val = float(f_dist.sf(F, dfn, dfd))
    return float(np.sqrt(max(r2, 0.0))), p_val, preds


def steiger_test(r12, r13, r23, n):
    r12 = np.clip(r12, -0.9999, 0.9999)
    r13 = np.clip(r13, -0.9999, 0.9999)
    r23 = np.clip(r23, -0.9999, 0.9999)
    z12, z13 = np.arctanh(r12), np.arctanh(r13)
    rsq_bar = (r12 ** 2 + r13 ** 2) / 2
    f = (1 - r23) / (2 * (1 - rsq_bar))
    h = (1 - f * rsq_bar) / (1 - rsq_bar)
    var = (2 * (1 - r23) / (n - 1)) * h
    if var <= 0:
        return 0.0, 1.0
    z = (z12 - z13) / np.sqrt(var)
    return float(z), float(2 * norm.sf(abs(z)))


def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "n.s."


def main():
    idrnn_path   = sorted(glob.glob(os.path.join(OUT_DIR, "latents_idrnn_step1_bestseed*.pt")))[0]
    vanilla_path = sorted(glob.glob(os.path.join(OUT_DIR, "latents_vanilla_bestseed*.pt")))[0]
    print(f"Loading IDRNN  : {idrnn_path}")
    print(f"Loading Vanilla: {vanilla_path}")

    d_i = torch.load(idrnn_path,   map_location="cpu", weights_only=False)
    d_v = torch.load(vanilla_path, map_location="cpu", weights_only=False)
    z_idrnn   = np.asarray(d_i["z"])
    h_vanilla = np.asarray(d_v["h"])
    subids    = np.asarray(d_i["subids"])
    assert np.array_equal(subids, np.asarray(d_v["subids"])), "subid orders disagree"
    seed_i = int(d_i["seed"]); seed_v = int(d_v["seed"])
    print(f"  IDRNN   seed {seed_i}, z shape {z_idrnn.shape}")
    print(f"  Vanilla seed {seed_v}, h shape {h_vanilla.shape}")

    pca_z = PCA(n_components=z_idrnn.shape[1]).fit(z_idrnn)
    pca_h = PCA(n_components=h_vanilla.shape[1]).fit(h_vanilla)
    print(f"  IDRNN z   explained var ratio: {np.round(pca_z.explained_variance_ratio_, 3)}")
    print(f"  Vanilla h explained var ratio: {np.round(pca_h.explained_variance_ratio_, 3)}")

    quest = pd.read_csv("data/finalQuestionnaireDataSession1.csv").set_index("ID")
    factors = (pd.read_csv("data/CFA_compound_questionnaire_factors_s1.csv")
                 .set_index("ID")[["AxDep", "posMood", "negMood", "Exp"]])
    quest = quest.join(factors, how="left")
    for k, (items, _) in SCALES.items():
        quest[k] = quest[items].mean(axis=1)

    all_rows = []
    for n_pc in [1, 2]:
        print(f"\n── n_pc={n_pc}  (matched predictor count) ──")
        Zp = pca_z.transform(z_idrnn)[:, :n_pc]
        Hp = pca_h.transform(h_vanilla)[:, :n_pc]
        for scale in SCALE_KEYS:
            y_all = quest.reindex(subids)[scale].values.astype(float)
            mask = ~np.isnan(y_all)
            n = int(mask.sum())
            if n < 30:
                continue
            yy = y_all[mask]
            Zi = Zp[mask]; Hv = Hp[mask]
            r_i, p_i, preds_i = insample_ols(Zi, yy)
            r_v, p_v, preds_v = insample_ols(Hv, yy)
            r_iv, _ = pearsonr(preds_i, preds_v)
            z_s, p_s = steiger_test(r_i, r_v, r_iv, n)
            print(f"  {scale:<10} n={n}  IDRNN r={r_i:+.3f} ({p_i:.3g} {sig_stars(p_i)})  "
                  f"Vanilla r={r_v:+.3f} ({p_v:.3g} {sig_stars(p_v)})  "
                  f"Steiger p={p_s:.3g} {sig_stars(p_s)}")
            all_rows.append(dict(
                n_pc=n_pc, scale=scale, n=n,
                r_idrnn=r_i, p_idrnn=p_i,
                r_vanilla=r_v, p_vanilla=p_v,
                r_iv=r_iv, z_steiger=z_s, p_steiger=p_s,
            ))

    decode_df = pd.DataFrame(all_rows)
    out_csv = os.path.join(OUT_DIR, "decoding_results_pca.csv")
    decode_df.to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv}")

    fig, axes = plt.subplots(1, 2, figsize=(17, 5.5), sharey=True)
    c1, c2 = "#4C72B0", "#DD8452"

    for ax, n_pc in zip(axes, [1, 2]):
        sub = decode_df[decode_df["n_pc"] == n_pc].reset_index(drop=True)
        labels = [SCALES[k][1] for k in sub["scale"]]
        x = np.arange(len(labels))
        w = 0.35
        ax.bar(x - w/2, sub["r_idrnn"].abs(), w,
               color=c1, alpha=0.85, edgecolor="k", linewidth=0.5,
               label=f"IDRNN PC1..{n_pc} (seed {seed_i})")
        ax.bar(x + w/2, sub["r_vanilla"].abs(), w,
               color=c2, alpha=0.85, edgecolor="k", linewidth=0.5,
               label=f"Vanilla PC1..{n_pc} (seed {seed_v})")

        for i, row in sub.iterrows():
            ax.text(i - w/2, abs(row["r_idrnn"]) + 0.005, sig_stars(row["p_idrnn"]),
                    ha="center", va="bottom", fontsize=8, fontweight="bold", color=c1)
            ax.text(i + w/2, abs(row["r_vanilla"]) + 0.005, sig_stars(row["p_vanilla"]),
                    ha="center", va="bottom", fontsize=8, fontweight="bold", color=c2)
            y_max = max(abs(row["r_idrnn"]), abs(row["r_vanilla"])) + 0.05
            ax.plot([i - w/2, i - w/2, i + w/2, i + w/2],
                    [y_max - 0.01, y_max, y_max, y_max - 0.01],
                    color="k", linewidth=0.8)
            stars = sig_stars(row["p_steiger"])
            ax.text(i, y_max + 0.005, stars, ha="center", va="bottom",
                    fontsize=8, color=("k" if stars != "n.s." else "grey"))

        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("|r|" if n_pc == 1 else "")
        ax.set_title(f"PCA → {n_pc} component" + ("s" if n_pc > 1 else ""),
                     fontweight="bold")
        ax.legend(fontsize=8, loc="upper right")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    fig.suptitle("PCA-reduced decoding — IDRNN z vs Vanilla h, matched #predictors",
                 fontweight="bold")
    fig.tight_layout()
    out_png = os.path.join(OUT_DIR, "step1_vs_vanilla_pca_decoding.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
