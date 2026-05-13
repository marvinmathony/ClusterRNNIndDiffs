#!/usr/bin/env python3
"""
Re-run only the questionnaire decoding + plotting step of
train_and_decode_thalmann_step1.py using the already-saved best-seed latents.
Fixes the bug where vanilla decoding was computed on IDRNN latents.
"""
import os
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
}
SCALE_KEYS = list(SCALES.keys())


def insample_ols(Z, y):
    n, p = Z.shape
    clf   = LinearRegression().fit(Z, y)
    preds = clf.predict(Z)
    ss_res = np.sum((y - preds) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2     = 1.0 - ss_res / ss_tot
    dfn, dfd = p, n - p - 1
    F      = (r2 / dfn) / ((1 - r2) / dfd) if r2 < 1 else np.inf
    p_val  = float(f_dist.sf(F, dfn, dfd))
    return float(np.sqrt(max(r2, 0.0))), p_val, preds


def steiger_test(r12, r13, r23, n):
    r12 = np.clip(r12, -0.9999, 0.9999)
    r13 = np.clip(r13, -0.9999, 0.9999)
    r23 = np.clip(r23, -0.9999, 0.9999)
    z12, z13 = np.arctanh(r12), np.arctanh(r13)
    rsq_bar = (r12**2 + r13**2) / 2
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


def _find_latents(pattern):
    import glob
    hits = sorted(glob.glob(os.path.join(OUT_DIR, pattern)))
    if not hits:
        raise FileNotFoundError(pattern)
    return hits[0]


def main():
    idrnn_path   = _find_latents("latents_idrnn_step1_bestseed*.pt")
    vanilla_path = _find_latents("latents_vanilla_bestseed*.pt")
    print(f"Loading IDRNN latents   : {idrnn_path}")
    print(f"Loading Vanilla latents : {vanilla_path}")

    d_i = torch.load(idrnn_path,   map_location="cpu", weights_only=False)
    d_v = torch.load(vanilla_path, map_location="cpu", weights_only=False)
    z_idrnn   = np.asarray(d_i["z"])       # (236, z_dim)
    h_vanilla = np.asarray(d_v["h"])       # (236, hidden)
    subids_i  = np.asarray(d_i["subids"])
    subids_v  = np.asarray(d_v["subids"])
    assert np.array_equal(subids_i, subids_v), "subid orders disagree"
    subids = subids_i
    best_idrnn_seed   = int(d_i["seed"])
    best_vanilla_seed = int(d_v["seed"])
    print(f"  IDRNN seed {best_idrnn_seed}, z shape {z_idrnn.shape}")
    print(f"  Vanilla seed {best_vanilla_seed}, h shape {h_vanilla.shape}")

    quest = pd.read_csv("data/finalQuestionnaireDataSession1.csv").set_index("ID")
    for k, (items, _) in SCALES.items():
        quest[k] = quest[items].mean(axis=1)

    print(f"\n── Decoding (in-sample OLS) ──")
    decode_rows = []
    for scale in SCALE_KEYS:
        y_all = quest.reindex(subids)[scale].values.astype(float)
        mask = ~np.isnan(y_all)
        if mask.sum() < 30:
            print(f"  {scale}: too few valid (n={mask.sum()}), skipping")
            continue
        Zi = z_idrnn[mask]; Hv = h_vanilla[mask]; yy = y_all[mask]
        r_i, p_i, preds_i = insample_ols(Zi, yy)
        r_v, p_v, preds_v = insample_ols(Hv, yy)
        r_iv, _ = pearsonr(preds_i, preds_v)
        z_s, p_s = steiger_test(r_i, r_v, r_iv, int(mask.sum()))
        print(f"  {scale:<10}  IDRNN r={r_i:+.3f} ({p_i:.3g} {sig_stars(p_i)})  "
              f"Vanilla r={r_v:+.3f} ({p_v:.3g} {sig_stars(p_v)})  "
              f"Steiger z={z_s:+.2f} p={p_s:.3g} {sig_stars(p_s)}")
        decode_rows.append({
            "scale": scale, "n": int(mask.sum()),
            "r_idrnn": r_i, "p_idrnn": p_i,
            "r_vanilla": r_v, "p_vanilla": p_v,
            "r_iv": r_iv, "z_steiger": z_s, "p_steiger": p_s,
        })
    decode_df = pd.DataFrame(decode_rows)
    decode_df.to_csv(os.path.join(OUT_DIR, "decoding_results.csv"), index=False)

    labels = [SCALES[k][1] for k in decode_df["scale"]]
    x = np.arange(len(labels))
    w = 0.35
    c1, c2 = "#4C72B0", "#DD8452"

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(x - w/2, decode_df["r_idrnn"].abs(), w,
           color=c1, alpha=0.85, edgecolor="k", linewidth=0.5,
           label=f"IDRNN step-1 lookup (seed {best_idrnn_seed})")
    ax.bar(x + w/2, decode_df["r_vanilla"].abs(), w,
           color=c2, alpha=0.85, edgecolor="k", linewidth=0.5,
           label=f"Vanilla h (seed {best_vanilla_seed})")

    for i, row in decode_df.iterrows():
        ax.text(i - w/2, abs(row["r_idrnn"]) + 0.01, sig_stars(row["p_idrnn"]),
                ha="center", va="bottom", fontsize=8, fontweight="bold", color=c1)
        ax.text(i + w/2, abs(row["r_vanilla"]) + 0.01, sig_stars(row["p_vanilla"]),
                ha="center", va="bottom", fontsize=8, fontweight="bold", color=c2)
        y_max = max(abs(row["r_idrnn"]), abs(row["r_vanilla"])) + 0.07
        ax.plot([i - w/2, i - w/2, i + w/2, i + w/2],
                [y_max - 0.01, y_max, y_max, y_max - 0.01], color="k", linewidth=0.8)
        stars = sig_stars(row["p_steiger"])
        ax.text(i, y_max + 0.005, stars, ha="center", va="bottom",
                fontsize=8, color=("k" if stars != "n.s." else "grey"))

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("|r|  (in-sample OLS)", fontsize=11)
    ax.set_title(
        f"Questionnaire decoding — step-1 lookup z vs vanilla h  "
        f"(all 236 subjects, no outer CV)",
        fontweight="bold",
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.set_ylim(0, min(1.0, ax.get_ylim()[1] + 0.1))
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out_png = os.path.join(OUT_DIR, "step1_vs_vanilla_decoding.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out_png}")
    print(f"Saved {os.path.join(OUT_DIR, 'decoding_results.csv')}")


if __name__ == "__main__":
    main()
