#!/usr/bin/env python3
"""Panel b/c decoding for the Thalmann z=3 canonical model.

Decodes per-subject targets from the canonical step-1 IDRNN latent z (one row
per subject) vs a DIMENSION-MATCHED vanilla hidden state (PCA(h) -> z_dim PC
scores, so both arch's regressors see exactly z_dim features).

Targets (per user request):
  • personality scales      : PANAS_PA, PANAS_NA, STICSA, PHQ, CEI, BIG5_open
  • composite personality   : CFA factors AxDep, posMood, negMood, Exp
  • working memory (NEW)     : WM_composite + the 3 spans (OS, SS, WMU recall)

Statistics per target (mirrors train_and_decode_thalmann_step1_s2_h10z3.py):
  • in-sample OLS  (sqrt(R^2), F-test p)            — k-biased, |r|
  • LOO RidgeCV    (out-of-sample Pearson r)        — unbiased signal test
  • Steiger's test (IDRNN vs dim-matched vanilla)

Inputs (canonical _full bundle, produced by extract_canonical_latents_thalmann.py):
  --idrnn_pt   final_plots/thalmann_z3_full/canonical/idrnn/latents_idrnn_canonical.pt
  --vanilla_pt final_plots/thalmann_z3_full/canonical/vanilla/latents_vanilla_canonical.pt
Both bundles must carry per-subject latents + a `subids` array (concat order of
data_thalmann_full/subids_full.npy).

Outputs (--out_dir, default final_plots/thalmann_z3_full/decoding/):
  decoding_results.csv
  panel_b_decoding.png / .pdf
"""
import argparse
import os
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneOut
from scipy.stats import pearsonr, norm, f as f_dist

# ── Targets ────────────────────────────────────────────────────────────────────
# (item columns, display label).  CFA factors are single precomputed columns.
SCALES = {
    "PANAS_PA":  ([f"PANAS_{i}"  for i in [0, 2, 4, 6, 8, 14, 16, 18]], "PANAS\nPos. Affect"),
    "PANAS_NA":  ([f"PANAS_{i}"  for i in [1, 3, 5, 7, 9, 11, 13, 15]], "PANAS\nNeg. Affect"),
    "STICSA":    ([f"STICSA_{i}" for i in range(22)],                   "STICSA\nAnxiety"),
    "PHQ":       ([f"PHQ_9_{i}"  for i in range(10)],                   "PHQ-9\nDepression"),
    "CEI":       ([f"CEI_{i}"    for i in range(4)],                    "CEI\nCuriosity"),
    "BIG5_open": ([f"BIG_5_{i}"  for i in range(6)],                    "BIG5\nOpenness"),
    "AxDep":     (["AxDep"],   "CFA\nAxDep"),
    "posMood":   (["posMood"], "CFA\nposMood"),
    "negMood":   (["negMood"], "CFA\nnegMood"),
    "Exp":       (["Exp"],     "CFA\nExploration"),
}
CFA_KEYS = {"AxDep", "posMood", "negMood", "Exp"}

# Working memory targets (data/wm-performance.csv, keyed by participant_id).
# Each span has two reps (_0, _1); we average them.  WM_composite = mean of the
# three span-recall accuracies.
WM_SPANS = {
    "WM_OS":  (["OS_recall_0",  "OS_recall_1"],  "WM\nOper. Span"),
    "WM_SS":  (["SS_recall_0",  "SS_recall_1"],  "WM\nSymm. Span"),
    "WM_WMU": (["WMU_recall_0", "WMU_recall_1"], "WM\nUpdating"),
}
WM_COMPOSITE_LABEL = "WM\nComposite"

# Order targets group-wise for the plot.
PERSONALITY = ["PANAS_PA", "PANAS_NA", "STICSA", "PHQ", "CEI", "BIG5_open"]
COMPOSITE   = ["AxDep", "posMood", "negMood", "Exp"]
WM_KEYS     = ["WM_composite", "WM_OS", "WM_SS", "WM_WMU"]
ALL_KEYS    = PERSONALITY + COMPOSITE + WM_KEYS


def label_for(key):
    if key in SCALES:
        return SCALES[key][1]
    if key == "WM_composite":
        return WM_COMPOSITE_LABEL
    return WM_SPANS[key][1]


# ── Target assembly ──────────────────────────────────────────────────────────
def load_targets(subids):
    """Return a DataFrame indexed by subids with all target columns.

    Personality (non-CFA): mean of S1+S2 where both present, else whichever.
    CFA factors: S1-only file.  WM: data/wm-performance.csv (single session of
    span tasks, two reps averaged)."""
    q1 = pd.read_csv("data/finalQuestionnaireDataSession1.csv").set_index("ID")
    q2 = pd.read_csv("data/finalQuestionnaireDataSession2.csv").set_index("ID")
    factors = (pd.read_csv("data/CFA_compound_questionnaire_factors_s1.csv")
                 .set_index("ID")[["AxDep", "posMood", "negMood", "Exp"]])
    wm = pd.read_csv("data/wm-performance.csv").set_index("participant_id")

    out = pd.DataFrame(index=subids)
    for key, (items, _) in SCALES.items():
        if key in CFA_KEYS:
            out[key] = factors.reindex(subids)[items[0]]
            continue
        s1 = q1.reindex(subids)[items].mean(axis=1)
        s2 = q2.reindex(subids)[items].mean(axis=1)
        out[key] = pd.concat([s1, s2], axis=1).mean(axis=1)

    # WM spans (mean of two reps) + composite
    span_means = {}
    for key, (cols, _) in WM_SPANS.items():
        span_means[key] = wm.reindex(subids)[cols].mean(axis=1)
        out[key] = span_means[key]
    out["WM_composite"] = pd.concat(span_means.values(), axis=1).mean(axis=1)
    return out


# ── Latent loading ───────────────────────────────────────────────────────────
def _load_latent_bundle(path, kind):
    d = torch.load(path, map_location="cpu", weights_only=False)
    subids = np.asarray(d["subids"]).astype(int)
    if kind == "idrnn":
        z = np.asarray(d.get("z_train", d.get("z")))            # (N, z_dim)
    else:
        z = np.asarray(d.get("h_avg_train", d.get("h")))        # (N, hidden)
    return z, subids


# ── Regressors ───────────────────────────────────────────────────────────────
def loo_ridge(Z, y, alphas=(0.001, 0.01, 0.1, 1, 10, 100, 1000)):
    loo = LeaveOneOut()
    preds = np.zeros(len(y))
    for tr, te in loo.split(Z):
        mu_z, sd_z = Z[tr].mean(0), Z[tr].std(0) + 1e-8
        mu_y, sd_y = y[tr].mean(),  y[tr].std()  + 1e-8
        clf = RidgeCV(alphas=list(alphas))
        clf.fit((Z[tr] - mu_z) / sd_z, (y[tr] - mu_y) / sd_y)
        preds[te] = clf.predict((Z[te] - mu_z) / sd_z) * sd_y + mu_y
    r, p = pearsonr(y, preds)
    return float(r), float(p), preds


def insample_ols(Z, y):
    n, p = Z.shape
    clf = LinearRegression().fit(Z, y)
    preds = clf.predict(Z)
    ss_res = np.sum((y - preds) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot
    dfn, dfd = p, n - p - 1
    F = (r2 / dfn) / ((1 - r2) / dfd) if (r2 < 1 and dfd > 0) else np.inf
    p_val = float(f_dist.sf(F, dfn, dfd)) if dfd > 0 else 1.0
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


def pc_match(h_raw, z_dim):
    """PCA(h) -> z_dim PC scores so the vanilla regressor sees exactly z_dim
    features (honest dim-match for decoding). If h already has <= z_dim cols,
    return centred unchanged."""
    if h_raw.shape[1] <= z_dim:
        return (h_raw - h_raw.mean(0)).astype(np.float32)
    return PCA(n_components=z_dim).fit_transform(h_raw).astype(np.float32)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idrnn_pt",
                    default="final_plots/thalmann_z3_full/canonical/idrnn/latents_idrnn_canonical.pt")
    ap.add_argument("--vanilla_pt",
                    default="final_plots/thalmann_z3_full/canonical/vanilla/latents_vanilla_canonical.pt")
    ap.add_argument("--out_dir", default="final_plots/thalmann_z3_full/decoding")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    z_idr, sub_idr = _load_latent_bundle(args.idrnn_pt, "idrnn")
    h_van, sub_van = _load_latent_bundle(args.vanilla_pt, "vanilla")
    Z_DIM = z_idr.shape[1]
    print(f"IDRNN z {z_idr.shape} (subids {sub_idr.shape}); "
          f"vanilla h {h_van.shape} (subids {sub_van.shape}); z_dim={Z_DIM}")

    # Align vanilla h onto IDRNN subid order
    v_idx = {int(s): i for i, s in enumerate(sub_van)}
    h_aligned = np.full((len(sub_idr), h_van.shape[1]), np.nan, dtype=np.float32)
    for i, s in enumerate(sub_idr):
        if int(s) in v_idx:
            h_aligned[i] = h_van[v_idx[int(s)]]

    # Dimension-match vanilla h -> Z_DIM PC scores
    finite_rows = np.all(np.isfinite(h_aligned), axis=1)
    h_dm = np.full((len(sub_idr), Z_DIM), np.nan, dtype=np.float32)
    h_dm[finite_rows] = pc_match(h_aligned[finite_rows], Z_DIM)

    targets = load_targets(sub_idr)

    rows = []
    for key in ALL_KEYS:
        y_all = targets[key].values.astype(float)
        mask = np.isfinite(y_all) & np.all(np.isfinite(z_idr), axis=1) & \
               np.all(np.isfinite(h_dm), axis=1)
        n = int(mask.sum())
        if n < 30:
            print(f"  {key:<13} too few valid (n={n}), skipping")
            continue
        Zi, Hv, yy = z_idr[mask], h_dm[mask], y_all[mask]

        r_i_ins, p_i_ins, pr_i_ins = insample_ols(Zi, yy)
        r_v_ins, p_v_ins, pr_v_ins = insample_ols(Hv, yy)
        r_iv_ins, _ = pearsonr(pr_i_ins, pr_v_ins)
        z_s_ins, p_s_ins = steiger_test(r_i_ins, r_v_ins, r_iv_ins, n)

        r_i_loo, p_i_loo, pr_i_loo = loo_ridge(Zi, yy)
        r_v_loo, p_v_loo, pr_v_loo = loo_ridge(Hv, yy)
        r_iv_loo, _ = pearsonr(pr_i_loo, pr_v_loo)
        z_s_loo, p_s_loo = steiger_test(r_i_loo, r_v_loo, r_iv_loo, n)

        # Honest out-of-sample decodability: LOO R^2 (<=0 => no signal; the
        # negative LOO pred-vs-target r is the no-information anti-corr artifact).
        def _r2(y, pred):
            return float(1.0 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2))
        r2_i_loo = _r2(yy, pr_i_loo)
        r2_v_loo = _r2(yy, pr_v_loo)

        chance_r = np.sqrt(Z_DIM / (n - 1))     # same k for both (dim-matched)
        print(f"  {key:<13} n={n:3d}  LOO: IDRNN r={r_i_loo:+.3f}{sig_stars(p_i_loo):>4} | "
              f"Van(dm) r={r_v_loo:+.3f}{sig_stars(p_v_loo):>4}  (chance≈{chance_r:.3f})")
        rows.append(dict(
            scale=key, group=("personality" if key in PERSONALITY else
                              "composite" if key in COMPOSITE else "wm"),
            n=n,
            r_idrnn=r_i_ins, p_idrnn=p_i_ins, r_vanilla=r_v_ins, p_vanilla=p_v_ins,
            r_iv=r_iv_ins, z_steiger=z_s_ins, p_steiger=p_s_ins,
            r_idrnn_loo=r_i_loo, p_idrnn_loo=p_i_loo,
            r_vanilla_loo=r_v_loo, p_vanilla_loo=p_v_loo,
            r_iv_loo=r_iv_loo, z_steiger_loo=z_s_loo, p_steiger_loo=p_s_loo,
            r2_idrnn_loo=r2_i_loo, r2_vanilla_loo=r2_v_loo,
            chance_r=chance_r,
        ))

    df = pd.DataFrame(rows)
    csv_p = os.path.join(args.out_dir, "decoding_results.csv")
    df.to_csv(csv_p, index=False)
    print(f"\nSaved {csv_p}  ({len(df)} targets)")

    _plot(df, Z_DIM, args.out_dir)


def _plot(df, z_dim, out_dir):
    labels = [label_for(k) for k in df["scale"]]
    x = np.arange(len(labels))
    w = 0.38
    c_idr, c_van = "#0272b2", "#ec6f00"     # nature Blue[3], Orange[3]

    fig, axes = plt.subplots(1, 2, figsize=(max(11, 0.9 * len(labels)), 5.2))
    panels = [
        (axes[0], "r_idrnn", "r_vanilla", "p_idrnn", "p_vanilla", "p_steiger",
         "In-sample OLS (R²-based, k-biased)", True),
        (axes[1], "r_idrnn_loo", "r_vanilla_loo", "p_idrnn_loo", "p_vanilla_loo",
         "p_steiger_loo", "LOO RidgeCV (unbiased — true signal)", False),
    ]
    for ax, ci, cvk, pi_, pv_, ps_, title, use_abs in panels:
        vi = df[ci].abs() if use_abs else df[ci]
        vv = df[cvk].abs() if use_abs else df[cvk]
        ax.bar(x - w/2, vi, w, color=c_idr, edgecolor="k", linewidth=0.5,
               label=f"IDRNN z (k={z_dim})")
        ax.bar(x + w/2, vv, w, color=c_van, edgecolor="k", linewidth=0.5,
               label=f"Vanilla h, dim-matched (k={z_dim})")
        for i, row in df.iterrows():
            ri = abs(row[ci]) if use_abs else row[ci]
            rv = abs(row[cvk]) if use_abs else row[cvk]
            ax.text(i - w/2, ri + 0.008*np.sign(ri or 1), sig_stars(row[pi_]),
                    ha="center", va="bottom" if ri >= 0 else "top", fontsize=7, color=c_idr)
            ax.text(i + w/2, rv + 0.008*np.sign(rv or 1), sig_stars(row[pv_]),
                    ha="center", va="bottom" if rv >= 0 else "top", fontsize=7, color=c_van)
        if not use_abs:
            ax.axhline(0, color="grey", lw=1, alpha=0.5)
            ch = df["chance_r"].mean()
            ax.axhline(ch, color="grey", ls="--", lw=1, alpha=0.6,
                       label=f"chance r≈{ch:.3f}")
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8, rotation=35, ha="right")
        ax.set_ylabel("|r|" if use_abs else "r (signed)", fontsize=10)
        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.legend(fontsize=8, loc="upper right")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        # group separators
        ng_p, ng_c = len(PERSONALITY), len(COMPOSITE)
        for xb in (ng_p - 0.5, ng_p + ng_c - 0.5):
            ax.axvline(xb, color="grey", lw=0.6, ls=":", alpha=0.6)
    axes[0].set_ylim(0, min(1.0, axes[0].get_ylim()[1] + 0.08))
    fig.suptitle("Thalmann z=3 — questionnaire + working-memory decoding "
                 "(canonical step-1 z vs dim-matched vanilla h)", fontweight="bold")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"panel_b_decoding.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir}/panel_b_decoding.png/.pdf")


if __name__ == "__main__":
    main()
