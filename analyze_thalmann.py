#!/usr/bin/env python3
"""
analyze_thalmann.py — Outer-CV analysis for the Thalmann two-task dataset.

Each participant is a test participant in exactly one outer fold.
Per-participant results (NLL, latents) are averaged across seeds, then
pooled across folds for N-participant statistics.

Analyses
--------
1. NLL comparison   — IDRNN vs Vanilla; paired t-test.
2. RSA              — IDRNN latent RDM vs questionnaire scale RDMs (Spearman r
                      + permutation test). Scales: PANAS_PA, PANAS_NA, STICSA,
                      PHQ_9, BIG_5, CEI, and a composite (all scales stacked).

Usage
-----
    python analyze_thalmann.py
    python analyze_thalmann.py --seeds 200,300 --folds 3
"""
import argparse
import os
import json
import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, spearmanr
from scipy.spatial.distance import pdist, squareform

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--seeds", type=str, default=None,
                    help="Comma-separated seeds, e.g. '200,300,400'")
parser.add_argument("--folds", type=int, default=3,
                    help="Number of outer folds (default 3)")
parser.add_argument("--dgp", type=str, default="thalmann")
parser.add_argument("--n_perm", type=int, default=5000,
                    help="Permutation iterations for RSA significance test")
args = parser.parse_args()

FILTER_SEEDS = (set(int(s) for s in args.seeds.split(",")) if args.seeds else None)
N_FOLDS      = args.folds
DGP          = args.dgp
N_PERM       = args.n_perm

DATA_DIR   = f"data_{DGP}"
IDRNN_BASE = f"runs_{DGP}"
VAN_BASE   = f"runs_vanilla_{DGP}"
PLOT_DIR   = f"plots_{DGP}"
QUEST_PATH = "data/finalQuestionnaireDataSession1.csv"

os.makedirs(PLOT_DIR, exist_ok=True)

# ── Colour palette ─────────────────────────────────────────────────────────────
BLUE_INDIV   = "#2a82c2"
ORANGE       = "#e1861f"

# ── Helpers ───────────────────────────────────────────────────────────────────
def discover_seeds(fold, base_dir):
    fold_dir = os.path.join(base_dir, f"fold{fold}")
    if not os.path.isdir(fold_dir):
        return []
    seeds = []
    for d in os.listdir(fold_dir):
        if d.startswith("seed_") and os.path.isdir(os.path.join(fold_dir, d)):
            s = int(d.split("_")[1])
            if FILTER_SEEDS is None or s in FILTER_SEEDS:
                seeds.append(s)
    return sorted(seeds)


# ── Load fold metadata ─────────────────────────────────────────────────────────
fold_test_subids  = {}
fold_train_subids = {}

for fold in range(N_FOLDS):
    fold_data_dir = os.path.join(DATA_DIR, f"fold{fold}")
    df_test_fold  = pd.read_csv(os.path.join(fold_data_dir, "df_test.csv"))
    df_train_fold = pd.read_csv(os.path.join(fold_data_dir, "df_train.csv"))
    fold_test_subids[fold]  = sorted(df_test_fold["subid"].unique())
    fold_train_subids[fold] = sorted(df_train_fold["subid"].unique())

all_subids   = sorted(set(s for subs in fold_test_subids.values() for s in subs))
subid_to_idx = {s: i for i, s in enumerate(all_subids)}
N            = len(all_subids)
print(f"Total participants across {N_FOLDS} folds: {N}")


# ── Questionnaire scales ───────────────────────────────────────────────────────
def compute_scale_scores(df_q):
    """
    Returns a DataFrame with one row per participant (indexed by ID) and one
    column per scale summary score.

    PANAS items 0-19:
      PA (positive affect) = even-indexed items (0,2,4,...,18)
      NA (negative affect) = odd-indexed  items (1,3,5,...,19)
    STICSA_0..21   → total sum
    PHQ_9_0..8     → total sum (item 9 = functional impairment, excluded)
    BIG_5_0..5     → total sum (brief Big-5 measure)
    CEI_0..3       → total sum
    """
    d = {}
    panas_even = [f"PANAS_{i}" for i in range(0, 20, 2)]   # 10 items
    panas_odd  = [f"PANAS_{i}" for i in range(1, 20, 2)]   # 10 items
    d["PANAS_PA"] = df_q[panas_even].sum(axis=1).values
    d["PANAS_NA"] = df_q[panas_odd].sum(axis=1).values

    sticsa_cols = [f"STICSA_{i}" for i in range(22)]
    d["STICSA"]  = df_q[sticsa_cols].sum(axis=1).values

    phq_cols = [f"PHQ_9_{i}" for i in range(9)]   # items 0-8
    d["PHQ_9"]   = df_q[phq_cols].sum(axis=1).values

    big5_cols = [f"BIG_5_{i}" for i in range(6)]
    d["BIG_5"]   = df_q[big5_cols].sum(axis=1).values

    cei_cols = [f"CEI_{i}" for i in range(4)]
    d["CEI"]     = df_q[cei_cols].sum(axis=1).values

    out = pd.DataFrame(d, index=df_q["ID"].values)
    return out


df_q     = pd.read_csv(QUEST_PATH)
scores_q = compute_scale_scores(df_q)   # indexed by participant ID

SCALE_NAMES   = ["PANAS_PA", "PANAS_NA", "STICSA", "PHQ_9", "BIG_5", "CEI"]
SCALE_LABELS  = {
    "PANAS_PA": "PANAS Positive Affect",
    "PANAS_NA": "PANAS Negative Affect",
    "STICSA":   "STICSA (anxiety)",
    "PHQ_9":    "PHQ-9 (depression)",
    "BIG_5":    "Big-5 (brief)",
    "CEI":      "CEI (curiosity)",
    "composite": "Composite (all scales)",
}
print(f"Questionnaire loaded: {len(df_q)} participants, scales: {SCALE_NAMES}")


# ── NLL aggregation ───────────────────────────────────────────────────────────
def aggregate_nll(nametag, base_dir):
    """Per-participant NLL averaged across seeds; aligned to all_subids."""
    acc = {s: [] for s in all_subids}
    for fold in range(N_FOLDS):
        fold_data_dir = os.path.join(DATA_DIR, f"fold{fold}")
        test_subids   = fold_test_subids[fold]
        for seed in discover_seeds(fold, base_dir):
            csv_path = os.path.join(fold_data_dir, f"seed_{seed}",
                                    f"rnn_results{nametag}.csv")
            if not os.path.exists(csv_path):
                continue
            df = pd.read_csv(csv_path)
            if "normalized_likelihood" not in df.columns:
                continue
            nlls = df["normalized_likelihood"].values
            for i, subid in enumerate(test_subids):
                if i < len(nlls):
                    acc[subid].append(nlls[i])
    nll = np.array([np.mean(acc[s]) if acc[s] else np.nan for s in all_subids])
    print(f"  [{nametag}] valid NLL: {int(np.isfinite(nll).sum())}/{N}")
    return nll


# ── Latent aggregation ────────────────────────────────────────────────────────
def aggregate_lats(nametag, base_dir, strategy="mean"):
    """
    Aggregate latent vectors per participant across seeds.
    Returns (N, z_dim) array; NaN rows if missing.

    strategy:
      "mean" — mean over all valid (non-zero) timesteps  (B, Bk*T, z_dim) → (B, z_dim)
      "last" — last valid (non-zero) timestep per participant  → (B, z_dim)
               For IDRNN this is the posterior after seeing all blocks, the most
               informed single estimate of z.
    """
    acc = {s: [] for s in all_subids}
    for fold in range(N_FOLDS):
        fold_data_dir = os.path.join(DATA_DIR, f"fold{fold}")
        test_subids   = fold_test_subids[fold]
        for seed in discover_seeds(fold, base_dir):
            lat_path = os.path.join(fold_data_dir, f"seed_{seed}",
                                    f"latents_tensor{nametag}.pt")
            if not os.path.exists(lat_path):
                continue
            lat = torch.load(lat_path, map_location="cpu").numpy()
            if lat.ndim > 2:
                valid = (lat != 0).any(axis=-1)           # (B, seq_len) bool
                if strategy == "last":
                    # Last valid timestep per participant in the flattened sequence
                    seq_len = lat.shape[1]
                    has_any  = valid.any(axis=-1)          # (B,)
                    last_idx = seq_len - 1 - np.argmax(valid[:, ::-1], axis=-1)  # (B,)
                    last_idx = np.where(has_any, last_idx, 0)
                    lat = lat[np.arange(lat.shape[0]), last_idx]   # (B, z_dim)
                else:  # "mean"
                    cnt = valid.sum(axis=1, keepdims=True).clip(min=1)
                    lat = (lat * valid[:, :, np.newaxis]).sum(axis=1) / cnt  # (B, z_dim)
            for i, subid in enumerate(test_subids):
                if i < lat.shape[0]:
                    acc[subid].append(lat[i])
    z_dim = next((acc[s][0].shape[0] for s in all_subids if acc[s]), None)
    if z_dim is None:
        return None
    lats = np.array([
        np.mean(acc[s], axis=0) if acc[s] else np.full(z_dim, np.nan)
        for s in all_subids
    ])
    n_valid = int(np.isfinite(lats).all(axis=1).sum())
    print(f"  [{nametag}|{strategy}] valid latents: {n_valid}/{N}")
    return lats


# ── RSA helpers ───────────────────────────────────────────────────────────────
def make_rdm(X):
    """(n, d) → upper-triangle condensed distance vector (Euclidean)."""
    return pdist(X, metric="euclidean")


def rsa_spearman(rdm_a, rdm_b):
    """Spearman r between two condensed RDMs."""
    r, _ = spearmanr(rdm_a, rdm_b)
    return r


def permutation_test_rsa(rdm_lat, rdm_target, n_perm=5000, rng=None):
    """
    Permutation test: permute rows/cols of target distance matrix, recompute
    correlation.  Returns (observed_r, p_value).
    """
    if rng is None:
        rng = np.random.default_rng(42)
    n = int(round((1 + np.sqrt(1 + 8 * len(rdm_target))) / 2))
    obs_r   = rsa_spearman(rdm_lat, rdm_target)
    sq_lat  = squareform(rdm_lat)
    sq_tgt  = squareform(rdm_target)
    null_rs = []
    for _ in range(n_perm):
        perm = rng.permutation(n)
        perm_sq = sq_tgt[np.ix_(perm, perm)]
        null_rs.append(rsa_spearman(rdm_lat, squareform(perm_sq)))
    p_val = (np.sum(np.array(null_rs) >= obs_r) + 1) / (n_perm + 1)
    return obs_r, p_val


# ── Run aggregation ────────────────────────────────────────────────────────────
print("\n── Aggregating IDRNN NLL ──")
idrnn_nll = aggregate_nll("latentmodel", IDRNN_BASE)

print("\n── Aggregating Vanilla NLL ──")
van_nll = aggregate_nll("vanilla", VAN_BASE)

print("\n── Aggregating IDRNN latents (mean over valid timesteps) ──")
idrnn_lats_mean = aggregate_lats("latentmodel", IDRNN_BASE, strategy="mean")

print("\n── Aggregating IDRNN latents (last valid timestep) ──")
idrnn_lats_last = aggregate_lats("latentmodel", IDRNN_BASE, strategy="last")


# ── NLL comparison ─────────────────────────────────────────────────────────────
print("\n── NLL comparison (IDRNN vs Vanilla) ──")
both_valid = np.isfinite(idrnn_nll) & np.isfinite(van_nll)
n_paired   = int(both_valid.sum())
print(f"  Paired participants: {n_paired}/{N}")

if n_paired > 1:
    t_stat, p_nll = ttest_rel(idrnn_nll[both_valid], van_nll[both_valid])
    d_nll = np.mean(idrnn_nll[both_valid] - van_nll[both_valid])
    print(f"  IDRNN NLL mean: {np.nanmean(idrnn_nll):.4f}  "
          f"Vanilla NLL mean: {np.nanmean(van_nll):.4f}")
    print(f"  Δ(IDRNN-Vanilla): {d_nll:.4f}  t={t_stat:.3f}  p={p_nll:.4f}")
else:
    t_stat = p_nll = d_nll = np.nan
    print("  Not enough paired data for t-test.")


# ── RSA with questionnaire data ────────────────────────────────────────────────
def run_rsa(lats, label):
    """
    Compute RSA between IDRNN latents and each questionnaire scale.
    Returns dict: scale -> {r, p_perm}, or {} if insufficient data.
    Also returns (rsa_mask, n_rsa, rsa_subids, quest_rsa, rdm_lat).
    """
    if lats is None:
        print(f"  [{label}] No latent data; skipping.")
        return {}, None, 0, None, None, None

    lat_valid   = np.isfinite(lats).all(axis=1)
    quest_valid = np.array([sid in scores_q.index for sid in all_subids])
    rsa_mask    = lat_valid & quest_valid
    n_rsa       = int(rsa_mask.sum())
    print(f"  [{label}] Participants in RSA: {n_rsa}/{N}")

    if n_rsa <= 2:
        print(f"  [{label}] Not enough participants.")
        return {}, rsa_mask, n_rsa, None, None, None

    rsa_subids = [s for s, m in zip(all_subids, rsa_mask) if m]
    lat_rsa    = lats[rsa_mask]
    quest_rsa  = scores_q.loc[rsa_subids, SCALE_NAMES]
    rdm_lat    = make_rdm(lat_rsa)

    results = {}
    rng = np.random.default_rng(0)
    for scale in SCALE_NAMES:
        rdm_q  = make_rdm(quest_rsa[[scale]].values)
        r, p   = permutation_test_rsa(rdm_lat, rdm_q, n_perm=N_PERM, rng=rng)
        results[scale] = {"r": float(r), "p_perm": float(p)}
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        print(f"    {scale:12s}: r={r:+.3f}  p_perm={p:.4f}  {sig}")

    rdm_composite = make_rdm(quest_rsa[SCALE_NAMES].values)
    r_c, p_c = permutation_test_rsa(rdm_lat, rdm_composite, n_perm=N_PERM, rng=rng)
    results["composite"] = {"r": float(r_c), "p_perm": float(p_c)}
    sig_c = "***" if p_c < 0.001 else "**" if p_c < 0.01 else "*" if p_c < 0.05 else "n.s."
    print(f"    {'composite':12s}: r={r_c:+.3f}  p_perm={p_c:.4f}  {sig_c}")

    return results, rsa_mask, n_rsa, rsa_subids, quest_rsa, rdm_lat


print("\n── RSA — IDRNN latents (mean over valid timesteps) ──")
rsa_results_mean, rsa_mask_mean, n_rsa_mean, \
    rsa_subids_mean, quest_rsa_mean, rdm_lat_mean = run_rsa(idrnn_lats_mean, "mean")

print("\n── RSA — IDRNN latents (last valid timestep) ──")
rsa_results_last, rsa_mask_last, n_rsa_last, \
    rsa_subids_last, quest_rsa_last, rdm_lat_last = run_rsa(idrnn_lats_last, "last")

# Convenience aliases for downstream plotting (prefer last, fall back to mean)
rsa_results = rsa_results_last if rsa_results_last else rsa_results_mean
n_rsa       = n_rsa_last if rsa_results_last else n_rsa_mean


# ── Plot 1: NLL comparison ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(4, 4))
if n_paired > 1:
    means  = [np.nanmean(idrnn_nll), np.nanmean(van_nll)]
    sems   = [np.nanstd(idrnn_nll) / np.sqrt(n_paired),
              np.nanstd(van_nll)   / np.sqrt(n_paired)]
    colors = [BLUE_INDIV, ORANGE]
    bars   = ax.bar(["IDRNN", "Vanilla"], means, color=colors,
                    yerr=sems, capsize=4, width=0.5)

    # Significance bar
    y_top = max(means) + max(sems) * 1.5
    sig_s = "***" if p_nll < 0.001 else "**" if p_nll < 0.01 else "*" if p_nll < 0.05 else "n.s."
    ax.plot([0, 0, 1, 1], [y_top, y_top + 0.02, y_top + 0.02, y_top], lw=1.5, c="k")
    ax.text(0.5, y_top + 0.025, sig_s, ha="center", va="bottom", fontsize=11)
ax.set_ylabel("Mean normalized log-likelihood")
ax.set_title(f"NLL comparison (N={n_paired})")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "nll_comparison.png"), dpi=150)
plt.close(fig)
print(f"  NLL plot → {PLOT_DIR}/nll_comparison.png")


# ── Plot 2: RSA bar chart — mean vs last, side by side ────────────────────────
scale_order = SCALE_NAMES + ["composite"]

_strategies = [
    ("mean", rsa_results_mean, "#7fb3d3", n_rsa_mean),
    ("last", rsa_results_last, BLUE_INDIV,  n_rsa_last),
]
_strategies = [(lbl, res, col, n) for lbl, res, col, n in _strategies if res]

if _strategies:
    labels = [SCALE_LABELS[s] for s in scale_order]
    n_scales  = len(scale_order)
    n_strat   = len(_strategies)
    bar_w     = 0.35 if n_strat == 2 else 0.5
    offsets   = np.linspace(-(n_strat - 1) * bar_w / 2,
                             (n_strat - 1) * bar_w / 2, n_strat)

    fig, ax = plt.subplots(figsize=(n_scales * 1.4 + 1, 4))
    ax.axhline(0, color="k", lw=0.8)

    for (strat_lbl, res, col, n_s), offset in zip(_strategies, offsets):
        r_vals = [res.get(s, {}).get("r", 0)      for s in scale_order]
        p_vals = [res.get(s, {}).get("p_perm", 1) for s in scale_order]
        xs     = np.arange(n_scales) + offset
        bar_colors = [col if p < 0.05 else "#cccccc" for p in p_vals]
        ax.bar(xs, r_vals, width=bar_w, color=bar_colors,
               label=f"IDRNN {strat_lbl} (N={n_s})", edgecolor="none")
        for i, (r, p) in enumerate(zip(r_vals, p_vals)):
            sig_s = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            if sig_s:
                ax.text(xs[i], r + 0.005 if r >= 0 else r - 0.01, sig_s,
                        ha="center", va="bottom" if r >= 0 else "top", fontsize=9)

    ax.set_xticks(np.arange(n_scales))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("RSA Spearman r")
    ax.set_title("RSA — IDRNN latents vs questionnaire scales\n(mean vs last timestep)")
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "rsa_questionnaire.png"), dpi=150)
    plt.close(fig)
    print(f"  RSA plot → {PLOT_DIR}/rsa_questionnaire.png")


# ── Plot 3: RSA scatter — best scale from 'last' strategy ─────────────────────
_scatter_res   = rsa_results_last if rsa_results_last else rsa_results_mean
_scatter_lats  = idrnn_lats_last  if rsa_results_last else idrnn_lats_mean
_scatter_qrsa  = quest_rsa_last   if rsa_results_last else quest_rsa_mean
_scatter_rdm   = rdm_lat_last     if rsa_results_last else rdm_lat_mean
_scatter_n     = n_rsa_last       if rsa_results_last else n_rsa_mean
_scatter_strat = "last"           if rsa_results_last else "mean"

if _scatter_res and _scatter_lats is not None and _scatter_n > 2:
    best_scale = max(_scatter_res, key=lambda s: abs(_scatter_res[s]["r"]))
    if best_scale != "composite":
        best_r = _scatter_res[best_scale]["r"]
        best_p = _scatter_res[best_scale]["p_perm"]
        rdm_q_sq   = squareform(make_rdm(_scatter_qrsa[[best_scale]].values))
        rdm_lat_sq = squareform(_scatter_rdm)

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.scatter(rdm_q_sq[np.triu_indices(_scatter_n, k=1)],
                   rdm_lat_sq[np.triu_indices(_scatter_n, k=1)],
                   s=2, alpha=0.15, color=BLUE_INDIV, rasterized=True)
        sig_s = "***" if best_p < 0.001 else "**" if best_p < 0.01 else "*" if best_p < 0.05 else "n.s."
        ax.set_xlabel(f"{SCALE_LABELS.get(best_scale, best_scale)} distance")
        ax.set_ylabel("Latent Euclidean distance")
        ax.set_title(f"RSA scatter — {best_scale} [{_scatter_strat}]\nr={best_r:+.3f}  {sig_s}")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, "rsa_scatter_best_scale.png"), dpi=150)
        plt.close(fig)
        print(f"  RSA scatter → {PLOT_DIR}/rsa_scatter_best_scale.png")


# ── Save summary JSON ──────────────────────────────────────────────────────────
summary = {
    "dgp":          DGP,
    "n_participants": N,
    "n_rsa":         n_rsa if rsa_results else 0,
    "nll": {
        "idrnn_mean":   float(np.nanmean(idrnn_nll)),
        "van_mean":     float(np.nanmean(van_nll)),
        "delta_mean":   float(d_nll) if np.isfinite(d_nll) else None,
        "t_stat":       float(t_stat) if np.isfinite(t_stat) else None,
        "p_value":      float(p_nll)  if np.isfinite(p_nll)  else None,
        "n_paired":     n_paired,
    },
    "rsa": rsa_results,
}
summary_path = os.path.join(PLOT_DIR, "analysis_summary.json")
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSummary saved → {summary_path}")
print("Done.")
