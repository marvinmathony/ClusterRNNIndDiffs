#!/usr/bin/env python3
"""
analyze_step1_latents_thalmann.py — Geometry and decodability of step-1
lookup-z embeddings for the Thalmann two-task dataset.

Step-1 z are the per-participant embeddings learned by LatentRNNz
(LookupEncoderZ) and stored in frozen_decoder/policy_model.pt.
They represent the "oracle" individual-difference signal available to
the model before the encoder is trained.

Plots
-----
1. Ridge-regression decodability: step-1 z → questionnaire scales
   (within-fold LOO-CV, z-scored X and y — same protocol as
   ridge_decode_thalmann.py so results are directly comparable)
2. Bar chart comparing step-1 vs step-2 decodability (upper bound)
3. PCA of step-1 z coloured by each questionnaire scale (fold 0)
4. Cross-seed consistency: mean ± std of pairwise-distance correlation
   across seeds within each fold

Usage
-----
    python analyze_step1_latents_thalmann.py
    python analyze_step1_latents_thalmann.py --dgp thalmann --folds 3
"""
import argparse, os, json
import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.decomposition import PCA
import warnings; warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--dgp",   type=str, default="thalmann")
parser.add_argument("--folds", type=int, default=3)
args = parser.parse_args()

DGP        = args.dgp
N_FOLDS    = args.folds
DATA_DIR   = f"data_{DGP}"
IDRNN_BASE = f"runs_{DGP}"
PLOT_DIR   = f"plots_{DGP}"
QUEST_PATH = "data/finalQuestionnaireDataSession1.csv"
ALPHAS     = np.logspace(-3, 4, 50)

os.makedirs(PLOT_DIR, exist_ok=True)

# ── Questionnaire subscales ───────────────────────────────────────────────────
def compute_scale_scores(df_q):
    d = {}
    d["PANAS_PA"] = df_q[[f"PANAS_{i}" for i in range(0, 20, 2)]].sum(axis=1).values
    d["PANAS_NA"] = df_q[[f"PANAS_{i}" for i in range(1, 20, 2)]].sum(axis=1).values
    d["STICSA"]   = df_q[[f"STICSA_{i}" for i in range(22)]].sum(axis=1).values
    d["PHQ_9"]    = df_q[[f"PHQ_9_{i}"  for i in range(9)]].sum(axis=1).values
    d["BIG_5"]    = df_q[[f"BIG_5_{i}"  for i in range(6)]].sum(axis=1).values
    d["CEI"]      = df_q[[f"CEI_{i}"    for i in range(4)]].sum(axis=1).values
    return pd.DataFrame(d, index=df_q["ID"].values)

SCALE_NAMES  = ["PANAS_PA", "PANAS_NA", "STICSA", "PHQ_9", "BIG_5", "CEI"]
SCALE_LABELS = {"PANAS_PA": "PANAS PA", "PANAS_NA": "PANAS NA",
                "STICSA": "STICSA", "PHQ_9": "PHQ-9", "BIG_5": "Big-5", "CEI": "CEI"}

df_q     = pd.read_csv(QUEST_PATH)
scores_q = compute_scale_scores(df_q)
print(f"Questionnaire loaded: {len(scores_q)} participants")

# ── Discover seeds ────────────────────────────────────────────────────────────
def discover_seeds(fold):
    fold_dir = os.path.join(IDRNN_BASE, f"fold{fold}")
    if not os.path.isdir(fold_dir):
        return []
    seeds = [int(d.split("_")[1])
             for d in os.listdir(fold_dir)
             if d.startswith("seed_") and os.path.isdir(os.path.join(fold_dir, d))]
    return sorted(seeds)

# ── Load step-1 z for a fold × seed ─────────────────────────────────────────
def load_step1_z(fold, seed):
    """
    Returns (z, subids):
        z       : np.ndarray (N_train, z_dim)
        subids  : list of int, training participant IDs in embedding order
    Returns (None, None) if file not found.
    """
    policy_path = os.path.join(IDRNN_BASE, f"fold{fold}",
                               f"seed_{seed}", "frozen_decoder", "policy_model.pt")
    if not os.path.exists(policy_path):
        return None, None
    sd = torch.load(policy_path, map_location="cpu")
    z  = sd["encoder.embed.weight"].numpy()     # (N_train, z_dim)

    df_train = pd.read_csv(os.path.join(DATA_DIR, f"fold{fold}", "df_train.csv"))
    subids   = df_train["subid"].tolist()       # length N_train, indexed order
    assert len(subids) == len(z), "subid count mismatch with z embedding"
    return z, subids

# ── Within-fold LOO ridge regression using step-1 z ──────────────────────────
def step1_within_fold_loo_ridge():
    """
    Same protocol as within_fold_loo_ridge in ridge_decode_thalmann.py but
    applied to step-1 lookup-z embeddings.
    """
    fold_data = {}
    for fold in range(N_FOLDS):
        seeds = discover_seeds(fold)
        acc = {}  # subid -> list of z vectors (across seeds)
        for seed in seeds:
            z, subids = load_step1_z(fold, seed)
            if z is None:
                continue
            for i, sid in enumerate(subids):
                acc.setdefault(sid, []).append(z[i])

        subs, lats = [], []
        for sid in sorted(acc):
            if sid in scores_q.index:
                subs.append(sid)
                lats.append(np.mean(acc[sid], axis=0))   # average across seeds

        if len(subs) < 4:
            continue
        fold_data[fold] = {
            "subids": subs,
            "X": np.array(lats),
            "y": scores_q.loc[subs, SCALE_NAMES].values.astype(float),
        }
        print(f"  fold {fold}: {len(subs)} training participants with questionnaire data,"
              f"  z_dim={fold_data[fold]['X'].shape[1]}")

    if not fold_data:
        return None

    scale_preds = {s: {"y_true": [], "y_pred": []} for s in SCALE_NAMES}
    for fd in fold_data.values():
        X_raw = fd["X"]
        mu_x  = X_raw.mean(axis=0)
        sig_x = X_raw.std(axis=0).clip(min=1e-8)
        X_z   = (X_raw - mu_x) / sig_x
        n_f   = len(fd["subids"])

        for si, scale in enumerate(SCALE_NAMES):
            y_raw = fd["y"][:, si]
            mu_y  = y_raw.mean(); sig_y = y_raw.std().clip(min=1e-8)
            y_z   = (y_raw - mu_y) / sig_y
            y_pred_z = np.full(n_f, np.nan)
            for i in range(n_f):
                tr = np.ones(n_f, dtype=bool); tr[i] = False
                ridge = RidgeCV(alphas=ALPHAS, cv=None)
                ridge.fit(X_z[tr], y_z[tr])
                y_pred_z[i] = ridge.predict(X_z[[i]])[0]
            scale_preds[scale]["y_true"].extend(y_raw.tolist())
            scale_preds[scale]["y_pred"].extend((y_pred_z * sig_y + mu_y).tolist())

    results = {}
    for scale in SCALE_NAMES:
        yt = np.array(scale_preds[scale]["y_true"])
        yp = np.array(scale_preds[scale]["y_pred"])
        if len(yt) < 5 or not np.isfinite(yp).all():
            results[scale] = {"r": np.nan, "p": np.nan}
            continue
        r, p = pearsonr(yt, yp)
        results[scale] = {"r": float(r), "p": float(p), "n": len(yt)}
    return results


# ── Load step-2 ridge results for comparison (if available) ──────────────────
def load_step2_results():
    path = os.path.join(PLOT_DIR, "ridge_decode_results.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        res = json.load(f)
    # Use "last" strategy if available, else "mean"
    strat = "last" if "last" in res else next(iter(res))
    model_res = res.get(strat, {}).get("IDRNN")
    if model_res is None:
        return None
    return {scale: model_res[scale] for scale in SCALE_NAMES if scale in model_res}


# ── Cross-seed consistency of step-1 z ───────────────────────────────────────
def compute_seed_consistency():
    """
    For each fold, compute the Spearman correlation between pairwise-distance
    matrices of step-1 z across pairs of seeds. Reports mean ± std.
    """
    corrs = []
    for fold in range(N_FOLDS):
        seeds = discover_seeds(fold)
        if len(seeds) < 2:
            continue
        all_z = []
        for seed in seeds:
            z, subids = load_step1_z(fold, seed)
            if z is None:
                continue
            # z-score to remove global scale differences
            z = (z - z.mean(axis=0)) / z.std(axis=0).clip(min=1e-8)
            all_z.append(z)
        if len(all_z) < 2:
            continue
        # All seeds should have same N_train; compute pairwise dist matrices
        for i in range(len(all_z)):
            for j in range(i + 1, len(all_z)):
                n = min(all_z[i].shape[0], all_z[j].shape[0])
                zi, zj = all_z[i][:n], all_z[j][:n]
                di = np.linalg.norm(zi[:, None] - zi[None], axis=-1).ravel()
                dj = np.linalg.norm(zj[:, None] - zj[None], axis=-1).ravel()
                r, _ = spearmanr(di, dj)
                corrs.append(r)
    if corrs:
        print(f"\nCross-seed consistency (Spearman r of pairwise-distance matrices):")
        print(f"  n_pairs={len(corrs)}  mean={np.mean(corrs):.3f}  std={np.std(corrs):.3f}")
    return corrs


# ── PCA geometry for a single fold ───────────────────────────────────────────
def pca_geometry_plots(fold=0):
    seeds = discover_seeds(fold)
    if not seeds:
        print(f"No seeds found for fold {fold}, skipping PCA.")
        return

    # Average z across seeds for this fold
    acc = {}
    for seed in seeds:
        z, subids = load_step1_z(fold, seed)
        if z is None:
            continue
        for i, sid in enumerate(subids):
            acc.setdefault(sid, []).append(z[i])

    subs = sorted(s for s in acc if s in scores_q.index)
    if len(subs) < 5:
        print(f"Too few participants with questionnaire data in fold {fold}.")
        return
    Z = np.array([np.mean(acc[s], axis=0) for s in subs])  # (N, z_dim)
    Z = (Z - Z.mean(0)) / Z.std(0).clip(min=1e-8)           # z-score

    pca = PCA(n_components=2)
    Z2  = pca.fit_transform(Z)
    var = pca.explained_variance_ratio_
    print(f"\nStep-1 z PCA (fold {fold}): PC1={var[0]:.2%}  PC2={var[1]:.2%}")

    scores_sub = scores_q.loc[subs, SCALE_NAMES].values.astype(float)  # (N, n_scales)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, scale, vals in zip(axes.ravel(), SCALE_NAMES, scores_sub.T):
        sc = ax.scatter(Z2[:, 0], Z2[:, 1], c=vals, cmap="RdBu_r",
                        s=18, alpha=0.8, linewidths=0)
        plt.colorbar(sc, ax=ax, shrink=0.8)
        ax.set_title(SCALE_LABELS[scale], fontsize=9)
        ax.set_xlabel(f"PC1 ({var[0]:.1%})", fontsize=7)
        ax.set_ylabel(f"PC2 ({var[1]:.1%})", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(f"Step-1 lookup-z PCA (fold {fold}, N={len(subs)})", fontsize=10)
    fig.tight_layout()
    out = os.path.join(PLOT_DIR, "step1_z_pca.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"PCA plot → {out}")


# ── Run analyses ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step-1 lookup-z: within-fold LOO ridge regression")
print("=" * 60)
step1_res = step1_within_fold_loo_ridge()
if step1_res:
    for scale in SCALE_NAMES:
        r, p = step1_res[scale].get("r", np.nan), step1_res[scale].get("p", np.nan)
        n    = step1_res[scale].get("n", 0)
        sig  = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        print(f"  {scale:12s}: r={r:+.3f}  p={p:.4f}  {sig}  (N={n})")

step2_res = load_step2_results()

# ── Plot 1: step-1 decodability bar chart + step-2 overlay ───────────────────
if step1_res:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.axhline(0, color="k", lw=0.8)
    xs    = np.arange(len(SCALE_NAMES))
    BLUE  = "#2a82c2"
    GREEN = "#2ca02c"
    GRAY  = "#aaaaaa"

    # Step-1 bars
    r1   = np.array([step1_res[s]["r"]  if step1_res[s] else np.nan for s in SCALE_NAMES])
    p1   = np.array([step1_res[s].get("p", 1.0) for s in SCALE_NAMES])
    cols = [GREEN if p < 0.05 else GRAY for p in p1]
    ax.bar(xs - 0.2, r1, 0.35, color=cols, label="Step-1 z (oracle)", alpha=0.9)
    for x, r, p in zip(xs - 0.2, r1, p1):
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        if sig:
            ax.text(x, r + 0.005 if r >= 0 else r - 0.01, sig,
                    ha="center", va="bottom" if r >= 0 else "top", fontsize=8)

    # Step-2 IDRNN bars (if available)
    if step2_res:
        r2   = np.array([step2_res[s]["r"]  if s in step2_res else np.nan for s in SCALE_NAMES])
        p2   = np.array([step2_res[s].get("p", 1.0) if s in step2_res else 1.0 for s in SCALE_NAMES])
        cols2 = [BLUE if p < 0.05 else GRAY for p in p2]
        ax.bar(xs + 0.2, r2, 0.35, color=cols2, label="Step-2 IDRNN", alpha=0.9)
        for x, r, p in zip(xs + 0.2, r2, p2):
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            if sig:
                ax.text(x, r + 0.005 if r >= 0 else r - 0.01, sig,
                        ha="center", va="bottom" if r >= 0 else "top", fontsize=8)

    ax.set_xticks(xs)
    ax.set_xticklabels([SCALE_LABELS[s] for s in SCALE_NAMES], rotation=30, ha="right")
    ax.set_ylabel("LOO-CV Pearson r")
    ax.set_title(f"Step-1 lookup-z vs Step-2 IDRNN decodability\n"
                 f"(within-fold LOO-CV, z-scored X and y)")
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out = os.path.join(PLOT_DIR, "step1_vs_step2_decodability.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nDecodability plot → {out}")

# ── Plot 2: PCA geometry ─────────────────────────────────────────────────────
pca_geometry_plots(fold=0)

# ── Cross-seed consistency ───────────────────────────────────────────────────
corrs = compute_seed_consistency()
if corrs:
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(corrs, bins=20, color="#2a82c2", edgecolor="white")
    ax.axvline(np.mean(corrs), color="k", lw=1.5,
               label=f"mean={np.mean(corrs):.3f}")
    ax.set_xlabel("Spearman r (pairwise-distance matrix)")
    ax.set_ylabel("Count")
    ax.set_title("Step-1 z cross-seed consistency")
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out = os.path.join(PLOT_DIR, "step1_z_seed_consistency.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Seed consistency plot → {out}")

# ── Save JSON ─────────────────────────────────────────────────────────────────
import json
out_json = os.path.join(PLOT_DIR, "step1_decode_results.json")
with open(out_json, "w") as f:
    json.dump({s: step1_res[s] for s in SCALE_NAMES} if step1_res else {}, f, indent=2)
print(f"\nResults saved → {out_json}")
print("Done.")
