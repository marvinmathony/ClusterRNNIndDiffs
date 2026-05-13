#!/usr/bin/env python3
"""
ridge_decode_thalmann.py — Ridge regression from IDRNN/Vanilla latents to
questionnaire scores for the Thalmann two-task dataset.

For each questionnaire scale:
  - LOO-CV Ridge regression: train on N-1 participants, predict held-out one.
  - Report Pearson r and R² between predicted and actual scores.
  - Compare IDRNN latents vs Vanilla latents.

Usage
-----
    python ridge_decode_thalmann.py
    python ridge_decode_thalmann.py --seeds 200,300 --strategy last
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
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--seeds",    type=str, default=None,
                    help="Comma-separated seeds, e.g. '200,300'")
parser.add_argument("--folds",    type=int, default=3)
parser.add_argument("--dgp",      type=str, default="thalmann")
parser.add_argument("--strategy", type=str, default="both",
                    choices=["mean", "last", "both"],
                    help="Latent aggregation strategy")
args = parser.parse_args()

FILTER_SEEDS = (set(int(s) for s in args.seeds.split(",")) if args.seeds else None)
N_FOLDS      = args.folds
DGP          = args.dgp
STRATEGY     = args.strategy

DATA_DIR   = f"data_{DGP}"
IDRNN_BASE = f"runs_{DGP}"
VAN_BASE   = f"runs_vanilla_{DGP}"
PLOT_DIR   = f"plots_{DGP}"
QUEST_PATH = "data/finalQuestionnaireDataSession1.csv"

os.makedirs(PLOT_DIR, exist_ok=True)

ALPHAS = np.logspace(-3, 4, 50)   # Ridge alpha grid

# ── Questionnaire scales ──────────────────────────────────────────────────────
def compute_scale_scores(df_q):
    d = {}
    panas_even = [f"PANAS_{i}" for i in range(0, 20, 2)]
    panas_odd  = [f"PANAS_{i}" for i in range(1, 20, 2)]
    d["PANAS_PA"] = df_q[panas_even].sum(axis=1).values
    d["PANAS_NA"] = df_q[panas_odd].sum(axis=1).values
    d["STICSA"]   = df_q[[f"STICSA_{i}" for i in range(22)]].sum(axis=1).values
    d["PHQ_9"]    = df_q[[f"PHQ_9_{i}"  for i in range(9)]].sum(axis=1).values
    d["BIG_5"]    = df_q[[f"BIG_5_{i}"  for i in range(6)]].sum(axis=1).values
    d["CEI"]      = df_q[[f"CEI_{i}"    for i in range(4)]].sum(axis=1).values
    return pd.DataFrame(d, index=df_q["ID"].values)

SCALE_NAMES  = ["PANAS_PA", "PANAS_NA", "STICSA", "PHQ_9", "BIG_5", "CEI"]
SCALE_LABELS = {
    "PANAS_PA": "PANAS PA",
    "PANAS_NA": "PANAS NA",
    "STICSA":   "STICSA",
    "PHQ_9":    "PHQ-9",
    "BIG_5":    "Big-5",
    "CEI":      "CEI",
}

df_q     = pd.read_csv(QUEST_PATH)
scores_q = compute_scale_scores(df_q)

# ── Fold metadata ─────────────────────────────────────────────────────────────
fold_test_subids = {}
for fold in range(N_FOLDS):
    df_test = pd.read_csv(os.path.join(DATA_DIR, f"fold{fold}", "df_test.csv"))
    fold_test_subids[fold] = sorted(df_test["subid"].unique())

all_subids   = sorted(set(s for subs in fold_test_subids.values() for s in subs))
subid_to_idx = {s: i for i, s in enumerate(all_subids)}
N            = len(all_subids)
print(f"Total participants: {N}  |  Questionnaire loaded: {len(scores_q)}")


# ── Seed discovery ────────────────────────────────────────────────────────────
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


# ── Latent loading ────────────────────────────────────────────────────────────
def aggregate_latents(nametag, base_dir, strategy="mean"):
    """
    Returns (N, z_dim) array of per-participant latent vectors,
    averaged across seeds. NaN rows for missing participants.
    """
    acc = {s: [] for s in all_subids}
    for fold in range(N_FOLDS):
        test_subids = fold_test_subids[fold]
        for seed in discover_seeds(fold, base_dir):
            lat_path = os.path.join(DATA_DIR, f"fold{fold}",
                                    f"seed_{seed}", f"latents_tensor{nametag}.pt")
            if not os.path.exists(lat_path):
                continue
            lat = torch.load(lat_path, map_location="cpu").numpy()  # (B, T, d) or (B, d)
            if lat.ndim == 3:
                valid = (lat != 0).any(axis=-1)   # (B, T) bool
                if strategy == "last":
                    has_any  = valid.any(axis=-1)
                    last_idx = lat.shape[1] - 1 - np.argmax(valid[:, ::-1], axis=-1)
                    last_idx = np.where(has_any, last_idx, 0)
                    lat = lat[np.arange(lat.shape[0]), last_idx]
                else:  # mean
                    cnt = valid.sum(axis=1, keepdims=True).clip(min=1)
                    lat = (lat * valid[:, :, np.newaxis]).sum(axis=1) / cnt
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
    valid = int(np.isfinite(lats).all(axis=1).sum())
    print(f"  [{nametag}|{strategy}] valid: {valid}/{N},  z_dim={z_dim}")
    return lats


# ── LOO Ridge regression ──────────────────────────────────────────────────────
def loo_ridge(X, y):
    """
    Leave-one-out cross-validated Ridge regression.
    X: (n, d), y: (n,)
    Returns (y_pred, r, r2, p_value)
    """
    n = len(y)
    y_pred = np.full(n, np.nan)
    for i in range(n):
        mask_train = np.ones(n, dtype=bool)
        mask_train[i] = False
        scaler = StandardScaler()
        X_tr   = scaler.fit_transform(X[mask_train])
        X_te   = scaler.transform(X[[i]])
        ridge  = RidgeCV(alphas=ALPHAS, cv=None)   # GCV (efficient LOO internally)
        ridge.fit(X_tr, y[mask_train])
        y_pred[i] = ridge.predict(X_te)[0]
    r, p = pearsonr(y, y_pred)
    r2   = r ** 2 * np.sign(r)          # signed R² so negative correlations show
    return y_pred, float(r), float(r2), float(p)


# ── Within-fold-whitened latent loading ───────────────────────────────────────
def aggregate_latents_fold_whitened(nametag, base_dir, strategy="mean"):
    """
    Load latents per fold, z-score within each fold (to remove cross-fold
    coordinate-system differences), then concatenate across folds.
    Returns (N, z_dim) array aligned to all_subids. NaN for missing participants.
    """
    fold_lats = {}   # fold -> {subid: [vectors across seeds]}
    for fold in range(N_FOLDS):
        test_subids = fold_test_subids[fold]
        fold_acc = {s: [] for s in test_subids}
        for seed in discover_seeds(fold, base_dir):
            lat_path = os.path.join(DATA_DIR, f"fold{fold}",
                                    f"seed_{seed}", f"latents_tensor{nametag}.pt")
            if not os.path.exists(lat_path):
                continue
            lat = torch.load(lat_path, map_location="cpu").numpy()
            if lat.ndim == 3:
                valid = (lat != 0).any(axis=-1)
                if strategy == "last":
                    has_any  = valid.any(axis=-1)
                    last_idx = lat.shape[1] - 1 - np.argmax(valid[:, ::-1], axis=-1)
                    last_idx = np.where(has_any, last_idx, 0)
                    lat = lat[np.arange(lat.shape[0]), last_idx]
                else:
                    cnt = valid.sum(axis=1, keepdims=True).clip(min=1)
                    lat = (lat * valid[:, :, np.newaxis]).sum(axis=1) / cnt
            for i, subid in enumerate(test_subids):
                if i < lat.shape[0]:
                    fold_acc[subid].append(lat[i])

        # Average across seeds, then z-score this fold's latents
        fold_mat = []
        fold_subs = []
        for subid in test_subids:
            if fold_acc[subid]:
                fold_mat.append(np.mean(fold_acc[subid], axis=0))
                fold_subs.append(subid)
        if not fold_mat:
            continue
        fold_mat = np.array(fold_mat)     # (n_fold, z_dim)
        # z-score each dimension within this fold (prevent cross-fold confound)
        mu  = fold_mat.mean(axis=0)
        sig = fold_mat.std(axis=0).clip(min=1e-8)
        fold_mat = (fold_mat - mu) / sig
        fold_lats[fold] = {s: v for s, v in zip(fold_subs, fold_mat)}

    z_dim = next((v.shape[0] for fd in fold_lats.values()
                  for v in fd.values()), None)
    if z_dim is None:
        return None
    lats = np.array([
        fold_lats.get(
            next((f for f in range(N_FOLDS) if s in fold_lats.get(f, {})), -1),
            {}
        ).get(s, np.full(z_dim, np.nan))
        for s in all_subids
    ])
    valid = int(np.isfinite(lats).all(axis=1).sum())
    print(f"  [{nametag}|{strategy}|whitened] valid: {valid}/{N},  z_dim={z_dim}")
    return lats


# ── Within-fold pooled regression ─────────────────────────────────────────────
def within_fold_loo_ridge(nametag, base_dir, strategy="mean"):
    """
    For each fold: load latents for test participants, z-score X and y
    within the fold, run LOO-CV Ridge, collect predictions.
    Pool predictions across folds → compute Pearson r per scale.

    Both X (latents) and y (scores) are z-scored within each fold so that
    the regression only exploits within-fold co-variation, not between-fold
    means (which are a confound of the outer-CV split).
    """
    # Collect per-participant (latent, questionnaire) pairs organized by fold
    fold_data = {}
    for fold in range(N_FOLDS):
        test_subids = fold_test_subids[fold]
        fold_acc = {s: [] for s in test_subids}
        for seed in discover_seeds(fold, base_dir):
            lat_path = os.path.join(DATA_DIR, f"fold{fold}",
                                    f"seed_{seed}", f"latents_tensor{nametag}.pt")
            if not os.path.exists(lat_path):
                continue
            lat = torch.load(lat_path, map_location="cpu").numpy()
            if lat.ndim == 3:
                valid = (lat != 0).any(axis=-1)
                if strategy == "last":
                    has_any  = valid.any(axis=-1)
                    last_idx = lat.shape[1] - 1 - np.argmax(valid[:, ::-1], axis=-1)
                    last_idx = np.where(has_any, last_idx, 0)
                    lat = lat[np.arange(lat.shape[0]), last_idx]
                else:
                    cnt = valid.sum(axis=1, keepdims=True).clip(min=1)
                    lat = (lat * valid[:, :, np.newaxis]).sum(axis=1) / cnt
            for i, subid in enumerate(test_subids):
                if i < lat.shape[0]:
                    fold_acc[subid].append(lat[i])

        subs, lats = [], []
        for subid in test_subids:
            if fold_acc[subid] and subid in scores_q.index:
                subs.append(subid)
                lats.append(np.mean(fold_acc[subid], axis=0))
        if not subs:
            continue
        fold_data[fold] = {
            "subids": subs,
            "X": np.array(lats),
            "y": scores_q.loc[subs, SCALE_NAMES].values.astype(float),
        }

    if not fold_data:
        return None

    z_dim  = next(iter(fold_data.values()))["X"].shape[1]
    n_total = sum(len(fd["subids"]) for fd in fold_data.values())
    print(f"  [{nametag}|{strategy}|within-fold] z_dim={z_dim}, "
          f"participants={n_total}")

    # LOO within each fold (both X and y z-scored within fold)
    scale_preds  = {s: {"y_true": [], "y_pred": []} for s in SCALE_NAMES}

    for fold, fd in fold_data.items():
        X_raw = fd["X"]
        n_f   = len(fd["subids"])
        if n_f < 4:
            continue

        # Z-score X within fold
        mu_x  = X_raw.mean(axis=0)
        sig_x = X_raw.std(axis=0).clip(min=1e-8)
        X_z   = (X_raw - mu_x) / sig_x

        for si, scale in enumerate(SCALE_NAMES):
            y_raw = fd["y"][:, si]
            mu_y  = y_raw.mean()
            sig_y = y_raw.std().clip(min=1e-8)
            y_z   = (y_raw - mu_y) / sig_y   # z-scored within fold

            y_pred_z = np.full(n_f, np.nan)
            for i in range(n_f):
                tr = np.ones(n_f, dtype=bool); tr[i] = False
                ridge = RidgeCV(alphas=ALPHAS, cv=None)
                ridge.fit(X_z[tr], y_z[tr])
                y_pred_z[i] = ridge.predict(X_z[[i]])[0]

            # Undo z-scoring to get predictions on original scale
            scale_preds[scale]["y_true"].extend(y_raw.tolist())
            scale_preds[scale]["y_pred"].extend((y_pred_z * sig_y + mu_y).tolist())

    results_per_scale = {}
    for scale in SCALE_NAMES:
        y_true = np.array(scale_preds[scale]["y_true"])
        y_pred = np.array(scale_preds[scale]["y_pred"])
        if len(y_true) < 5 or not np.isfinite(y_pred).all():
            results_per_scale[scale] = {"r": np.nan, "r2": np.nan, "p": np.nan}
            continue
        r, p = pearsonr(y_true, y_pred)
        r2   = r ** 2 * np.sign(r)
        results_per_scale[scale] = {"r": float(r), "r2": float(r2),
                                     "p": float(p), "n": len(y_true)}
    return results_per_scale


# ── Main analysis ─────────────────────────────────────────────────────────────
strategies = ["mean", "last"] if STRATEGY == "both" else [STRATEGY]

results = {}   # strategy -> model -> scale -> {r, r2, p}

for strat in strategies:
    results[strat] = {}
    print(f"\n{'='*60}")
    print(f"Strategy: {strat}  (within-fold LOO, both X and y z-scored)")
    print(f"{'='*60}")

    for model_label, nametag, base_dir in [
        ("IDRNN",   "latentmodel", IDRNN_BASE),
        ("Vanilla", "vanilla",     VAN_BASE),
    ]:
        print(f"\n  -- {model_label} --")
        model_res = within_fold_loo_ridge(nametag, base_dir, strategy=strat)
        if model_res is None:
            print("  No latents found, skipping.")
            continue

        for scale in SCALE_NAMES:
            res = model_res[scale]
            r, p = res.get("r", np.nan), res.get("p", np.nan)
            r2   = res.get("r2", np.nan)
            sig  = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            print(f"    {scale:12s}:  r={r:+.3f}  R²={r2:+.3f}  p={p:.4f}  {sig}")

        results[strat][model_label] = model_res


# ── Plot: bar chart of r values, IDRNN vs Vanilla, per strategy ──────────────
BLUE   = "#2a82c2"
ORANGE = "#e1861f"

for strat in strategies:
    if not results[strat]:
        continue
    fig, ax = plt.subplots(figsize=(len(SCALE_NAMES) * 1.4 + 1, 4))
    ax.axhline(0, color="k", lw=0.8)

    n_models = len(results[strat])
    bar_w    = 0.35 if n_models == 2 else 0.5
    offsets  = np.linspace(-(n_models - 1) * bar_w / 2,
                            (n_models - 1) * bar_w / 2, n_models)
    colors   = [BLUE, ORANGE]

    for (model_label, model_res), offset, col in zip(
            results[strat].items(), offsets, colors):
        xs     = np.arange(len(SCALE_NAMES)) + offset
        r_vals = [model_res[s]["r"]  for s in SCALE_NAMES]
        p_vals = [model_res[s]["p"]  for s in SCALE_NAMES]
        bar_cols = [col if p < 0.05 else "#cccccc" for p in p_vals]
        ax.bar(xs, r_vals, width=bar_w, color=bar_cols,
               label=model_label, edgecolor="none")
        for x, r, p in zip(xs, r_vals, p_vals):
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            if sig:
                ax.text(x, r + 0.005 if r >= 0 else r - 0.01, sig,
                        ha="center", va="bottom" if r >= 0 else "top", fontsize=9)

    ax.set_xticks(np.arange(len(SCALE_NAMES)))
    ax.set_xticklabels([SCALE_LABELS[s] for s in SCALE_NAMES],
                       rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("LOO-CV Pearson r")
    ax.set_title(f"Ridge regression: latents → questionnaire scales\n"
                 f"(strategy={strat}, LOO-CV, N≈{N})")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out_path = os.path.join(PLOT_DIR, f"ridge_decode_{strat}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\n  Plot saved → {out_path}")


# ── Save JSON ─────────────────────────────────────────────────────────────────
out_json = os.path.join(PLOT_DIR, "ridge_decode_results.json")
with open(out_json, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved → {out_json}")
print("Done.")
