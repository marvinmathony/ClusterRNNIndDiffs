#!/usr/bin/env python3
"""
Outer-CV analysis for IDRNN vs Vanilla (Sloutsky data).

Each participant is a test participant in exactly one outer fold.
For analyses that yield a per-participant result (NLL, decoding correctness),
we run the analysis for every seed and then average the result for each
participant across seeds.  This gives N participant-level observations for
between-architecture statistics.

Analyses:
  1. NLL comparison    — per-participant NLL averaged across seeds;
                         paired t-test across N participants
  2. Age-group decoding — LOOCV RBF-SVM run per seed on each fold's test
                          participants; per-participant correctness averaged
                          across seeds; Wilcoxon signed-rank test
  3. RSA               — latents averaged per participant across seeds;
                         latent RDM vs age RDM (Spearman r)
  4. Summary JSON

Usage:
    python analyze_outer_cv.py
    python analyze_outer_cv.py --seeds 200,300,400 --folds 3
"""
import pingouin as pg
import argparse
import os
import json
import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.stats import ttest_rel, wilcoxon, sem as scipy_sem, spearmanr, pearsonr
from scipy.spatial.distance import pdist
from sklearn.svm import SVC
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import balanced_accuracy_score
from sklearn.decomposition import PCA
try:
    from sloutsky_cog_model import unpack_params as _unpack_cog_params
except ImportError:
    _unpack_cog_params = None

# ── Colour palette (mirrors analyze_synthetic_multi_dataset.py) ───────────────
GREY_COMMON  = "#a19f9f"
GREY_INDIV   = "#545454"
GREEN_COMMON = "#93cd90"
GREEN_INDIV  = "#3ba83b"
BLUE_COMMON  = "#a0c4e8"
BLUE_INDIV   = "#2a82c2"
ORANGE       = "#e1861f"

# Random-model NLL reference — set after DGP is known (below)

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--seeds", type=str, default=None,
                    help="Comma-separated seeds to restrict, e.g. '200,300,400'")
parser.add_argument("--vanilla_seeds", type=str, default=None,
                    help="Optional separate vanilla seed restriction "
                         "(falls back to --seeds if unset).")
parser.add_argument("--folds", type=int, default=3,
                    help="Number of outer folds (default 3)")
parser.add_argument("--dgp", type=str, default="sloutsky",
                    help="Dataset DGP name: 'sloutsky' (default) or 'dezfouli'")
parser.add_argument("--run_suffix", type=str, default=None,
                    help="If set, look under runs_{DGP}_{suffix}/, "
                         "runs_vanilla_{DGP}_{suffix}/, and "
                         "data_{DGP}/foldF/seed_S/{suffix}/ for retrained runs.")
parser.add_argument("--top_k_by_specificity", type=int, default=None,
                    help="For IDRNN bars: per fold, average NLL/decoding only "
                         "across the top-K seeds (ranked by step1_specificity "
                         "read from each seed's config.json). None = use all "
                         "(default).")
parser.add_argument("--plot_suffix", type=str, default=None,
                    help="Optional suffix appended to all output PNG/JSON "
                         "filenames so v2 outputs don't overwrite v1 ones.")
parser.add_argument("--cog_em_csv", type=str, default="cog_model_results.csv",
                    help="Per-fold cog-model EM NLL filename (default: "
                         "'cog_model_results.csv'). Use to compare alternative "
                         "marginal-LL evaluations without overwriting originals.")
args = parser.parse_args()
FILTER_SEEDS = (set(int(s) for s in args.seeds.split(",")) if args.seeds else None)
FILTER_VANILLA_SEEDS = (set(int(s) for s in args.vanilla_seeds.split(","))
                        if args.vanilla_seeds else None)
N_FOLDS = args.folds

# ── Configuration ─────────────────────────────────────────────────────────────
DGP        = args.dgp
DATA_DIR   = f"data_{DGP}"
RUN_SUFFIX = args.run_suffix
SEED_SUB   = RUN_SUFFIX if RUN_SUFFIX else ""    # subdir under data_*/foldF/seed_S/
_RUNS_TAIL = f"_{RUN_SUFFIX}" if RUN_SUFFIX else ""
IDRNN_BASE = f"runs_{DGP}{_RUNS_TAIL}"
VAN_BASE   = f"runs_vanilla_{DGP}{_RUNS_TAIL}"
PLOT_DIR   = f"plots_{DGP}"
os.makedirs(PLOT_DIR, exist_ok=True)
TOP_K_BY_SPEC = args.top_k_by_specificity
PLOT_TAG = (f"_{args.plot_suffix}" if args.plot_suffix
            else (f"_{RUN_SUFFIX}" if RUN_SUFFIX else ""))

# Random-model NLL reference (depends on number of options)
if DGP == "dezfouli":
    K_OPTIONS = 2   # binary choice (2 options per trial)
    T_TRIALS  = None
    RANDOM_NLL = np.log(K_OPTIONS)   # per-trial reference
else:
    K_OPTIONS = 4
    T_TRIALS  = 30
    RANDOM_NLL = T_TRIALS * np.log(K_OPTIONS)   # ≈ 41.59

_seed_tag = ("_seeds" + "-".join(str(s) for s in sorted(FILTER_SEEDS))
             if FILTER_SEEDS else "")
_seed_tag = _seed_tag + PLOT_TAG       # append v2 suffix so v2 outputs don't
                                        # overwrite v1 PNG/JSON files

C_PARAMS     = [0.01, 0.1, 1, 10, 100]
GAMMA_PARAMS = [0.01, 0.1, 1, "scale", "auto"]


# ── Helpers ───────────────────────────────────────────────────────────────────
def discover_seeds(fold, base_dir):
    """Return sorted list of seed ints found in base_dir/fold{fold}/.
    Vanilla bases use FILTER_VANILLA_SEEDS when set, IDRNN/CP use FILTER_SEEDS."""
    fold_dir = os.path.join(base_dir, f"fold{fold}")
    if not os.path.isdir(fold_dir):
        return []
    _is_vanilla = "vanilla" in os.path.basename(base_dir)
    _fset = (FILTER_VANILLA_SEEDS
             if _is_vanilla and FILTER_VANILLA_SEEDS is not None
             else FILTER_SEEDS)
    seeds = []
    for d in os.listdir(fold_dir):
        if d.startswith("seed_") and os.path.isdir(os.path.join(fold_dir, d)):
            s = int(d.split("_")[1])
            if _fset is None or s in _fset:
                seeds.append(s)
    return sorted(seeds)


def _rank_seeds_by_specificity(fold, base_dir, seeds, top_k):
    """Read step1_specificity from each seed's config.json; return top_k seeds
    ranked by specificity desc.  Falls back to all input seeds (in seed order)
    if no specificity values are available."""
    if top_k is None or not seeds:
        return list(seeds)
    fold_dir = os.path.join(base_dir, f"fold{fold}")
    spec_per_seed = {}
    for s in seeds:
        cfg_p = os.path.join(fold_dir, f"seed_{s}", "config.json")
        if not os.path.exists(cfg_p):
            continue
        try:
            with open(cfg_p) as f:
                cfg = json.load(f)
            sp = cfg.get("step1_specificity")
            if sp is not None:
                spec_per_seed[s] = float(sp)
        except Exception:
            pass
    if not spec_per_seed:
        return list(seeds)
    ordered = sorted(spec_per_seed.items(), key=lambda kv: -kv[1])
    return [s for s, _ in ordered[:top_k]]


def rbfsvm_traintest(X_train, y_train, X_test):
    """
    Grid search on training data, fit on all training participants,
    predict on test participants.  Returns per-test-participant predictions.
    """
    param_grid = {"svc__C": C_PARAMS, "svc__gamma": GAMMA_PARAMS}
    pipe = make_pipeline(StandardScaler(), SVC(kernel="rbf"))
    gs   = GridSearchCV(pipe, param_grid, cv=3, n_jobs=-1)
    gs.fit(X_train, y_train)
    best = gs.best_estimator_.get_params()
    clf  = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=best["svc__C"], gamma=best["svc__gamma"])
    )
    clf.fit(X_train, y_train)
    return clf.predict(X_test)


def logreg_traintest(X_train, y_train, X_test):
    """
    Same training logic as rbfsvm_traintest but with L2 logistic regression.
    Grid search over C on training data, fit, predict on test participants.
    """
    param_grid = {"logisticregression__C": C_PARAMS}
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    gs   = GridSearchCV(pipe, param_grid, cv=3, n_jobs=-1)
    gs.fit(X_train, y_train)
    best_C = gs.best_estimator_.get_params()["logisticregression__C"]
    clf  = make_pipeline(StandardScaler(),
                         LogisticRegression(C=best_C, max_iter=1000))
    clf.fit(X_train, y_train)
    return clf.predict(X_test)


# ── Load participant metadata from fold test CSVs ─────────────────────────────
# Label column: 'age' for sloutsky, 'diag' for dezfouli
LABEL_COL = "diag" if DGP == "dezfouli" else "age"

all_subid_to_label = {}   # subid (int) -> label string
fold_test_subids   = {}   # fold -> sorted list of test subids
fold_train_subids  = {}   # fold -> sorted list of train subids

for fold in range(N_FOLDS):
    fold_data_dir = os.path.join(DATA_DIR, f"fold{fold}")
    df_test_fold  = pd.read_csv(os.path.join(fold_data_dir, "df_test.csv"))
    df_train_fold = pd.read_csv(os.path.join(fold_data_dir, "df_train.csv"))
    fold_test_subids[fold]  = sorted(df_test_fold["subid"].unique())
    fold_train_subids[fold] = sorted(df_train_fold["subid"].unique())
    for subid in fold_test_subids[fold]:
        label = df_test_fold.loc[df_test_fold["subid"] == subid, LABEL_COL].iloc[0]
        all_subid_to_label[subid] = label

# Exclude participants with degenerate behaviour (always choose option 0)
EXCLUDE_SUBIDS = {59} if DGP == "sloutsky" else set()

all_subids   = sorted(s for s in all_subid_to_label.keys() if s not in EXCLUDE_SUBIDS)
subid_to_idx = {s: i for i, s in enumerate(all_subids)}
N = len(all_subids)
print(f"Total participants across {N_FOLDS} folds: {N} (excluded: {EXCLUDE_SUBIDS})")

# Group order: fixed for sloutsky, auto-discovered for dezfouli
if DGP == "sloutsky":
    GROUP_ORDER  = ["young_child", "old_child", "adult"]
    GROUP_LABELS = ["Young children", "Older children", "Adults"]
    GROUP_COLORS = ["#5B9BD5", "#ED7D31", "#A5A5A5"]
else:
    GROUP_ORDER  = sorted(set(all_subid_to_label.values()))
    GROUP_LABELS = GROUP_ORDER
    _palette     = ["#5B9BD5", "#ED7D31", "#A5A5A5", "#70AD47", "#FF0000"]
    GROUP_COLORS = _palette[:len(GROUP_ORDER)]

group_map = {g: i for i, g in enumerate(GROUP_ORDER)}

# Human-readable label for plot titles / file names
GROUP_LABEL_STR = "Diagnosis-group" if DGP == "dezfouli" else "Age-group"
DECODING_TAG    = "diag_decoding"   if DGP == "dezfouli" else "age_decoding"
RSA_LABEL_STR   = "group"           if DGP == "dezfouli" else "age"

# Keep backward-compatible aliases used elsewhere in the script
all_subid_to_age = all_subid_to_label
y_age_group = np.array([all_subid_to_label[s] for s in all_subids])
y_age       = np.array([group_map[g] for g in y_age_group])

# Pre-compute per-fold labels aligned to fold_test/train_subids
fold_y_age       = {fold: np.array([group_map[all_subid_to_label[s]] for s in fold_test_subids[fold]])
                    for fold in range(N_FOLDS)}
fold_y_age_train = {fold: np.array([group_map[all_subid_to_label[s]] for s in fold_train_subids[fold]])
                    for fold in range(N_FOLDS)}


# ── 1. NLL: per-participant, averaged across seeds ────────────────────────────
def aggregate_nll(nametag, base_dir, use_specificity=False, spec_base_dir=None):
    """
    Collect NLL for every participant from their fold, averaged across seeds.
    Returns (N,) array aligned to all_subids; NaN if missing.

    use_specificity: if True and TOP_K_BY_SPEC is set, per-fold rank seeds by
                     step1_specificity (read from configs in `spec_base_dir`,
                     defaulting to `base_dir`) and keep only the top-K.
    """
    acc = {s: [] for s in all_subids}
    spec_dir = spec_base_dir or base_dir
    for fold in range(N_FOLDS):
        fold_data_dir = os.path.join(DATA_DIR, f"fold{fold}")
        test_subids   = fold_test_subids[fold]
        seeds_fold = discover_seeds(fold, base_dir)
        if use_specificity and TOP_K_BY_SPEC is not None:
            seeds_fold = _rank_seeds_by_specificity(
                fold, spec_dir, seeds_fold, TOP_K_BY_SPEC)
            print(f"  [fold {fold} {nametag}] top-{TOP_K_BY_SPEC} by "
                  f"step1_specificity: {seeds_fold}")
        for seed in seeds_fold:
            csv_path = os.path.join(fold_data_dir, f"seed_{seed}", SEED_SUB,
                                    f"rnn_results{nametag}.csv")
            if not os.path.exists(csv_path):
                continue
            df = pd.read_csv(csv_path)
            if "normalized_likelihood" not in df.columns:
                continue
            nlls = df["normalized_likelihood"].values
            for i, subid in enumerate(test_subids):
                if subid in EXCLUDE_SUBIDS:
                    continue
                if i < len(nlls):
                    acc[subid].append(nlls[i])

    nll = np.array([np.mean(acc[s]) if acc[s] else np.nan for s in all_subids])
    print(f"  [{nametag}] valid NLL: {int(np.isfinite(nll).sum())}/{N}")
    return nll


# ── 2. Decoding: train→test per seed, average per-participant correctness
def aggregate_decoding(nametag, base_dir, decoder=rbfsvm_traintest,
                       use_specificity=False):
    """
    For each fold × seed:
      - Grid search on training participants' latents
      - Fit SVM on all training participants
      - Predict on test participants → per-participant correctness (0/1)

    For each participant: average correctness across seeds in their fold.

    Returns
    -------
    correct_mean : (N,) float in [0, 1] — NaN if no data for that participant
    """
    acc = {s: [] for s in all_subids}   # subid -> list of 0/1 per seed

    for fold in range(N_FOLDS):
        fold_data_dir  = os.path.join(DATA_DIR, f"fold{fold}")
        test_subids    = fold_test_subids[fold]
        y_test_fold    = fold_y_age[fold]
        y_train_fold   = fold_y_age_train[fold]
        seeds          = discover_seeds(fold, base_dir)
        if use_specificity and TOP_K_BY_SPEC is not None:
            seeds = _rank_seeds_by_specificity(
                fold, base_dir, seeds, TOP_K_BY_SPEC)

        for seed in seeds:
            lat_test_path  = os.path.join(fold_data_dir, f"seed_{seed}", SEED_SUB,
                                          f"latents_tensor{nametag}.pt")
            lat_train_path = os.path.join(fold_data_dir, f"seed_{seed}", SEED_SUB,
                                          f"latents_tensor{nametag}_traindata.pt")
            if not os.path.exists(lat_test_path) or not os.path.exists(lat_train_path):
                continue

            lat_test  = torch.load(lat_test_path,  map_location="cpu").numpy()
            lat_train = torch.load(lat_train_path, map_location="cpu").numpy()
            # Average over blocks/timestep dimension if present (matches decoding_logistic.py)
            if lat_test.ndim  > 2: lat_test  = lat_test.mean(axis=1)
            if lat_train.ndim > 2: lat_train = lat_train.mean(axis=1)

            if lat_test.shape[0] != len(test_subids):
                print(f"  Warning: fold {fold} seed {seed}: "
                      f"test lat n={lat_test.shape[0]} ≠ {len(test_subids)}, skipping")
                continue

            print(f"  fold {fold} seed {seed}: train→test {decoder.__name__} "
                  f"(train n={lat_train.shape[0]}, test n={lat_test.shape[0]}) ...",
                  flush=True)
            preds = decoder(lat_train, y_train_fold, lat_test)
            for i, subid in enumerate(test_subids):
                if subid in acc:
                    acc[subid].append(int(preds[i] == y_test_fold[i]))

    correct_mean = np.array([
        np.mean(acc[s]) if acc[s] else np.nan for s in all_subids
    ])
    n_valid = int(np.isfinite(correct_mean).sum())
    print(f"  [{nametag}] participants with decoding data: {n_valid}/{N}")
    return correct_mean


# ── Block-structure latent aggregation helpers ────────────────────────────────
_block_bk = None
_block_t  = None


def _load_block_dims():
    """Load (Bk, T) from the first available fold's xin_test.npy."""
    global _block_bk, _block_t
    if _block_bk is not None:
        return _block_bk, _block_t
    for fold in range(N_FOLDS):
        p = os.path.join(DATA_DIR, f"fold{fold}", "xin_test.npy")
        if os.path.exists(p):
            x = np.load(p, mmap_mode="r")   # (B, Bk, T, in_dim)
            _block_bk, _block_t = int(x.shape[1]), int(x.shape[2])
            print(f"  [block dims] Bk={_block_bk}, T={_block_t} per block")
            return _block_bk, _block_t
    return None, None


def _last_per_block(lat_raw, bk, t_per_block):
    """(B, Bk*T, z_dim) → (B, Bk, z_dim)  taking the last *valid* (non-padding) timestep per block.

    mu_tensor is initialised to zeros and only filled for non-padded trials, so
    padding positions are exactly zero vectors.  We find the last non-zero row
    in each block; if a block is entirely zero (degenerate), we leave it as zeros.
    """
    B, _, z_dim = lat_raw.shape
    lat_blocks = lat_raw.reshape(B, bk, t_per_block, z_dim)  # (B, Bk, T, z_dim)

    # valid[b, k, t] = True if at least one z-dimension is non-zero
    valid = (lat_blocks != 0).any(axis=-1)          # (B, Bk, T) bool
    has_any = valid.any(axis=-1)                     # (B, Bk) — block has ≥1 valid trial

    # Flip T, take argmax (= index of first True in flipped = last True in original)
    last_idx = t_per_block - 1 - np.argmax(valid[:, :, ::-1], axis=-1)  # (B, Bk)
    # argmax returns 0 for all-False rows; those are masked out below anyway

    # Vectorised gather
    b_idx  = np.arange(B)[:, None]    # (B, 1)
    bk_idx = np.arange(bk)[None, :]   # (1, Bk)
    result = lat_blocks[b_idx, bk_idx, last_idx]    # (B, Bk, z_dim)

    result[~has_any] = 0.0             # zero out degenerate all-padding blocks
    return result


def _reduce_for_strategy(lat_train_raw, lat_test_raw, strategy, bk, t_per_block):
    """
    Reduce (B, Bk*T, z_dim) latents to (B, features) according to strategy.
    PCA is fitted on training blocks and applied to test blocks.

    Strategies
    ----------
    last_participant : globally last valid latent per participant → (B, z_dim)
    mean_participant : mean over all valid latents per participant → (B, z_dim)
    last_block_mean  : last timestep per block, averaged across blocks → (B, z_dim)
    pca_last         : PCA on block latents (fit on train), last block's PC scores → (B, n_pcs)
    pca_mean         : same PCA, mean PC scores across blocks → (B, n_pcs)
    concat           : last timestep per block, all blocks concatenated → (B, Bk*z_dim)
    """
    if lat_train_raw.ndim == 2:
        return lat_train_raw, lat_test_raw   # already (B, z_dim)

    if strategy == "last_participant":
        # Globally last valid (non-zero) timestep per participant in the flattened Bk*T sequence.
        # For IDRNN with cross-block hidden state this is the richest single posterior estimate.
        def _last_valid_global(lat):
            B, seq_len, z_dim = lat.shape
            valid = (lat != 0).any(axis=-1)          # (B, Bk*T) bool
            # argmax on reversed axis gives index of first True = last True in original
            last_idx = seq_len - 1 - np.argmax(valid[:, ::-1], axis=-1)   # (B,)
            # Clamp: if a participant has NO valid trial, fall back to index 0
            has_any  = valid.any(axis=-1)             # (B,) bool
            last_idx = np.where(has_any, last_idx, 0)
            return lat[np.arange(B), last_idx]        # (B, z_dim)
        return _last_valid_global(lat_train_raw), _last_valid_global(lat_test_raw)

    if strategy == "mean_participant":
        # Mean over all valid (non-zero) timesteps in the flattened Bk*T dimension.
        def _masked_mean(lat):
            valid = (lat != 0).any(axis=-1)           # (B, Bk*T) bool
            lat_sum = (lat * valid[:, :, np.newaxis]).sum(axis=1)   # (B, z_dim)
            cnt     = valid.sum(axis=1, keepdims=True).clip(min=1)  # (B, 1)
            return lat_sum / cnt
        return _masked_mean(lat_train_raw), _masked_mean(lat_test_raw)

    train_blocks = _last_per_block(lat_train_raw, bk, t_per_block)  # (B_tr, Bk, z_dim)
    test_blocks  = _last_per_block(lat_test_raw,  bk, t_per_block)  # (B_te, Bk, z_dim)

    if strategy == "last_block_mean":
        return train_blocks.mean(axis=1), test_blocks.mean(axis=1)

    elif strategy == "concat":
        B_tr, B_te = train_blocks.shape[0], test_blocks.shape[0]
        return train_blocks.reshape(B_tr, -1), test_blocks.reshape(B_te, -1)

    elif strategy in ("pca_last", "pca_mean"):
        B_tr, _, z_dim = train_blocks.shape
        B_te = test_blocks.shape[0]
        n_pcs = max(1, min(bk - 1, z_dim, 8))
        pca = PCA(n_components=n_pcs)
        pca.fit(train_blocks.reshape(-1, z_dim))     # fit on (B_tr * Bk, z_dim)
        proj_tr = pca.transform(train_blocks.reshape(-1, z_dim)).reshape(B_tr, bk, n_pcs)
        proj_te = pca.transform(test_blocks.reshape(-1, z_dim)).reshape(B_te, bk, n_pcs)
        if strategy == "pca_last":
            return proj_tr[:, -1, :], proj_te[:, -1, :]
        else:
            return proj_tr.mean(axis=1), proj_te.mean(axis=1)

    # fallback
    return lat_train_raw.mean(axis=1), lat_test_raw.mean(axis=1)


BLOCK_STRATEGIES = ["last_participant", "mean_participant", "last_block_mean", "pca_last", "pca_mean", "concat"]
STRATEGY_LABELS  = {
    "last_participant": "Last\ntimestep",
    "mean_participant": "Mean\n(valid)",
    "last_block_mean": "Last-block\nmean",
    "pca_last":        "PCA\n(last block)",
    "pca_mean":        "PCA\n(mean block)",
    "concat":          "Concat\nblocks",
    "mean":            "Time\nmean",
}


def aggregate_decoding_multi(nametag, base_dir, decoder=rbfsvm_traintest,
                              use_specificity=False):
    """
    Like aggregate_decoding but for block-structured DGPs runs all 4 aggregation
    strategies in a single pass (loading each tensor once per fold × seed).

    For non-block DGPs falls back to a single 'mean' strategy.

    Returns dict: strategy_name → (N,) correctness array.
    """
    if DGP != "dezfouli":
        return {"mean": aggregate_decoding(nametag, base_dir, decoder,
                                           use_specificity=use_specificity)}

    bk, t_per_block = _load_block_dims()
    if bk is None:
        return {"mean": aggregate_decoding(nametag, base_dir, decoder,
                                           use_specificity=use_specificity)}

    acc = {s: {sub: [] for sub in all_subids} for s in BLOCK_STRATEGIES}

    for fold in range(N_FOLDS):
        fold_data_dir = os.path.join(DATA_DIR, f"fold{fold}")
        test_subids   = fold_test_subids[fold]
        y_test_fold   = fold_y_age[fold]
        y_train_fold  = fold_y_age_train[fold]
        seeds         = discover_seeds(fold, base_dir)
        if use_specificity and TOP_K_BY_SPEC is not None:
            seeds = _rank_seeds_by_specificity(
                fold, base_dir, seeds, TOP_K_BY_SPEC)

        for seed in seeds:
            lat_test_path  = os.path.join(fold_data_dir, f"seed_{seed}", SEED_SUB,
                                          f"latents_tensor{nametag}.pt")
            lat_train_path = os.path.join(fold_data_dir, f"seed_{seed}", SEED_SUB,
                                          f"latents_tensor{nametag}_traindata.pt")
            if not os.path.exists(lat_test_path) or not os.path.exists(lat_train_path):
                continue

            lat_test_raw  = torch.load(lat_test_path,  map_location="cpu").numpy()
            lat_train_raw = torch.load(lat_train_path, map_location="cpu").numpy()

            if lat_test_raw.shape[0] != len(test_subids):
                print(f"  Warning: fold {fold} seed {seed}: "
                      f"test lat n={lat_test_raw.shape[0]} ≠ {len(test_subids)}, skipping")
                continue

            for strategy in BLOCK_STRATEGIES:
                lat_tr, lat_te = _reduce_for_strategy(
                    lat_train_raw, lat_test_raw, strategy, bk, t_per_block
                )
                preds = decoder(lat_tr, y_train_fold, lat_te)
                for i, subid in enumerate(test_subids):
                    if subid in acc[strategy]:
                        acc[strategy][subid].append(int(preds[i] == y_test_fold[i]))

            print(f"  fold {fold} seed {seed} [{nametag} | {decoder.__name__}]: done",
                  flush=True)

    results = {}
    for s in BLOCK_STRATEGIES:
        arr = np.array([
            np.mean(acc[s][sub]) if acc[s][sub] else np.nan for sub in all_subids
        ])
        n_valid = int(np.isfinite(arr).sum())
        print(f"  [{nametag} | {s}] participants with data: {n_valid}/{N}")
        results[s] = arr
    return results


# ── 3. Latents averaged per participant (for RSA) ─────────────────────────────
def aggregate_lats(nametag, base_dir, use_specificity=False):
    """
    Average latent vectors per participant across seeds.
    Returns (N, z_dim) array; NaN rows if missing.
    """
    acc = {s: [] for s in all_subids}
    for fold in range(N_FOLDS):
        fold_data_dir = os.path.join(DATA_DIR, f"fold{fold}")
        test_subids   = fold_test_subids[fold]
        seeds_fold = discover_seeds(fold, base_dir)
        if use_specificity and TOP_K_BY_SPEC is not None:
            seeds_fold = _rank_seeds_by_specificity(
                fold, base_dir, seeds_fold, TOP_K_BY_SPEC)
        for seed in seeds_fold:
            lat_path = os.path.join(fold_data_dir, f"seed_{seed}", SEED_SUB,
                                    f"latents_tensor{nametag}.pt")
            if not os.path.exists(lat_path):
                continue
            lat = torch.load(lat_path, map_location="cpu").numpy()
            if lat.ndim > 2:
                lat = lat.mean(axis=1)
            for i, subid in enumerate(test_subids):
                if subid in acc and i < lat.shape[0]:
                    acc[subid].append(lat[i])

    z_dim = next((acc[s][0].shape[0] for s in all_subids if acc[s]), None)
    if z_dim is None:
        return None
    lats = np.array([
        np.mean(acc[s], axis=0) if acc[s] else np.full(z_dim, np.nan)
        for s in all_subids
    ])
    n_valid = int(np.isfinite(lats).all(axis=1).sum())
    print(f"  [{nametag}] valid latents: {n_valid}/{N}")
    return lats


# ── 4. Cog-model NLL: one CSV per fold, aligned by subid ─────────────────────
def aggregate_cog_nll(csv_filename, label=""):
    """
    Load per-participant NLL from fold-specific CSV files saved by
    sloutsky_cog_model.py / sloutsky_ill_specified_cog_model.py.
    Expects columns: 'subid', 'normalized_likelihood'.
    Returns (N,) array aligned to all_subids; NaN if missing.
    """
    nll = np.full(N, np.nan)
    for fold in range(N_FOLDS):
        csv_path = os.path.join(DATA_DIR, f"fold{fold}", csv_filename)
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        if "subid" not in df.columns or "normalized_likelihood" not in df.columns:
            print(f"  [{label}] {csv_path}: missing required columns, skipping")
            continue
        for _, row in df.iterrows():
            s = int(row["subid"])
            if s in subid_to_idx:
                nll[subid_to_idx[s]] = row["normalized_likelihood"]
    valid = int(np.isfinite(nll).sum())
    tag = label or csv_filename
    print(f"  [{tag}] valid: {valid}/{N}")
    return nll


# ── 5. True cog params for all N participants (from fold em_results.npz) ──────
COG_PARAM_NAMES  = ["theta", "b_value_train", "b_uncertain_train",
                    "b_lag_train", "b_novelty_train"]
COG_PARAM_LABELS = {"theta": "θ", "b_value_train": "w value",
                    "b_uncertain_train": "w uncert", "b_lag_train": "w lag",
                    "b_novelty_train": "w novelty"}

def load_true_cog_params():
    """
    Load MAP-estimated cog params for all N test participants across folds.
    Returns dict: param_name -> (N,) array aligned to all_subids.
    """
    true_p = {pn: np.full(N, np.nan) for pn in COG_PARAM_NAMES}
    for fold in range(N_FOLDS):
        em_path = os.path.join(DATA_DIR, f"fold{fold}", "em_results.npz")
        if not os.path.exists(em_path):
            print(f"  [cog_params] missing {em_path}, skipping fold {fold}")
            continue
        em = np.load(em_path, allow_pickle=True)
        h_test   = em["h_all_test"]           # (n_test, 5)
        p_test   = em["participants_test"].tolist()
        for i, subid in enumerate(p_test):
            if subid in subid_to_idx:
                p = _unpack_cog_params(h_test[i])
                for pn in COG_PARAM_NAMES:
                    true_p[pn][subid_to_idx[subid]] = p[pn]
    valid = int(np.isfinite(true_p["theta"]).sum())
    print(f"  [true_cog_params] valid: {valid}/{N}")
    return true_p


# ── 6. Cog-param decoding: Ridge train→test per fold×seed ────────────────────
def _ridge_traintest(X_train, y_train, X_test):
    alphas = np.logspace(-3, 3, 20)
    pipe = make_pipeline(StandardScaler(),
                         RidgeCV(alphas=alphas, cv=min(5, len(X_train))))
    pipe.fit(X_train, y_train)
    return pipe.predict(X_test)


def aggregate_cog_param_decoding(rnn_nametag, rnn_base_dir, n_pcs=None,
                                  use_last=False, use_specificity=False):
    """
    For each fold × seed:
      - Load train latents (latents_tensor{nametag}_traindata.pt)
      - Load test  latents (latents_tensor{nametag}.pt)
      - Reduce time dimension: last timestep if use_last=True, else mean
      - If n_pcs is set: fit PCA on train latents, project both train and test
      - Load train/test cog params from em_results.npz for that fold
      - Fit Ridge on (projected) train latents, predict test → per-participant predicted value

    Average predictions per participant across seeds.
    Returns dict: param_name -> (N,) seed-averaged predicted values (NaN if missing).
    """
    pred_acc = {s: {pn: [] for pn in COG_PARAM_NAMES} for s in all_subids}

    for fold in range(N_FOLDS):
        fold_data_dir   = os.path.join(DATA_DIR, f"fold{fold}")
        test_subs_fold  = fold_test_subids[fold]
        train_subs_fold = fold_train_subids[fold]

        em_path = os.path.join(fold_data_dir, "em_results.npz")
        if not os.path.exists(em_path):
            print(f"  [{rnn_nametag}] fold {fold}: missing em_results.npz, skipping")
            continue
        em = np.load(em_path, allow_pickle=True)

        # Build cog-param matrices ordered to match fold_{train,test}_subids
        def _params_ordered(h_all, participants, ordered_subs):
            idx_map = {int(s): i for i, s in enumerate(participants)}
            rows = []
            for s in ordered_subs:
                p = _unpack_cog_params(h_all[idx_map[s]])
                rows.append([p[pn] for pn in COG_PARAM_NAMES])
            return np.array(rows)   # (n, 5)

        cog_train = _params_ordered(em["h_all_train"],
                                    em["participants_train"].tolist(),
                                    train_subs_fold)
        cog_test  = _params_ordered(em["h_all_test"],
                                    em["participants_test"].tolist(),
                                    test_subs_fold)

        seeds_fold = discover_seeds(fold, rnn_base_dir)
        if use_specificity and TOP_K_BY_SPEC is not None:
            seeds_fold = _rank_seeds_by_specificity(
                fold, rnn_base_dir, seeds_fold, TOP_K_BY_SPEC)
        for seed in seeds_fold:
            lat_test_path  = os.path.join(fold_data_dir, f"seed_{seed}", SEED_SUB,
                                          f"latents_tensor{rnn_nametag}.pt")
            lat_train_path = os.path.join(fold_data_dir, f"seed_{seed}", SEED_SUB,
                                          f"latents_tensor{rnn_nametag}_traindata.pt")
            if not os.path.exists(lat_test_path) or not os.path.exists(lat_train_path):
                continue

            lat_test  = torch.load(lat_test_path,  map_location="cpu").numpy()
            lat_train = torch.load(lat_train_path, map_location="cpu").numpy()
            if lat_test.ndim  > 2:
                lat_test  = lat_test[:, -1, :]  if use_last else lat_test.mean(axis=1)
            if lat_train.ndim > 2:
                lat_train = lat_train[:, -1, :] if use_last else lat_train.mean(axis=1)

            if lat_test.shape[0] != len(test_subs_fold):
                continue

            # Optional PCA: fit on training latents, project both sets
            if n_pcs is not None:
                n_pcs_eff = min(n_pcs, lat_train.shape[1], lat_train.shape[0] - 1)
                pca = PCA(n_components=n_pcs_eff)
                lat_train = pca.fit_transform(lat_train)
                lat_test  = pca.transform(lat_test)

            for j, pn in enumerate(COG_PARAM_NAMES):
                preds = _ridge_traintest(lat_train, cog_train[:, j], lat_test)
                for i, subid in enumerate(test_subs_fold):
                    if subid in pred_acc:
                        pred_acc[subid][pn].append(float(preds[i]))

    pred_avg = {
        pn: np.array([
            np.mean(pred_acc[s][pn]) if pred_acc[s][pn] else np.nan
            for s in all_subids
        ])
        for pn in COG_PARAM_NAMES
    }
    valid = int(np.isfinite(pred_avg["theta"]).sum())
    print(f"  [{rnn_nametag}] cog-param decoding: {valid}/{N} participants")
    return pred_avg


# ── Run aggregation ───────────────────────────────────────────────────────────
print("\n── Aggregating IDRNN ──")
idrnn_nll     = aggregate_nll("latentmodel", IDRNN_BASE, use_specificity=True)
print("\n── Aggregating Vanilla ──")
van_nll       = aggregate_nll("vanilla",      VAN_BASE)

print("\n── Aggregating RNN common-process ──")
# CP RNN NLL CSVs sit in the IDRNN run dirs, so seed selection by IDRNN
# step1 specificity is the natural choice here too.
rnn_cp_nll = aggregate_nll("_common_process", IDRNN_BASE, use_specificity=True)

if DGP in ("sloutsky", "dezfouli"):
    print("\n── Aggregating cognitive model NLLs (from fold CSVs) ──")
    cog_em_nll  = aggregate_cog_nll(args.cog_em_csv,               "CogModel_EM")
    cog_cp_nll  = aggregate_cog_nll("cog_model_cp_results.csv",    "CogModel_CP")
    ill_em_nll  = aggregate_cog_nll("ill_specified_map_results.csv","IllSpec_EM")
    ill_cp_nll  = aggregate_cog_nll("ill_specified_cp_results.csv", "IllSpec_CP")
else:
    cog_em_nll = cog_cp_nll = ill_em_nll = ill_cp_nll = np.full(N, np.nan)

print("\n── Decoding IDRNN — all strategies (RBF-SVM) ──")
idrnn_results_svm = aggregate_decoding_multi("latentmodel", IDRNN_BASE,
                                              decoder=rbfsvm_traintest,
                                              use_specificity=True)
print("\n── Decoding Vanilla — all strategies (RBF-SVM) ──")
van_results_svm   = aggregate_decoding_multi("vanilla",      VAN_BASE,
                                              decoder=rbfsvm_traintest)

print("\n── Decoding IDRNN — all strategies (Logistic Regression) ──")
idrnn_results_lr  = aggregate_decoding_multi("latentmodel", IDRNN_BASE,
                                              decoder=logreg_traintest,
                                              use_specificity=True)
print("\n── Decoding Vanilla — all strategies (Logistic Regression) ──")
van_results_lr    = aggregate_decoding_multi("vanilla",      VAN_BASE,
                                              decoder=logreg_traintest)

# Primary strategy for downstream stats.
# last_participant: globally last valid latent per participant — for IDRNN with cross-block
# hidden state this is the richest single estimate (posterior after seeing all blocks);
# for Vanilla it's the last trial of the last block (blocks are independent anyway).
_primary = "last_participant" if DGP == "dezfouli" else "mean"
idrnn_correct    = idrnn_results_svm[_primary]
van_correct      = van_results_svm[_primary]
idrnn_correct_lr = idrnn_results_lr[_primary]
van_correct_lr   = van_results_lr[_primary]

print("\n── Latents for RSA (averaged per participant) ──")
idrnn_lats = aggregate_lats("latentmodel", IDRNN_BASE, use_specificity=True)
van_lats   = aggregate_lats("vanilla",      VAN_BASE)

if DGP == "sloutsky":
    print("\n── True cog params (from fold em_results.npz) ──")
    true_cog = load_true_cog_params()

    print("\n── Cog-param decoding IDRNN (full latents, last timestep) ──")
    idrnn_cog_pred = aggregate_cog_param_decoding("latentmodel", IDRNN_BASE, use_last=True)
    print("\n── Cog-param decoding Vanilla (full latents, time mean) ──")
    van_cog_pred   = aggregate_cog_param_decoding("vanilla",      VAN_BASE,  use_last=False)
else:
    true_cog = idrnn_cog_pred = van_cog_pred = None

if DGP == "sloutsky":
    print("\n── Cog-param decoding IDRNN (PC1, last timestep) ──")
    idrnn_cog_pred_pc1 = aggregate_cog_param_decoding("latentmodel", IDRNN_BASE,
                                                       n_pcs=1, use_last=True)
    print("\n── Cog-param decoding Vanilla (PC1, time mean) ──")
    van_cog_pred_pc1   = aggregate_cog_param_decoding("vanilla",      VAN_BASE,
                                                       n_pcs=1, use_last=False)
else:
    idrnn_cog_pred_pc1 = van_cog_pred_pc1 = None


# ── Shared plot helper ────────────────────────────────────────────────────────
def _sig_bar(ax, x1, x2, y, h, p):
    s = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c="k")
    ax.text((x1 + x2) / 2, y + h, s, ha="center", va="bottom", fontsize=10)


# ── Plot 1: Multi-model NLL comparison ───────────────────────────────────────
print(f"\n── NLL comparison (N={N} participants) ──")

GROUP_GAP = 0.5
BAR_W_NLL = 0.55

# (label, nll_array, color) — None sentinel inserts a group gap
_nll_groups = [
    ("Ill-spec.\nCP",  ill_cp_nll,  GREY_COMMON),
    ("Ill-spec.\nEM",  ill_em_nll,  GREY_INDIV),
    ("Cog model\nCP",  cog_cp_nll,  GREEN_COMMON),
    ("Cog model\nEM",  cog_em_nll,  GREEN_INDIV),
    ("CP RNN",   rnn_cp_nll,  BLUE_COMMON),
    ("IDRNN",          idrnn_nll,   BLUE_INDIV),
    ("Vanilla\nRNN",   van_nll,     ORANGE),
]

x_pos_nll, x_labels_nll, x_data_nll, x_colors_nll = [], [], [], []
cur_x = 0.0
for entry in _nll_groups:
    lbl, data, col = entry
    if data is None or not np.isfinite(data).any():
        print(f"  Skipping '{lbl}': no valid data")
        cur_x += 1.0
        continue
    x_pos_nll.append(cur_x)
    x_labels_nll.append(lbl)
    x_data_nll.append(data)
    x_colors_nll.append(col)
    cur_x += 1.0

_means_nll = [np.nanmean(d) for d in x_data_nll]
_sems_nll  = [float(scipy_sem(d[np.isfinite(d)])) for d in x_data_nll]

for lbl, m, s in zip(x_labels_nll, _means_nll, _sems_nll):
    print(f"  {lbl.replace(chr(10),' '):25s}: {m:.3f} ± {s:.3f}")

def _paired_stat(d1, d2):
    mask = np.isfinite(d1) & np.isfinite(d2)
    if mask.sum() < 4:
        return float("nan"), float("nan"), float("nan")
    t, p = ttest_rel(d1[mask], d2[mask])
    bf = float(pg.ttest(d1[mask], d2[mask], paired=True)["BF10"].iloc[0])
    return float(t), float(p), bf

def _idx_of(substr):
    for k, lbl in enumerate(x_labels_nll):
        if substr in lbl.replace("\n", " "):
            return k
    return None

print("\n  Key paired comparisons:")
for a, b in [("IDRNN", "Vanilla RNN"), ("IDRNN", "CP RNN"), ("IDRNN", "Cog model EM"),
             ("Cog model CP", "Cog model EM"),
             ("Ill-spec. CP", "Ill-spec. EM")]:
    ia, ib = _idx_of(a), _idx_of(b)
    if ia is None or ib is None:
        continue
    t, p, bf = _paired_stat(x_data_nll[ia], x_data_nll[ib])
    print(f"    {a} vs {b}: t={t:.3f}  p={p:.4f}  BF10={bf:.3f}")

fig_nll, ax_nll = plt.subplots(figsize=(max(9, 1.4 * len(x_pos_nll)), 6))

ax_nll.bar(x_pos_nll, _means_nll, width=BAR_W_NLL,
           color=x_colors_nll, alpha=0.9, zorder=2,
           error_kw={"elinewidth": 1, "capthick": 1})
ax_nll.errorbar(x_pos_nll, _means_nll, yerr=_sems_nll, fmt="none",
                capsize=3, ecolor="k", elinewidth=1, zorder=5)

# Participant dots (jittered)
np.random.seed(42)
for xi, data in zip(x_pos_nll, x_data_nll):
    fin = data[np.isfinite(data)]
    jit = np.random.normal(0, 0.04, size=len(fin))
    ax_nll.scatter(xi + jit, fin, alpha=0.4, c="k", s=15, zorder=10, edgecolors="none")

# Random-model reference line
ax_nll.axhline(RANDOM_NLL, linestyle="--", color="black", linewidth=1.5,
               label=f"Random model ({RANDOM_NLL:.1f})", zorder=3)

# Significance bars
_ymax_nll = max(m + s for m, s in zip(_means_nll, _sems_nll))
_h_nll    = (max(_means_nll) - min(_means_nll)) * 0.025

def _nll_sig_bar(label_a, label_b, level=0):
    ia, ib = _idx_of(label_a), _idx_of(label_b)
    if ia is None or ib is None:
        return
    _, p, _ = _paired_stat(x_data_nll[ia], x_data_nll[ib])
    y = _ymax_nll + _h_nll * (2 + 4 * level)
    _sig_bar(ax_nll, x_pos_nll[ia], x_pos_nll[ib], y, _h_nll, p)

_nll_sig_bar("Cog model CP", "Cog model EM", level=0)
_nll_sig_bar("Ill-spec. CP", "Ill-spec. EM", level=0)
_nll_sig_bar("IDRNN", "Vanilla RNN",         level=0)
_nll_sig_bar("IDRNN", "CP RNN",          level=1)
_nll_sig_bar("Cog model EM", "IDRNN",         level=2)

ax_nll.set_xticks(x_pos_nll)
ax_nll.set_xticklabels(x_labels_nll, fontsize=9)
ax_nll.set_ylabel("Mean negative log-likelihood per trial", fontsize=12)

# Y-axis: auto-scale around data
_y_bottom = min(_means_nll) - (max(_means_nll) - min(_means_nll)) * 0.5
_y_top    = max(max(_means_nll), RANDOM_NLL) * 1.5
ax_nll.set_ylim(bottom=max(0, _y_bottom), top=_y_top)

# Legend 1 — model type (horizontal, upper left)
_legend1_handles = [
    Patch(facecolor=GREY_INDIV,  label="Ill-specified model"),
    Patch(facecolor=GREEN_INDIV, label="Cog. model"),
    Patch(facecolor=BLUE_INDIV,  label="ID RNN"),
    Patch(facecolor=ORANGE,      label="Vanilla RNN"),
    Line2D([0], [0], color="black", lw=1.5, linestyle="--", label="Random model"),
]
legend1 = ax_nll.legend(handles=_legend1_handles, title="Model type",
                        loc="upper left", ncol=len(_legend1_handles),
                        frameon=False, fontsize=9)
ax_nll.add_artist(legend1)

# Legend 2 — fit type, placed just below legend1 (upper left, second row)
_legend2_handles = [
    Patch(facecolor="#c0c0c0", label="Common process (CP)"),
    Patch(facecolor="#606060", label="Individual differences (ID/EM)"),
]
legend2 = ax_nll.legend(handles=_legend2_handles, title="Fit type",
                        loc="upper left", ncol=len(_legend2_handles),
                        frameon=False, fontsize=9,
                        bbox_to_anchor=(0, 0.88))
ax_nll.add_artist(legend2)

ax_nll.spines["top"].set_visible(False)
ax_nll.spines["right"].set_visible(False)
fig_nll.tight_layout()
nll_path = os.path.join(PLOT_DIR, f"outer_cv_nll{_seed_tag}.png")
fig_nll.savefig(nll_path, dpi=300, bbox_inches="tight")
print(f"NLL plot saved → {nll_path}")
plt.close(fig_nll)

# Also print the IDRNN vs Vanilla stats with BF for the summary
paired_nll  = np.isfinite(idrnn_nll) & np.isfinite(van_nll)
t_nll, p_nll, bf10_nll = _paired_stat(idrnn_nll, van_nll)

# ── Plot 2: Age-group decoding ────────────────────────────────────────────────
# Statistics: Wilcoxon signed-rank on per-participant mean correctness
paired_dec = np.isfinite(idrnn_correct) & np.isfinite(van_correct)
print(f"\n── Decoding comparison (N={paired_dec.sum()} participants) ──")

wilcoxon_stat, p_dec = float("nan"), float("nan")
if paired_dec.sum() >= 6:
    wilcoxon_stat, p_dec = wilcoxon(
        idrnn_correct[paired_dec], van_correct[paired_dec],
        alternative="greater"
    )
    t_dec, p_dec_t = ttest_rel(idrnn_correct[paired_dec], van_correct[paired_dec],
                                alternative="greater")
    result = pg.ttest(idrnn_correct[paired_dec], van_correct[paired_dec],
                  paired=True, alternative="greater")
    print(result[["T", "p-val", "BF10"]])
    print(f"  IDRNN   mean correctness = {idrnn_correct[paired_dec].mean():.3f} "
          f"± {scipy_sem(idrnn_correct[paired_dec]):.3f}")
    print(f"  Vanilla mean correctness = {van_correct[paired_dec].mean():.3f}  "
          f"± {scipy_sem(van_correct[paired_dec]):.3f}")
    print(f"  Wilcoxon (one-sided IDRNN>Vanilla): W={wilcoxon_stat:.1f}  p={p_dec:.4f}")
    print(f"  Paired t-test (one-sided):           t={t_dec:.3f}  p={p_dec_t:.4f}")

# Accuracy metric: overall for dezfouli (matches paper's reported 52%),
# balanced for sloutsky (unequal group sizes).
# Paper reference: Dezfouli et al. overall classification rate = 52%, chance = 33%.
PAPER_ACC_REFERENCE = 0.52 if DGP == "dezfouli" else None

def balanced_acc_from_correct(correct):
    """Balanced accuracy: per-class mean correctness, then macro-average."""
    per_class = [correct[y_age == c].mean() for c in range(len(GROUP_ORDER))
                 if (y_age == c).any()]
    return float(np.mean(per_class)) if per_class else float("nan")

def overall_acc_from_correct(correct):
    """Overall accuracy: mean correctness across all participants."""
    valid = correct[np.isfinite(correct)]
    return float(valid.mean()) if len(valid) > 0 else float("nan")

_acc_fn    = overall_acc_from_correct if DGP == "dezfouli" else balanced_acc_from_correct
_acc_label = "Overall accuracy" if DGP == "dezfouli" else "Balanced accuracy"

ba_idrnn    = _acc_fn(idrnn_correct[paired_dec]    if paired_dec.any() else idrnn_correct)
ba_van      = _acc_fn(van_correct[paired_dec]      if paired_dec.any() else van_correct)

# Logistic regression stats
paired_lr = np.isfinite(idrnn_correct_lr) & np.isfinite(van_correct_lr)
print(f"\n── Logistic Regression decoding comparison (N={paired_lr.sum()} participants) ──")
p_dec_lr = float("nan")
if paired_lr.sum() >= 6:
    _, p_dec_lr = wilcoxon(idrnn_correct_lr[paired_lr], van_correct_lr[paired_lr],
                           alternative="greater")
    t_lr, p_lr_t = ttest_rel(idrnn_correct_lr[paired_lr], van_correct_lr[paired_lr],
                              alternative="greater")
    result_lr = pg.ttest(idrnn_correct_lr[paired_lr], van_correct_lr[paired_lr],
                         paired=True, alternative="greater")
    print(result_lr[["T", "p-val", "BF10"]])
    print(f"  IDRNN   mean correctness (LR) = {idrnn_correct_lr[paired_lr].mean():.3f} "
          f"± {scipy_sem(idrnn_correct_lr[paired_lr]):.3f}")
    print(f"  Vanilla mean correctness (LR) = {van_correct_lr[paired_lr].mean():.3f}  "
          f"± {scipy_sem(van_correct_lr[paired_lr]):.3f}")
    print(f"  Wilcoxon one-sided: p={p_dec_lr:.4f}  t={t_lr:.3f}  p={p_lr_t:.4f}")

ba_idrnn_lr = _acc_fn(idrnn_correct_lr[paired_lr] if paired_lr.any() else idrnn_correct_lr)
ba_van_lr   = _acc_fn(van_correct_lr[paired_lr]   if paired_lr.any() else van_correct_lr)

# Plot: 2-panel, RBF-SVM | Logistic Regression
chance = 1.0 / len(GROUP_ORDER)
fig, axes = plt.subplots(1, 2, figsize=(8, 5), sharey=True)
for ax, (ba_i, ba_v), (ci, cv), p_w, title in [
    (axes[0], (ba_idrnn, ba_van),       (idrnn_correct[paired_dec], van_correct[paired_dec]),
     p_dec,    "RBF-SVM"),
    (axes[1], (ba_idrnn_lr, ba_van_lr), (idrnn_correct_lr[paired_lr], van_correct_lr[paired_lr]),
     p_dec_lr, "Logistic Regression"),
]:
    ax.bar([0, 1], [ba_i, ba_v], width=0.5,
           color=[BLUE_INDIV, ORANGE], alpha=0.85, zorder=2)
    ax.axhline(chance, color="k", ls=":", lw=1, alpha=0.6, label=f"Chance ({chance:.2f})")
    if PAPER_ACC_REFERENCE is not None:
        ax.axhline(PAPER_ACC_REFERENCE, color="firebrick", ls="--", lw=1.2, alpha=0.7,
                   label=f"Paper ({PAPER_ACC_REFERENCE:.0%})")
    rng2 = np.random.default_rng(1)
    for xi, data in zip([0, 1], [ci, cv]):
        jit = rng2.uniform(-0.12, 0.12, size=data.shape[0])
        ax.scatter(xi + jit, data, alpha=0.35, c="k", s=14, zorder=10, edgecolors="none")
    if np.isfinite(p_w):
        ymax = max(ba_i, ba_v) + 0.08
        _sig_bar(ax, 0, 1, ymax, 0.02, p_w)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["IDRNN", "Vanilla RNN"], fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].set_ylabel(f"{_acc_label} (per participant, avg over seeds)", fontsize=10)
fig.suptitle(f"{GROUP_LABEL_STR} decoding from latents\nOuter {N_FOLDS}-fold CV · N={paired_dec.sum()} participants",
             fontsize=11, fontweight="bold")
fig.tight_layout()
dec_path = os.path.join(PLOT_DIR, f"outer_cv_{DECODING_TAG}{_seed_tag}.png")
fig.savefig(dec_path, dpi=150)
print(f"{GROUP_LABEL_STR} decoding plot saved → {dec_path}")
plt.close(fig)


# ── Plot 2b: multi-strategy decoding comparison (block-structured DGPs only) ──
if DGP == "dezfouli":
    strategies  = BLOCK_STRATEGIES
    x           = np.arange(len(strategies))
    width       = 0.35

    fig_ms, axes_ms = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, idrnn_res, van_res, title in [
        (axes_ms[0], idrnn_results_svm, van_results_svm, "RBF-SVM"),
        (axes_ms[1], idrnn_results_lr,  van_results_lr,  "Logistic Regression"),
    ]:
        ba_idrnn_all = [_acc_fn(idrnn_res[s]) for s in strategies]
        ba_van_all   = [_acc_fn(van_res[s])   for s in strategies]

        ax.bar(x - width / 2, ba_idrnn_all, width,
               color=BLUE_INDIV, alpha=0.85, label="IDRNN", zorder=2)
        ax.bar(x + width / 2, ba_van_all, width,
               color=ORANGE, alpha=0.85, label="Vanilla RNN", zorder=2)
        ax.axhline(chance, color="k", ls=":", lw=1, alpha=0.6,
                   label=f"Chance ({chance:.2f})")
        if PAPER_ACC_REFERENCE is not None:
            ax.axhline(PAPER_ACC_REFERENCE, color="firebrick", ls="--", lw=1.2, alpha=0.7,
                       label=f"Paper ({PAPER_ACC_REFERENCE:.0%})")

        rng_ms = np.random.default_rng(2)
        for xi, s in enumerate(strategies):
            ci = idrnn_res[s]
            cv = van_res[s]
            paired_s = np.isfinite(ci) & np.isfinite(cv)
            if not paired_s.any():
                continue
            ax.scatter(xi - width / 2 + rng_ms.uniform(-0.09, 0.09, ci[paired_s].shape),
                       ci[paired_s], alpha=0.3, c="k", s=10, zorder=10, edgecolors="none")
            ax.scatter(xi + width / 2 + rng_ms.uniform(-0.09, 0.09, cv[paired_s].shape),
                       cv[paired_s], alpha=0.3, c="k", s=10, zorder=10, edgecolors="none")
            if paired_s.sum() >= 6:
                _, p_s = wilcoxon(ci[paired_s], cv[paired_s], alternative="greater")
                ymax = max(ba_idrnn_all[xi], ba_van_all[xi]) + 0.07
                _sig_bar(ax, xi - width / 2, xi + width / 2, ymax, 0.015, p_s)

        ax.set_xticks(x)
        ax.set_xticklabels([STRATEGY_LABELS.get(s, s) for s in strategies], fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes_ms[0].set_ylabel(f"{_acc_label} (per participant, avg over seeds)", fontsize=10)
    fig_ms.suptitle(
        f"{GROUP_LABEL_STR} decoding — latent aggregation strategies\n"
        f"Outer {N_FOLDS}-fold CV · N={N} participants",
        fontsize=11, fontweight="bold",
    )
    fig_ms.tight_layout()
    ms_path = os.path.join(PLOT_DIR, f"outer_cv_{DECODING_TAG}_strategies{_seed_tag}.png")
    fig_ms.savefig(ms_path, dpi=150)
    print(f"Multi-strategy decoding plot saved → {ms_path}")
    plt.close(fig_ms)


# ── Plot 3: RSA ───────────────────────────────────────────────────────────────
def rsa_age(lats, label=""):
    if lats is None:
        return float("nan")
    valid = np.isfinite(lats).all(axis=1)
    if valid.sum() < 4:
        return float("nan")
    rdm_lat = pdist(lats[valid],                    metric="euclidean")
    rdm_age = pdist(y_age[valid].reshape(-1, 1),    metric="euclidean")
    r, _ = spearmanr(rdm_lat, rdm_age)
    print(f"RSA [{label}] {RSA_LABEL_STR} Spearman r = {r:.3f}")
    return float(r)


r_rsa_idrnn = rsa_age(idrnn_lats, "IDRNN")
r_rsa_van   = rsa_age(van_lats,   "Vanilla")

fig, ax = plt.subplots(figsize=(4, 4))
ax.bar([0, 1], [r_rsa_idrnn, r_rsa_van], width=0.5,
       color=[BLUE_INDIV, ORANGE], alpha=0.85)
ax.axhline(0, color="k", lw=0.8, ls=":")
ax.set_xticks([0, 1])
ax.set_xticklabels(["IDRNN", "Vanilla RNN"], fontsize=11)
ax.set_ylabel("Spearman r (RSA)", fontsize=11)
ax.set_title(f"RSA: latent RDM vs {RSA_LABEL_STR} RDM\nOuter CV, all {N} participants", fontsize=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
rsa_path = os.path.join(PLOT_DIR, f"outer_cv_rsa{_seed_tag}.png")
fig.savefig(rsa_path, dpi=150)
print(f"RSA plot saved → {rsa_path}")
plt.close(fig)


# ── Plot 4: Cog-param decoding — per-participant z-score product ──────────────
#
# Per-participant score: z_pred_i × z_true_i  (z-scored across all N participants)
# Mean of these scores = Pearson r.  Each participant gets a signed scalar:
#   positive → predicted in the correct direction relative to the group
#   negative → predicted in the wrong direction
# Statistics: paired Wilcoxon (one-sided IDRNN > Vanilla) per cog param.
if DGP == "sloutsky":
    print("\n── Cog-param decoding (per-participant z-score product, N=94) ──")

def _participant_decoding_scores(pred_dict, true_dict):
    """
    Returns dict: param_name -> (N,) per-participant score = z_pred × z_true.
    Mean across valid participants equals Pearson r.
    """
    scores = {}
    for pn in COG_PARAM_NAMES:
        pred = pred_dict[pn].copy()
        true = true_dict[pn].copy()
        valid = np.isfinite(pred) & np.isfinite(true)
        s = np.full(N, np.nan)
        if valid.sum() >= 4:
            p_v, t_v = pred[valid], true[valid]
            z_p = (p_v - p_v.mean()) / (p_v.std() + 1e-12)
            z_t = (t_v - t_v.mean()) / (t_v.std() + 1e-12)
            s[valid] = z_p * z_t
        scores[pn] = s
    return scores

cog_dec_stats = {}
idrnn_scores = {}
van_scores   = {}
if DGP == "sloutsky":
  idrnn_scores = _participant_decoding_scores(idrnn_cog_pred, true_cog)
  van_scores   = _participant_decoding_scores(van_cog_pred,   true_cog)

  print(f"  {'Param':12s}  {'IDRNN mean':>12s}  {'Vanilla mean':>12s}  "
        f"{'W':>8s}  {'p (W)':>8s}  {'t':>7s}  {'p (t)':>7s}  {'BF10':>7s}")
if DGP == "sloutsky":
    for pn in COG_PARAM_NAMES:
        si = idrnn_scores[pn]
        sv = van_scores[pn]
        paired = np.isfinite(si) & np.isfinite(sv)
        if paired.sum() >= 6:
            W, p_w = wilcoxon(si[paired], sv[paired], alternative="greater")
            t_v, p_t = ttest_rel(si[paired], sv[paired], alternative="greater")
            bf = float(pg.ttest(si[paired], sv[paired],
                                paired=True, alternative="greater")["BF10"].iloc[0])
        else:
            W, p_w, t_v, p_t, bf = (float("nan"),) * 5
        cog_dec_stats[pn] = dict(W=W, p_w=p_w, t=t_v, p_t=p_t, bf10=bf,
                                  idrnn_mean=float(np.nanmean(si)),
                                  van_mean=float(np.nanmean(sv)))
        print(f"  {COG_PARAM_LABELS[pn]:12s}  {np.nanmean(si):+12.3f}  {np.nanmean(sv):+12.3f}  "
              f"{W:8.1f}  {p_w:8.4f}  {t_v:+7.3f}  {p_t:7.4f}  {bf:7.3f}")

    fig_dec, axes_dec = plt.subplots(1, len(COG_PARAM_NAMES),
                                      figsize=(3.2 * len(COG_PARAM_NAMES), 5),
                                      sharey=True)
    rng_cog = np.random.default_rng(7)

    for i, (pn, ax_d) in enumerate(zip(COG_PARAM_NAMES, axes_dec)):
        si = idrnn_scores[pn]
        sv = van_scores[pn]
        paired = np.isfinite(si) & np.isfinite(sv)

        means = [np.nanmean(si[paired]), np.nanmean(sv[paired])]
        sems  = [float(scipy_sem(si[paired])), float(scipy_sem(sv[paired]))]
        ax_d.bar([0, 1], means, width=0.5, color=[BLUE_INDIV, ORANGE],
                 alpha=0.85, zorder=2, label=["IDRNN", "Vanilla"] if i == 0 else ["_", "_"])
        ax_d.errorbar([0, 1], means, yerr=sems, fmt="none",
                      capsize=4, ecolor="k", elinewidth=1.2, zorder=5)

        for xi, scores_sub in [(0, si[paired]), (1, sv[paired])]:
            jit = rng_cog.uniform(-0.12, 0.12, size=scores_sub.shape[0])
            ax_d.scatter(xi + jit, scores_sub, alpha=0.3, c="k", s=12,
                         zorder=10, edgecolors="none")

        ax_d.axhline(0, color="k", lw=0.8, ls=":")

        p_w = cog_dec_stats[pn]["p_w"]
        if np.isfinite(p_w):
            ymax_d = max(m + s for m, s in zip(means, sems))
            h_d    = max(abs(np.nanmax(si[paired])), abs(np.nanmax(sv[paired]))) * 0.05
            _sig_bar(ax_d, 0, 1, ymax_d + h_d, h_d, p_w)

        ax_d.set_xticks([0, 1])
        ax_d.set_xticklabels(["IDRNN", "Van."], fontsize=8)
        ax_d.set_title(COG_PARAM_LABELS[pn], fontsize=10)
        ax_d.spines["top"].set_visible(False)
        ax_d.spines["right"].set_visible(False)

    axes_dec[0].set_ylabel("z_pred × z_true  (per participant)", fontsize=10)
    fig_dec.suptitle(
        f"Decoding cognitive parameters from latents — Outer {N_FOLDS}-fold CV\n"
        f"N={N} participants · Ridge train→test · dots = participants · "
        f"sig bar = Wilcoxon one-sided (IDRNN > Vanilla)",
        fontsize=10, fontweight="bold"
    )
    fig_dec.tight_layout()
    dec_cog_path = os.path.join(PLOT_DIR, f"outer_cv_cog_decoding{_seed_tag}.png")
    fig_dec.savefig(dec_cog_path, dpi=150)
    print(f"Cog-param decoding plot saved → {dec_cog_path}")
    plt.close(fig_dec)

    # ── Plot 4b: Cog-param decoding — PC1 only, same layout ──────────────────────
    print("\n── Cog-param decoding PC1 (per-participant z-score product, N=94) ──")

    idrnn_scores_pc1 = _participant_decoding_scores(idrnn_cog_pred_pc1, true_cog)
    van_scores_pc1   = _participant_decoding_scores(van_cog_pred_pc1,   true_cog)

    cog_dec_stats_pc1 = {}
    print(f"  {'Param':12s}  {'IDRNN mean':>12s}  {'Vanilla mean':>12s}  "
          f"{'W':>8s}  {'p (W)':>8s}  {'BF10':>7s}")
    for pn in COG_PARAM_NAMES:
        si = idrnn_scores_pc1[pn]
        sv = van_scores_pc1[pn]
        paired = np.isfinite(si) & np.isfinite(sv)
        if paired.sum() >= 6:
            W, p_w = wilcoxon(si[paired], sv[paired], alternative="greater")
            t_v, p_t = ttest_rel(si[paired], sv[paired], alternative="greater")
            bf = float(pg.ttest(si[paired], sv[paired],
                                paired=True, alternative="greater")["BF10"].iloc[0])
        else:
            W, p_w, t_v, p_t, bf = (float("nan"),) * 5
        cog_dec_stats_pc1[pn] = dict(W=W, p_w=p_w, t=t_v, p_t=p_t, bf10=bf,
                                      idrnn_mean=float(np.nanmean(si)),
                                      van_mean=float(np.nanmean(sv)))
        print(f"  {COG_PARAM_LABELS[pn]:12s}  {np.nanmean(si):+12.3f}  {np.nanmean(sv):+12.3f}  "
              f"{W:8.1f}  {p_w:8.4f}  {bf:7.3f}")

    # ── Combined comparison plot: full latents vs PC1, 2 rows ──────────────────
    fig_cmp, axes_cmp = plt.subplots(
        2, len(COG_PARAM_NAMES),
        figsize=(3.2 * len(COG_PARAM_NAMES), 9),
        sharey="row"
    )
    rng_cmp = np.random.default_rng(8)

    for row, (scores_i, scores_v, stats, row_label) in enumerate([
        (idrnn_scores,     van_scores,     cog_dec_stats,     "Full latents"),
        (idrnn_scores_pc1, van_scores_pc1, cog_dec_stats_pc1, "PC1 only"),
    ]):
        for col, pn in enumerate(COG_PARAM_NAMES):
            ax = axes_cmp[row, col]
            si = scores_i[pn]
            sv = scores_v[pn]
            paired = np.isfinite(si) & np.isfinite(sv)

            means = [np.nanmean(si[paired]), np.nanmean(sv[paired])]
            sems  = [float(scipy_sem(si[paired])), float(scipy_sem(sv[paired]))]
            ax.bar([0, 1], means, width=0.5, color=[BLUE_INDIV, ORANGE],
                   alpha=0.85, zorder=2)
            ax.errorbar([0, 1], means, yerr=sems, fmt="none",
                        capsize=4, ecolor="k", elinewidth=1.2, zorder=5)
            for xi, sc in [(0, si[paired]), (1, sv[paired])]:
                jit = rng_cmp.uniform(-0.12, 0.12, size=sc.shape[0])
                ax.scatter(xi + jit, sc, alpha=0.3, c="k", s=12,
                           zorder=10, edgecolors="none")
            ax.axhline(0, color="k", lw=0.8, ls=":")

            p_w = stats[pn]["p_w"]
            if np.isfinite(p_w):
                ymax_c = max(m + s for m, s in zip(means, sems))
                h_c    = max(abs(np.nanmax(si[paired])),
                             abs(np.nanmax(sv[paired]))) * 0.05
                _sig_bar(ax, 0, 1, ymax_c + h_c, h_c, p_w)

            ax.set_xticks([0, 1])
            ax.set_xticklabels(["IDRNN", "Van."], fontsize=8)
            if col == 0:
                ax.set_ylabel(f"{row_label}\nz_pred × z_true", fontsize=9)
            if row == 0:
                ax.set_title(COG_PARAM_LABELS[pn], fontsize=10)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    fig_cmp.suptitle(
        f"Cog-param decoding: full latents vs PC1\n"
        f"Outer {N_FOLDS}-fold CV · N={N} participants · Wilcoxon one-sided sig bars",
        fontsize=10, fontweight="bold"
    )
    fig_cmp.tight_layout()
    cmp_path = os.path.join(PLOT_DIR, f"outer_cv_cog_decoding_vs_pc1{_seed_tag}.png")
    fig_cmp.savefig(cmp_path, dpi=150)
    print(f"Full vs PC1 comparison plot saved → {cmp_path}")
    plt.close(fig_cmp)

    # ── Plot 5: w_novelty scatter (predicted vs true, all 94 participants) ──────
    pn_star = "b_novelty_train"
    pred_nov_i = idrnn_cog_pred[pn_star]
    pred_nov_v = van_cog_pred[pn_star]
    true_nov   = true_cog[pn_star]

    fig_nov, ax_nov = plt.subplots(figsize=(5, 4.5))

    valid = np.isfinite(pred_nov_i) & np.isfinite(true_nov)
    sc = ax_nov.scatter(true_nov[valid], pred_nov_i[valid],
                        c=true_nov[valid], cmap="viridis",
                        s=40, alpha=0.8, edgecolors="k", linewidths=0.4, zorder=3)
    plt.colorbar(sc, ax=ax_nov, label="novelty param (true)")
    if valid.sum() >= 3:
        coef = np.polyfit(true_nov[valid], pred_nov_i[valid], 1)
        xr   = np.linspace(true_nov[valid].min(), true_nov[valid].max(), 100)
        ax_nov.plot(xr, np.poly1d(coef)(xr), "--", color="red", lw=1.5, zorder=2)
        r, _ = pearsonr(true_nov[valid], pred_nov_i[valid])
        ax_nov.set_title(f"IDRNN\nr = {r:.3f}", fontsize=10)
    ax_nov.set_xlabel("true w novelty", fontsize=9)
    ax_nov.set_ylabel("predicted w novelty", fontsize=9)
    ax_nov.spines["top"].set_visible(False)
    ax_nov.spines["right"].set_visible(False)

    fig_nov.tight_layout()
    nov_path = os.path.join(PLOT_DIR, f"outer_cv_novelty_scatter{_seed_tag}.png")
    fig_nov.savefig(nov_path, dpi=150)
    print(f"Novelty scatter saved → {nov_path}")
    plt.close(fig_nov)

    cog_dec_stats_pc1 = cog_dec_stats_pc1  # keep in scope for summary


# ── Save summary JSON ─────────────────────────────────────────────────────────
summary = {
    "n_participants": N,
    "n_folds":        N_FOLDS,
    "nll": {
        "idrnn_mean":   float(np.nanmean(idrnn_nll)),
        "van_mean":     float(np.nanmean(van_nll)),
        "paired_t":     float(t_nll),
        "paired_p":     float(p_nll),
        "paired_bf10":  float(bf10_nll),
        "n_paired":     int(paired_nll.sum()),
        "models": {lbl.replace("\n"," "): float(np.nanmean(d))
                   for lbl, d in zip(x_labels_nll, x_data_nll)},
    },
    f"{DECODING_TAG}": {
        "idrnn_mean_correct": float(np.nanmean(idrnn_correct)),
        "van_mean_correct":   float(np.nanmean(van_correct)),
        "idrnn_bal_acc":      ba_idrnn,
        "van_bal_acc":        ba_van,
        "wilcoxon_stat":      float(wilcoxon_stat) if np.isfinite(wilcoxon_stat) else None,
        "wilcoxon_p":         float(p_dec)         if np.isfinite(p_dec)         else None,
        "n_paired":           int(paired_dec.sum()),
    },
    f"rsa_{RSA_LABEL_STR}": {
        "idrnn":   r_rsa_idrnn,
        "vanilla": r_rsa_van,
    },
    **({"cog_param_decoding_full": {pn: cog_dec_stats[pn] for pn in COG_PARAM_NAMES},
        "cog_param_decoding_pc1":  {pn: cog_dec_stats_pc1[pn] for pn in COG_PARAM_NAMES},
       } if DGP == "sloutsky" else {}),
}
json_path = os.path.join(PLOT_DIR, f"outer_cv_summary{_seed_tag}.json")
with open(json_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSummary saved → {json_path}")


# ── Per-seed specificity vs decoding correlation (dezfouli / IDRNN only) ──────
#
# Goal: after NLL-based CV model selection (cv_selected_epoch), compute for each seed:
#   (a) reconstruction specificity on the test participants of each fold
#   (b) decoding accuracy from that seed's latents (single seed, not averaged)
# Then correlate specificity with decoding accuracy across seeds via Spearman r.
# This validates whether specificity loss is a useful proxy for representation quality.
#
# Specificity is computed using the decoder with matched vs permuted z:
#   - z_matched = last valid latent per participant (from saved latents_tensor)
#   - matched_loss    = NLL of test choices when decoder uses own z
#   - mismatched_loss = NLL of test choices when decoder uses another participant's z
#   - specificity = mean(mismatched_loss − matched_loss)
if DGP == "dezfouli":
    import copy
    import torch.nn.functional as _F
    from compute_reconstruction_specificity import (
        load_model_config as _load_model_config,
        create_model_from_config as _create_model_from_config,
        load_model_checkpoint as _load_model_checkpoint,
    )

    def _specificity_from_saved_latents(mdl, z_matched_np, xin_4d, c_4d, n_perm=30):
        """
        Compute reconstruction specificity using pre-computed z from saved latents.

        z_matched_np : (B, z_dim) numpy — last valid latent per participant
        xin_4d       : (B, Bk, T, in_dim) tensor
        c_4d         : (B, Bk, T) tensor  (padding = -100)
        """
        B   = z_matched_np.shape[0]
        bk  = xin_4d.shape[1]
        z_m = torch.from_numpy(z_matched_np).float()

        def _nll_for_z(z):
            """NLL across all valid trials for each participant using given z (B, z_dim)."""
            per_part = torch.zeros(B)
            h = None
            for b in range(bk):
                seq_b = xin_4d[:, b]   # (B, T, in_dim)
                c_b   = c_4d[:, b].long()  # (B, T)
                logits_b, h = mdl.decoder(seq_b, z, hidden=h)
                probs_b = _F.softmax(logits_b, dim=-1)
                valid_b = (c_b >= 0)
                c_safe  = c_b.clone()
                c_safe[~valid_b] = 0
                chosen  = probs_b.gather(-1, c_safe.unsqueeze(-1)).squeeze(-1)
                chosen[~valid_b] = 1.0   # log(1)=0 → no contribution
                per_part -= chosen.clamp(min=1e-8).log().sum(dim=1)
            return per_part  # (B,)

        with torch.no_grad():
            matched = _nll_for_z(z_m)
            perm_losses = []
            for _ in range(n_perm):
                perm = torch.randperm(B)
                same = perm == torch.arange(B)
                while same.any():
                    perm[same] = (perm[same] + 1) % B
                    same = perm == torch.arange(B)
                perm_losses.append(_nll_for_z(z_m[perm]))
            mismatched = torch.stack(perm_losses).mean(dim=0)

        return (mismatched - matched).mean().item()

    print("\n── Per-seed: specificity + decoding (IDRNN · dezfouli) ──")
    _bk, _t_blk = _load_block_dims()

    _spec_by_seed  = {}   # seed -> [specificity_fold0, specificity_fold1, ...]
    _dec_by_seed_s = {}   # seed -> [acc_fold0, acc_fold1, ...]   (SVM)
    _dec_by_seed_l = {}   # seed -> [acc_fold0, acc_fold1, ...]   (LR)

    for _fold in range(N_FOLDS):
        _fold_data_dir = os.path.join(DATA_DIR, f"fold{_fold}")
        _test_subs     = fold_test_subids[_fold]
        _y_test        = fold_y_age[_fold]
        _y_train       = fold_y_age_train[_fold]
        _seeds_fold    = discover_seeds(_fold, IDRNN_BASE)

        # Load fold data once (needed for specificity computation)
        _xin_path = os.path.join(_fold_data_dir, "xin_test.npy")
        _c_path   = os.path.join(_fold_data_dir, "c_test.npy")
        if not (os.path.exists(_xin_path) and os.path.exists(_c_path)):
            print(f"  fold {_fold}: xin_test / c_test missing, skipping")
            continue
        _xin_fold = torch.from_numpy(np.load(_xin_path)).float()
        _c_fold   = torch.from_numpy(np.load(_c_path)).float()

        for _seed in _seeds_fold:
            _lat_te_path  = os.path.join(_fold_data_dir, f"seed_{_seed}", SEED_SUB,
                                         "latents_tensorlatentmodel.pt")
            _lat_tr_path  = os.path.join(_fold_data_dir, f"seed_{_seed}", SEED_SUB,
                                         "latents_tensorlatentmodel_traindata.pt")
            _run_dir      = os.path.join(IDRNN_BASE, f"fold{_fold}", f"seed_{_seed}")
            _cfg_path     = os.path.join(_run_dir, "config.json")

            if not all(os.path.exists(p) for p in [_lat_te_path, _lat_tr_path, _cfg_path]):
                continue

            _lat_te_raw = torch.load(_lat_te_path,  map_location="cpu").numpy()
            _lat_tr_raw = torch.load(_lat_tr_path,  map_location="cpu").numpy()

            if _lat_te_raw.shape[0] != len(_test_subs):
                print(f"  fold {_fold} seed {_seed}: shape mismatch, skipping")
                continue

            # ── Decoding accuracy for this single seed ──
            _lat_tr_red, _lat_te_red = _reduce_for_strategy(
                _lat_tr_raw, _lat_te_raw, _primary, _bk, _t_blk
            )
            for _clf_fn, _dec_dict in [(rbfsvm_traintest,  _dec_by_seed_s),
                                        (logreg_traintest,  _dec_by_seed_l)]:
                _preds = _clf_fn(_lat_tr_red, _y_train, _lat_te_red)
                _acc   = overall_acc_from_correct(
                    np.array([int(_preds[i] == _y_test[i]) for i in range(len(_test_subs))],
                             dtype=float)
                )
                _dec_dict.setdefault(_seed, []).append(_acc)

            # ── Reconstruction specificity via model + saved z ──
            with open(_cfg_path) as _fh:
                _cfg_s = json.load(_fh)
            _cv_epoch = _cfg_s.get("cv_selected_epoch")
            if _cv_epoch is None:
                continue
            _ckpt_path   = os.path.join(_run_dir, "checkpoints", f"epoch{_cv_epoch:04d}.pt")
            _frozen_path = os.path.join(_run_dir, "frozen_decoder", "policy_model.pt")
            if not (os.path.exists(_ckpt_path) and os.path.exists(_frozen_path)):
                continue

            try:
                _mcfg = _load_model_config(_run_dir)
                _mdl  = _create_model_from_config(
                    _mcfg, n_participants=_lat_te_raw.shape[0],
                    device="cpu", frozen_decoder_path=_frozen_path
                )
                _mdl  = _load_model_checkpoint(_ckpt_path, _mdl, "cpu")

                # z = last valid latent per participant (from saved tensor)
                _z_last = _reduce_for_strategy(
                    _lat_te_raw, _lat_te_raw, "last_participant", _bk, _t_blk
                )[0]   # (B, z_dim) numpy

                _spec = _specificity_from_saved_latents(
                    _mdl, _z_last, _xin_fold, _c_fold, n_perm=30
                )
                _spec_by_seed.setdefault(_seed, []).append(_spec)
                print(f"  fold {_fold} seed {_seed}: spec={_spec:.4f}", flush=True)
            except Exception as _ex:
                print(f"  fold {_fold} seed {_seed}: specificity failed — {_ex}", flush=True)

    # Average across folds per seed and compute Spearman correlation
    _seeds_union = sorted(
        set(_spec_by_seed) & set(_dec_by_seed_s) & set(_dec_by_seed_l)
    )
    print(f"\n  Seeds with both specificity and decoding data: {len(_seeds_union)}")

    if len(_seeds_union) >= 3:
        _spec_avg  = np.array([np.mean(_spec_by_seed[s])  for s in _seeds_union])
        _dec_avg_s = np.array([np.mean(_dec_by_seed_s[s]) for s in _seeds_union])
        _dec_avg_l = np.array([np.mean(_dec_by_seed_l[s]) for s in _seeds_union])

        _r_s, _p_s = spearmanr(_spec_avg, _dec_avg_s)
        _r_l, _p_l = spearmanr(_spec_avg, _dec_avg_l)

        print(f"\n  Specificity vs {_acc_label} (N={len(_seeds_union)} seeds):")
        print(f"    SVM:  Spearman r={_r_s:.3f}  p={_p_s:.4f}")
        print(f"    LR:   Spearman r={_r_l:.3f}  p={_p_l:.4f}")
        for _s in _seeds_union:
            print(f"    seed {_s}: spec={np.mean(_spec_by_seed[_s]):.4f}  "
                  f"svm_acc={np.mean(_dec_by_seed_s[_s]):.3f}  "
                  f"lr_acc={np.mean(_dec_by_seed_l[_s]):.3f}")

        # Scatter plot
        _fig_sc, _axes_sc = plt.subplots(1, 2, figsize=(10, 4.5))
        for _ax, _dec_vals, _r_val, _p_val, _title in [
            (_axes_sc[0], _dec_avg_s, _r_s, _p_s, "RBF-SVM"),
            (_axes_sc[1], _dec_avg_l, _r_l, _p_l, "Logistic Regression"),
        ]:
            _ax.scatter(_spec_avg, _dec_vals, c=BLUE_INDIV, s=70, alpha=0.85, zorder=5)
            for _i, _s in enumerate(_seeds_union):
                _ax.annotate(str(_s), (_spec_avg[_i], _dec_vals[_i]),
                             fontsize=7, ha="left", xytext=(3, 0),
                             textcoords="offset points")
            if len(_seeds_union) >= 3:
                _coef = np.polyfit(_spec_avg, _dec_vals, 1)
                _xr   = np.linspace(_spec_avg.min(), _spec_avg.max(), 100)
                _ax.plot(_xr, np.poly1d(_coef)(_xr), "--", color="grey", lw=1.5)
            if PAPER_ACC_REFERENCE is not None:
                _ax.axhline(PAPER_ACC_REFERENCE, color="firebrick", ls="--", lw=1.2,
                            alpha=0.7, label=f"Paper ({PAPER_ACC_REFERENCE:.0%})")
                _ax.legend(fontsize=8)
            _ax.set_xlabel("Reconstruction specificity (fold-avg)", fontsize=10)
            _ax.set_ylabel(f"{_acc_label} (fold-avg, single seed)", fontsize=10)
            _ax.set_title(f"{_title}\nSpearman r={_r_val:.3f}  p={_p_val:.3f}", fontsize=10)
            _ax.spines["top"].set_visible(False)
            _ax.spines["right"].set_visible(False)

        _fig_sc.suptitle(
            f"Specificity vs {GROUP_LABEL_STR} decoding — per seed (IDRNN)\n"
            f"N={len(_seeds_union)} seeds · {N_FOLDS}-fold CV · "
            f"latent aggregation: {_primary}",
            fontsize=11, fontweight="bold",
        )
        _fig_sc.tight_layout()
        _sc_path = os.path.join(PLOT_DIR, f"outer_cv_specificity_vs_decoding{_seed_tag}.png")
        _fig_sc.savefig(_sc_path, dpi=150)
        print(f"Specificity-decoding scatter saved → {_sc_path}")
        plt.close(_fig_sc)

        # Append to summary JSON
        summary["specificity_vs_decoding"] = {
            "n_seeds":         len(_seeds_union),
            "seeds":           list(_seeds_union),
            "spearman_r_svm":  float(_r_s),
            "spearman_p_svm":  float(_p_s),
            "spearman_r_lr":   float(_r_l),
            "spearman_p_lr":   float(_p_l),
            "per_seed": {
                str(_s): {
                    "specificity":    float(np.mean(_spec_by_seed[_s])),
                    "decoding_svm":   float(np.mean(_dec_by_seed_s[_s])),
                    "decoding_lr":    float(np.mean(_dec_by_seed_l[_s])),
                } for _s in _seeds_union
            },
        }
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary (with specificity-decoding) updated → {json_path}")
    else:
        print(f"  Not enough seeds ({len(_seeds_union)}) for correlation — skipping plot.")
