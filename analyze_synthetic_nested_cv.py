"""Nested-CV analysis for synthetic datasets.

Consumes the per-fold per-seed testing outputs produced by
testing_script.py --run_suffix nested_cv:
  data_dataset{ID}/fold{F}/seed_{S}/nested_cv/
    rnn_results{latentmodel,vanilla}.csv  — per-session normalized_likelihood
    latents_tensor{latentmodel,vanilla}.pt — (N_test, T, z_or_h_dim)

Produces three headline metrics, one per (dataset, arch), aggregated across
the 3 outer folds:
  1. Held-out NLL : pool per-session NLL across folds; paired Wilcoxon
                    across the N total test sessions
  2. Alpha RSA    : per-fold Spearman r(latent RDM, alpha RDM); mean across folds
  3. Alpha decoding : within-fold LOO ridge from latents → alpha; pool
                      per-subject OOS predictions across folds, one Pearson r

Outputs (per dataset):
  final_plots/synthetic/dataset{ID}/nested_cv_analysis.png
  final_plots/synthetic/dataset{ID}/nested_cv_summary.json

Cross-dataset summary (if multiple datasets requested):
  final_plots/synthetic/nested_cv_multi_dataset.png
  final_plots/synthetic/nested_cv_multi_dataset_summary.json

The script also runs the *canonical* (full-train-set, no-fold) model's
RSA + decoding for reference and includes those in the summary JSON.
"""

from __future__ import annotations
import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import pearsonr, spearmanr, wilcoxon
from scipy.spatial.distance import pdist
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import LeaveOneOut
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score


N_OUTER_FOLDS = 3
TOP_K_BY_SPEC = 5     # how many IDRNN seeds to average over (by step1_specificity)
PCA_N_COMPONENTS = 3  # equalises decoding capacity across IDRNN (5-dim z) and
                      # Vanilla (20-dim hidden) — PCA is fit on the train-subject
                      # latents of each fold and applied to the test-subject latents.


# ── Helpers ────────────────────────────────────────────────────────────────────
def _seed_dirs(base: str) -> List[Tuple[int, str]]:
    out = []
    if not os.path.isdir(base):
        return out
    for d in sorted(os.listdir(base)):
        if d.startswith("seed_") and os.path.isdir(os.path.join(base, d)):
            try:
                out.append((int(d.split("_")[1]), os.path.join(base, d)))
            except ValueError:
                pass
    return out


def _rank_by_specificity(seed_dirs: List[Tuple[int, str]],
                          top_k: int) -> List[Tuple[int, str]]:
    scored = []
    for seed, d in seed_dirs:
        cfg_p = os.path.join(d, "config.json")
        if not os.path.exists(cfg_p):
            continue
        try:
            cfg = json.load(open(cfg_p))
            sp = cfg.get("step1_specificity")
            if sp is not None:
                scored.append((float(sp), seed, d))
        except Exception:
            pass
    if not scored:
        return seed_dirs[:top_k]
    scored.sort(key=lambda t: -t[0])
    return [(s, d) for _, s, d in scored[:top_k]]


RUN_SUFFIX = "nested_cv"          # IDRNN run-suffix; overridden by --run_suffix
VANILLA_RUN_SUFFIX = "nested_cv"  # vanilla has no z dim, so z-anchored sweeps
                                  # reuse the free-z vanilla runs (always plain
                                  # "nested_cv").  Set in main() from RUN_SUFFIX.


def _data_seed_dir(dataset_id: int, fold: int, seed: int) -> str:
    return f"data_dataset{dataset_id}/fold{fold}/seed_{seed}/{RUN_SUFFIX}"


def _vanilla_data_seed_dir(dataset_id: int, fold: int, seed: int) -> str:
    return f"data_dataset{dataset_id}/fold{fold}/seed_{seed}/{VANILLA_RUN_SUFFIX}"


def _load_latents_per_subject(dataset_id: int, fold: int, arch: str,
                               seed_dirs: List[Tuple[int, str]],
                               split: str = "test") -> Optional[np.ndarray]:
    """Returns (N_seeds, N_subj, dim) per-subject mean latents over time,
    or None if no seed has the expected file.

    split: 'test' loads latents_tensor{name}.pt (held-out subjects);
           'train' loads latents_tensor{name}_traindata.pt (training subjects).
    """
    name = "latentmodel" if arch == "idrnn" else "vanilla"
    suffix = "_traindata" if split == "train" else ""
    _dir_fn = _data_seed_dir if arch == "idrnn" else _vanilla_data_seed_dir
    Zs = []
    for seed, _ in seed_dirs:
        p = os.path.join(_dir_fn(dataset_id, fold, seed),
                          f"latents_tensor{name}{suffix}.pt")
        if not os.path.exists(p):
            continue
        try:
            t = torch.load(p, map_location="cpu")
            if hasattr(t, "numpy"):
                arr = t.detach().numpy()
            else:
                arr = np.array(t)
        except Exception:
            continue
        if arr.ndim != 3:
            continue
        Zs.append(arr.mean(axis=1))   # per-subject time-mean → (N, dim)
    if not Zs:
        return None
    shapes = {z.shape for z in Zs}
    if len(shapes) != 1:
        return None
    return np.stack(Zs, axis=0)


def _pca_project(Z_train: np.ndarray, Z_test: np.ndarray,
                  n_components: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit StandardScaler + PCA on Z_train; transform both.  Returns
    (Z_train_pcs, Z_test_pcs, explained_variance_ratio).

    n_components is clipped to min(dim, n_train) - 1 if the raw dim is smaller.
    """
    n_pc = min(n_components, Z_train.shape[1], Z_train.shape[0] - 1)
    scaler = StandardScaler()
    Zs_tr = scaler.fit_transform(Z_train)
    Zs_te = scaler.transform(Z_test)
    pca = PCA(n_components=n_pc)
    Zp_tr = pca.fit_transform(Zs_tr)
    Zp_te = pca.transform(Zs_te)
    return Zp_tr, Zp_te, pca.explained_variance_ratio_


def _load_per_session_nll(dataset_id: int, fold: int, arch: str,
                           seed_dirs: List[Tuple[int, str]]) -> Optional[pd.DataFrame]:
    """Returns DataFrame with columns [session, nll_per_trial, seed] (one row per
    test session per seed).  Reads rnn_results{name}.csv which contains
    normalized_likelihood = exp(-nll_per_trial) per session."""
    name = "latentmodel" if arch == "idrnn" else "vanilla"
    _dir_fn = _data_seed_dir if arch == "idrnn" else _vanilla_data_seed_dir
    rows = []
    for seed, _ in seed_dirs:
        p = os.path.join(_dir_fn(dataset_id, fold, seed),
                          f"rnn_results{name}.csv")
        if not os.path.exists(p):
            continue
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if "normalized_likelihood" not in df.columns:
            continue
        df = df.copy()
        # NB: `compute_rnn_likelihoods_torch` saves PER-TRIAL NLL into the
        # "normalized_likelihood" column (misnamed).  Use it directly.
        df["nll_per_trial"] = df["normalized_likelihood"].astype(float)
        df["seed"] = seed
        rows.append(df[["session", "nll_per_trial", "seed"]])
    if not rows:
        return None
    return pd.concat(rows, ignore_index=True)


def _loo_ridge(Z: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, np.ndarray]:
    """LOO ridge regression; returns (Pearson r, p, R², per-subject OOS preds).

    Why both r AND R²:
      - When there's no signal, RidgeCV converges to heavy regularisation;
        predictions collapse to ~train_mean_y per LOO split, but those
        per-split means carry a built-in anti-correlation with truth:
          pred_i ≈ (N·mean − y_i) / (N−1) = const − y_i / (N−1)
        so naive Pearson r reports a spurious large *negative* value.
      - R² (sklearn r2_score) is ≤ 0 when the model is worse than predicting
        the global mean.  It does NOT show the LOO-no-signal artifact, so
        R² ≈ 0 (or negative) is the honest diagnostic of "no decodable signal".
    """
    preds = np.zeros(len(y))
    for tr, te in LeaveOneOut().split(Z):
        mu_z, sd_z = Z[tr].mean(0), Z[tr].std(0) + 1e-8
        mu_y, sd_y = y[tr].mean(), y[tr].std() + 1e-8
        clf = RidgeCV(alphas=[0.1, 1, 10, 100, 1000])
        clf.fit((Z[tr] - mu_z) / sd_z, (y[tr] - mu_y) / sd_y)
        preds[te] = clf.predict((Z[te] - mu_z) / sd_z) * sd_y + mu_y
    r, p = pearsonr(y, preds)
    r2 = float(r2_score(y, preds))
    return float(r), float(p), r2, preds


def _rsa(Z: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Spearman r between pairwise latent-distance vector and pairwise |Δalpha|."""
    if len(y) < 4:
        return float("nan"), 1.0
    Z_rdm = pdist(Z, metric="euclidean")
    y_rdm = pdist(y[:, None], metric="cityblock")
    r, p = spearmanr(Z_rdm, y_rdm)
    return float(r), float(p)


# ── Per-dataset analysis ──────────────────────────────────────────────────────
def analyze_dataset(dataset_id: int, n_vanilla_seeds: Optional[int] = None) -> Dict:
    out = {"dataset_id": dataset_id, "folds": {}, "summary": {}}

    # Pool per-subject results across folds for decoding
    pooled_y_idrnn,  pooled_pred_idrnn  = [], []
    pooled_y_vanilla, pooled_pred_vanilla = [], []
    rsa_per_fold = {"idrnn": [], "vanilla": []}

    # NLL: collect per-session rows from each fold × seed
    nll_rows = []

    for fold in range(N_OUTER_FOLDS):
        fold_dir   = f"data_dataset{dataset_id}/fold{fold}"
        param_p    = os.path.join(fold_dir, "true_param_test.csv")
        if not os.path.exists(param_p):
            print(f"  [dataset{dataset_id}/fold{fold}] no true_param_test.csv — skip")
            continue
        alpha = pd.read_csv(param_p)["alphaP_list"].to_numpy()    # (N_test_subj,)

        fold_rec = {"alpha_n": int(len(alpha))}

        def _analyze_arch(arch: str, seed_dirs_for_seeds: List[Tuple[int, str]],
                          match_dim: Optional[int] = None):
            """Per-seed PCA→RSA→decode → average metrics across seeds.

            We never average latent tensors across seeds because IDRNN's z has
            a rotation/sign ambiguity (rotating z and inversely rotating the
            decoder weights leaves choices unchanged).  Two seeds typically
            converge to different rotated frames, so elementwise averaging
            collapses signal toward zero.  Computing the metric per seed and
            averaging the metrics (which are rotation-invariant scalars) is
            the standard fix.

            `match_dim`: number of PCs to project to.  When set (e.g. the IDRNN
            z_dim), the arch is decoded from exactly this many PCs so the
            IDRNN-vs-vanilla comparison is dimension-matched.  None → fall back
            to min(PCA_N_COMPONENTS, latent_dim).
            """
            Z_test = _load_latents_per_subject(dataset_id, fold, arch,
                                                seed_dirs_for_seeds, split="test")
            Z_train = _load_latents_per_subject(dataset_id, fold, arch,
                                                 seed_dirs_for_seeds, split="train")
            if Z_test is None or Z_train is None:
                return {"error": f"no {arch} latents loaded"}, None, None

            n_pc = match_dim if match_dim is not None else PCA_N_COMPONENTS
            rsa_list, dec_list, r2_list, pred_list, ev_list, predvar_list = [], [], [], [], [], []
            for s_idx in range(Z_test.shape[0]):
                Zp_tr, Zp_te, var_ratio = _pca_project(
                    Z_train[s_idx], Z_test[s_idx], n_pc)
                r_rsa, _ = _rsa(Zp_te, alpha)
                r_dec, _, r2_dec, preds = _loo_ridge(Zp_te, alpha)
                rsa_list.append(r_rsa)
                dec_list.append(r_dec)
                r2_list.append(r2_dec)
                pred_list.append(preds)
                ev_list.append(var_ratio)
                # Diagnostic: prediction variance / target variance.  Small ratio
                # (< ~0.1) flags the LOO-no-signal regime where Pearson r is
                # spuriously negative.
                predvar_list.append(float(preds.std() / (alpha.std() + 1e-12)))

            rsa_arr = np.array(rsa_list, dtype=float)
            dec_arr = np.array(dec_list, dtype=float)
            r2_arr  = np.array(r2_list, dtype=float)
            preds_avg = np.mean(np.stack(pred_list, axis=0), axis=0)
            r_dec_avgpred, p_dec_avgpred = pearsonr(alpha, preds_avg)
            r2_avgpred = float(r2_score(alpha, preds_avg))

            return ({
                "n_seeds_used":  int(Z_test.shape[0]),
                "raw_dim":       int(Z_test.shape[2]),
                "pca_n_components":       int(Zp_te.shape[1]),
                "pca_explained_variance": [float(v) for v in np.mean(np.stack(ev_list, 0), axis=0)],
                # Per-seed metrics
                "rsa_r_per_seed":     [float(x) for x in rsa_arr],
                "decode_r_per_seed":  [float(x) for x in dec_arr],
                "decode_r2_per_seed": [float(x) for x in r2_arr],
                "pred_to_target_std_ratio_per_seed": predvar_list,
                # Mean ± SEM across seeds within fold
                "rsa_r_mean_across_seeds":    float(rsa_arr.mean()),
                "rsa_r_sem_across_seeds":     float(rsa_arr.std(ddof=1) / np.sqrt(len(rsa_arr))) if len(rsa_arr) > 1 else 0.0,
                "decode_r_mean_across_seeds": float(dec_arr.mean()),
                "decode_r_sem_across_seeds":  float(dec_arr.std(ddof=1) / np.sqrt(len(dec_arr))) if len(dec_arr) > 1 else 0.0,
                "decode_r2_mean_across_seeds": float(r2_arr.mean()),
                "decode_r2_sem_across_seeds":  float(r2_arr.std(ddof=1) / np.sqrt(len(r2_arr))) if len(r2_arr) > 1 else 0.0,
                # On the seed-averaged predictions (predictions live in alpha-space)
                "decode_r_avg_preds":  float(r_dec_avgpred),
                "decode_p_avg_preds":  float(p_dec_avgpred),
                "decode_r2_avg_preds": r2_avgpred,
            }, float(rsa_arr.mean()), preds_avg)

        # ── IDRNN ──
        idr_base = f"runs_dataset{dataset_id}_{RUN_SUFFIX}/fold{fold}"
        idr_seeds_all = _seed_dirs(idr_base)
        idr_seeds = _rank_by_specificity(idr_seeds_all, TOP_K_BY_SPEC)
        idr_rec, idr_rsa_r, idr_preds = _analyze_arch("idrnn", idr_seeds)
        fold_rec["idrnn"] = idr_rec
        if idr_rsa_r is not None:
            rsa_per_fold["idrnn"].append(idr_rsa_r)
            pooled_y_idrnn.append(alpha);  pooled_pred_idrnn.append(idr_preds)

        # Dimension-match target: the number of PCs IDRNN was actually decoded
        # from (= min(PCA_N_COMPONENTS, z_dim)).  Vanilla is projected to the
        # SAME count so the comparison is at equal representational capacity.
        match_dim = idr_rec.get("pca_n_components") if "error" not in idr_rec else None

        idr_nll_df = _load_per_session_nll(dataset_id, fold, "idrnn", idr_seeds_all)
        if idr_nll_df is not None:
            agg = idr_nll_df.groupby("session", as_index=False)["nll_per_trial"].mean()
            agg["arch"] = "idrnn"; agg["fold"] = fold
            nll_rows.append(agg)

        # ── Vanilla ──
        # Vanilla has no z dim, so the z-anchored sweeps reuse the free-z vanilla
        # runs.  When RUN_SUFFIX has a z/spec suffix, strip it for the vanilla path.
        van_base = f"runs_vanilla_dataset{dataset_id}_{VANILLA_RUN_SUFFIX}/fold{fold}"
        van_seeds_all = _seed_dirs(van_base)
        # Optional subsample to match IDRNN's TOP_K so seed-count asymmetry
        # isn't confounded with arch differences.  Uses the first n seeds by
        # seed-number order (deterministic).
        if n_vanilla_seeds is not None and len(van_seeds_all) > n_vanilla_seeds:
            van_seeds_used = van_seeds_all[:n_vanilla_seeds]
        else:
            van_seeds_used = van_seeds_all
        van_rec, van_rsa_r, van_preds = _analyze_arch("vanilla", van_seeds_used,
                                                       match_dim=match_dim)
        fold_rec["vanilla"] = van_rec
        if van_rsa_r is not None:
            rsa_per_fold["vanilla"].append(van_rsa_r)
            pooled_y_vanilla.append(alpha);  pooled_pred_vanilla.append(van_preds)

        van_nll_df = _load_per_session_nll(dataset_id, fold, "vanilla", van_seeds_all)
        if van_nll_df is not None:
            agg = van_nll_df.groupby("session", as_index=False)["nll_per_trial"].mean()
            agg["arch"] = "vanilla"; agg["fold"] = fold
            nll_rows.append(agg)

        out["folds"][f"fold{fold}"] = fold_rec

    # ── Summary across folds ─────────────────────────────────────────────────
    s: Dict = {}

    if pooled_y_idrnn:
        y_all  = np.concatenate(pooled_y_idrnn)
        pr_all = np.concatenate(pooled_pred_idrnn)
        r, p = pearsonr(y_all, pr_all)
        s["idrnn_decode_pooled_r"] = float(r)
        s["idrnn_decode_pooled_p"] = float(p)
        s["idrnn_decode_n"]        = int(len(y_all))
    if pooled_y_vanilla:
        y_all  = np.concatenate(pooled_y_vanilla)
        pr_all = np.concatenate(pooled_pred_vanilla)
        r, p = pearsonr(y_all, pr_all)
        s["vanilla_decode_pooled_r"] = float(r)
        s["vanilla_decode_pooled_p"] = float(p)
        s["vanilla_decode_n"]        = int(len(y_all))

    for arch in ("idrnn", "vanilla"):
        rs = rsa_per_fold[arch]
        if rs:
            s[f"{arch}_rsa_mean"]  = float(np.mean(rs))
            s[f"{arch}_rsa_sem"]   = float(np.std(rs, ddof=1) / np.sqrt(len(rs))) if len(rs) > 1 else 0.0
            s[f"{arch}_rsa_per_fold"] = [float(x) for x in rs]

    # NLL paired test (per-session, mean across seeds)
    if nll_rows:
        nll_df = pd.concat(nll_rows, ignore_index=True)
        # pivot to (session, fold) x arch
        wide = (nll_df.pivot_table(index=["session", "fold"], columns="arch",
                                    values="nll_per_trial").dropna())
        if not wide.empty:
            s["nll_idrnn_mean"]   = float(wide["idrnn"].mean())
            s["nll_vanilla_mean"] = float(wide["vanilla"].mean())
            s["nll_idrnn_sem"]    = float(wide["idrnn"].sem())
            s["nll_vanilla_sem"]  = float(wide["vanilla"].sem())
            s["n_sessions_paired"] = int(len(wide))
            try:
                W, pval = wilcoxon(wide["idrnn"], wide["vanilla"])
                s["nll_wilcoxon_W"] = float(W)
                s["nll_wilcoxon_p"] = float(pval)
            except Exception as e:
                s["nll_wilcoxon_error"] = str(e)

    out["summary"] = s
    return out


# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_dataset(rec: Dict, out_path: str) -> None:
    s = rec.get("summary", {})
    if not s:
        return

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    archs = ["idrnn", "vanilla"]
    colors = {"idrnn": "#2a82c2", "vanilla": "#a19f9f"}
    labels = {"idrnn": "IDRNN", "vanilla": "Vanilla"}

    # ── NLL ──
    ax = axes[0]
    vals = [s.get(f"nll_{a}_mean", np.nan) for a in archs]
    errs = [s.get(f"nll_{a}_sem", 0.0)    for a in archs]
    bars = ax.bar([0, 1], vals, yerr=errs, color=[colors[a] for a in archs],
                  edgecolor="black", capsize=4)
    ax.set_xticks([0, 1]); ax.set_xticklabels([labels[a] for a in archs])
    ax.set_ylabel("Held-out NLL per trial")
    title = f"Held-out NLL (N={s.get('n_sessions_paired', '?')} sessions)"
    if "nll_wilcoxon_p" in s:
        title += f"\nWilcoxon p={s['nll_wilcoxon_p']:.1e}"
    ax.set_title(title)

    # ── RSA ──
    ax = axes[1]
    rsas_idr = s.get("idrnn_rsa_per_fold", [])
    rsas_van = s.get("vanilla_rsa_per_fold", [])
    if rsas_idr or rsas_van:
        for i, (a, rs) in enumerate([("idrnn", rsas_idr), ("vanilla", rsas_van)]):
            ax.scatter([i] * len(rs), rs, s=80, color=colors[a],
                       edgecolor="black", zorder=3)
            if rs:
                ax.bar(i, np.mean(rs), color=colors[a], alpha=0.4)
    ax.set_xticks([0, 1]); ax.set_xticklabels([labels[a] for a in archs])
    ax.set_ylabel("RSA Spearman r (latent vs α)")
    ax.set_title("Alpha-recovery RSA (per fold)")
    ax.axhline(0, color="grey", lw=0.5)

    # ── Decoding ──
    ax = axes[2]
    rs = [s.get(f"{a}_decode_pooled_r", np.nan) for a in archs]
    ax.bar([0, 1], rs, color=[colors[a] for a in archs], edgecolor="black")
    for i, a in enumerate(archs):
        n = s.get(f"{a}_decode_n", 0)
        p = s.get(f"{a}_decode_pooled_p", 1.0)
        ax.text(i, max(rs) * 1.02 if rs[i] > 0 else 0.02,
                f"N={n}\np={p:.1e}", ha="center", fontsize=8)
    ax.set_xticks([0, 1]); ax.set_xticklabels([labels[a] for a in archs])
    ax.set_ylabel("Pearson r")
    ax.set_title("Alpha decoding (LOO ridge, pooled)")
    ax.axhline(0, color="grey", lw=0.5)

    fig.suptitle(f"Synthetic dataset {rec['dataset_id']} — nested-CV outer-fold analysis",
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path}")


# ── Multi-dataset summary ─────────────────────────────────────────────────────
def plot_multi(records: List[Dict], out_path: str) -> None:
    if len(records) < 2:
        return
    ds_ids = [r["dataset_id"] for r in records]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    metric_specs = [
        ("nll_{a}_mean",          "Held-out NLL/trial", False),
        ("{a}_rsa_mean",          "RSA r",              False),
        ("{a}_decode_pooled_r",   "Decoding r",         False),
    ]
    colors = {"idrnn": "#2a82c2", "vanilla": "#a19f9f"}
    xs = np.arange(len(ds_ids))
    width = 0.35

    for ax, (key_tmpl, ylabel, _) in zip(axes, metric_specs):
        for i, arch in enumerate(("idrnn", "vanilla")):
            vals = [r["summary"].get(key_tmpl.format(a=arch), np.nan) for r in records]
            ax.bar(xs + (i - 0.5) * width, vals, width=width, color=colors[arch],
                    edgecolor="black", label=arch.upper())
        ax.set_xticks(xs); ax.set_xticklabels([f"ds{d}" for d in ds_ids])
        ax.set_ylabel(ylabel)
        ax.axhline(0, color="grey", lw=0.5)
        ax.legend()

    fig.suptitle("Synthetic nested-CV — multi-dataset comparison", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    global RUN_SUFFIX, VANILLA_RUN_SUFFIX
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_ids", type=int, nargs="+", default=[0, 2, 5, 10, 17])
    ap.add_argument("--n_vanilla_seeds", type=int, default=None,
                    help="Subsample vanilla seeds to this count (to match IDRNN's "
                          "TOP_K=5).  None = use all 30 seeds.")
    ap.add_argument("--run_suffix", default="nested_cv",
                    help="IDRNN run-suffix to analyze: 'nested_cv' (free-z), "
                          "'nested_cv_z1' (z=1 NLL), 'nested_cv_z1_spec' "
                          "(z=1 spec).  Vanilla always reads the free-z runs "
                          "since it has no z dim.")
    ap.add_argument("--out_tag", default=None,
                    help="Subdir tag under final_plots/synthetic/ for outputs. "
                          "Defaults to the run_suffix so sweeps don't overwrite.")
    args = ap.parse_args()

    RUN_SUFFIX = args.run_suffix
    VANILLA_RUN_SUFFIX = "nested_cv"   # vanilla has no z; always free-z runs
    out_tag = args.out_tag or args.run_suffix
    base_out = f"final_plots/synthetic/{out_tag}"

    records = []
    for did in args.dataset_ids:
        print(f"\n── dataset {did} [{RUN_SUFFIX}] ──")
        rec = analyze_dataset(did, n_vanilla_seeds=args.n_vanilla_seeds)
        rec["run_suffix"] = RUN_SUFFIX
        out_dir = f"{base_out}/dataset{did}"
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "nested_cv_summary.json"), "w") as f:
            json.dump(rec, f, indent=2)
        plot_dataset(rec, os.path.join(out_dir, "nested_cv_analysis.png"))
        records.append(rec)

    if len(records) > 1:
        os.makedirs(base_out, exist_ok=True)
        with open(os.path.join(base_out, "nested_cv_multi_dataset_summary.json"), "w") as f:
            json.dump([r["summary"] for r in records], f, indent=2)
        plot_multi(records, os.path.join(base_out, "nested_cv_multi_dataset.png"))


if __name__ == "__main__":
    main()
