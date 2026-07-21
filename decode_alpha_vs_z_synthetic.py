"""Does the IDRNN's α-encoding improve when the bottleneck is relaxed?

For each (dataset, fold, z-value), pick the top-3 Optuna trials by cv_val_loss,
load the trained IDRNN encoder, forward-pass on the held-out test subjects,
PCA to 3 components, and report:
  - LOO ridge R² (honest "is there signal?")
  - LOO ridge Pearson r (back-compat)
  - RSA Spearman r vs α
  - pred/y std ratio (collapse diagnostic)

Per-seed metrics, averaged across the chosen trials within each fold (rotation-
invariant aggregation).  No latent averaging.

Output:
  final_plots/synthetic/decode_alpha_vs_z.png
  final_plots/synthetic/decode_alpha_vs_z.csv
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import warnings
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import pdist
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import LeaveOneOut
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

sys.path.insert(0, ".")
from modelsandtraining import IDRNN, Decoder, LatentRNN_secondstep


# ── Metrics ───────────────────────────────────────────────────────────────────
def _pca_project(Z_train: np.ndarray, Z_test: np.ndarray, n_components: int):
    n_pc = min(n_components, Z_train.shape[1], Z_train.shape[0] - 1)
    sc = StandardScaler(); Zs_tr = sc.fit_transform(Z_train); Zs_te = sc.transform(Z_test)
    p = PCA(n_components=n_pc); return p.fit_transform(Zs_tr), p.transform(Zs_te), p.explained_variance_ratio_


def _loo_ridge(Z, y):
    preds = np.zeros(len(y))
    for tr, te in LeaveOneOut().split(Z):
        mu_z, sd_z = Z[tr].mean(0), Z[tr].std(0) + 1e-8
        mu_y, sd_y = y[tr].mean(), y[tr].std() + 1e-8
        clf = RidgeCV(alphas=[0.1, 1, 10, 100, 1000])
        clf.fit((Z[tr] - mu_z) / sd_z, (y[tr] - mu_y) / sd_y)
        preds[te] = clf.predict((Z[te] - mu_z) / sd_z) * sd_y + mu_y
    r, _ = pearsonr(y, preds)
    return float(r), float(r2_score(y, preds)), preds


def _rsa(Z, y):
    if len(y) < 4: return float("nan")
    z_rdm = pdist(Z); y_rdm = pdist(y[:, None], metric="cityblock")
    r, _ = spearmanr(z_rdm, y_rdm); return float(r)


# ── Optuna-trial lookup ───────────────────────────────────────────────────────
def find_trials_by_z(dataset_id: int, fold: int, z_value: int,
                      top_k: int = 3) -> List[Dict[str, Any]]:
    """Top-K Optuna IDRNN trials matching (dataset, fold, z) by cv_val_loss."""
    pattern = (f"optuna_runs/synthetic/dataset{dataset_id}/idrnn/"
               f"fold{fold}/trial_*/config.json")
    import glob
    hits = []
    for cfg_p in glob.glob(pattern):
        try:
            cfg = json.load(open(cfg_p))
            if cfg.get("z") != z_value:
                continue
            if cfg.get("cv_val_loss") is None:
                continue
            hits.append({
                "trial_dir":   os.path.dirname(cfg_p),
                "cv_val_loss": float(cfg["cv_val_loss"]),
                "cfg":         cfg,
            })
        except Exception:
            continue
    hits.sort(key=lambda h: h["cv_val_loss"])
    return hits[:top_k]


# ── Load model + extract latents ──────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_idrnn_from_config(cfg: Dict[str, Any]) -> torch.nn.Module:
    mc = cfg["model_config"]
    encoder = IDRNN(in_dim=mc["enc_in_dim"], z_dim=mc["z_dim"],
                    hid=mc["enc_hidden"],
                    n_tasks=mc.get("n_tasks"),
                    task_emb_dim=mc.get("task_emb_dim", 0))
    decoder = Decoder(in_dim=mc["dec_in_dim"], z_dim=mc["z_dim"],
                      hid=mc["hidden"], A=mc["A"])
    for p in decoder.parameters():
        p.requires_grad = False
    model = LatentRNN_secondstep(
        encoder=encoder, hid=mc["hidden"], z_dim=mc["z_dim"],
        in_dim=mc["dec_in_dim"], A=mc["A"], decoder=decoder,
        n_tasks=mc.get("n_tasks"), task_emb_dim=mc.get("task_emb_dim", 0),
        reinit_decoder_per_block=mc.get("reinit_decoder_per_block", False),
    )
    return model.to(DEVICE)


def _latents_for_trial(trial_dir: str, xin_train: torch.Tensor,
                        xin_test: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    cfg_path = os.path.join(trial_dir, "config.json")
    cfg = json.load(open(cfg_path))
    epoch = cfg.get("cv_selected_epoch")
    if epoch is None:
        return None, None
    ckpt = os.path.join(trial_dir, "checkpoints", f"epoch{epoch:04d}.pt")
    if not os.path.exists(ckpt):
        return None, None
    model = _build_idrnn_from_config(cfg)
    sd = torch.load(ckpt, map_location=DEVICE)
    model.load_state_dict(sd, strict=False)
    model.eval()
    with torch.no_grad():
        # IDRNN encoder returns (mu, lv) given block tensor
        # Need to match the shape expected by IDRNN.forward(blocks_tensor)
        # Synthetic xin is (N, T, in_dim) — add a block dim
        x_tr = xin_train.unsqueeze(1) if xin_train.dim() == 3 else xin_train
        x_te = xin_test.unsqueeze(1)  if xin_test.dim()  == 3 else xin_test
        z_tr, _ = model.encoder(x_tr.to(DEVICE), return_per_timestep=False)
        z_te, _ = model.encoder(x_te.to(DEVICE),  return_per_timestep=False)
    # z is (B, z_dim) for return_per_timestep=False
    return z_tr.detach().cpu().numpy(), z_te.detach().cpu().numpy()


# ── Per-dataset sweep ─────────────────────────────────────────────────────────
def analyze(dataset_id: int, z_values: List[int], top_k: int) -> pd.DataFrame:
    rows = []
    xin_test_root = np.load(f"data_dataset{dataset_id}/xin_test.npy")  # for shape only; reload per fold
    for fold in range(3):
        # Per-fold test/train data + alpha
        xin_train = torch.from_numpy(np.load(
            f"data_dataset{dataset_id}/fold{fold}/xin_train.npy")).float()
        xin_test  = torch.from_numpy(np.load(
            f"data_dataset{dataset_id}/fold{fold}/xin_test.npy")).float()
        alpha = pd.read_csv(
            f"data_dataset{dataset_id}/fold{fold}/true_param_test.csv"
        )["alphaP_list"].to_numpy()

        for z in z_values:
            trials = find_trials_by_z(dataset_id, fold, z, top_k=top_k)
            if not trials:
                continue
            r2s, rs, rsas, ratios = [], [], [], []
            for t in trials:
                z_tr, z_te = _latents_for_trial(t["trial_dir"], xin_train, xin_test)
                if z_tr is None or z_te is None:
                    continue
                Zp_tr, Zp_te, _ = _pca_project(z_tr, z_te, min(3, z))
                r_dec, r2_dec, preds = _loo_ridge(Zp_te, alpha)
                r_rsa = _rsa(Zp_te, alpha)
                r2s.append(r2_dec); rs.append(r_dec); rsas.append(r_rsa)
                ratios.append(float(preds.std() / (alpha.std() + 1e-12)))
            if not r2s:
                continue
            rows.append({
                "dataset": dataset_id, "fold": fold, "z": z,
                "n_trials": len(r2s),
                "R2_mean": np.mean(r2s),    "R2_sem": np.std(r2s, ddof=1)/np.sqrt(len(r2s)) if len(r2s)>1 else 0,
                "r_mean":  np.mean(rs),     "r_sem":  np.std(rs, ddof=1)/np.sqrt(len(rs)) if len(rs)>1 else 0,
                "rsa_mean":np.mean(rsas),   "rsa_sem":np.std(rsas, ddof=1)/np.sqrt(len(rsas)) if len(rsas)>1 else 0,
                "pred_to_y_std_mean": np.mean(ratios),
            })
            print(f"  ds{dataset_id}/fold{fold}/z={z}: "
                  f"n={len(r2s)} R²={np.mean(r2s):+.3f} r={np.mean(rs):+.3f} "
                  f"RSA={np.mean(rsas):+.3f}  pred/y_std={np.mean(ratios):.3f}")
    return pd.DataFrame(rows)


# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_z_vs_decode(df: pd.DataFrame, out: str):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    metrics = [("R2_mean", "R2_sem", "LOO ridge R²", axes[0]),
               ("rsa_mean", "rsa_sem", "RSA Spearman r (latent vs α)", axes[1]),
               ("pred_to_y_std_mean", None, "pred std / α std (signal-presence)", axes[2])]
    datasets = sorted(df["dataset"].unique())
    cmap = matplotlib.cm.get_cmap("viridis", len(datasets))
    for col, sem_col, title, ax in metrics:
        for i, d in enumerate(datasets):
            for fold in (0, 1, 2):
                sub = df[(df["dataset"]==d) & (df["fold"]==fold)].sort_values("z")
                if sub.empty: continue
                label = f"ds{d}" if fold == 0 else None
                if sem_col and sem_col in sub.columns:
                    ax.errorbar(sub["z"], sub[col], yerr=sub[sem_col],
                                color=cmap(i), alpha=0.4, marker="o", ms=4, label=label, lw=0.8)
                else:
                    ax.plot(sub["z"], sub[col], color=cmap(i), alpha=0.4,
                            marker="o", ms=4, label=label, lw=0.8)
        ax.set_xlabel("IDRNN z-dim")
        ax.set_ylabel(title)
        ax.set_title(title, fontweight="bold")
        ax.axhline(0, color="grey", lw=0.5)
        ax.legend(loc="best", fontsize=8)
    fig.suptitle("IDRNN α-decoding vs latent bottleneck (z-dim) — synthetic",
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_ids", type=int, nargs="+", default=[0, 2, 5, 10, 17])
    ap.add_argument("--z_values", type=int, nargs="+", default=[1, 2, 3, 5, 8])
    ap.add_argument("--top_k", type=int, default=3,
                    help="Top-K Optuna trials per (ds, fold, z) by cv_val_loss")
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=UserWarning)

    print(f"Datasets: {args.dataset_ids}  z values: {args.z_values}  top-{args.top_k} trials per cell")
    dfs = []
    for did in args.dataset_ids:
        print(f"\n── dataset {did} ──")
        df = analyze(did, args.z_values, args.top_k)
        if not df.empty:
            dfs.append(df)
    if not dfs:
        print("No data to plot."); return
    df = pd.concat(dfs, ignore_index=True)
    os.makedirs("final_plots/synthetic", exist_ok=True)
    df.to_csv("final_plots/synthetic/decode_alpha_vs_z.csv", index=False)
    plot_z_vs_decode(df, "final_plots/synthetic/decode_alpha_vs_z.png")
    print(f"\nSaved -> final_plots/synthetic/decode_alpha_vs_z.{{csv,png}}")


if __name__ == "__main__":
    main()
