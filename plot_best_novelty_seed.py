#!/usr/bin/env python3
"""
Find the IDRNN seed whose latent dimensions have the highest per-component
Pearson |r| with w_novelty (from the EM cognitive model fit), then plot its
latents in the same format as plot_sloutsky_cog_weights.py.

Epoch selection: reads LOSO epochs from analyze_across_seeds_summary.json
(produced by analyze_across_seeds.py) so LOSO doesn't need to re-run.
Falls back to LOSO inline if the summary file is not found.

Usage:
    python plot_best_novelty_seed.py
    python plot_best_novelty_seed.py --seeds 12,50,76,100,142
    python plot_best_novelty_seed.py --epoch 1500  # fixed epoch, skip LOSO
"""

import argparse
import os
import json
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.decomposition import PCA

from compute_reconstruction_specificity import (
    load_model_config, create_model_from_config,
    load_model_checkpoint, compute_reconstruction_loss,
)

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=None,
                    help="Fixed epoch for all seeds (bypass LOSO selection)")
parser.add_argument("--seeds", type=str, default=None,
                    help="Comma-separated seeds, e.g. '12,50,76,100,142'")
args = parser.parse_args()
FIXED_EPOCH  = args.epoch
FILTER_SEEDS = (set(int(s) for s in args.seeds.split(",")) if args.seeds else None)

# ── Configuration ─────────────────────────────────────────────────────────────
DGP       = "sloutsky"
DATA_DIR  = f"data_{DGP}"
IDRNN_DIR = f"runs_{DGP}"
VAN_DIR   = f"runs_vanilla_{DGP}"
PLOT_DIR  = f"plots_{DGP}"
MIN_EPOCH = 100
MAX_EPOCH = 3000
os.makedirs(PLOT_DIR, exist_ok=True)

GROUP_ORDER  = ["young_child", "old_child", "adult"]
GROUP_LABELS = ["Young\nchildren", "Older\nchildren", "Adults"]
GROUP_COLORS = ["#5B9BD5", "#ED7D31", "#A5A5A5"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Load sequence data ────────────────────────────────────────────────────────
xin_test  = torch.from_numpy(np.load(f"{DATA_DIR}/xin_test.npy")).float().to(device)
xin_train = torch.from_numpy(np.load(f"{DATA_DIR}/xin_train.npy")).float().to(device)
c_train   = torch.from_numpy(np.load(f"{DATA_DIR}/c_train.npy")).float().to(device)

_enc_train_path = f"{DATA_DIR}/xin_enc_train.npy"
xin_enc_train   = (torch.from_numpy(np.load(_enc_train_path)).float().to(device)
                   if os.path.exists(_enc_train_path) else None)

B_test  = xin_test.shape[0]
B_train = xin_train.shape[0]

# ── Load participant metadata and EM cognitive model params ───────────────────
df_test  = pd.read_csv(f"{DATA_DIR}/df_test.csv")
df_full  = pd.read_csv(f"{DATA_DIR}/exp2_train_all_participants.csv")
test_participants_sorted = np.array(sorted(df_test["subid"].unique()))
assert len(test_participants_sorted) == B_test

age_map = df_full.groupby("subid")["age"].first().to_dict()

em           = np.load(f"{DATA_DIR}/em_results.npz", allow_pickle=True)
h_all        = em["h_all"]        # (P_all, 5) — all participants
participants = em["participants"]  # (P_all,)

def softmax(x):
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / ex.sum()

# w_novelty aligned to test-split order (same logic as plot_sloutsky_cog_weights.py)
em_id_to_idx = {sid: i for i, sid in enumerate(participants)}
w_novelty, age_labels, valid_mask = [], [], []
for sid in test_participants_sorted:
    if sid in em_id_to_idx:
        idx = em_id_to_idx[sid]
        w_novelty.append(softmax(h_all[idx, 1:5])[3])
        age_labels.append(age_map.get(sid, "unknown"))
        valid_mask.append(True)
    else:
        w_novelty.append(np.nan)
        age_labels.append("unknown")
        valid_mask.append(False)

w_novelty  = np.array(w_novelty)
age_labels = np.array(age_labels)
valid      = np.array(valid_mask)

# ── LOSO epoch selection (inline — only used if summary JSON missing) ─────────
def list_checkpoints(base_dir, seed):
    ckpt_dir = os.path.join(base_dir, f"seed_{seed}", "checkpoints")
    if not os.path.exists(ckpt_dir):
        return []
    return sorted([
        int(f.replace("epoch", "").replace(".pt", ""))
        for f in os.listdir(ckpt_dir)
        if f.startswith("epoch") and f.endswith(".pt")
    ])


def compute_nll_for_seed_epoch(base_dir, seed, epoch, is_latent,
                                xin_data, c_data, xin_enc, B):
    run_dir     = os.path.join(base_dir, f"seed_{seed}")
    ckpt_path   = os.path.join(run_dir, "checkpoints", f"epoch{epoch:04d}.pt")
    frozen_path = os.path.join(run_dir, "frozen_decoder", "policy_model.pt")
    if not os.path.exists(ckpt_path):
        return None
    if is_latent and not os.path.exists(frozen_path):
        return None
    try:
        model_config = load_model_config(run_dir)
        model = create_model_from_config(
            model_config, n_participants=B, device=device,
            frozen_decoder_path=frozen_path if is_latent else None
        )
        model = load_model_checkpoint(ckpt_path, model, device)
        x_enc = xin_enc if xin_enc is not None else xin_data
        with torch.no_grad():
            mu, _ = model.encoder(x_enc.unsqueeze(1), return_per_timestep=False)
        loss = compute_reconstruction_loss(
            model, xin_data, c_data, z_latent=mu, is_latent_model=True
        )
        return float(loss.mean().item())
    except Exception as e:
        print(f"  NLL failed seed={seed} epoch={epoch}: {e}")
        return None


def loso_select_epoch(base_dir, seeds, held_out_seed):
    other_seeds  = [s for s in seeds if s != held_out_seed]
    epochs_sets  = {s: set(e for e in list_checkpoints(base_dir, s)
                            if MIN_EPOCH <= e <= MAX_EPOCH) for s in other_seeds}
    if not epochs_sets:
        return None
    common_epochs = sorted(set.intersection(*epochs_sets.values()))
    best_epoch, best_nll = None, float("inf")
    for epoch in common_epochs:
        nlls = [compute_nll_for_seed_epoch(base_dir, s, epoch, True,
                                            xin_train, c_train, xin_enc_train, B_train)
                for s in other_seeds]
        nlls = [n for n in nlls if n is not None]
        if nlls:
            mean_nll = float(np.mean(nlls))
            if mean_nll < best_nll:
                best_nll   = mean_nll
                best_epoch = epoch
    return best_epoch

# ── Resolve seeds and their epochs ───────────────────────────────────────────
all_seeds = sorted([
    int(d.split("_")[1])
    for d in os.listdir(IDRNN_DIR)
    if d.startswith("seed_")
    and (FILTER_SEEDS is None or int(d.split("_")[1]) in FILTER_SEEDS)
])
print(f"Seeds: {all_seeds}")

# Try to read epochs from the summary JSON first
_seed_tag   = ("_seeds" + "-".join(str(s) for s in sorted(FILTER_SEEDS))
               if FILTER_SEEDS else "")
summary_path = os.path.join(PLOT_DIR, f"analyze_across_seeds_summary{_seed_tag}.json")

seed_epochs = {}
if FIXED_EPOCH is not None:
    seed_epochs = {s: FIXED_EPOCH for s in all_seeds}
    print(f"Using fixed epoch {FIXED_EPOCH} for all seeds.")
elif os.path.exists(summary_path):
    with open(summary_path) as f:
        summary = json.load(f)
    raw = summary.get("idrnn", {}).get("epochs", {})
    seed_epochs = {int(k): v for k, v in raw.items() if int(k) in all_seeds}
    print(f"Loaded LOSO epochs from {summary_path}: {seed_epochs}")
else:
    print(f"Summary JSON not found at {summary_path}. Running LOSO inline …")
    for seed in all_seeds:
        ep = loso_select_epoch(IDRNN_DIR, all_seeds, seed)
        if ep is not None:
            seed_epochs[seed] = ep
            print(f"  Seed {seed}: LOSO epoch = {ep}")

# ── Latent extraction ─────────────────────────────────────────────────────────
def get_latent_array(seed, epoch):
    """Return (B_test, z_dim) last-timestep mu for IDRNN."""
    run_dir     = os.path.join(IDRNN_DIR, f"seed_{seed}")
    ckpt_path   = os.path.join(run_dir, "checkpoints", f"epoch{epoch:04d}.pt")
    frozen_path = os.path.join(run_dir, "frozen_decoder", "policy_model.pt")
    cfg_path    = os.path.join(run_dir, "config.json")

    with open(cfg_path) as f:
        cfg = json.load(f)
    enc_in_dim = cfg.get("model_config", {}).get("enc_in_dim", cfg["in_dim"])
    xenc = xin_test if enc_in_dim == cfg["in_dim"] else xin_test   # same_enc_dec=True for sloutsky

    model_config = load_model_config(run_dir)
    model = create_model_from_config(
        model_config, n_participants=B_test, device=device,
        frozen_decoder_path=frozen_path
    )
    model = load_model_checkpoint(ckpt_path, model, device)
    model.eval()

    with torch.no_grad():
        mu, _ = model.encoder(xenc.unsqueeze(1), return_per_timestep=True)
    return mu.squeeze(1)[:, -1, :].cpu().numpy()   # (B_test, z_dim)

# ── Find seed with best per-component |r| with w_novelty ─────────────────────
print("\nComputing per-dimension novelty correlations …")
best_seed, best_max_r, best_dim, best_epoch_used = None, -1.0, None, None
latents_per_seed = {}

for seed, epoch in seed_epochs.items():
    try:
        lat = get_latent_array(seed, epoch)      # (B_test, z_dim)
    except Exception as e:
        print(f"  Seed {seed}: latent extraction failed — {e}")
        continue
    latents_per_seed[seed] = lat

    # compute |r| between w_novelty and each latent dimension
    lat_valid    = lat[valid]
    wnov_valid   = w_novelty[valid]
    rs = []
    for dim in range(lat_valid.shape[1]):
        z_dim_vals = lat_valid[:, dim]
        fin = np.isfinite(z_dim_vals) & np.isfinite(wnov_valid)
        if fin.sum() > 2:
            r, _ = pearsonr(z_dim_vals[fin], wnov_valid[fin])
            rs.append(abs(r))
        else:
            rs.append(0.0)

    max_r    = max(rs)
    best_dim_this = int(np.argmax(rs))
    print(f"  Seed {seed} (epoch {epoch}): max |r(w_novelty, z_i)| = {max_r:.3f}  "
          f"(dim {best_dim_this})")

    if max_r > best_max_r:
        best_max_r      = max_r
        best_seed       = seed
        best_dim        = best_dim_this
        best_epoch_used = epoch

print(f"\n→ Best seed: {best_seed}  epoch: {best_epoch_used}  "
      f"dim: {best_dim}  max |r| = {best_max_r:.3f}")

# ── Reproduce plot_sloutsky_cog_weights.py format ────────────────────────────
lat_best    = latents_per_seed[best_seed]   # (B_test, z_dim)
z_dim_total = lat_best.shape[1]

# Apply valid mask
lat_best_v  = lat_best[valid]
w_nov_v     = w_novelty[valid]
age_v       = age_labels[valid]

# PCA (same as plot_sloutsky_cog_weights.py)
n_components = min(2, z_dim_total)
pca          = PCA(n_components=n_components)
z_reduced    = pca.fit_transform(lat_best_v)   # (N, n_components)
explained    = pca.explained_variance_ratio_

color_map   = {g: c for g, c in zip(GROUP_ORDER, GROUP_COLORS)}
point_colors = np.array([color_map.get(a, "#888888") for a in age_v])

if n_components >= 2:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"IDRNN: Novelty weight vs latent space  "
        f"(seed {best_seed}, epoch {best_epoch_used}, best z-dim={best_dim}, |r|={best_max_r:.3f})",
        fontsize=12,
    )

    # Panel 0: PCA scatter coloured by w_novelty
    sc = axes[0].scatter(
        z_reduced[:, 0], z_reduced[:, 1],
        c=w_nov_v, cmap="viridis",
        alpha=0.8, edgecolors="k", linewidths=0.4, s=50,
    )
    axes[0].set_xlabel(f"PC1 ({explained[0]*100:.1f}% var)", fontsize=11)
    axes[0].set_ylabel(f"PC2 ({explained[1]*100:.1f}% var)", fontsize=11)
    axes[0].set_title("Latent PCA\n(colour = w_novelty)", fontsize=11)
    plt.colorbar(sc, ax=axes[0], label="w_novelty")
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    # Panels 1 & 2: w_novelty vs PC1 / PC2, coloured by age group
    for panel_idx, pc_idx in enumerate([0, 1], start=1):
        ax       = axes[panel_idx]
        pc_label = f"PC{pc_idx+1} ({explained[pc_idx]*100:.1f}% var)"

        for grp, col, lbl in zip(GROUP_ORDER, GROUP_COLORS, GROUP_LABELS):
            mask = age_v == grp
            ax.scatter(w_nov_v[mask], z_reduced[mask, pc_idx],
                       color=col, alpha=0.75, s=50,
                       edgecolors="none", label=lbl.replace("\n", " "))

        # Regression line + Pearson r
        fin = np.isfinite(w_nov_v) & np.isfinite(z_reduced[:, pc_idx])
        if fin.sum() > 2:
            r, pval = pearsonr(w_nov_v[fin], z_reduced[fin, pc_idx])
            coef    = np.polyfit(w_nov_v[fin], z_reduced[fin, pc_idx], 1)
            x_line  = np.linspace(w_nov_v[fin].min(), w_nov_v[fin].max(), 100)
            ax.plot(x_line, np.poly1d(coef)(x_line), "r--",
                    linewidth=2, label=f"r = {r:.3f}, p = {pval:.3f}")

        ax.set_xlabel("w_novelty (EM fit)", fontsize=11)
        ax.set_ylabel(pc_label, fontsize=11)
        ax.set_title(f"w_novelty vs {pc_label.split(' ')[0]}", fontsize=11)
        ax.legend(fontsize=8, frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

else:
    # 1-D latent space
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle(
        f"IDRNN: Novelty weight vs latent  "
        f"(seed {best_seed}, epoch {best_epoch_used})",
        fontsize=12,
    )
    for grp, col, lbl in zip(GROUP_ORDER, GROUP_COLORS, GROUP_LABELS):
        mask = age_v == grp
        ax.scatter(w_nov_v[mask], z_reduced[mask, 0],
                   color=col, alpha=0.75, s=50,
                   edgecolors="none", label=lbl.replace("\n", " "))
    fin = np.isfinite(w_nov_v) & np.isfinite(z_reduced[:, 0])
    if fin.sum() > 2:
        r, pval = pearsonr(w_nov_v[fin], z_reduced[fin, 0])
        coef    = np.polyfit(w_nov_v[fin], z_reduced[fin, 0], 1)
        x_line  = np.linspace(w_nov_v[fin].min(), w_nov_v[fin].max(), 100)
        ax.plot(x_line, np.poly1d(coef)(x_line), "r--",
                linewidth=2, label=f"r = {r:.3f}, p = {pval:.3f}")
    ax.set_xlabel("w_novelty (EM fit)", fontsize=11)
    ax.set_ylabel("Latent z", fontsize=11)
    ax.legend(fontsize=9, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout()
out_path = os.path.join(
    PLOT_DIR,
    f"novelty_best_seed{best_seed}_epoch{best_epoch_used}{_seed_tag}.png",
)
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved: {out_path}")
plt.close()
