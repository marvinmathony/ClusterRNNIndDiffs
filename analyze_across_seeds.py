#!/usr/bin/env python3
"""
RSA and cognitive-model weight correlations across seeds.

Uses the same LOSO train-NLL epoch selection as decode_across_seeds.py:
for each held-out seed s, the best epoch is chosen by minimising mean
train-NLL across the other seeds.

Per seed computes:
  1. RSA: Spearman r between latent RDM and each behavioural RDM
         (age, w_novelty, w_value, w_uncert, theta, cog_nll) — IDRNN + Vanilla
  2. Cog-weight Pearson r: correlate PC1 of latents with each EM-fitted
         parameter (theta, w_value, w_uncert, w_lag, w_novelty) — IDRNN + Vanilla

Results are averaged across seeds and plotted with mean ± SEM + seed scatter.

Usage:
    python analyze_across_seeds.py                      # LOSO, all seeds
    python analyze_across_seeds.py --seeds 12,50,76     # restrict seeds
    python analyze_across_seeds.py --epoch 1500         # fixed epoch, skip LOSO
"""

import argparse
import os
import json
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr, pearsonr
from sklearn.decomposition import PCA

from modelsandtraining import IDRNN, Decoder, LatentRNN_secondstep, AblatedRNN
from compute_reconstruction_specificity import (
    load_model_config, create_model_from_config,
    load_model_checkpoint, compute_reconstruction_loss,
)
from sloutsky_cog_model import unpack_params

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Load sequence data ────────────────────────────────────────────────────────
xin_test  = torch.from_numpy(np.load(f"{DATA_DIR}/xin_test.npy")).float().to(device)
c_test    = torch.from_numpy(np.load(f"{DATA_DIR}/c_test.npy")).float().to(device)
xin_train = torch.from_numpy(np.load(f"{DATA_DIR}/xin_train.npy")).float().to(device)
c_train   = torch.from_numpy(np.load(f"{DATA_DIR}/c_train.npy")).float().to(device)

_enc_train_path = f"{DATA_DIR}/xin_enc_train.npy"
xin_enc_train   = (torch.from_numpy(np.load(_enc_train_path)).float().to(device)
                   if os.path.exists(_enc_train_path) else None)

B_test  = xin_test.shape[0]
B_train = xin_train.shape[0]

# ── Behavioural labels & cognitive model parameters ───────────────────────────
df_test  = pd.read_csv(f"{DATA_DIR}/df_test.csv")
df_unique = df_test.drop_duplicates("subid").sort_values("subid").reset_index(drop=True)
test_subids = sorted(df_test["subid"].unique())
assert len(test_subids) == B_test

# Binary age (matches analyze_rsa_sloutsky.py convention)
y_age = np.where(df_unique["age"] == "young_child", 0, 1)

# EM cognitive model parameters (test participants only)
em = np.load(f"{DATA_DIR}/em_results.npz", allow_pickle=True)
h_all_test       = em["h_all_test"]
participants_test = em["participants_test"]
assert list(participants_test) == test_subids, \
    "EM participants_test order doesn't match test_subids"

cog_params    = [unpack_params(h_all_test[i]) for i in range(B_test)]
theta_vals    = np.array([p["theta"]              for p in cog_params])
w_value_vals  = np.array([p["b_value_train"]      for p in cog_params])
w_uncert_vals = np.array([p["b_uncertain_train"]  for p in cog_params])
w_lag_vals    = np.array([p["b_lag_train"]        for p in cog_params])
w_novel_vals  = np.array([p["b_novelty_train"]    for p in cog_params])

# Cog model NLL
df_cog   = pd.read_csv(f"{DATA_DIR}/cog_model_results.csv")
nll_vals = df_cog.sort_values("session")["normalized_likelihood"].values
assert len(nll_vals) == B_test

# ── Target RDMs for RSA ───────────────────────────────────────────────────────
def make_rdm_vec(values):
    return pdist(values.reshape(-1, 1), metric="euclidean")

targets = {
    "age":      make_rdm_vec(y_age),
    "w_novelty":make_rdm_vec(w_novel_vals),
    "w_value":  make_rdm_vec(w_value_vals),
    "w_uncert": make_rdm_vec(w_uncert_vals),
    "theta":    make_rdm_vec(theta_vals),
    "cog_nll":  make_rdm_vec(nll_vals),
}

# Cognitive variables for direct Pearson correlation with PC1
cog_vars = {
    "theta":    theta_vals,
    "w_value":  w_value_vals,
    "w_uncert": w_uncert_vals,
    "w_lag":    w_lag_vals,
    "w_novelty":w_novel_vals,
}

# ── LOSO epoch-selection helpers ──────────────────────────────────────────────
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
    """Mean train NLL for one seed at one epoch."""
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
        if is_latent:
            x_enc = xin_enc if xin_enc is not None else xin_data
            with torch.no_grad():
                mu, _ = model.encoder(x_enc.unsqueeze(1), return_per_timestep=False)
            loss = compute_reconstruction_loss(
                model, xin_data, c_data, z_latent=mu, is_latent_model=True
            )
        else:
            loss = compute_reconstruction_loss(
                model, xin_data, c_data, is_latent_model=False
            )
        return float(loss.mean().item())
    except Exception as e:
        print(f"  NLL failed seed={seed} epoch={epoch}: {e}")
        return None


def loso_select_epoch(base_dir, seeds, held_out_seed, is_latent):
    """Return (best_epoch, mean_nll) for held_out_seed via train NLL of other seeds."""
    other_seeds = [s for s in seeds if s != held_out_seed]
    epochs_sets = {
        s: set(e for e in list_checkpoints(base_dir, s) if MIN_EPOCH <= e <= MAX_EPOCH)
        for s in other_seeds
    }
    if not epochs_sets:
        return None, None
    common_epochs = sorted(set.intersection(*epochs_sets.values()))
    if not common_epochs:
        return None, None

    best_epoch, best_nll = None, float("inf")
    for epoch in common_epochs:
        nlls = [compute_nll_for_seed_epoch(base_dir, s, epoch, is_latent,
                                            xin_train, c_train, xin_enc_train, B_train)
                for s in other_seeds]
        nlls = [n for n in nlls if n is not None]
        if nlls:
            mean_nll = float(np.mean(nlls))
            if mean_nll < best_nll:
                best_nll   = mean_nll
                best_epoch = epoch
    return best_epoch, best_nll


# ── Latent extraction ─────────────────────────────────────────────────────────
def get_latent_array(base_dir, seed, epoch, is_latent):
    """
    (B, D) float array.
    IDRNN  → last-timestep mu from the sequence encoder.
    Vanilla → time-averaged GRU hidden states.
    """
    run_dir  = os.path.join(base_dir, f"seed_{seed}")
    cfg_path = os.path.join(run_dir, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)

    frozen_path  = os.path.join(run_dir, "frozen_decoder", "policy_model.pt")
    ckpt_path    = os.path.join(run_dir, "checkpoints", f"epoch{epoch:04d}.pt")
    model_config = load_model_config(run_dir)
    model = create_model_from_config(
        model_config, n_participants=B_test, device=device,
        frozen_decoder_path=frozen_path if is_latent else None
    )
    model = load_model_checkpoint(ckpt_path, model, device)
    model.eval()

    with torch.no_grad():
        if is_latent:
            # Encoder uses full xin if enc_in_dim == in_dim (same_enc_dec=True)
            enc_in_dim = cfg.get("model_config", {}).get("enc_in_dim", cfg["in_dim"])
            xenc = xin_test if enc_in_dim == cfg["in_dim"] else xin_test
            mu, _ = model.encoder(xenc.unsqueeze(1), return_per_timestep=True)
            return mu.squeeze(1)[:, -1, :].cpu().numpy()   # (B, z_dim)
        else:
            _, _, hidden = model(xin_test)
            return hidden.mean(dim=1).cpu().numpy()         # (B, hid)


# ── Per-model analysis loop ───────────────────────────────────────────────────
def run_analysis(base_dir, is_latent, label):
    seeds = sorted([
        int(d.split("_")[1])
        for d in os.listdir(base_dir)
        if d.startswith("seed_")
        and (FILTER_SEEDS is None or int(d.split("_")[1]) in FILTER_SEEDS)
    ])
    print(f"\n{'='*70}")
    print(f"{label}  |  seeds: {seeds}")
    if FIXED_EPOCH is not None:
        print(f"  Fixed epoch: {FIXED_EPOCH}")
    print(f"{'='*70}")

    rsa_per_seed     = {t: [] for t in targets}
    cogcorr_per_seed = {v: [] for v in cog_vars}
    epoch_per_seed   = {}
    lat_per_seed     = {}   # cache for best-latent saving

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")

        if FIXED_EPOCH is not None:
            epoch = FIXED_EPOCH
            ckpt = os.path.join(base_dir, f"seed_{seed}", "checkpoints",
                                f"epoch{epoch:04d}.pt")
            if not os.path.exists(ckpt):
                print(f"  Checkpoint for epoch {epoch} not found. Skipping.")
                continue
            print(f"  Using fixed epoch: {epoch}")
        else:
            print(f"  Running LOSO over {len(seeds)-1} other seeds …")
            epoch, nll = loso_select_epoch(base_dir, seeds, seed, is_latent)
            if epoch is None:
                print(f"  LOSO selection failed. Skipping.")
                continue
            print(f"  LOSO epoch: {epoch}  (mean train NLL = {nll:.4f})")

        epoch_per_seed[seed] = epoch

        try:
            lat = get_latent_array(base_dir, seed, epoch, is_latent)
        except Exception as e:
            print(f"  Latent extraction failed: {e}")
            continue
        lat_per_seed[seed] = lat

        # ── RSA ──────────────────────────────────────────────────────────────
        rdm = pdist(lat, metric="euclidean")
        for tname, trdm in targets.items():
            r, _ = spearmanr(rdm, trdm)
            rsa_per_seed[tname].append(r)
        print("  RSA: " + "  ".join(f"{t}={rsa_per_seed[t][-1]:+.3f}" for t in targets))

        # ── Cog-weight Pearson r with PC1 ─────────────────────────────────────
        n_comp = min(2, lat.shape[1])
        pca    = PCA(n_components=n_comp)
        z_pc   = pca.fit_transform(lat)    # (B, n_comp)
        pc1    = z_pc[:, 0]
        for vname, vals in cog_vars.items():
            finite = np.isfinite(pc1) & np.isfinite(vals)
            r = float(pearsonr(pc1[finite], vals[finite])[0]) if finite.sum() > 2 else float("nan")
            cogcorr_per_seed[vname].append(r)
        print("  Cog r(PC1): " + "  ".join(f"{v}={cogcorr_per_seed[v][-1]:+.3f}" for v in cog_vars))

    return (
        {t: np.array(v) for t, v in rsa_per_seed.items()},
        {v: np.array(r) for v, r in cogcorr_per_seed.items()},
        epoch_per_seed,
        lat_per_seed,
    )


rsa_idrnn,  cogcorr_idrnn,  epochs_idrnn, lats_idrnn = run_analysis(IDRNN_DIR, True,  "IDRNN")
rsa_van,    cogcorr_van,    epochs_van,   lats_van   = run_analysis(VAN_DIR,   False, "Vanilla")

# ── Save best-novelty latents ─────────────────────────────────────────────────
def save_best_novelty_latent(lats, cogcorr, epochs, nametag):
    """Pick the seed with highest |r(PC1, w_novelty)| and save its latent."""
    nov_r = cogcorr.get("w_novelty", np.array([]))
    seeds = [s for s in lats]
    if not seeds or len(nov_r) == 0:
        print(f"  [{nametag}] No latents to save.")
        return
    # align cogcorr order with lats order (both follow seed insertion order)
    nov_r_abs  = np.abs(nov_r)
    best_idx   = int(np.argmax(nov_r_abs))
    best_seed  = seeds[best_idx]
    best_epoch = epochs[best_seed]
    best_lat   = lats[best_seed]

    save_path = os.path.join(DATA_DIR, f"latents_tensor{nametag}_best_novelty.pt")
    torch.save(torch.from_numpy(best_lat).float(), save_path)
    print(f"  [{nametag}] Best novelty seed: {best_seed}  epoch: {best_epoch}  "
          f"|r(PC1,w_novelty)|={nov_r_abs[best_idx]:.3f}  → saved to {save_path}")

    meta_path = save_path.replace(".pt", "_meta.json")
    with open(meta_path, "w") as f:
        json.dump({"seed": best_seed, "epoch": best_epoch,
                   "r_pc1_novelty": float(nov_r[best_idx])}, f, indent=2)
    print(f"  [{nametag}] Metadata → {meta_path}")

print("\n── Saving best-novelty latents ──")
save_best_novelty_latent(lats_idrnn, cogcorr_idrnn, epochs_idrnn, "latentmodel")
save_best_novelty_latent(lats_van,   cogcorr_van,   epochs_van,   "vanilla")

# ── Shared plot helpers ───────────────────────────────────────────────────────
_seed_tag = ("_seeds" + "-".join(str(s) for s in sorted(FILTER_SEEDS))
             if FILTER_SEEDS else "")

def scatter_seeds(ax, xi, vals, rng, jitter=0.05):
    """Overlay individual seed points at position xi."""
    jit = rng.uniform(-jitter, jitter, size=len(vals))
    ax.scatter(xi + jit, vals, color="k", s=18, zorder=5, alpha=0.7)


def grouped_bar_plot(ax, keys, idrnn_dict, van_dict, key_labels,
                     ylabel, title, primary_keys=()):
    """Grouped IDRNN/Vanilla bars with SEM error bars and seed scatter."""
    x   = np.arange(len(keys))
    w   = 0.35
    rng = np.random.default_rng(0)

    for i, k in enumerate(keys):
        iv = idrnn_dict[k]
        vv = van_dict[k]
        alpha = 0.95 if k in primary_keys else 0.5

        mi  = float(np.nanmean(iv)) if len(iv) else 0
        sei = float(np.nanstd(iv) / np.sqrt(np.sum(~np.isnan(iv)))) if len(iv) > 1 else 0
        mv  = float(np.nanmean(vv)) if len(vv) else 0
        sev = float(np.nanstd(vv) / np.sqrt(np.sum(~np.isnan(vv)))) if len(vv) > 1 else 0

        ax.bar(i - w/2, mi, w, color="#4C72B0", alpha=alpha,
               label="IDRNN"   if i == 0 else "_")
        ax.bar(i + w/2, mv, w, color="#DD8452", alpha=alpha,
               label="Vanilla" if i == 0 else "_")
        ax.errorbar([i - w/2, i + w/2], [mi, mv], yerr=[sei, sev],
                    fmt="none", color="k", capsize=4, lw=1.2)

        scatter_seeds(ax, i - w/2, iv, rng)
        scatter_seeds(ax, i + w/2, vv, rng)

    ax.axhline(0, color="k", lw=0.8, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels([key_labels.get(k, k) for k in keys])
    for lbl, k in zip(ax.get_xticklabels(), keys):
        if k in primary_keys:
            lbl.set_fontweight("bold")
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ── Plot 1: RSA ───────────────────────────────────────────────────────────────
target_labels = {
    "age":       "Age",
    "w_novelty": "w novelty",
    "w_value":   "w value",
    "w_uncert":  "w uncert",
    "theta":     "θ",
    "cog_nll":   "Cog NLL",
}
fig1, ax1 = plt.subplots(figsize=(10, 4))
grouped_bar_plot(
    ax1,
    keys        = list(targets.keys()),
    idrnn_dict  = rsa_idrnn,
    van_dict    = rsa_van,
    key_labels  = target_labels,
    ylabel      = "Spearman r (RSA)",
    title       = ("RSA: latent RDM vs behavioural RDMs\n"
                   "(LOSO train-NLL epoch, mean ± SEM)"),
    primary_keys = {"age", "w_novelty"},
)
fig1.tight_layout()
rsa_path = os.path.join(PLOT_DIR, f"rsa_across_seeds{_seed_tag}.png")
fig1.savefig(rsa_path, dpi=150)
print(f"\nRSA plot saved → {rsa_path}")
plt.close(fig1)

# ── Plot 2: Cog-weight Pearson r with PC1 ─────────────────────────────────────
cog_labels = {
    "theta":    "θ",
    "w_value":  "w value",
    "w_uncert": "w uncert",
    "w_lag":    "w lag",
    "w_novelty":"w novelty",
}
fig2, ax2 = plt.subplots(figsize=(8, 4))
grouped_bar_plot(
    ax2,
    keys         = list(cog_vars.keys()),
    idrnn_dict   = cogcorr_idrnn,
    van_dict     = cogcorr_van,
    key_labels   = cog_labels,
    ylabel       = "Pearson r  (latent PC1 vs cog param)",
    title        = ("Cog model weights vs latent PC1\n"
                    "(LOSO train-NLL epoch, mean ± SEM)"),
    primary_keys = {"w_novelty"},
)
fig2.tight_layout()
cog_path = os.path.join(PLOT_DIR, f"cogcorr_across_seeds{_seed_tag}.png")
fig2.savefig(cog_path, dpi=150)
print(f"Cog-weight correlation plot saved → {cog_path}")
plt.close(fig2)

# ── Save summary JSON ─────────────────────────────────────────────────────────
summary = {
    "idrnn": {
        "epochs":  {str(s): e for s, e in epochs_idrnn.items()},
        "rsa":     {t: rsa_idrnn[t].tolist()     for t in targets},
        "cogcorr": {v: cogcorr_idrnn[v].tolist() for v in cog_vars},
    },
    "vanilla": {
        "epochs":  {str(s): e for s, e in epochs_van.items()},
        "rsa":     {t: rsa_van[t].tolist()     for t in targets},
        "cogcorr": {v: cogcorr_van[v].tolist() for v in cog_vars},
    },
}
json_path = os.path.join(PLOT_DIR, f"analyze_across_seeds_summary{_seed_tag}.json")
with open(json_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"Summary saved → {json_path}")
