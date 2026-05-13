#!/usr/bin/env python3
"""
RSA analysis for IDRNN latents on Sloutsky data.

Correlates the latent representational dissimilarity matrix (RDM) with:
  a) Age RDM          (ordinal: young_child=0, old_child=1, adult=2)
  b) Novelty weight   (w_novelty from EM cognitive model fit, data_sloutsky/em_results.npz)
  c) Value weight     (w_value from EM fit)
  d) Uncertainty weight (w_uncert from EM fit)
  e) Temperature      (theta from EM fit)
  f) Cog model NLL    (per-participant model fit quality)

Sweeps all checkpoint epochs and plots RSA correlation over training.
Also reports permutation-test significance at the selected epoch.

Usage:
    python analyze_rsa_sloutsky.py
    python analyze_rsa_sloutsky.py --runs_dir hp_search_runs_sloutsky/lmbd_0.005_z_8
    python analyze_rsa_sloutsky.py --epoch 600 --no_sweep   # single epoch only
"""

import os
import sys
import json
import pickle
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from itertools import combinations
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from modelsandtraining import IDRNN, Decoder, LatentRNN_secondstep, AblatedRNN
from compute_reconstruction_specificity import (
    load_model_config, create_model_from_config, load_model_checkpoint
)
from sloutsky_cog_model import unpack_params

# ── Defaults ─────────────────────────────────────────────────────────────────
DATA_DIR   = "data_sloutsky"
SEEDS      = [12, 50, 76, 100, 142]
STEP       = 100          # evaluate every N epochs
PERM_N     = 1000         # permutations for significance test
OUT_DIR    = "plots_sloutsky"

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--runs_dir", default="runs_sloutsky",
                    help="Directory containing seed_*/checkpoints subdirs")
parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
parser.add_argument("--seed", type=int, default=None,
                    help="Single seed to use (overrides --seeds; required for sweep)")
parser.add_argument("--step", type=int, default=STEP,
                    help="Checkpoint interval to evaluate")
parser.add_argument("--epoch", type=int, default=None,
                    help="Evaluate a single epoch only (overrides sweep)")
parser.add_argument("--no_sweep", action="store_true",
                    help="Skip epoch sweep, only compute at --epoch")
parser.add_argument("--perm_n", type=int, default=PERM_N)
parser.add_argument("--out_dir", default=OUT_DIR)
parser.add_argument("--vanilla_dir", default="runs_vanilla_sloutsky",
                    help="Vanilla model run directory for comparison")
parser.add_argument("--vanilla_epoch", type=int, default=None,
                    help="Epoch for vanilla (default: from best_epoch_by_loss.json)")
parser.add_argument("--vanilla_seed", type=int, default=None,
                    help="Seed for vanilla (default: from best_epoch_by_loss.json)")
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
device = torch.device("cpu")

# ── Load test data ────────────────────────────────────────────────────────────
xin_test = torch.from_numpy(np.load(f"{DATA_DIR}/xin_test.npy")).float()
xenc     = xin_test.unsqueeze(1)   # (B, 1, T, enc_in_dim)
B        = xin_test.shape[0]

df_all  = pd.read_csv(f"{DATA_DIR}/exp2_train_all_participants.csv")
df_test = pd.read_csv(f"{DATA_DIR}/df_test.csv")

# Test participant order: sorted coded subids (matches xin_test row order)
test_subids = sorted(df_test["subid"].unique())
assert len(test_subids) == B, f"Expected {B} participants, got {len(test_subids)}"

# Age labels (ordinal)
#age_map  = df_all.drop_duplicates("subid").set_index("subid")["age"].to_dict()
#age_ord  = {"young_child": 0, "old_child": 1, "adult": 2}
#y_age    = np.array([age_ord[age_map[s]] for s in test_subids])
df_unique = df_test.drop_duplicates(subset="subid").sort_values("subid").reset_index(drop=True)
y_age = np.where(df_unique["age"] == "young_child", 0, 1)


# Cognitive model parameters from EM fit (em_results.npz)
# h_all_test[i] are unconstrained params aligned with sorted test_subids order
em = np.load(f"{DATA_DIR}/em_results.npz", allow_pickle=True)
h_all_test       = em["h_all_test"]          # (29, 5) unconstrained
participants_test = em["participants_test"]   # should match test_subids

assert list(participants_test) == test_subids, \
    "EM participants_test ordering doesn't match test_subids"

# Transform to interpretable parameters
cog_params = [unpack_params(h_all_test[i]) for i in range(B)]
theta_vals   = np.array([p["theta"]              for p in cog_params])
w_value_vals = np.array([p["b_value_train"]       for p in cog_params])
w_uncert_vals= np.array([p["b_uncertain_train"]   for p in cog_params])
w_lag_vals   = np.array([p["b_lag_train"]         for p in cog_params])
w_novel_vals = np.array([p["b_novelty_train"]     for p in cog_params])

# NLL from cognitive model fit (marginal LL under group prior)
df_cog   = pd.read_csv(f"{DATA_DIR}/cog_model_results.csv")
nll_vals = df_cog.sort_values("session")["normalized_likelihood"].values
assert len(nll_vals) == B

print(f"Test participants: {B}")
#print(f"Age distribution:   {dict(zip(['young_child','old_child','adult'], np.bincount(y_age)))}")
print(f"Age distribution:   {dict(zip(['young_child','older'], np.bincount(y_age)))}")
print(f"Theta range:        [{theta_vals.min():.3f}, {theta_vals.max():.3f}]")
print(f"w_novelty range:    [{w_novel_vals.min():.3f}, {w_novel_vals.max():.3f}]")
print(f"w_value range:      [{w_value_vals.min():.3f}, {w_value_vals.max():.3f}]")
print(f"w_uncert range:     [{w_uncert_vals.min():.3f}, {w_uncert_vals.max():.3f}]")
print(f"NLL range:          [{nll_vals.min():.2f}, {nll_vals.max():.2f}]")

# ── Build target RDMs ─────────────────────────────────────────────────────────
def make_rdm_vec(values, metric="euclidean"):
    """Vectorize upper triangle of pairwise distance matrix."""
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    return pdist(values, metric=metric)

targets = {
    "age":      make_rdm_vec(y_age.reshape(-1, 1)),
    "w_novelty":make_rdm_vec(w_novel_vals.reshape(-1, 1)),
    "w_value":  make_rdm_vec(w_value_vals.reshape(-1, 1)),
    "w_uncert": make_rdm_vec(w_uncert_vals.reshape(-1, 1)),
    "theta":    make_rdm_vec(theta_vals.reshape(-1, 1)),
    "nll":      make_rdm_vec(nll_vals.reshape(-1, 1)),
}

# ── Model loading helpers ─────────────────────────────────────────────────────
def load_model(seed, epoch):
    run_dir  = os.path.join(args.runs_dir, f"seed_{seed}")
    ckpt     = os.path.join(run_dir, "checkpoints", f"epoch{epoch:04d}.pt")
    dec_path = os.path.join(run_dir, "frozen_decoder", "policy_model.pt")
    if not os.path.exists(ckpt) or not os.path.exists(dec_path):
        return None
    cfg   = load_model_config(run_dir)
    model = create_model_from_config(cfg, n_participants=B, device=device,
                                     frozen_decoder_path=dec_path)
    return load_model_checkpoint(ckpt, model, device)

def extract_latents(model):
    model.eval()
    with torch.no_grad():
        mu, _ = model.encoder(xenc, return_per_timestep=True)  # (B,1,T,z)
        return mu.squeeze(1)[:, -1, :].cpu().numpy() #mu.squeeze(1).mean(dim=1).cpu().numpy()           # (B, z)

def latent_rdm_vec(lat):
    return pdist(lat, metric="euclidean")

# ── Vanilla model helpers ─────────────────────────────────────────────────────
def load_vanilla_model(seed, epoch):
    run_dir = os.path.join(args.vanilla_dir, f"seed_{seed}")
    ckpt    = os.path.join(run_dir, "checkpoints", f"epoch{epoch:04d}.pt")
    if not os.path.exists(ckpt):
        return None
    cfg   = load_model_config(run_dir)
    model = create_model_from_config(cfg, n_participants=B, device=device)
    return load_model_checkpoint(ckpt, model, device)

def extract_vanilla_hidden(model):
    """Return time-averaged GRU hidden states: (B, hid)."""
    model.eval()
    with torch.no_grad():
        _, _, hidden_tr = model(xin_test)  # hidden_tr: (B, T, hid)
        return hidden_tr.mean(dim=1).cpu().numpy()

# ── Permutation test ──────────────────────────────────────────────────────────
def permutation_rsa(latent_vec, target_vec, n_perm=1000, seed=0):
    """One-sided permutation test: how often does shuffled latent_vec beat observed r?"""
    rng = np.random.default_rng(seed)
    n   = int((1 + np.sqrt(1 + 8 * len(latent_vec))) / 2)  # recover N from N*(N-1)/2
    obs_r, _ = spearmanr(latent_vec, target_vec)

    null = np.empty(n_perm)
    for i in range(n_perm):
        perm_idx = rng.permutation(n)
        # rebuild permuted RDM rows/cols
        mat = squareform(latent_vec)
        mat_perm = mat[np.ix_(perm_idx, perm_idx)] #np.ix_ switches subject indices in both dimensions, preserving structure
        null[i], _ = spearmanr(squareform(mat_perm, checks=False), target_vec)

    p = np.mean(null >= obs_r)
    return obs_r, p, null

# ── Single seed selection ─────────────────────────────────────────────────────
# Resolve which single seed to use for the sweep
# Priority: --seed > best_seed from nll JSON > first in --seeds
spec_json_early = os.path.join(args.runs_dir, "best_epoch_by_nll.json")
if args.seed is not None:
    sweep_seed = args.seed
elif os.path.exists(spec_json_early):
    with open(spec_json_early) as f:
        sweep_seed = json.load(f).get("best_seed", args.seeds[0])
else:
    sweep_seed = args.seeds[0]
print(f"Using seed {sweep_seed} for sweep")

# ── Epoch list ────────────────────────────────────────────────────────────────
def available_epochs(seed):
    ckpt_dir = os.path.join(args.runs_dir, f"seed_{seed}", "checkpoints")
    if not os.path.exists(ckpt_dir):
        return []
    eps = []
    for f in os.listdir(ckpt_dir):
        if f.startswith("epoch") and f.endswith(".pt"):
            eps.append(int(f.replace("epoch", "").replace(".pt", "")))
    return sorted(eps)

if args.epoch is not None and args.no_sweep:
    epochs_to_eval = [args.epoch]
else:
    all_eps = set(available_epochs(sweep_seed))
    if args.epoch:
        all_eps.add(args.epoch)
    epochs_to_eval = sorted(e for e in all_eps if e % args.step == 0 or e == args.epoch)

print(f"\nEpochs to evaluate: {epochs_to_eval[:5]} ... {epochs_to_eval[-5:]} "
      f"(total={len(epochs_to_eval)})")

# ── Main sweep ────────────────────────────────────────────────────────────────
results = []   # list of dicts per epoch

for epoch in tqdm(epochs_to_eval, desc="Epochs"):
    model = load_model(sweep_seed, epoch)
    if model is None:
        continue
    lat = extract_latents(model)
    rdm = latent_rdm_vec(lat)

    row = {"epoch": epoch}
    for tname, trdm in targets.items():
        r, p = spearmanr(rdm, trdm)
        row[f"r_{tname}"] = r
        row[f"p_{tname}"] = p
        if epoch == epochs_to_eval[-1]:
            print(f"  Epoch {epoch}: {tname} r={r:.3f} p={p:.4f}")
    results.append(row)

df_res = pd.DataFrame(results).sort_values("epoch").reset_index(drop=True)
print("\n" + df_res[["epoch"] + [c for c in df_res.columns if c.startswith("r_")]].to_string())

# ── Permutation test at best epoch per target ────────────────────────────────
best_epochs = {}
for tname in targets:
    col = f"r_{tname}"
    idx = df_res[col].idxmax() #return index of first occurence of maximum
    best_epochs[tname] = df_res.loc[idx, "epoch"]
    print(f"Best epoch for {tname} RSA: {best_epochs[tname]} "
          f"(r={df_res.loc[idx, col]:.3f})")

primary_targets = ["age", "w_novelty"]
tested_epochs   = set()
for ep_label in primary_targets:
    ep = best_epochs[ep_label]
    if ep in tested_epochs:
        continue
    tested_epochs.add(ep)

    m = load_model(sweep_seed, ep)
    if m is None:
        continue
    rdm_lat = latent_rdm_vec(extract_latents(m))

    print(f"\nPermutation test at epoch {ep} (best for {ep_label}, n_perm={args.perm_n}):")
    for tname, trdm in targets.items():
        r_obs, p_perm, null = permutation_rsa(rdm_lat, trdm, n_perm=args.perm_n)
        print(f"  {tname:12s}: r={r_obs:.3f}, p_perm={p_perm:.4f}")

# ── Plots ─────────────────────────────────────────────────────────────────────
target_labels = {
    "age":       "Age (ordinal)",
    "w_novelty": "w novelty",
    "w_value":   "w value",
    "w_uncert":  "w uncertainty",
    "theta":     "θ (temperature)",
    "nll":       "Cog model NLL",
}
colors = {
    "age":       "#2196F3",
    "w_novelty": "#E91E63",
    "w_value":   "#4CAF50",
    "w_uncert":  "#FF9800",
    "theta":     "#9C27B0",
    "nll":       "#795548",
}

n_targets = len(targets)
ncols = 3
nrows = (n_targets + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
axes = np.array(axes).flatten()

for ax, (tname, tlabel) in zip(axes, target_labels.items()):
    col = f"r_{tname}"
    ax.plot(df_res["epoch"], df_res[col], color=colors[tname], lw=2)
    ax.axhline(0, color="k", lw=0.8, ls=":")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Spearman r (RSA)")
    ax.set_title(f"Latent RDM ↔ {tlabel}")

# Hide unused subplots
for ax in axes[n_targets:]:
    ax.set_visible(False)

plt.suptitle(f"RSA: IDRNN latents vs. behavioural variables\n({args.runs_dir}, seed {sweep_seed})", y=1.02)
plt.tight_layout()
out_path = os.path.join(args.out_dir, "rsa_sloutsky.png")
plt.savefig(out_path, bbox_inches="tight")
print(f"\nPlot saved to {out_path}")

# Save numeric results
csv_path = os.path.join(args.out_dir, "rsa_sloutsky.csv")
df_res.to_csv(csv_path, index=False)
print(f"Results saved to {csv_path}")

# ── Single-epoch RSA comparison: IDRNN (specificity epoch) vs Vanilla ─────────
print("\n" + "="*70)
print("SINGLE-EPOCH RSA COMPARISON: IDRNN vs Vanilla")
print("="*70)

# Determine IDRNN epoch: --epoch overrides specificity JSON
spec_json = os.path.join(args.runs_dir, "best_epoch_by_nll.json")
spec_info = {}
if os.path.exists(spec_json):
    with open(spec_json) as f:
        spec_info = json.load(f)

if args.epoch:
    idrnn_epoch = args.epoch
    print(f"IDRNN epoch (from --epoch): {idrnn_epoch}")
elif spec_info:
    idrnn_epoch = spec_info["best_epoch"]
    print(f"IDRNN epoch (by specificity): {idrnn_epoch}")
else:
    idrnn_epoch = epochs_to_eval[-1]
    print(f"IDRNN epoch (fallback): {idrnn_epoch}")

# Determine vanilla epoch/seed from best_epoch_by_loss.json
loss_json = os.path.join(args.vanilla_dir, "best_epoch_by_loss.json")
if os.path.exists(loss_json):
    with open(loss_json) as f:
        loss_info = json.load(f)
    van_epoch = args.vanilla_epoch if args.vanilla_epoch else (args.epoch if args.epoch else loss_info["best_epoch"])
    van_seed  = args.vanilla_seed  if args.vanilla_seed  else loss_info["best_seed"]
    print(f"Vanilla epoch (by loss):      {van_epoch}  seed: {van_seed}")
else:
    van_epoch = args.vanilla_epoch if args.vanilla_epoch else idrnn_epoch
    van_seed  = args.vanilla_seed  if args.vanilla_seed  else args.seeds[0]
    print(f"Vanilla epoch (fallback): {van_epoch}  seed: {van_seed}")

idrnn_seed = sweep_seed
print(f"IDRNN seed: {idrnn_seed}")

# Build IDRNN RDM at specificity-selected epoch, best seed only
print(f"\nLoading IDRNN checkpoint (epoch {idrnn_epoch}, seed {idrnn_seed})...")
m = load_model(idrnn_seed, idrnn_epoch)
if m is None:
    print("  WARNING: no IDRNN checkpoint found — skipping IDRNN comparison")
    idrnn_rdm = None
else:
    idrnn_rdm = latent_rdm_vec(extract_latents(m))
    print(f"  Loaded seed {idrnn_seed}")

# Build Vanilla RDM at loss-selected epoch, best seed only
print(f"\nLoading Vanilla checkpoint (epoch {van_epoch}, seed {van_seed})...")
m = load_vanilla_model(van_seed, van_epoch)
if m is None:
    print("  WARNING: no Vanilla checkpoint found — skipping Vanilla comparison")
    van_rdm = None
else:
    van_rdm = latent_rdm_vec(extract_vanilla_hidden(m))
    print(f"  Loaded seed {van_seed}")

# Print comparison table + permutation tests
comparison_rows = []
print(f"\n{'Target':14s}  {'IDRNN r':>8s}  {'IDRNN p':>8s}  {'Vanilla r':>10s}  {'Vanilla p':>10s}")
print("-" * 60)
for tname, trdm in targets.items():
    row = {"target": tname}
    if idrnn_rdm is not None:
        r_i, p_i, _ = permutation_rsa(idrnn_rdm, trdm, n_perm=args.perm_n)
        row["idrnn_r"] = r_i;  row["idrnn_p"] = p_i
    else:
        r_i = p_i = float("nan")
        row["idrnn_r"] = r_i;  row["idrnn_p"] = p_i

    if van_rdm is not None:
        r_v, p_v, _ = permutation_rsa(van_rdm, trdm, n_perm=args.perm_n)
        row["van_r"] = r_v;  row["van_p"] = p_v
    else:
        r_v = p_v = float("nan")
        row["van_r"] = r_v;  row["van_p"] = p_v

    print(f"{tname:14s}  {r_i:+8.3f}  {p_i:8.4f}  {r_v:+10.3f}  {p_v:10.4f}")
    comparison_rows.append(row)

df_comp = pd.DataFrame(comparison_rows)
comp_csv = os.path.join(args.out_dir, "rsa_comparison_idrnn_vs_vanilla.csv")
df_comp.to_csv(comp_csv, index=False)
print(f"\nComparison table saved to {comp_csv}")

# Bar plot: IDRNN vs Vanilla — all targets, age & w_novelty highlighted
target_order = list(targets.keys())
tlabels_short = {"age":"Age","w_novelty":"w novelty","w_value":"w value",
                 "w_uncert":"w uncert","theta":"θ","nll":"NLL"}
primary = {"age", "w_novelty"}

def sig_star(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "n.s."

x  = np.arange(len(target_order))
w  = 0.35
fig2, ax2 = plt.subplots(figsize=(9, 4))

for i, t in enumerate(target_order):
    alpha_i = 0.95 if t in primary else 0.45
    alpha_v = 0.95 if t in primary else 0.45
    r_i = df_comp.loc[df_comp.target==t,"idrnn_r"].values[0]
    r_v = df_comp.loc[df_comp.target==t,"van_r"].values[0]
    p_i = df_comp.loc[df_comp.target==t,"idrnn_p"].values[0]
    p_v = df_comp.loc[df_comp.target==t,"van_p"].values[0]

    bar_i = ax2.bar(i - w/2, r_i, w, color="#2196F3", alpha=alpha_i,
                    label="IDRNN" if i == 0 else "_")
    bar_v = ax2.bar(i + w/2, r_v, w, color="#FF9800", alpha=alpha_v,
                    label="Vanilla" if i == 0 else "_")

    ax2.text(i - w/2, r_i + 0.01, sig_star(p_i), ha="center", va="bottom", fontsize=8)
    ax2.text(i + w/2, r_v + 0.01, sig_star(p_v), ha="center", va="bottom", fontsize=8)

    # Bold x-tick for primary targets
    if t in primary:
        ax2.get_xticklabels()  # tick labels set below

ax2.axhline(0, color="k", lw=0.8)
ax2.set_xticks(x)
ticklabels = ax2.set_xticklabels([tlabels_short.get(t, t) for t in target_order])
for lbl, t in zip(ticklabels, target_order):
    if t in primary:
        lbl.set_fontweight("bold")

ax2.set_ylabel("Spearman r (RSA)")
ax2.set_title(f"RSA: IDRNN (seed {idrnn_seed}, ep {idrnn_epoch}) vs "
              f"Vanilla (seed {van_seed}, ep {van_epoch})\n"
              "(bold = primary targets; * p<.05  ** p<.01  *** p<.001)")
ax2.legend(fontsize=9)
plt.tight_layout()
bar_path = os.path.join(args.out_dir, "rsa_comparison_idrnn_vs_vanilla.png")
fig2.savefig(bar_path, bbox_inches="tight")
print(f"Bar plot saved to {bar_path}")
