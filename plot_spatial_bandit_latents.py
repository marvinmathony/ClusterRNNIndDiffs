#!/usr/bin/env python3
"""
Plot latent representations for the spatial bandit dataset.
Colors points by age_months (continuous) and decodes age from latents.

Each "observation" in the latent tensor is one (participant, round) block.
Age decoding uses GroupKFold so that all rounds of a participant end up
in the same fold (prevents data leakage).

Usage:
    python plot_spatial_bandit_latents.py --latent True --dim_reduction pca --avg False
    python plot_spatial_bandit_latents.py --latent False --dim_reduction tsne --avg True
"""

import os
import argparse
import json
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr

parser = argparse.ArgumentParser()
parser.add_argument('--latent',        type=lambda x: x.lower() == 'true', default=True)
parser.add_argument('--dim_reduction', type=str, default='pca', choices=['pca', 'tsne'])
parser.add_argument('--avg',           type=lambda x: x.lower() == 'true', default=False,
                    help="Average latents over time (True) or use last time step (False)")
args = parser.parse_args()

DATA_DIR = "data_spatial_bandit"
PLOT_DIR = "plots_spatial_bandit"
os.makedirs(PLOT_DIR, exist_ok=True)

model_type = "latentmodel" if args.latent else "vanilla"
model_name = "IDRNN"       if args.latent else "Vanilla"

print(f"\n{'='*60}")
print(f"PLOTTING LATENTS — SPATIAL BANDIT  |  {model_name}")
print(f"Dim reduction: {args.dim_reduction}  |  Avg: {args.avg}")
print(f"{'='*60}\n")

# ---------------------------------------------------------------------------
# Load latent tensor and sequence-level metadata
# df_test has one row per (participant, round) = one row per sequence
# ---------------------------------------------------------------------------
latent_path = os.path.join(DATA_DIR, f"latents_tensor{model_type}.pt")
if not os.path.exists(latent_path):
    raise FileNotFoundError(
        f"Latent tensor not found at {latent_path}. Run testing_script.py first.")

latent_tensor = torch.load(latent_path, map_location='cpu')
print(f"Loaded latent tensor: {latent_tensor.shape}  (n_sequences, T, z_dim)")

df_test = pd.read_csv(os.path.join(DATA_DIR, "df_test.csv"))

# One row per (participant, round) sequence
age_months  = df_test["age_months"].to_numpy(dtype=float)   # (n_sequences,)
participant  = df_test["subid"].to_numpy()                    # group labels for CV

n_seq        = len(age_months)
n_part       = len(np.unique(participant))
print(f"Sequences:    {n_seq}  ({n_part} unique participants)")
print(f"Age range:    {age_months.min():.1f} – {age_months.max():.1f} months")

# ---------------------------------------------------------------------------
# Extract per-sequence latent vectors (mean over time or last timestep)
# ---------------------------------------------------------------------------
if args.avg:
    latents = latent_tensor.mean(dim=1) if latent_tensor.dim() == 3 else latent_tensor
    title_suffix = "average"
else:
    latents = latent_tensor[:, -1, :] if latent_tensor.dim() == 3 else latent_tensor
    title_suffix = "last"

latents_np = latents.cpu().numpy()   # (n_sequences, z_dim)
z_dim = latents_np.shape[1]
print(f"Latents shape: {latents_np.shape}")

# ---------------------------------------------------------------------------
# Dimensionality reduction (sequence level)
# ---------------------------------------------------------------------------
if z_dim == 1:
    rng = np.random.default_rng(0)
    latents_2d = np.column_stack([latents_np[:, 0],
                                  rng.standard_normal(n_seq) * 0.05])
elif args.dim_reduction == "pca":
    reducer = PCA(n_components=min(2, z_dim), random_state=42)
    latents_2d = reducer.fit_transform(latents_np)
    print(f"PCA explained var: {reducer.explained_variance_ratio_}")
else:  # tsne
    reducer = TSNE(n_components=2, perplexity=min(30, n_seq - 1), random_state=42)
    latents_2d = reducer.fit_transform(latents_np)

# ---------------------------------------------------------------------------
# Plot 1: 2D scatter coloured by age_months (sequence level)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))
norm = Normalize(vmin=age_months.min(), vmax=age_months.max())
sc = ax.scatter(latents_2d[:, 0], latents_2d[:, 1],
                c=age_months, cmap='viridis', norm=norm,
                alpha=0.6, edgecolors='k', linewidth=0.3, s=30)
plt.colorbar(sc, ax=ax, label="Age (months)")
ax.set_xlabel(f"{args.dim_reduction.upper()} Component 1")
ax.set_ylabel(f"{args.dim_reduction.upper()} Component 2")
ax.set_title(f"{model_name} — Spatial Bandit Latents\n"
             f"{title_suffix.capitalize()} latents, {args.dim_reduction.upper()}\n"
             f"({n_seq} sequences from {n_part} participants)")
ax.grid(True, alpha=0.3)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
fname = os.path.join(PLOT_DIR, f"latents_{model_type}_{args.dim_reduction}_{title_suffix}.png")
plt.savefig(fname, dpi=150, bbox_inches='tight')
plt.close()
print(f"Scatter plot saved: {fname}")

# ---------------------------------------------------------------------------
# Plot 2: Individual latent dims / PCs vs age_months
# ---------------------------------------------------------------------------
def plot_latent_dims_vs_age(latents_np, age_months, participant, plot_dir,
                             model_name, use_pca, title_suffix):
    scaler = StandardScaler()
    lat = scaler.fit_transform(latents_np)
    if use_pca and z_dim > 1:
        pca = PCA(n_components=min(6, z_dim))
        lat = pca.fit_transform(lat)
        ev = pca.explained_variance_ratio_
        labels = [f"PC{i+1} ({ev[i]*100:.1f}%)" for i in range(lat.shape[1])]
        suffix = "pca"
    else:
        labels = [f"z{i+1}" for i in range(lat.shape[1])]
        suffix = "raw"

    n_comp = lat.shape[1]
    n_cols = min(3, n_comp)
    n_rows = (n_comp + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows), squeeze=False)
    axes = axes.flatten()

    for i in range(n_comp):
        ax = axes[i]
        vals = lat[:, i]
        r_p, p_p = pearsonr(age_months, vals)
        sig = "***" if p_p < 0.001 else "**" if p_p < 0.01 else "*" if p_p < 0.05 else "n.s."
        sc = ax.scatter(age_months, vals, c=age_months, cmap='viridis',
                        alpha=0.5, edgecolors='k', linewidth=0.2, s=20)
        m, b = np.polyfit(age_months, vals, 1)
        x_line = np.linspace(age_months.min(), age_months.max(), 100)
        ax.plot(x_line, m*x_line + b, 'r--', linewidth=1.5)
        ax.set_xlabel("Age (months)")
        ax.set_ylabel(labels[i])
        ax.set_title(f"{labels[i]}\nr={r_p:.3f}, p={p_p:.3f} ({sig})")
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    for j in range(n_comp, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"{model_name} — {'PCA Components' if use_pca else 'Raw Latent Dims'} "
                 f"vs Age\nSpatial Bandit, {title_suffix} latents "
                 f"({n_seq} sequences)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    out = os.path.join(plot_dir,
                       f"latent_vs_age_{model_name.lower()}_{suffix}_{title_suffix}.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


plot_latent_dims_vs_age(latents_np, age_months, participant, PLOT_DIR,
                         model_name, use_pca=False, title_suffix=title_suffix)
if z_dim > 1:
    plot_latent_dims_vs_age(latents_np, age_months, participant, PLOT_DIR,
                             model_name, use_pca=True, title_suffix=title_suffix)

# ---------------------------------------------------------------------------
# Age decoding: Ridge regression with GroupKFold (split by participant)
# Each fold keeps all rounds of a participant together to prevent leakage.
# ---------------------------------------------------------------------------
def decode_age_cv(latents_np, age_months, participant, model_name, title_suffix, plot_dir,
                  n_splits=5, seed=42):
    scaler = StandardScaler()
    X = scaler.fit_transform(latents_np)
    y = age_months
    groups = participant

    gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])

    y_pred_all = np.zeros_like(y)
    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        ridge.fit(X[train_idx], y[train_idx])
        y_pred_all[test_idx] = ridge.predict(X[test_idx])

    mae  = np.mean(np.abs(y - y_pred_all))
    r2   = 1 - np.sum((y - y_pred_all)**2) / np.sum((y - np.mean(y))**2)
    r_p, p_p = pearsonr(y, y_pred_all)
    sig  = "***" if p_p < 0.001 else "**" if p_p < 0.01 else "*" if p_p < 0.05 else "n.s."

    print(f"\n[Age decoding (sequence-level) — {model_name}, {title_suffix} latents]")
    print(f"  MAE:       {mae:.2f} months")
    print(f"  R²:        {r2:.3f}")
    print(f"  Pearson r: {r_p:.3f}  (p={p_p:.4f})")
    print(f"  CV:        GroupKFold(n_splits={gkf.n_splits}), split by participant")

    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(y, y_pred_all, c=y, cmap='viridis', alpha=0.6,
                    edgecolors='k', linewidth=0.3, s=30)
    plt.colorbar(sc, ax=ax, label="Actual age (months)")
    lo, hi = min(y.min(), y_pred_all.min()), max(y.max(), y_pred_all.max())
    ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1.5, label='Perfect prediction')
    ax.set_xlabel("Actual age (months)")
    ax.set_ylabel("Predicted age (months)")
    ax.set_title(f"{model_name} — Age Decoding (sequence level)\n"
                 f"MAE={mae:.1f} months, R²={r2:.3f}, r={r_p:.3f} ({sig})\n"
                 f"GroupKFold CV (split by participant)")
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    fname = os.path.join(plot_dir,
                         f"age_decoding_{model_name.lower()}_{title_suffix}.png")
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")

    return {"mae": mae, "r2": r2, "pearson_r": r_p, "pearson_p": p_p,
            "n_sequences": int(len(y)), "n_participants": int(len(np.unique(groups))),
            "model": model_name, "title_suffix": title_suffix}


# Sequence-level decoding (GroupKFold by participant)
decode_results = decode_age_cv(
    latents_np, age_months, participant, model_name, title_suffix, PLOT_DIR)

# Also do participant-level decoding: average latents over rounds per participant
# then run a regular KFold (now observations are independent)
def decode_age_participant_level(latents_np, age_months, participant,
                                  model_name, title_suffix, plot_dir):
    from sklearn.model_selection import KFold
    from scipy.stats import pearsonr as pearsonr_

    unique_participants = np.unique(participant)
    lat_part  = np.array([latents_np[participant == p].mean(axis=0) for p in unique_participants])
    age_part  = np.array([age_months[participant == p].mean()        for p in unique_participants])

    scaler = StandardScaler()
    X = scaler.fit_transform(lat_part)
    y = age_part

    kf    = KFold(n_splits=min(5, len(y)), shuffle=True, random_state=42)
    ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])

    y_pred = np.zeros_like(y)
    for tr, te in kf.split(X):
        ridge.fit(X[tr], y[tr])
        y_pred[te] = ridge.predict(X[te])

    mae = np.mean(np.abs(y - y_pred))
    r2  = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
    r_p, p_p = pearsonr_(y, y_pred)
    sig = "***" if p_p < 0.001 else "**" if p_p < 0.01 else "*" if p_p < 0.05 else "n.s."

    print(f"\n[Age decoding (participant-level avg) — {model_name}, {title_suffix} latents]")
    print(f"  N participants: {len(y)}")
    print(f"  MAE:       {mae:.2f} months")
    print(f"  R²:        {r2:.3f}")
    print(f"  Pearson r: {r_p:.3f}  (p={p_p:.4f})")

    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(y, y_pred, c=y, cmap='viridis', alpha=0.8,
                    edgecolors='k', linewidth=0.4, s=60)
    plt.colorbar(sc, ax=ax, label="Actual age (months)")
    lo, hi = min(y.min(), y_pred.min()), max(y.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1.5)
    ax.set_xlabel("Actual age (months)")
    ax.set_ylabel("Predicted age (months)")
    ax.set_title(f"{model_name} — Age Decoding (participant avg)\n"
                 f"MAE={mae:.1f} months, R²={r2:.3f}, r={r_p:.3f} ({sig})")
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    fname = os.path.join(plot_dir,
                         f"age_decoding_{model_name.lower()}_{title_suffix}_participant_avg.png")
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")

    return {"mae": mae, "r2": r2, "pearson_r": r_p, "pearson_p": p_p,
            "n_participants": int(len(y)), "level": "participant_avg",
            "model": model_name, "title_suffix": title_suffix}


decode_results_part = decode_age_participant_level(
    latents_np, age_months, participant, model_name, title_suffix, PLOT_DIR)

# Save both decoding results
combined = {"sequence_level": decode_results, "participant_level": decode_results_part}
decode_json = os.path.join(DATA_DIR, f"age_decoding_{model_type}_{title_suffix}.json")
with open(decode_json, "w") as f:
    json.dump(combined, f, indent=2)
print(f"\nDecoding results saved: {decode_json}")

# ---------------------------------------------------------------------------
# Likelihood comparison (when both models have been tested)
# ---------------------------------------------------------------------------
def plot_likelihood_comparison(data_dir, plot_dir):
    from scipy.stats import ttest_rel, sem
    try:
        idrnn_df   = pd.read_csv(os.path.join(data_dir, "rnn_resultslatentmodel.csv"))
        vanilla_df = pd.read_csv(os.path.join(data_dir, "rnn_resultsvanilla.csv"))
    except FileNotFoundError as e:
        print(f"Skipping likelihood comparison: {e}")
        return

    idrnn_ll   = idrnn_df["normalized_likelihood"].values
    vanilla_ll = vanilla_df["normalized_likelihood"].values
    models     = ["IDRNN", "Vanilla RNN"]
    all_lls    = [idrnn_ll, vanilla_ll]
    colors_bar = ['#2a82c2', '#e1861f']
    means      = [np.mean(ll) for ll in all_lls]
    sems_vals  = [sem(ll)     for ll in all_lls]
    x_pos      = np.arange(len(models))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.bar(x_pos, means, color=colors_bar, alpha=0.85, width=0.6)
    rng = np.random.default_rng(42)
    for i, ll in enumerate(all_lls):
        ax.scatter(x_pos[i] + rng.uniform(-0.15, 0.15, size=len(ll)),
                   ll, alpha=0.5, c='black', s=15, zorder=10)
    ax.errorbar(x_pos, means, yerr=sems_vals, fmt='none', capsize=5,
                ecolor='black', elinewidth=1, zorder=11)
    _, p = ttest_rel(idrnn_ll, vanilla_ll)
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    ymax = max(m + s for m, s in zip(means, sems_vals))
    h = ymax * 0.02
    ax.plot([0, 0, 1, 1], [ymax+h, ymax+2*h, ymax+2*h, ymax+h], lw=1.5, c='black')
    ax.text(0.5, ymax+2*h, sig, ha='center', va='bottom', fontsize=12)
    ax.set_ylabel("Mean NLL per Sequence (lower = better)")
    ax.set_title("Model Likelihood — Spatial Bandit")
    ax.set_xticks(x_pos); ax.set_xticklabels(models)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    fname = os.path.join(plot_dir, "likelihood_comparison.png")
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Likelihood comparison saved: {fname}")


if args.latent:
    plot_likelihood_comparison(DATA_DIR, PLOT_DIR)

print("\nDone.")
