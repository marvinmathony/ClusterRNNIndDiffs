#!/usr/bin/env python3
"""
Plot latent representations for Sloutsky (human) data.
Colors points by age group (children vs adults) based on game version.
"""

import os
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA
from scipy.stats import ttest_ind, ttest_rel, sem, pointbiserialr

# Parse arguments
parser = argparse.ArgumentParser(description="Plot latent representations for Sloutsky data")
parser.add_argument('--latent', type=lambda x: x.lower() == 'true', default=True,
                    help="latent (IDRNN) or vanilla modeling")
parser.add_argument('--dim_reduction', type=str, default='pca', choices=['pca', 'tsne', 'mds'],
                    help="Dimensionality reduction method")
parser.add_argument('--avg', type=lambda x: x.lower() == 'true', default=True,
                    help="Use average latents (True) or last latent (False)")
args = parser.parse_args()

# Configuration
DATA_DIR = "data_sloutsky"
PLOT_DIR = "plots_sloutsky"
os.makedirs(PLOT_DIR, exist_ok=True)

# Model type
model_type = "latentmodel" if args.latent else "vanilla"
model_name = "IDRNN" if args.latent else "Vanilla"

print(f"\n{'='*60}")
print(f"PLOTTING LATENTS FOR SLOUTSKY DATA")
print(f"Model: {model_name}")
print(f"Dim reduction: {args.dim_reduction}")
print(f"Average latents: {args.avg}")
print(f"{'='*60}\n")

# Load latent tensor
latent_path = f"{DATA_DIR}/latents_tensor{model_type}.pt"
if not os.path.exists(latent_path):
    raise FileNotFoundError(f"Latent tensor not found at {latent_path}. Run testing_script.py first.")

latent_tensor = torch.load(latent_path, map_location='cpu')
print(f"Loaded latent tensor with shape: {latent_tensor.shape}")

# Load test dataframe for group information
df_test = pd.read_csv(f"{DATA_DIR}/df_test.csv")

# Color mapping: game version -> age group
# Coin CollectorV6 and V5 = children (0), Coin Collector = adults (1)
color_map = {
    "Coin CollectorV6": 0,
    "Coin CollectorV5": 0,
    "Coin Collector": 1
}

# Get unique subjects and their groups
df_unique = df_test.drop_duplicates(subset="subid")
group_vals = df_unique["game"].values
colors = [color_map.get(val, 0) for val in group_vals]

print(f"Number of participants: {len(colors)}")
print(f"Children (0): {colors.count(0)}, Adults (1): {colors.count(1)}")

# Prepare latents for dimensionality reduction
if args.avg:
    # Average over time dimension
    if latent_tensor.dim() == 3:
        latents = latent_tensor.mean(dim=1)  # (B, T, z_dim) -> (B, z_dim)
    else:
        latents = latent_tensor
    title_suffix = "average"
else:
    # Use last time step
    if latent_tensor.dim() == 3:
        latents = latent_tensor[:, -1, :]  # (B, T, z_dim) -> (B, z_dim)
    else:
        latents = latent_tensor
    title_suffix = "last"

latents_np = latents.cpu().numpy()
print(f"Latents shape after processing: {latents_np.shape}")

# Apply dimensionality reduction
z_dim = latents_np.shape[1]

# Handle case where z_dim is too small for 2D reduction
if z_dim == 1:
    print(f"Latent dimension is 1 - skipping {args.dim_reduction.upper()} (cannot reduce to 2D)")
    # Create pseudo-2D by adding jittered y-axis
    latents_reduced = np.column_stack([latents_np[:, 0], np.random.randn(len(latents_np)) * 0.1])
    skip_2d_reduction = True
elif args.dim_reduction == "pca":
    n_components = min(2, z_dim)
    reducer = PCA(n_components=n_components)
    latents_reduced = reducer.fit_transform(latents_np)
    explained_var = reducer.explained_variance_ratio_
    if n_components == 2:
        print(f"PCA explained variance: {explained_var[0]:.2%}, {explained_var[1]:.2%}")
    else:
        print(f"PCA explained variance: {explained_var[0]:.2%}")
        # Pad to 2D for plotting
        latents_reduced = np.column_stack([latents_reduced, np.random.randn(len(latents_reduced)) * 0.1])
    skip_2d_reduction = False
elif args.dim_reduction == "tsne":
    n_components = min(2, z_dim)
    reducer = TSNE(n_components=n_components, perplexity=min(30, len(colors)-1), random_state=42)
    latents_reduced = reducer.fit_transform(latents_np)
    if n_components == 1:
        latents_reduced = np.column_stack([latents_reduced, np.random.randn(len(latents_reduced)) * 0.1])
    skip_2d_reduction = False
elif args.dim_reduction == "mds":
    n_components = min(2, z_dim)
    reducer = MDS(n_components=n_components, random_state=42, dissimilarity="euclidean", n_init=4, max_iter=300)
    latents_reduced = reducer.fit_transform(latents_np)
    if n_components == 1:
        latents_reduced = np.column_stack([latents_reduced, np.random.randn(len(latents_reduced)) * 0.1])
    skip_2d_reduction = False

# Create plot
fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(latents_reduced[:, 0], latents_reduced[:, 1],
                     c=colors, cmap='coolwarm', alpha=0.7, edgecolors='k', linewidth=0.5)

# Add colorbar with labels
cbar = plt.colorbar(scatter, ax=ax, ticks=[0, 1])
cbar.ax.set_yticklabels(['Children', 'Adults'])

# Labels and title
if z_dim == 1:
    ax.set_xlabel("Latent Value (z)")
    ax.set_ylabel("Jitter (for visualization)")
    ax.set_title(f"{model_name} Latent Representations (Sloutsky)\n{title_suffix.capitalize()} latents, 1D")
else:
    ax.set_xlabel(f"{args.dim_reduction.upper()} Component 1")
    ax.set_ylabel(f"{args.dim_reduction.upper()} Component 2")
    ax.set_title(f"{model_name} Latent Representations (Sloutsky)\n{title_suffix.capitalize()} latents, {args.dim_reduction.upper()}")
ax.grid(True, alpha=0.3)

# Save plot
plot_filename = f"{PLOT_DIR}/latents_{model_type}_{args.dim_reduction}_{title_suffix}.png"
plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
plt.close()

print(f"\nPlot saved to: {plot_filename}")

# Also create a 1D plot if latent dimension is 1
if latents_np.shape[1] == 1:
    fig, ax = plt.subplots(figsize=(10, 3))
    y_jitter = 0.02 * np.random.randn(len(colors))
    scatter = ax.scatter(latents_np[:, 0], y_jitter, c=colors, cmap='coolwarm',
                        alpha=0.7, edgecolors='k', linewidth=0.5)
    cbar = plt.colorbar(scatter, ax=ax, ticks=[0, 1])
    cbar.ax.set_yticklabels(['Children', 'Adults'])
    ax.set_xlabel("Latent Value")
    ax.set_yticks([])
    ax.set_title(f"{model_name} 1D Latent Representation (Sloutsky)")

    plot_filename_1d = f"{PLOT_DIR}/latents_{model_type}_1d.png"
    plt.savefig(plot_filename_1d, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"1D plot saved to: {plot_filename_1d}")

# ============================================================================
# NEW PLOTS: Individual PCs vs Age Group + Likelihood Comparison
# ============================================================================

def plot_pc_vs_age_group(latents_np, colors, group_labels, plot_dir, model_name, use_pca=True):
    """
    Plot individual latent dimensions or principal components vs age group (children/adults).
    Similar to alpha vs z plots in analyze_synthetic_multi_dataset.py.

    Args:
        latents_np: Latent representations (n_participants, z_dim)
        colors: Age group labels (0=children, 1=adults)
        group_labels: List of group names for labeling
        plot_dir: Directory to save plots
        model_name: Name of the model (IDRNN or Vanilla)
        use_pca: If True, apply PCA; if False, plot raw latent dimensions
    """

    colors = np.array(colors)
    z_dim = latents_np.shape[1]

    if use_pca and z_dim > 1:
        # Apply PCA
        pca = PCA(n_components=min(3, z_dim))
        latents_transformed = pca.fit_transform(latents_np)
        explained_var = pca.explained_variance_ratio_
        n_components = latents_transformed.shape[1]
        component_labels = [f'PC{i+1} ({explained_var[i]*100:.1f}%)' for i in range(n_components)]
        suffix = "pca"
    else:
        # Use raw latent dimensions
        latents_transformed = latents_np
        n_components = z_dim
        component_labels = [f'z{i+1}' for i in range(n_components)]
        suffix = "raw"

    # Create figure with subplots for each component
    n_cols = min(3, n_components)
    n_rows = (n_components + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows), squeeze=False)
    axes = axes.flatten()

    for i in range(n_components):
        ax = axes[i]
        pc_values = latents_transformed[:, i]

        # Separate by age group
        children_vals = pc_values[colors == 0]
        adults_vals = pc_values[colors == 1]

        # Add jitter for visualization
        jitter_children = np.random.uniform(-0.15, 0.15, size=len(children_vals))
        jitter_adults = np.random.uniform(-0.15, 0.15, size=len(adults_vals))

        # Scatter plot with jitter
        ax.scatter(jitter_children, children_vals, c='#1f77b4', alpha=0.7,
                   label=f'Children (n={len(children_vals)})', edgecolors='k', linewidths=0.5)
        ax.scatter(1 + jitter_adults, adults_vals, c='#d62728', alpha=0.7,
                   label=f'Adults (n={len(adults_vals)})', edgecolors='k', linewidths=0.5)

        # Add means with error bars
        mean_children = np.mean(children_vals)
        mean_adults = np.mean(adults_vals)
        sem_children = np.std(children_vals) / np.sqrt(len(children_vals))
        sem_adults = np.std(adults_vals) / np.sqrt(len(adults_vals))

        ax.errorbar([0], [mean_children], yerr=[sem_children], fmt='s', color='#1f77b4',
                   markersize=12, capsize=5, capthick=2, markeredgecolor='black', markeredgewidth=1.5)
        ax.errorbar([1], [mean_adults], yerr=[sem_adults], fmt='s', color='#d62728',
                   markersize=12, capsize=5, capthick=2, markeredgecolor='black', markeredgewidth=1.5)

        # Statistical test
        t_stat, p_val = ttest_ind(children_vals, adults_vals)
        r_pb, p_pb = pointbiserialr(colors, pc_values)

        # Significance annotation
        sig_str = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."

        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Children', 'Adults'])
        ax.set_ylabel(component_labels[i])
        ax.set_title(f'{component_labels[i]}\nt={t_stat:.2f}, p={p_val:.3f} ({sig_str})\nr_pb={r_pb:.3f}')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Hide unused subplots
    for i in range(n_components, len(axes)):
        axes[i].set_visible(False)

    title_type = "PCA Components" if use_pca else "Raw Latent Dimensions"
    fig.suptitle(f'{model_name}: {title_type} vs Age Group', fontsize=14, fontweight='bold')
    plt.tight_layout()

    plot_filename = f"{plot_dir}/latent_vs_age_{model_name.lower()}_{suffix}_{title_suffix}.png"
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved {title_type} vs Age Group plot to: {plot_filename}")


def add_sig_bar(ax, x1, x2, y, h, p_val):
    """Draw a significance bracket between x1 and x2."""
    sig_str = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c='black')
    ax.text((x1 + x2) / 2, y + h, sig_str, ha='center', va='bottom', fontsize=10)


def plot_likelihood_comparison(data_dir, plot_dir):
    """
    Plot likelihood comparison between IDRNN, Vanilla RNN, and the cognitive
    model (EM-fitted, marginal NLL via logsumexp over group prior).
    Lower NLL = better.
    """

    # Load RNN results
    try:
        rnn_idrnn_df  = pd.read_csv(f"{data_dir}/rnn_resultslatentmodel.csv")
        rnn_vanilla_df = pd.read_csv(f"{data_dir}/rnn_resultsvanilla.csv")
    except FileNotFoundError as e:
        print(f"Warning: Could not load RNN results files: {e}")
        return

    idrnn_ll  = rnn_idrnn_df["normalized_likelihood"].values
    vanilla_ll = rnn_vanilla_df["normalized_likelihood"].values

    # Load cognitive model marginal NLL (optional)
    cog_path = f"{data_dir}/cog_model_results.csv"
    cog_ll = None
    if os.path.exists(cog_path):
        cog_df = pd.read_csv(cog_path)
        cog_ll = cog_df["normalized_likelihood"].values
    else:
        print(f"Note: {cog_path} not found — cognitive model bar will be omitted.")

    # Build model list dynamically
    model_data = [
        ("IDRNN",       idrnn_ll,  '#2a82c2'),
        ("Vanilla RNN", vanilla_ll, '#e1861f'),
    ]
    if cog_ll is not None:
        model_data.append(("Cog. Model\n(EM)", cog_ll, '#3ba83b'))

    models      = [m[0] for m in model_data]
    all_lls     = [m[1] for m in model_data]
    bar_colors  = [m[2] for m in model_data]
    means = [np.mean(ll) for ll in all_lls]
    sems_vals  = [sem(ll)    for ll in all_lls]

    x_pos = np.arange(len(models))

    fig, ax = plt.subplots(figsize=(4 + 2 * len(models), 6))

    ax.bar(x_pos, means, color=bar_colors, alpha=0.85, width=0.6)

    # Individual participant points with jitter
    rng = np.random.default_rng(42)
    for i, ll in enumerate(all_lls):
        jitter = rng.uniform(-0.15, 0.15, size=len(ll))
        ax.scatter(x_pos[i] + jitter, ll, alpha=0.5, c='black', s=20,
                   zorder=10, edgecolors='none')

    # Error bars
    ax.errorbar(x_pos, means, yerr=sems_vals, fmt='none', capsize=5, capthick=1,
                ecolor='black', elinewidth=1, zorder=11)

    # Significance bars: IDRNN vs Vanilla, and IDRNN vs Cog Model
    ymax = max(m + s for m, s in zip(means, sems_vals))
    h = ymax * 0.02

    _, p_iv = ttest_rel(idrnn_ll, vanilla_ll)
    add_sig_bar(ax, 0, 1, ymax + h, h, p_iv)

    if cog_ll is not None and len(cog_ll) == len(idrnn_ll):
        _, p_ic = ttest_rel(idrnn_ll, cog_ll)
        add_sig_bar(ax, 0, 2, ymax + 4 * h, h, p_ic)

    ax.set_ylabel('Mean Negative Log Likelihood per Participant', fontsize=12)
    ax.set_title('Model Likelihood Comparison (Sloutsky Data)\nLower is Better',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Stats text box
    stats_lines = [f"{name}: {m:.2f} ± {s:.2f}"
                   for name, m, s in zip(models, means, sems_vals)]
    ax.text(0.98, 0.98, "\n".join(stats_lines), transform=ax.transAxes,
            fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plot_filename = f"{plot_dir}/likelihood_comparison.png"
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved likelihood comparison plot to: {plot_filename}")
    for name, m, s in zip(models, means, sems_vals):
        print(f"  {name:20s}: {m:.2f} ± {s:.2f}")


# Generate the new plots
print("\n" + "="*60)
print("GENERATING ADDITIONAL PLOTS")
print("="*60)

# Plot 1a: Raw latent dimensions vs Age Group
print("\nPlotting raw latent dimensions vs age group...")
plot_pc_vs_age_group(latents_np, colors, ['Children', 'Adults'], PLOT_DIR, model_name, use_pca=False)

# Plot 1b: PCA components vs Age Group (only if z_dim > 1, otherwise identical to raw)
if latents_np.shape[1] > 1:
    print("\nPlotting PCA components vs age group...")
    plot_pc_vs_age_group(latents_np, colors, ['Children', 'Adults'], PLOT_DIR, model_name, use_pca=True)

# Plot 2: Likelihood comparison (only needs to be done once, not per model type)
if args.latent:  # Only plot once when running IDRNN
    print("\nPlotting likelihood comparison...")
    plot_likelihood_comparison(DATA_DIR, PLOT_DIR)

print("\nDone!")
