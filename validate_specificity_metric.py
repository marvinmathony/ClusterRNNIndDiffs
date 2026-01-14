#!/usr/bin/env python3
"""
Validate the reconstruction specificity metric by plotting its correlation
with ground truth RSA alignment across all datasets.

This script answers: "Can specificity be used to select the best IDRNN checkpoint?"

Uses epoch-level mean specificity (from precomputed JSON) vs mean GT RSA correlation.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import pdist
import argparse
import json


def load_ground_truth_rsa(data_dir):
    """Load ground truth parameters and compute RSA vector."""
    test_params = pd.read_csv(f"{data_dir}/true_test_parameter_values.csv")
    params = test_params["alphaP_list"].values.reshape(-1, 1)
    dist_vec = pdist(params, metric="euclidean")
    return dist_vec


def collect_epoch_level_data(dataset_id, min_epoch=1000):
    """
    Collect mean specificity and mean GT RSA correlation per epoch.
    Uses precomputed specificity JSON.

    Returns list of dicts with: dataset, epoch, specificity, gt_rsa_corr, gt_rsa_std
    """
    base_dir = f"runs_dataset{dataset_id}"
    data_dir = f"data_dataset{dataset_id}"
    seeds = [12, 50, 76, 100, 142]

    if not os.path.exists(base_dir) or not os.path.exists(data_dir):
        print(f"Dataset {dataset_id} not found, skipping...")
        return []

    # Load ground truth RSA
    try:
        gt_rsa = load_ground_truth_rsa(data_dir)
    except Exception as e:
        print(f"Failed to load ground truth for dataset {dataset_id}: {e}")
        return []

    # Load specificity data
    spec_path = os.path.join(base_dir, "reconstruction_specificity.json")
    if not os.path.exists(spec_path):
        print(f"No specificity data for dataset {dataset_id}")
        return []

    with open(spec_path, 'r') as f:
        spec_data = json.load(f)

    epochs = spec_data['epochs']
    epochs = [e for e in epochs if e >= min_epoch]

    results = []
    for epoch in epochs:
        epoch_str = str(epoch)
        if epoch_str not in spec_data.get('per_epoch', {}):
            continue

        specificity = spec_data['per_epoch'][epoch_str]['mean_specificity']
        specificity_std = spec_data['per_epoch'][epoch_str].get('std_specificity', 0)

        # Compute mean GT correlation across seeds
        seed_corrs = []
        for seed in seeds:
            rsa_path = os.path.join(base_dir, f"seed_{seed}", "rsa", f"epoch_{epoch:04d}.npy")
            if os.path.exists(rsa_path):
                model_rsa = np.load(rsa_path)
                r = np.corrcoef(gt_rsa, model_rsa)[0, 1]
                seed_corrs.append(r)

        if seed_corrs:
            results.append({
                'dataset': dataset_id,
                'epoch': epoch,
                'specificity': specificity,
                'specificity_std': specificity_std,
                'gt_rsa_corr': np.mean(seed_corrs),
                'gt_rsa_std': np.std(seed_corrs),
                'n_seeds': len(seed_corrs)
            })

    return results


def plot_specificity_vs_gt_correlation(all_results, output_path="plots/specificity_validation.png"):
    """
    Create validation plot showing specificity vs GT RSA correlation.
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    if not all_results:
        print("No valid results to plot!")
        return

    df = pd.DataFrame(all_results)
    datasets = sorted(df['dataset'].unique())
    n_datasets = len(datasets)

    if n_datasets == 0:
        print("No datasets found!")
        return

    # Create figure
    fig, axes = plt.subplots(1, n_datasets + 1, figsize=(4 * (n_datasets + 1), 4))
    if n_datasets == 1:
        axes = [axes, axes]  # Handle single dataset case

    # Colors for epochs (gradient from light to dark)
    cmap = plt.cm.viridis

    all_specificity = []
    all_gt_corr = []
    dataset_correlations = {}

    # Per-dataset plots
    for idx, dataset_id in enumerate(datasets):
        ax = axes[idx]
        subset = df[df['dataset'] == dataset_id].sort_values('epoch')

        # Normalize epochs for coloring
        epochs = subset['epoch'].values
        epoch_norm = (epochs - epochs.min()) / (epochs.max() - epochs.min() + 1e-8)

        # Scatter with error bars
        for i, (_, row) in enumerate(subset.iterrows()):
            color = cmap(epoch_norm[i])
            ax.errorbar(row['specificity'], row['gt_rsa_corr'],
                       xerr=row['specificity_std'], yerr=row['gt_rsa_std'],
                       fmt='o', color=color, markersize=8, alpha=0.7,
                       capsize=3, elinewidth=1)

        # Compute correlation
        r, p = pearsonr(subset['specificity'], subset['gt_rsa_corr'])
        dataset_correlations[dataset_id] = {'r': r, 'p': p, 'n': len(subset)}

        ax.set_xlabel('Reconstruction Specificity')
        ax.set_ylabel('GT RSA Correlation')
        ax.set_title(f'Dataset {dataset_id}\nr={r:.3f} (p={p:.4f})')

        # Add trend line
        z = np.polyfit(subset['specificity'], subset['gt_rsa_corr'], 1)
        p_line = np.poly1d(z)
        x_line = np.linspace(subset['specificity'].min(), subset['specificity'].max(), 100)
        ax.plot(x_line, p_line(x_line), 'k--', alpha=0.5, linewidth=2)

        # Add colorbar to show epoch progression
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(epochs.min(), epochs.max()))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Epoch', fontsize=8)

        all_specificity.extend(subset['specificity'].tolist())
        all_gt_corr.extend(subset['gt_rsa_corr'].tolist())

    # Combined plot
    ax = axes[-1]
    colors = plt.cm.tab10(np.linspace(0, 1, n_datasets))
    for i, dataset_id in enumerate(datasets):
        subset = df[df['dataset'] == dataset_id]
        ax.scatter(subset['specificity'], subset['gt_rsa_corr'],
                  alpha=0.7, s=50, c=[colors[i]], label=f'Dataset {dataset_id}')

    r_all, p_all = pearsonr(all_specificity, all_gt_corr)
    rho_all, p_rho_all = spearmanr(all_specificity, all_gt_corr)

    ax.set_xlabel('Reconstruction Specificity')
    ax.set_ylabel('GT RSA Correlation')
    ax.set_title(f'All Datasets Combined\nr={r_all:.3f} (p={p_all:.2e})')
    ax.legend(fontsize=8, loc='lower right')

    # Add trend line
    z = np.polyfit(all_specificity, all_gt_corr, 1)
    p_line = np.poly1d(z)
    x_line = np.linspace(min(all_specificity), max(all_specificity), 100)
    ax.plot(x_line, p_line(x_line), 'k--', alpha=0.5, linewidth=2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved validation plot to {output_path}")

    # Print summary statistics
    print("\n" + "="*80)
    print("VALIDATION SUMMARY: Specificity vs Ground Truth RSA Correlation")
    print("="*80)

    for dataset_id in datasets:
        stats = dataset_correlations[dataset_id]
        print(f"Dataset {dataset_id}: Pearson r = {stats['r']:.3f} (p = {stats['p']:.4f}), N = {stats['n']} epochs")

    print(f"\nCombined:    Pearson r = {r_all:.3f} (p = {p_all:.2e}), N = {len(df)}")
    print(f"             Spearman rho = {rho_all:.3f} (p = {p_rho_all:.2e})")

    # Checkpoint selection analysis
    print("\n" + "-"*80)
    print("EPOCH SELECTION ANALYSIS (using epoch-level means)")
    print("-"*80)

    for dataset_id in datasets:
        subset = df[df['dataset'] == dataset_id]

        # Best by specificity
        best_by_spec = subset.loc[subset['specificity'].idxmax()]

        # Best by GT (oracle)
        best_by_gt = subset.loc[subset['gt_rsa_corr'].idxmax()]

        print(f"\nDataset {dataset_id}:")
        print(f"  Oracle (best GT):     epoch {int(best_by_gt['epoch'])}, GT={best_by_gt['gt_rsa_corr']:.4f}")
        print(f"  By specificity:       epoch {int(best_by_spec['epoch'])}, GT={best_by_spec['gt_rsa_corr']:.4f}")
        print(f"  Gap: {best_by_gt['gt_rsa_corr'] - best_by_spec['gt_rsa_corr']:.4f}")

    return dataset_correlations


def main():
    parser = argparse.ArgumentParser(description="Validate specificity metric against GT RSA")
    parser.add_argument('--datasets', type=int, nargs='+', default=[0, 1, 2, 3, 4],
                        help="Dataset IDs to analyze")
    parser.add_argument('--min_epoch', type=int, default=1000,
                        help="Minimum epoch to consider")
    parser.add_argument('--output', type=str, default="plots/specificity_validation.png",
                        help="Output path for the plot")
    args = parser.parse_args()

    print("Collecting epoch-level data across datasets...")
    all_results = []

    for dataset_id in args.datasets:
        print(f"\nProcessing dataset {dataset_id}...")
        results = collect_epoch_level_data(dataset_id, min_epoch=args.min_epoch)
        all_results.extend(results)
        print(f"  Collected {len(results)} epochs")

    print(f"\nTotal epochs across all datasets: {len(all_results)}")

    if len(all_results) == 0:
        print("\nNo data found. Run compute_reconstruction_specificity.py first.")
        return

    # Create validation plot
    plot_specificity_vs_gt_correlation(all_results, output_path=args.output)


if __name__ == "__main__":
    main()
