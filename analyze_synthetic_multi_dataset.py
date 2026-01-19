"""
Aggregated analysis script for multi-dataset experiments.

This script collects results from all datasets and creates plots showing:
1. RSA correlation with ground truth (aggregated across datasets)
2. Model likelihoods (aggregated across datasets)
3. Per-dataset breakdown showing consistency of trends
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plot_functions as plf
import os
import json
from scipy.stats import ttest_rel, sem
from modelsandtraining import vectorize_rsa
import torch

# Configuration
N_DATASETS = 5
SEEDS = [12, 50, 76, 100, 142]

def load_dataset_results(dataset_id):
    """Load all results for a given dataset."""
    data_dir = f"data_dataset{dataset_id}"
    runs_dir = f"runs_dataset{dataset_id}"
    runs_vanilla_dir = f"runs_vanilla_dataset{dataset_id}"

    results = {}

    # Load parameter values
    test_parameter_df = pd.read_csv(f"{data_dir}/true_test_parameter_values.csv")
    params = test_parameter_df["alphaP_list"].values
    results['params'] = params

    # Load model evaluation dataframes
    try:
        model_eval_df = pd.read_csv(f"{data_dir}/model_eval_dfvanilla.csv")
        results['model_eval_df'] = model_eval_df
    except FileNotFoundError:
        print(f"Warning: model_eval_df not found for dataset {dataset_id}")
        results['model_eval_df'] = None

    # Load RNN results
    try:
        rnn_ID_df = pd.read_csv(f"{data_dir}/rnn_resultslatentmodel.csv")
        rnn_common_process_df = pd.read_csv(f"{data_dir}/rnn_results_common_process.csv")
        rnn_vanilla_df = pd.read_csv(f"{data_dir}/rnn_resultsvanilla.csv")

        results['rnn_ID_df'] = rnn_ID_df
        results['rnn_common_process_df'] = rnn_common_process_df
        results['rnn_vanilla_df'] = rnn_vanilla_df
    except FileNotFoundError as e:
        print(f"Warning: RNN results not found for dataset {dataset_id}: {e}")
        results['rnn_ID_df'] = None
        results['rnn_common_process_df'] = None
        results['rnn_vanilla_df'] = None

    # Load best epoch info
    try:
        with open(os.path.join(runs_dir, "best_epoch_by_specificity.json"), "r") as f:
            latent_meta = json.load(f)
        with open(os.path.join(runs_vanilla_dir, "best_epoch_by_rsa.json"), "r") as f:
            vanilla_meta = json.load(f)

        results['latent_best_epoch'] = latent_meta["best_epoch"]
        results['latent_best_seed'] = latent_meta["best_seed"] #76  # Using fixed seed as in original
        results['vanilla_best_epoch'] = vanilla_meta["best_epoch"]
        results['vanilla_best_seed'] = vanilla_meta["best_seed"]
    except FileNotFoundError as e:
        print(f"Warning: Best epoch metadata not found for dataset {dataset_id}: {e}")
        return None

    # Load latent tensors
    try:
        latent_tensor = torch.load(f"{data_dir}/latents_tensorlatentmodel{results['latent_best_seed']}.pt")
        results['latent_tensor'] = latent_tensor
    except FileNotFoundError:
        print(f"Warning: Latent tensor not found for dataset {dataset_id}")
        results['latent_tensor'] = None

    # Compute RSA for ground truth parameters
    distance_matrix_params, params_order = plf.rsa_latents(
        latents=params,
        metric="euclidean",
        title="test_params",
        reduction="entire",
        plot=False,
        original_data=True,
        cluster_order=False
    )
    vec_params = vectorize_rsa(distance_matrix_params)
    results['vec_params'] = vec_params

    # Load RSA vectors for models
    def load_rsa_vector(seed, epoch, base_directory):
        rsa_path = os.path.join(base_directory, f"seed_{seed}", "rsa", f"epoch_{epoch:04d}.npy")
        return np.load(rsa_path)

    try:
        latent_rsa = load_rsa_vector(
            results['latent_best_seed'],
            results['latent_best_epoch'],
            runs_dir
        )
        vanilla_rsa = load_rsa_vector(
            results['vanilla_best_seed'],
            results['vanilla_best_epoch'],
            runs_vanilla_dir
        )

        results['latent_rsa_corr'] = np.corrcoef(vec_params, latent_rsa)[0, 1]
        results['vanilla_rsa_corr'] = np.corrcoef(vec_params, vanilla_rsa)[0, 1]
    except FileNotFoundError as e:
        print(f"Warning: Could not load RSA vectors for dataset {dataset_id}: {e}")
        results['latent_rsa_corr'] = None
        results['vanilla_rsa_corr'] = None

    return results

def aggregate_rsa_correlations(all_results):
    """Aggregate RSA correlations across datasets."""
    latent_corrs = []
    vanilla_corrs = []

    for dataset_id, results in all_results.items():
        if results and results['latent_rsa_corr'] is not None:
            latent_corrs.append(results['latent_rsa_corr'])
            vanilla_corrs.append(results['vanilla_rsa_corr'])

    return {
        'IDRNN': {
            'corrs': np.array(latent_corrs),
            'mean': np.mean(latent_corrs),
            'sem': sem(latent_corrs)
        },
        'vanilla': {
            'corrs': np.array(vanilla_corrs),
            'mean': np.mean(vanilla_corrs),
            'sem': sem(vanilla_corrs)
        }
    }

def aggregate_model_likelihoods(all_results):
    """Aggregate model likelihoods across datasets."""
    aggregated = {
        'Q_common': [],
        'Q_MAP': [],
        'FQ_common': [],
        'FQ_MAP': [],
        'RNN_common': [],
        'RNN_ID': [],
        'RNN_vanilla': [],
        'True_model': []
    }

    for dataset_id, results in all_results.items():
        if results is None or results['model_eval_df'] is None:
            continue

        model_eval_df = results['model_eval_df']

        # Extract likelihoods for each model
        try:
            Q_common = model_eval_df[model_eval_df["model"] == "Q (common fit)"]["normalized_likelihood"].values
            Q_MAP = model_eval_df[model_eval_df["model"] == "Q (MAP)"]["normalized_likelihood"].values
            FQ_common = model_eval_df[model_eval_df["model"] == "FQ (common fit)"]["normalized_likelihood"].values
            FQ_MAP = model_eval_df[model_eval_df["model"] == "FQ (MAP)"]["normalized_likelihood"].values
            True_model = model_eval_df[model_eval_df["model"] == "True model"]["normalized_likelihood"].values

            aggregated['Q_common'].append(np.mean(Q_common))
            aggregated['Q_MAP'].append(np.mean(Q_MAP))
            aggregated['FQ_common'].append(np.mean(FQ_common))
            aggregated['FQ_MAP'].append(np.mean(FQ_MAP))
            aggregated['True_model'].append(np.mean(True_model))
        except Exception as e:
            print(f"Warning: Could not extract cognitive model likelihoods for dataset {dataset_id}: {e}")

        # Extract RNN likelihoods
        if results['rnn_common_process_df'] is not None:
            try:
                rnn_common = results['rnn_common_process_df'][
                    results['rnn_common_process_df']["model"] == "common_process_RNN"
                ]["normalized_likelihood"].values
                aggregated['RNN_common'].append(np.mean(rnn_common))
            except:
                pass

        if results['rnn_ID_df'] is not None:
            try:
                rnn_ID = results['rnn_ID_df'][
                    results['rnn_ID_df']["model"] == "IDRNN"
                ]["normalized_likelihood"].values
                aggregated['RNN_ID'].append(np.mean(rnn_ID))
            except:
                pass

        if results['rnn_vanilla_df'] is not None:
            try:
                rnn_vanilla = results['rnn_vanilla_df'][
                    results['rnn_vanilla_df']["model"] == "vanillaRNN"
                ]["normalized_likelihood"].values
                aggregated['RNN_vanilla'].append(np.mean(rnn_vanilla))
            except:
                pass

    # Compute means and SEMs
    summary = {}
    for key, values in aggregated.items():
        if len(values) > 0:
            summary[key] = {
                'values': np.array(values),
                'mean': np.mean(values),
                'sem': sem(values)
            }
        else:
            summary[key] = {
                'values': np.array([]),
                'mean': np.nan,
                'sem': np.nan
            }

    return summary

def p_to_stars(p):
    """Convert p-value to significance stars."""
    if p < 0.001: return "***"
    elif p < 0.01: return "**"
    elif p < 0.05: return "*"
    else: return "n.s."

def add_sig(ax, x1, x2, y, h, p):
    """Draw significance bars from x1 to x2 at height y with annotation."""
    stars = p_to_stars(p)
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c='black')
    ax.text((x1 + x2) / 2, y + h, stars, ha='center', va='bottom', color='black')

def plot_rsa_aggregated(rsa_dict, output_dir):
    """Plot aggregated RSA correlations."""
    fig, ax = plt.subplots(figsize=(8, 6))

    models = ['IDRNN', 'vanilla']
    means = [rsa_dict['IDRNN']['mean'], rsa_dict['vanilla']['mean']]
    sems = [rsa_dict['IDRNN']['sem'], rsa_dict['vanilla']['sem']]

    bar_colors = ['tab:blue', 'tab:orange']
    x_pos = np.arange(len(models))

    # Draw bars without error bars first (for cleaner look)
    bars = ax.bar(
        x_pos,
        means,
        color=bar_colors,
        alpha=0.7,
        width=0.6
    )

    # Add individual dataset points with jitter
    np.random.seed(42)  # For reproducible jitter
    for i, model in enumerate(models):
        corrs = rsa_dict[model]['corrs']
        jitter = np.random.uniform(-0.15, 0.15, size=len(corrs))
        ax.scatter(x_pos[i] + jitter, corrs, alpha=0.8, c='black', s=50, zorder=10,
                   edgecolors='white', linewidths=0.5, label='Individual datasets' if i == 0 else None)

    # Add SEM error bars on top (as separate elements for visibility)
    ax.errorbar(x_pos, means, yerr=sems, fmt='none', capsize=8, capthick=2,
                ecolor='black', elinewidth=2, zorder=11)

    # Add mean value annotations
    for i, (mean, sem) in enumerate(zip(means, sems)):
        ax.text(x_pos[i], mean + sems[i] + 0.02, f'{mean:.3f}±{sem:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('Correlation with ground truth', fontsize=12)
    ax.set_title(f'RSA: Model Latent Geometry vs Ground Truth\n(Aggregated across {N_DATASETS} datasets)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models)
    ax.set_ylim(bottom=0)
    ax.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/aggregated_rsa_correlation.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved aggregated RSA plot to {output_dir}/aggregated_rsa_correlation.png")

def plot_likelihoods_aggregated(likelihood_summary, output_dir):
    """Plot aggregated model likelihoods."""
    fig, ax = plt.subplots(figsize=(12, 6))

    models = ['Q CP', 'Q MAP', 'FQ CP', 'FQ MAP', 'RNNCP', 'RNNID', 'VanillaRNN']
    means = [
        likelihood_summary['Q_common']['mean'],
        likelihood_summary['Q_MAP']['mean'],
        likelihood_summary['FQ_common']['mean'],
        likelihood_summary['FQ_MAP']['mean'],
        likelihood_summary['RNN_common']['mean'],
        likelihood_summary['RNN_ID']['mean'],
        likelihood_summary['RNN_vanilla']['mean']
    ]
    sems = [
        likelihood_summary['Q_common']['sem'],
        likelihood_summary['Q_MAP']['sem'],
        likelihood_summary['FQ_common']['sem'],
        likelihood_summary['FQ_MAP']['sem'],
        likelihood_summary['RNN_common']['sem'],
        likelihood_summary['RNN_ID']['sem'],
        likelihood_summary['RNN_vanilla']['sem']
    ]

    bar_labels = ['common fit', 'individual differences', '_common fit', '_individual differences',
                  '_common fit', '_individual differences', 'vanilla NN']
    bar_colors = ['tab:green', 'tab:blue', 'tab:green', 'tab:blue', 'tab:green', 'tab:blue', 'tab:orange']

    bars = ax.bar(models, means, yerr=sems, capsize=5, color=bar_colors, alpha=0.8, label=bar_labels)

    # Add individual dataset points
    keys = ['Q_common', 'Q_MAP', 'FQ_common', 'FQ_MAP', 'RNN_common', 'RNN_ID', 'RNN_vanilla']
    for i, key in enumerate(keys):
        values = likelihood_summary[key]['values']
        if len(values) > 0:
            x = np.random.normal(i, 0.04, size=len(values))
            ax.scatter(x, values, alpha=0.6, c='black', s=30, zorder=10)

    # Add true model reference line
    true_model_mean = likelihood_summary['True_model']['mean']
    ax.axhline(true_model_mean, linestyle='--', color='black', linewidth=1.5, label='True model')

    # Statistical tests (if enough data)
    try:
        if len(likelihood_summary['Q_common']['values']) > 1:
            _, p_q = ttest_rel(likelihood_summary['Q_common']['values'],
                              likelihood_summary['Q_MAP']['values'])
            _, p_fq = ttest_rel(likelihood_summary['FQ_common']['values'],
                               likelihood_summary['FQ_MAP']['values'])
            _, p_rnn = ttest_rel(likelihood_summary['RNN_common']['values'],
                                likelihood_summary['RNN_ID']['values'])
            _, p_rnn_vanilla = ttest_rel(likelihood_summary['RNN_vanilla']['values'],
                                        likelihood_summary['RNN_ID']['values'])

            # Add significance bars
            ymax = max([m for m in means if not np.isnan(m)])
            h = ymax * 0.01

            add_sig(ax, 0, 1, ymax + h*2, h, p_q)
            add_sig(ax, 2, 3, ymax + h*2, h, p_fq)
            add_sig(ax, 4, 5, ymax + h*2, h, p_rnn)
            add_sig(ax, 6, 5, ymax + h*2.3, h, p_rnn_vanilla)
    except Exception as e:
        print(f"Warning: Could not compute statistical tests: {e}")

    ax.set_ylabel('Mean log likelihood per participant', fontsize=12)
    ax.set_title(f'Model Performance Comparison\n(Aggregated across {N_DATASETS} datasets)',
                 fontsize=14, fontweight='bold')
    ax.legend(title='Model types', loc='lower left')
    ax.set_ylim(bottom=true_model_mean - 5)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/aggregated_model_likelihoods.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved aggregated likelihood plot to {output_dir}/aggregated_model_likelihoods.png")

def plot_per_dataset_breakdown(all_results, output_dir):
    """Create plots showing per-dataset breakdown of key metrics."""
    dataset_ids = sorted(all_results.keys())

    # Plot 1: RSA correlations per dataset
    fig, ax = plt.subplots(figsize=(10, 6))

    # Track which datasets actually have RSA data
    valid_rsa_datasets = [d for d in dataset_ids
                          if all_results[d] and all_results[d]['latent_rsa_corr'] is not None]
    latent_corrs = [all_results[d]['latent_rsa_corr'] for d in valid_rsa_datasets]
    vanilla_corrs = [all_results[d]['vanilla_rsa_corr'] for d in valid_rsa_datasets]

    x = np.arange(len(valid_rsa_datasets))
    width = 0.35

    ax.bar(x - width/2, latent_corrs, width, label='IDRNN', color='tab:blue', alpha=0.8)
    ax.bar(x + width/2, vanilla_corrs, width, label='Vanilla RNN', color='tab:orange', alpha=0.8)

    ax.set_xlabel('Dataset ID', fontsize=12)
    ax.set_ylabel('RSA Correlation', fontsize=12)
    ax.set_title('RSA Correlation per Dataset', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'D{d}' for d in valid_rsa_datasets])
    ax.legend()
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/per_dataset_rsa.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved per-dataset RSA plot to {output_dir}/per_dataset_rsa.png")

    # Plot 2: Model likelihoods per dataset (for RNN models)
    fig, ax = plt.subplots(figsize=(10, 6))

    rnn_id_lls = []
    rnn_common_lls = []
    rnn_vanilla_lls = []
    valid_ll_datasets = []

    for dataset_id in dataset_ids:
        if all_results[dataset_id] is None:
            continue

        # Only include datasets that have all three RNN results
        if (all_results[dataset_id]['rnn_ID_df'] is not None and
            all_results[dataset_id]['rnn_common_process_df'] is not None and
            all_results[dataset_id]['rnn_vanilla_df'] is not None):

            ll_id = all_results[dataset_id]['rnn_ID_df'][
                all_results[dataset_id]['rnn_ID_df']["model"] == "IDRNN"
            ]["normalized_likelihood"].mean()
            rnn_id_lls.append(ll_id)

            ll_common = all_results[dataset_id]['rnn_common_process_df'][
                all_results[dataset_id]['rnn_common_process_df']["model"] == "common_process_RNN"
            ]["normalized_likelihood"].mean()
            rnn_common_lls.append(ll_common)

            ll_vanilla = all_results[dataset_id]['rnn_vanilla_df'][
                all_results[dataset_id]['rnn_vanilla_df']["model"] == "vanillaRNN"
            ]["normalized_likelihood"].mean()
            rnn_vanilla_lls.append(ll_vanilla)

            valid_ll_datasets.append(dataset_id)

    x = np.arange(len(valid_ll_datasets))
    width = 0.25

    ax.bar(x - width, rnn_common_lls, width, label='RNN Common Process', color='tab:green', alpha=0.8)
    ax.bar(x, rnn_id_lls, width, label='IDRNN', color='tab:blue', alpha=0.8)
    ax.bar(x + width, rnn_vanilla_lls, width, label='Vanilla RNN', color='tab:orange', alpha=0.8)

    ax.set_xlabel('Dataset ID', fontsize=12)
    ax.set_ylabel('Mean Log Likelihood', fontsize=12)
    ax.set_title('RNN Model Performance per Dataset', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'D{d}' for d in valid_ll_datasets])
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"{output_dir}/per_dataset_likelihoods.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved per-dataset likelihood plot to {output_dir}/per_dataset_likelihoods.png")

def main():
    """Main analysis function."""
    print("="*80)
    print("MULTI-DATASET AGGREGATED ANALYSIS")
    print("="*80)

    # Create output directory
    output_dir = "plots/multi_dataset"
    os.makedirs(output_dir, exist_ok=True)

    # Load all dataset results
    print(f"\nLoading results from {N_DATASETS} datasets...")
    all_results = {}
    for dataset_id in range(N_DATASETS):
        print(f"  Loading dataset {dataset_id}...")
        results = load_dataset_results(dataset_id)
        all_results[dataset_id] = results

    successful_datasets = sum(1 for r in all_results.values() if r is not None)
    print(f"\n✅ Successfully loaded {successful_datasets}/{N_DATASETS} datasets")

    # Aggregate RSA correlations
    print("\nAggregating RSA correlations...")
    rsa_dict = aggregate_rsa_correlations(all_results)

    print(f"\nRSA Correlation Results:")
    print(f"  IDRNN:       {rsa_dict['IDRNN']['mean']:.3f} ± {rsa_dict['IDRNN']['sem']:.3f}")
    print(f"  Vanilla RNN: {rsa_dict['vanilla']['mean']:.3f} ± {rsa_dict['vanilla']['sem']:.3f}")

    # Aggregate model likelihoods
    print("\nAggregating model likelihoods...")
    likelihood_summary = aggregate_model_likelihoods(all_results)

    print(f"\nModel Likelihood Results:")
    for key, data in likelihood_summary.items():
        if not np.isnan(data['mean']):
            print(f"  {key:15s}: {data['mean']:6.2f} ± {data['sem']:.2f}")

    # Generate plots
    print("\nGenerating plots...")
    plot_rsa_aggregated(rsa_dict, output_dir)
    plot_likelihoods_aggregated(likelihood_summary, output_dir)
    plot_per_dataset_breakdown(all_results, output_dir)

    # Save summary statistics
    summary_path = f"{output_dir}/summary_statistics.txt"
    with open(summary_path, "w") as f:
        f.write("MULTI-DATASET AGGREGATED RESULTS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Number of datasets: {N_DATASETS}\n")
        f.write(f"Successfully loaded: {successful_datasets}\n")
        f.write(f"Random seeds per dataset: {SEEDS}\n\n")

        f.write("RSA CORRELATIONS:\n")
        f.write(f"  IDRNN:       {rsa_dict['IDRNN']['mean']:.4f} ± {rsa_dict['IDRNN']['sem']:.4f}\n")
        f.write(f"  Vanilla RNN: {rsa_dict['vanilla']['mean']:.4f} ± {rsa_dict['vanilla']['sem']:.4f}\n\n")

        f.write("MODEL LIKELIHOODS:\n")
        for key, data in likelihood_summary.items():
            if not np.isnan(data['mean']):
                f.write(f"  {key:20s}: {data['mean']:8.4f} ± {data['sem']:.4f}\n")

    print(f"\n✅ Saved summary statistics to {summary_path}")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nResults saved in: {output_dir}/")
    print("  - aggregated_rsa_correlation.png")
    print("  - aggregated_model_likelihoods.png")
    print("  - per_dataset_rsa.png")
    print("  - per_dataset_likelihoods.png")
    print("  - summary_statistics.txt")

if __name__ == "__main__":
    main()
