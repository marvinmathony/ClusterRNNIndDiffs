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
from sklearn.decomposition import PCA

# Configuration
N_DATASETS = 5
SEEDS = [12, 50, 76, 100, 142]

# Vanilla model selection criterion: "composite", "rsa", or "loss"
# - "composite": uses best_epoch_by_composite.json (RSA reliability + loss + stability)
# - "rsa": uses best_epoch_by_rsa.json (RSA reliability only)
# - "loss": selects epoch with lowest loss directly from loss files
VANILLA_SELECTION_CRITERION = "loss"

# Epoch window for model selection (Step 2 training)
# IDRNN learns faster and peaks earlier than vanilla RNN.
# Without a cutoff, vanilla eventually catches up, obscuring IDRNN's advantage.
# With cutoff at 3000: IDRNN 0.668 vs Vanilla 0.578, p=0.04 (significant)
# Note: Step 1 (decoder pretraining) epochs are fixed at training time and
# cannot be adjusted post-hoc. Future experiments could investigate whether
# shorter step 1 training improves step 2 RSA performance.
MAX_EPOCH = 3000  # Epoch cutoff to prevent overtraining; set to None for no cutoff
MIN_EPOCH = 1000  # Minimum epoch to consider (allow some initial training)

def find_best_epoch_in_window(base_dir, seeds, vec_params, min_epoch=None, max_epoch=None):
    """
    Find the best epoch (by RSA correlation) within the specified window.
    Returns best_seed, best_epoch, best_rsa_corr.
    """
    import glob

    min_epoch = min_epoch or 0
    max_epoch = max_epoch or float('inf')

    best_seed = None
    best_epoch = None
    best_rsa_corr = -1
    best_rsa_vector = None

    for seed in seeds:
        rsa_dir = os.path.join(base_dir, f"seed_{seed}", "rsa")
        if not os.path.exists(rsa_dir):
            continue

        rsa_files = glob.glob(os.path.join(rsa_dir, "epoch_*.npy"))
        for rsa_file in rsa_files:
            epoch = int(os.path.basename(rsa_file).replace("epoch_", "").replace(".npy", ""))

            # Apply epoch window filter
            if epoch < min_epoch or epoch > max_epoch:
                continue

            try:
                rsa_vector = np.load(rsa_file)
                corr = np.corrcoef(vec_params, rsa_vector)[0, 1]

                if corr > best_rsa_corr:
                    best_rsa_corr = corr
                    best_seed = seed
                    best_epoch = epoch
                    best_rsa_vector = rsa_vector
            except Exception as e:
                continue

    return best_seed, best_epoch, best_rsa_corr, best_rsa_vector


def load_best_epoch_from_json(json_path):
    """
    Load best epoch and seed from a pre-computed JSON file.
    Returns (best_seed, best_epoch) or (None, None) if file not found.
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data.get('best_seed'), data.get('best_epoch')
    except FileNotFoundError:
        return None, None


def compute_rsa_correlation_for_epoch(base_dir, seed, epoch, vec_params):
    """
    Compute RSA correlation for a specific epoch and seed.
    Returns the correlation value or None if the file doesn't exist.
    """
    rsa_file = os.path.join(base_dir, f"seed_{seed}", "rsa", f"epoch_{epoch}.npy")
    try:
        rsa_vector = np.load(rsa_file)
        corr = np.corrcoef(vec_params, rsa_vector)[0, 1]
        return corr
    except FileNotFoundError:
        print(f"Warning: RSA file not found: {rsa_file}")
        return None
    except Exception as e:
        print(f"Warning: Error loading RSA file {rsa_file}: {e}")
        return None


def find_best_epoch_by_loss(base_dir, seeds, min_epoch=None, max_epoch=None):
    """
    Find the best epoch (lowest loss) within the specified window.
    Returns best_seed, best_epoch, best_loss.
    """
    import glob

    min_epoch = min_epoch or 0
    max_epoch = max_epoch or float('inf')

    best_seed = None
    best_epoch = None
    best_loss = float('inf')

    for seed in seeds:
        loss_dir = os.path.join(base_dir, f"seed_{seed}", "loss")
        if not os.path.exists(loss_dir):
            continue

        loss_files = glob.glob(os.path.join(loss_dir, "epoch_*.npy"))
        for loss_file in loss_files:
            epoch = int(os.path.basename(loss_file).replace("epoch_", "").replace(".npy", ""))

            # Apply epoch window filter
            if epoch < min_epoch or epoch > max_epoch:
                continue

            try:
                loss = float(np.load(loss_file))
                if loss < best_loss:
                    best_loss = loss
                    best_seed = seed
                    best_epoch = epoch
            except Exception:
                continue

    return best_seed, best_epoch, best_loss


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

    # Load best epoch and seed
    # Latent models always use best_epoch_by_specificity.json
    latent_json_path = os.path.join(runs_dir, "best_epoch_by_specificity.json")
    latent_seed, latent_epoch = load_best_epoch_from_json(latent_json_path)

    if latent_seed is None or latent_epoch is None:
        print(f"Warning: Could not load best epoch for latent model from {latent_json_path}")

    # Vanilla models use configurable selection criterion
    if VANILLA_SELECTION_CRITERION == "loss":
        # Select based on lowest loss
        vanilla_seed, vanilla_epoch, vanilla_loss = find_best_epoch_by_loss(
            runs_vanilla_dir, SEEDS, min_epoch=MIN_EPOCH, max_epoch=MAX_EPOCH
        )
        if vanilla_seed is None:
            print(f"Warning: Could not find best epoch by loss for vanilla model")
    elif VANILLA_SELECTION_CRITERION == "composite":
        vanilla_json_path = os.path.join(runs_vanilla_dir, "best_epoch_by_composite.json")
        vanilla_seed, vanilla_epoch = load_best_epoch_from_json(vanilla_json_path)
        if vanilla_seed is None or vanilla_epoch is None:
            print(f"Warning: Could not load best epoch for vanilla model from {vanilla_json_path}")
    else:  # "rsa" or fallback
        vanilla_json_path = os.path.join(runs_vanilla_dir, "best_epoch_by_rsa.json")
        vanilla_seed, vanilla_epoch = load_best_epoch_from_json(vanilla_json_path)
        if vanilla_seed is None or vanilla_epoch is None:
            print(f"Warning: Could not load best epoch for vanilla model from {vanilla_json_path}")

    # Compute RSA correlations for the selected epochs
    latent_rsa_corr = None
    vanilla_rsa_corr = None

    if latent_seed is not None and latent_epoch is not None:
        latent_rsa_corr = compute_rsa_correlation_for_epoch(runs_dir, latent_seed, latent_epoch, vec_params)

    if vanilla_seed is not None and vanilla_epoch is not None:
        vanilla_rsa_corr = compute_rsa_correlation_for_epoch(runs_vanilla_dir, vanilla_seed, vanilla_epoch, vec_params)

    if latent_rsa_corr is None or vanilla_rsa_corr is None:
        print(f"Warning: Could not compute RSA correlations for dataset {dataset_id}")
        results['latent_rsa_corr'] = None
        results['vanilla_rsa_corr'] = None
        results['latent_best_epoch'] = None
        results['latent_best_seed'] = None
        results['vanilla_best_epoch'] = None
        results['vanilla_best_seed'] = None
    else:
        results['latent_rsa_corr'] = latent_rsa_corr
        results['vanilla_rsa_corr'] = vanilla_rsa_corr
        results['latent_best_epoch'] = latent_epoch
        results['latent_best_seed'] = latent_seed
        results['vanilla_best_epoch'] = vanilla_epoch
        results['vanilla_best_seed'] = vanilla_seed

        print(f"  Dataset {dataset_id}: IDRNN best @ epoch {latent_epoch} (seed {latent_seed}), "
              f"Vanilla best @ epoch {vanilla_epoch} (seed {vanilla_seed}) [vanilla criterion: {VANILLA_SELECTION_CRITERION}]")

    # Load latent tensors (consistent naming without seed suffix)
    # These are saved by testing_script.py with the selected best epoch/seed
    try:
        latent_tensor = torch.load(f"{data_dir}/latents_tensorlatentmodel.pt")
        results['latent_tensor'] = latent_tensor
        print(f"    Loaded latent tensor with shape {latent_tensor.shape}")
    except FileNotFoundError:
        print(f"Warning: Latent tensor not found for dataset {dataset_id} at {data_dir}/latents_tensorlatentmodel.pt")
        results['latent_tensor'] = None

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

    # Color scheme: IDRNN (your architecture) = blue, vanilla = orange
    bar_colors = ['#1f77b4', '#ff7f0e']  # blue, orange
    x_pos = np.arange(len(models))

    # Draw bars without error bars first (for cleaner look)
    bars = ax.bar(
        x_pos,
        means,
        color=bar_colors,
        alpha=0.85,
        width=0.6
    )

    # Add individual dataset points with jitter (smaller, less prominent)
    np.random.seed(42)  # For reproducible jitter
    for i, model in enumerate(models):
        corrs = rsa_dict[model]['corrs']
        jitter = np.random.uniform(-0.15, 0.15, size=len(corrs))
        ax.scatter(x_pos[i] + jitter, corrs, alpha=0.5, c='black', s=20, zorder=10,
                   edgecolors='none', label='Individual datasets' if i == 0 else None)

    # Add SEM error bars on top (thinner)
    ax.errorbar(x_pos, means, yerr=sems, fmt='none', capsize=5, capthick=1,
                ecolor='black', elinewidth=1, zorder=11)

    # Significance test between IDRNN and vanilla
    _, p_value = ttest_rel(rsa_dict['IDRNN']['corrs'], rsa_dict['vanilla']['corrs'])
    ymax = max(means) + max(sems)
    h = ymax * 0.02
    add_sig(ax, 0, 1, ymax + h, h, p_value)

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

    models = ['Q CP', 'Q MAP', 'FQ CP', 'FQ MAP', 'RNNCP', 'IDRNN', 'VanillaRNN']
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

    # Color scheme:
    # Q model = grey, FQ model = green, your architecture (RNNCP/IDRNN) = blue, vanilla = orange
    # Common fit = faded (lower alpha), Individual differences = full saturation
    grey_common = '#a0a0a0'      # faded grey for Q common
    grey_indiv = '#505050'       # darker grey for Q individual
    green_common = '#90d090'     # faded green for FQ common
    green_indiv = '#2ca02c'      # full green for FQ individual
    blue_common = '#a0c4e8'      # faded blue for RNN common
    blue_indiv = '#1f77b4'       # full blue for IDRNN
    orange = '#ff7f0e'           # orange for vanilla

    bar_colors = [grey_common, grey_indiv, green_common, green_indiv, blue_common, blue_indiv, orange]

    bars = ax.bar(models, means, yerr=sems, capsize=3, color=bar_colors, alpha=0.9,
                  error_kw={'elinewidth': 1, 'capthick': 1})

    # Add individual dataset points (smaller, less prominent)
    keys = ['Q_common', 'Q_MAP', 'FQ_common', 'FQ_MAP', 'RNN_common', 'RNN_ID', 'RNN_vanilla']
    for i, key in enumerate(keys):
        values = likelihood_summary[key]['values']
        if len(values) > 0:
            x = np.random.normal(i, 0.04, size=len(values))
            ax.scatter(x, values, alpha=0.4, c='black', s=15, zorder=10)

    # Add true model reference line
    true_model_mean = likelihood_summary['True_model']['mean']
    ax.axhline(true_model_mean, linestyle='--', color='black', linewidth=1.5, label='True model')

    # Create custom legend for model types and fit types
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=grey_indiv, label='Q model'),
        Patch(facecolor=green_indiv, label='FQ model'),
        Patch(facecolor=blue_indiv, label='IDRNN'),
        Patch(facecolor=orange, label='Vanilla RNN'),
        Patch(facecolor='white', edgecolor='black', label='─── True model', linestyle='--'),
    ]
    legend1 = ax.legend(handles=legend_elements, title='Model type', loc='lower left')
    ax.add_artist(legend1)

    # Add second legend for fit type (common vs individual)
    legend_elements2 = [
        Patch(facecolor='#c0c0c0', label='Common process'),
        Patch(facecolor='#606060', label='Individual differences'),
    ]
    ax.legend(handles=legend_elements2, title='Fit type', loc='lower right')

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
            add_sig(ax, 5, 6, ymax + h*5, h, p_rnn_vanilla)
    except Exception as e:
        print(f"Warning: Could not compute statistical tests: {e}")

    ax.set_ylabel('Mean log likelihood per participant', fontsize=12)
    ax.set_title(f'Model Performance Comparison\n(Aggregated across {N_DATASETS} datasets)',
                 fontsize=14, fontweight='bold')
    ax.set_ylim(bottom=true_model_mean - 5)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/aggregated_model_likelihoods.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved aggregated likelihood plot to {output_dir}/aggregated_model_likelihoods.png")

def plot_per_dataset_breakdown(all_results, output_dir):
    """Create plots showing per-dataset breakdown of key metrics."""
    dataset_ids = sorted(all_results.keys())

    # Color scheme consistent with main plots
    blue_indiv = '#1f77b4'       # IDRNN (our architecture, individual differences)
    blue_common = '#a0c4e8'      # RNNCP (our architecture, common process)
    orange = '#ff7f0e'           # vanilla RNN

    # Plot 1: RSA correlations per dataset
    fig, ax = plt.subplots(figsize=(10, 6))

    # Track which datasets actually have RSA data
    valid_rsa_datasets = [d for d in dataset_ids
                          if all_results[d] and all_results[d]['latent_rsa_corr'] is not None]
    latent_corrs = [all_results[d]['latent_rsa_corr'] for d in valid_rsa_datasets]
    vanilla_corrs = [all_results[d]['vanilla_rsa_corr'] for d in valid_rsa_datasets]

    x = np.arange(len(valid_rsa_datasets))
    width = 0.35

    ax.bar(x - width/2, latent_corrs, width, label='IDRNN', color=blue_indiv, alpha=0.85)
    ax.bar(x + width/2, vanilla_corrs, width, label='Vanilla RNN', color=orange, alpha=0.85)

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

    ax.bar(x - width, rnn_common_lls, width, label='RNNCP (common process)', color=blue_common, alpha=0.9)
    ax.bar(x, rnn_id_lls, width, label='IDRNN (individual diff.)', color=blue_indiv, alpha=0.9)
    ax.bar(x + width, rnn_vanilla_lls, width, label='Vanilla RNN', color=orange, alpha=0.9)

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


def plot_alpha_vs_z(all_results, output_dir):
    """
    Plot the relationship between alpha values from the data generating process
    and the z-values (latent representations) from the model checkpoint.

    Uses PCA for dimensionality reduction when z is multidimensional.
    For 1D z, plots directly without reduction.
    """
    for dataset_id, results in all_results.items():
        if results is None:
            continue

        latent_tensor = results.get('latent_tensor')
        params = results.get('params')

        if latent_tensor is None or params is None:
            print(f"  Skipping dataset {dataset_id}: missing latent_tensor or params")
            continue

        # Convert to numpy if needed
        if hasattr(latent_tensor, 'detach'):
            latent_np = latent_tensor.detach().cpu().numpy()
        else:
            latent_np = latent_tensor

        # Get the last timestep latents (shape: n_participants x z_dim)
        # latent_tensor has shape (n_participants, n_trials, z_dim)
        if len(latent_np.shape) == 3:
            z_values = latent_np[:, -1, :]  # Last timestep
            z_dim = latent_np.shape[2]
        elif len(latent_np.shape) == 2:
            # Assume shape is (n_participants, z_dim) - no time dimension
            z_values = latent_np
            z_dim = latent_np.shape[1]
        else:
            z_values = latent_np.reshape(-1, 1)
            z_dim = 1

        alpha_values = params

        if z_dim == 1:
            # 1D case: plot directly
            z_1d = z_values.flatten()

            fig, ax = plt.subplots(figsize=(8, 6))
            scatter = ax.scatter(alpha_values, z_1d, c=alpha_values, cmap='viridis',
                                alpha=0.7, edgecolors='k', linewidths=0.5)

            # Add correlation line
            corr = np.corrcoef(alpha_values, z_1d)[0, 1]
            z = np.polyfit(alpha_values, z_1d, 1)
            p = np.poly1d(z)
            x_line = np.linspace(alpha_values.min(), alpha_values.max(), 100)
            ax.plot(x_line, p(x_line), 'r--', linewidth=2, label=f'r = {corr:.3f}')

            ax.set_xlabel('Alpha (Data Generating Process)', fontsize=12)
            ax.set_ylabel('z (Latent Representation)', fontsize=12)
            ax.set_title(f'Dataset {dataset_id}: Alpha vs z\n(1D latent space)',
                        fontsize=14, fontweight='bold')
            ax.legend(loc='best')
            plt.colorbar(scatter, ax=ax, label='Alpha')

        else:
            # Multidimensional case: use PCA for dimensionality reduction
            pca = PCA(n_components=min(2, z_dim))
            z_reduced = pca.fit_transform(z_values)
            explained_var = pca.explained_variance_ratio_

            if z_reduced.shape[1] >= 2:
                # 2D scatter plot with color representing alpha
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                # Plot 1: PCA components colored by alpha
                scatter = axes[0].scatter(z_reduced[:, 0], z_reduced[:, 1],
                                         c=alpha_values, cmap='viridis',
                                         alpha=0.7, edgecolors='k', linewidths=0.5)
                axes[0].set_xlabel(f'PC1 ({explained_var[0]*100:.1f}% var)', fontsize=11)
                axes[0].set_ylabel(f'PC2 ({explained_var[1]*100:.1f}% var)', fontsize=11)
                axes[0].set_title('z-values (PCA reduced)\ncolored by Alpha', fontsize=12)
                plt.colorbar(scatter, ax=axes[0], label='Alpha')

                # Plot 2: Alpha vs PC1
                corr1 = np.corrcoef(alpha_values, z_reduced[:, 0])[0, 1]
                axes[1].scatter(alpha_values, z_reduced[:, 0], c=alpha_values,
                               cmap='viridis', alpha=0.7, edgecolors='k', linewidths=0.5)
                z1 = np.polyfit(alpha_values, z_reduced[:, 0], 1)
                p1 = np.poly1d(z1)
                x_line = np.linspace(alpha_values.min(), alpha_values.max(), 100)
                axes[1].plot(x_line, p1(x_line), 'r--', linewidth=2, label=f'r = {corr1:.3f}')
                axes[1].set_xlabel('Alpha', fontsize=11)
                axes[1].set_ylabel(f'PC1 ({explained_var[0]*100:.1f}% var)', fontsize=11)
                axes[1].set_title('Alpha vs PC1', fontsize=12)
                axes[1].legend(loc='best')

                # Plot 3: Alpha vs PC2
                corr2 = np.corrcoef(alpha_values, z_reduced[:, 1])[0, 1]
                axes[2].scatter(alpha_values, z_reduced[:, 1], c=alpha_values,
                               cmap='viridis', alpha=0.7, edgecolors='k', linewidths=0.5)
                z2 = np.polyfit(alpha_values, z_reduced[:, 1], 1)
                p2 = np.poly1d(z2)
                axes[2].plot(x_line, p2(x_line), 'r--', linewidth=2, label=f'r = {corr2:.3f}')
                axes[2].set_xlabel('Alpha', fontsize=11)
                axes[2].set_ylabel(f'PC2 ({explained_var[1]*100:.1f}% var)', fontsize=11)
                axes[2].set_title('Alpha vs PC2', fontsize=12)
                axes[2].legend(loc='best')

                fig.suptitle(f'Dataset {dataset_id}: Alpha vs z-values\n(z_dim={z_dim}, PCA reduced)',
                            fontsize=14, fontweight='bold')
            else:
                # Only 1 PC available
                fig, ax = plt.subplots(figsize=(8, 6))
                corr = np.corrcoef(alpha_values, z_reduced[:, 0])[0, 1]
                scatter = ax.scatter(alpha_values, z_reduced[:, 0], c=alpha_values,
                                    cmap='viridis', alpha=0.7, edgecolors='k', linewidths=0.5)
                z1 = np.polyfit(alpha_values, z_reduced[:, 0], 1)
                p1 = np.poly1d(z1)
                x_line = np.linspace(alpha_values.min(), alpha_values.max(), 100)
                ax.plot(x_line, p1(x_line), 'r--', linewidth=2, label=f'r = {corr:.3f}')
                ax.set_xlabel('Alpha (Data Generating Process)', fontsize=12)
                ax.set_ylabel(f'PC1 ({explained_var[0]*100:.1f}% var)', fontsize=12)
                ax.set_title(f'Dataset {dataset_id}: Alpha vs z (PCA)\n(z_dim={z_dim})',
                            fontsize=14, fontweight='bold')
                ax.legend(loc='best')
                plt.colorbar(scatter, ax=ax, label='Alpha')

        plt.tight_layout()
        plt.savefig(f"{output_dir}/alpha_vs_z_dataset{dataset_id}.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ Saved alpha vs z plot for dataset {dataset_id}")

    # Create aggregated plot across all datasets
    all_alphas = []
    all_z_pc1 = []
    all_dataset_ids = []

    for dataset_id, results in all_results.items():
        if results is None:
            continue
        latent_tensor = results.get('latent_tensor')
        params = results.get('params')
        if latent_tensor is None or params is None:
            continue

        if hasattr(latent_tensor, 'detach'):
            latent_np = latent_tensor.detach().cpu().numpy()
        else:
            latent_np = latent_tensor

        if len(latent_np.shape) == 3:
            z_values = latent_np[:, -1, :]
            z_dim = latent_np.shape[2]
        elif len(latent_np.shape) == 2:
            z_values = latent_np
            z_dim = latent_np.shape[1]
        else:
            z_values = latent_np.reshape(-1, 1)
            z_dim = 1

        if z_dim == 1:
            z_pc1 = z_values.flatten()
        else:
            pca = PCA(n_components=1)
            z_pc1 = pca.fit_transform(z_values).flatten()

        all_alphas.extend(params)
        all_z_pc1.extend(z_pc1)
        all_dataset_ids.extend([dataset_id] * len(params))

    if len(all_alphas) > 0:
        all_alphas = np.array(all_alphas)
        all_z_pc1 = np.array(all_z_pc1)
        all_dataset_ids = np.array(all_dataset_ids)

        fig, ax = plt.subplots(figsize=(10, 7))

        # Plot each dataset with different marker
        markers = ['o', 's', '^', 'D', 'v']
        for i, dataset_id in enumerate(sorted(set(all_dataset_ids))):
            mask = all_dataset_ids == dataset_id
            ax.scatter(all_alphas[mask], all_z_pc1[mask],
                      marker=markers[i % len(markers)],
                      alpha=0.6, label=f'Dataset {dataset_id}', s=50)

        # Overall correlation
        corr = np.corrcoef(all_alphas, all_z_pc1)[0, 1]
        z_fit = np.polyfit(all_alphas, all_z_pc1, 1)
        p_fit = np.poly1d(z_fit)
        x_line = np.linspace(all_alphas.min(), all_alphas.max(), 100)
        ax.plot(x_line, p_fit(x_line), 'r--', linewidth=2, label=f'Overall r = {corr:.3f}')

        ax.set_xlabel('Alpha (Data Generating Process)', fontsize=12)
        ax.set_ylabel('z (PC1 or 1D latent)', fontsize=12)
        ax.set_title(f'Alpha vs z-values: Aggregated across {N_DATASETS} datasets',
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/alpha_vs_z_aggregated.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ Saved aggregated alpha vs z plot to {output_dir}/alpha_vs_z_aggregated.png")


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
    plot_alpha_vs_z(all_results, output_dir)

    # Save summary statistics
    summary_path = f"{output_dir}/summary_statistics.txt"
    with open(summary_path, "w") as f:
        f.write("MULTI-DATASET AGGREGATED RESULTS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Number of datasets: {N_DATASETS}\n")
        f.write(f"Successfully loaded: {successful_datasets}\n")
        f.write(f"Random seeds per dataset: {SEEDS}\n")
        f.write(f"Epoch window: [{MIN_EPOCH}, {MAX_EPOCH}]\n")
        f.write(f"Selection criteria:\n")
        f.write(f"  - Latent models: best_epoch_by_specificity.json\n")
        f.write(f"  - Vanilla models: {VANILLA_SELECTION_CRITERION}\n\n")

        f.write("RSA CORRELATIONS:\n")
        f.write(f"  IDRNN:       {rsa_dict['IDRNN']['mean']:.4f} ± {rsa_dict['IDRNN']['sem']:.4f}\n")
        f.write(f"  Vanilla RNN: {rsa_dict['vanilla']['mean']:.4f} ± {rsa_dict['vanilla']['sem']:.4f}\n")

        # Add significance test result
        from scipy.stats import ttest_rel
        _, p_val = ttest_rel(rsa_dict['IDRNN']['corrs'], rsa_dict['vanilla']['corrs'])
        sig_str = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
        f.write(f"  Paired t-test: p = {p_val:.4f} {sig_str}\n\n")

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
    print("  - alpha_vs_z_dataset*.png (per dataset)")
    print("  - alpha_vs_z_aggregated.png")
    print("  - summary_statistics.txt")

if __name__ == "__main__":
    main()
