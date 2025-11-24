import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE, MDS
from scipy.spatial.distance import pdist, squareform
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.preprocessing import StandardScaler
import torch
from sklearn.decomposition import PCA

def group_models(models):
    """
    Groups model names into categories based on their prefixes.

    Parameters:
    - models (list): A list of model names.

    Returns:
    - dict: A dictionary where keys are model categories and values are lists of corresponding models.
    """
    model_groups = {}

    for model in models:
        if model == "GRU":
            key = "GRU"
        elif model == "True model":
            key = "True model"
        else:
            key = model.split(" (")[0]  # Extract prefix before " (common fit)", " (individual fit)", etc.

        if key not in model_groups:
            model_groups[key] = []
        
        model_groups[key].append(model)

    return model_groups

import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_rel

import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_rel

# def compare_model_performance(model_eval_df, models, file_id='model_comparison', save_dir='./Figs/'):
#     """
#     Compare predictive accuracy (normalized likelihood) across different models and save the figure.

#     Parameters:
#     - model_eval_df (DataFrame): DataFrame containing model evaluation results with columns:
#         'model' (str): Model name.
#         'session' (int): Session identifier.
#         'normalized_likelihood' (float): Normalized likelihood for each session.
#     - models (list): A list of model names.
#     - file_id (str): Identifier for the saved file.
#     - save_dir (str): Directory where the figure will be saved.

#     Returns:
#     - fig (matplotlib.figure.Figure): The generated figure.
#     """

#     # Ensure save directory exists
#     os.makedirs(save_dir, exist_ok=True)

#     # Define color mapping for fits
#     color_mapping = {
#         'common fit': 'blue',
#         'individual fit': 'orange',
#         'MAP': 'green', 
#         'GRU': 'red',        # GRU in a prominent color
#         'True model': 'gray' # True model in a less prominent color
#     }

#     # Generate model groups
#     model_groups = group_models(models)

#     # Extract session-wise GRU data for comparison
#     if 'GRU' not in model_eval_df['model'].values:
#         print("GRU model not found in model_eval_df.")
#         return None

#     gru_values = model_eval_df[model_eval_df['model'] == 'GRU'].sort_values(by='session')['normalized_likelihood'].values

#     # Extract True model values for horizontal line plot
#     true_model_values = None
#     if 'True model' in model_eval_df['model'].values:
#         true_model_values = model_eval_df[model_eval_df['model'] == 'True model']['normalized_likelihood'].values
#         true_model_mean = np.mean(true_model_values)

#     # Aggregate session-wise data
#     aggregated_data = {}
#     for group, sub_models in model_groups.items():
#         for sub_model in sub_models:
#             if sub_model in model_eval_df['model'].values:
#                 nl_values = model_eval_df[model_eval_df['model'] == sub_model].sort_values(by='session')['normalized_likelihood'].values
#                 aggregated_data.setdefault(group, {}).setdefault(sub_model, []).extend(nl_values)

#     # Initialize the plot
#     fig, ax = plt.subplots(figsize=(10, 6))

#     # If True model exists, plot it as a horizontal line
#     if true_model_values is not None:
#         ax.axhline(true_model_mean, color='lightgray', linestyle="--", linewidth=2)
#         # ax.text(1.05, true_model_mean, "Ground truth", fontsize=12, color='gray', ha='left', va='center', transform=ax.transData)

#     # Plot models and perform t-tests
#     positions = []
#     means = []
#     errors = []
#     labels = []
#     current_position = 0
#     significant_models = []  # Models significantly worse than GRU

#     for group, sub_models in model_groups.items():
#         if group == "True model":  # Skip True model from individual plotting
#             continue

#         sub_positions = []
#         for sub_model in sub_models:
#             if sub_model in aggregated_data.get(group, {}):
#                 nl_values = aggregated_data[group][sub_model]
#                 mean = np.mean(nl_values)
#                 error = np.std(nl_values) / np.sqrt(len(nl_values))
            
#                 color = (
#                     color_mapping['common fit'] if 'common fit' in sub_model else
#                     color_mapping['individual fit'] if 'individual fit' in sub_model else
#                     color_mapping['MAP'] if 'MAP' in sub_model else
#                     color_mapping[group]  # GRU or other models
#                 )
            
#                 # Perform paired t-test against GRU
#                 t_stat, p_value = ttest_rel(nl_values, gru_values)
#                 print(f"{sub_model} vs. GRU: t = {t_stat:.3f}, p = {p_value:.3f}")

#                 # If the model is significantly worse (p < 0.05), mark it
#                 if p_value < 0.05 and mean < np.mean(gru_values):
#                     significant_models.append((current_position, mean + 1.1 * error))

#                 # Plot individual sub-model
#                 ax.errorbar(
#                     current_position, mean, yerr=error, fmt='o', 
#                     color=color, ecolor=color, elinewidth=1.5, capsize=4, markersize=8
#                 )
            
#                 sub_positions.append(current_position)
#                 current_position += 0.1  # Close spacing for common and individual fits
    
#         # Compute the X-axis position for the group label
#         positions.append(np.mean(sub_positions))
#         labels.append(group)
#         current_position += 0.5  # Add spacing between model groups

#     # Adjust X-axis and labels
#     ax.set_xticks(positions)
#     ax.set_xticklabels(labels, rotation=0, fontsize=12)
#     ax.set_ylabel("Predictive Accuracy (Normalized Likelihood)", fontsize=14)
#     #ax.set_title("Model Fits", fontsize=16)

#     # Add asterisks for models significantly worse than GRU
#     for pos, height in significant_models:
#         ax.text(pos, height, '*', fontsize=14, ha='center', color='black')

#     ax.text(
#         1.0, 1.01,  # X and Y position (slightly outside the top-right)
#         "* Lower predictive accuracy than GRU (p < 0.05, paired t-test)",
#         fontsize=12, color='black', ha='right', va='bottom', transform=ax.transAxes
#     )


#     # Add a legend for common and individual fits
#     legend_elements = [
#         plt.Line2D([0], [0], marker='o', color=color_mapping['common fit'], label='Common fit', markersize=8),
#         plt.Line2D([0], [0], marker='o', color=color_mapping['individual fit'], label='Individual (ML)', markersize=8),
#         plt.Line2D([0], [0], marker='o', color=color_mapping['MAP'], label='Individual (MAP)', markersize=8) 
#     ]
#     ax.legend(handles=legend_elements, loc='upper left', fontsize=12, frameon=False)

#     # Adjust layout and save the figure
#     plt.tight_layout()
#     file_path = os.path.join(save_dir, f"{file_id}.eps")
#     plt.savefig(file_path, format='eps', dpi=300)
#     print(f"Figure saved at {file_path}")

#     # Show the plot
#     plt.show()

#     return fig  # Return the figure object

import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_rel

def compare_model_performance(model_eval_df, models, file_id='model_comparison', save_dir='./Figs/',
                              ylim=[0.5, 0.7],
                              plot_individual_ML=False, fig_width=8, x_tick_fontsize=4):
    """
    Compare predictive accuracy (normalized likelihood) across different models and save the figure.
    """

    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import ttest_rel

    os.makedirs(save_dir, exist_ok=True)

    # Updated color mapping
    color_mapping = {
    'common fit': '#1f77b4',         # blue for common fits
    'individual fit': '#2ca02c',     # green for individual fits
    'MAP': '#2ca02c',                # same green tone for MAP fits
    'common_process_RNN': '#ff7f0e', # orange for common_process_RNN
    'IDRNN': '#9467bd',              # violet reference
    'True model': 'lightgray'
    }

    # Generate model groups
    model_groups = group_models(models)

    # --- Extract reference values (now IDRNN) ---
    if 'IDRNN' not in model_eval_df['model'].values:
        print("IDRNN model not found in model_eval_df.")
        return None

    ref_values = model_eval_df[model_eval_df['model'] == 'IDRNN'] \
                    .sort_values(by='session')['normalized_likelihood'].values

    # Extract True model mean for reference line
    true_model_mean = None
    if 'True model' in model_eval_df['model'].values:
        true_model_mean = model_eval_df[model_eval_df['model'] == 'True model']['normalized_likelihood'].mean()

    # Aggregate per-model likelihoods
    aggregated_data = {}
    for group, sub_models in model_groups.items():
        for sub_model in sub_models:
            if sub_model in model_eval_df['model'].values:
                nl_values = model_eval_df[model_eval_df['model'] == sub_model] \
                                .sort_values(by='session')['normalized_likelihood'].values
                aggregated_data.setdefault(group, {}).setdefault(sub_model, []).extend(nl_values)

    # --- Initialize figure ---
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    # Plot True model line
    if true_model_mean is not None:
        ax.axhline(true_model_mean, color='gray', linestyle="--", linewidth=2, alpha=0.5)
        ax.text(1.2, true_model_mean + 0.005, "Ground truth",
                fontsize=12, color='gray', ha='left', va='center', transform=ax.transData)

    positions, labels, significant_models = [], [], []
    current_position = 0

    # --- Plot each group ---
    for group, sub_models in model_groups.items():
        if group == "True model":
            continue

        sub_positions = []
        for sub_model in sub_models:
            if sub_model not in aggregated_data.get(group, {}):
                continue
            if not plot_individual_ML and 'individual fit' in sub_model:
                continue

            nl_values = aggregated_data[group][sub_model]
            mean = np.mean(nl_values)
            error = np.std(nl_values) / np.sqrt(len(nl_values))

            color = (
                color_mapping['common fit'] if 'common fit' in sub_model else
                color_mapping['individual fit'] if 'individual fit' in sub_model else
                color_mapping['MAP'] if 'MAP' in sub_model else
                color_mapping['IDRNN'] if 'IDRNN' in sub_model else 
                color_mapping['common_process_RNN'] 
                     
                )

            # Paired t-test vs IDRNN
            t_stat, p_value = ttest_rel(nl_values, ref_values)
            print(f"{sub_model} vs. IDRNN: t = {t_stat:.3f}, p = {p_value:.3f}")

            # Mark significant models that perform worse than IDRNN
            if p_value < 0.05 and mean < np.mean(ref_values):
                significant_models.append((current_position, mean + 1.1 * error))

            # Plot means + error bars
            ax.errorbar(current_position, mean, yerr=error, fmt='o',
                        color=color, ecolor=color, elinewidth=1.5, capsize=4, markersize=8)
            sub_positions.append(current_position)
            current_position += 0.1

        if sub_positions:
            positions.append(np.mean(sub_positions))
            labels.append(group)
            current_position += 0.5

    # --- Axes & labels ---
    ax.set_ylim(ylim[0], ylim[1])
    ax.set_xlim(min(positions) - 0.3, max(positions) + 0.3)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=8, rotation=30, ha='right')
    ax.set_ylabel("Predictive Accuracy", fontsize=16)

    # Significance asterisks
    for pos, height in significant_models:
        ax.text(pos, height, '*', fontsize=14, ha='center', color='black')

    ax.text(1.0, 1.01,
            "* Lower predictive accuracy than IDRNN (p < 0.05, paired t-test)",
            fontsize=12, color='black', ha='right', va='bottom', transform=ax.transAxes)

    # --- Legend ---
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color=color_mapping['common fit'],
                   label='Cognitive model (common fit)', markersize=8),
        plt.Line2D([0], [0], marker='o', color=color_mapping['individual fit'],
                   label='Cognitive model (individual fit)', markersize=8),
        plt.Line2D([0], [0], marker='o', color=color_mapping['IDRNN'],
                   label='IDRNN (reference)', markersize=8),
        plt.Line2D([0], [0], marker='o', color=color_mapping['common_process_RNN'],
                   label='common_process_RNN', markersize=8)
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=12, frameon=False)


    # --- Save & show ---
    plt.tight_layout()
    file_path = os.path.join(save_dir, f"{file_id}.png")
    plt.savefig(file_path, dpi=300)
    print(f"✅ Figure saved at {file_path}")
    plt.show()

    return fig

def compare_model_performance_original(model_eval_df, models, file_id='model_comparison', save_dir='./Figs/', 
                              ylim = [0.5, 0.7], 
                              plot_individual_ML=False, fig_width=8, x_tick_fontsize=16):
    """
    Compare predictive accuracy (normalized likelihood) across different models and save the figure.

    Parameters:
    - model_eval_df (DataFrame): DataFrame containing model evaluation results with columns:
        'model' (str): Model name.
        'session' (int): Session identifier.
        'normalized_likelihood' (float): Normalized likelihood for each session.
    - models (list): A list of model names.
    - file_id (str): Identifier for the saved file.
    - save_dir (str): Directory where the figure will be saved.
    - plot_individual_fit (bool): Whether to plot 'Individual (ML)'. Default is False.
    - fig_width (int): Width of the figure. Default is 8.

    Returns:
    - fig (matplotlib.figure.Figure): The generated figure.
    """

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Define color mapping for fits
    color_mapping = {
        'common fit': 'C0',
        'individual fit': 'C3',  # Will be removed from legend if plot_individual_fit=False
        'MAP': 'C2', 
        'GRU': 'C4', 
        'True model': 'lightgray' # True model in a less prominent color
    }

    # Generate model groups
    model_groups = group_models(models)

    # Extract session-wise GRU data for comparison
    if 'GRU' not in model_eval_df['model'].values:
        print("GRU model not found in model_eval_df.")
        return None

    gru_values = model_eval_df[model_eval_df['model'] == 'GRU'].sort_values(by='session')['normalized_likelihood'].values

    # Extract True model values for horizontal line plot
    true_model_values = None
    if 'True model' in model_eval_df['model'].values:
        true_model_values = model_eval_df[model_eval_df['model'] == 'True model']['normalized_likelihood'].values
        true_model_mean = np.mean(true_model_values)

    # Aggregate session-wise data
    aggregated_data = {}
    for group, sub_models in model_groups.items():
        for sub_model in sub_models:
            if sub_model in model_eval_df['model'].values:
                nl_values = model_eval_df[model_eval_df['model'] == sub_model].sort_values(by='session')['normalized_likelihood'].values
                aggregated_data.setdefault(group, {}).setdefault(sub_model, []).extend(nl_values)

    # Initialize the plot with a smaller width
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    # If True model exists, plot it as a horizontal line
    if true_model_values is not None:
        ax.axhline(true_model_mean, color='gray', linestyle="--", linewidth=2, alpha=0.5)
        ax.text(1.2, true_model_mean+0.005, "Ground truth", fontsize=12, color='gray', ha='left', va='center', transform=ax.transData)

    # Plot models and perform t-tests
    positions = []
    means = []
    errors = []
    labels = []
    current_position = 0
    significant_models = []  # Models significantly worse than GRU

    for group, sub_models in model_groups.items():
        if group == "True model":  # Skip True model from individual plotting
            continue

        sub_positions = []
        for sub_model in sub_models:
            if sub_model in aggregated_data.get(group, {}):
                if not plot_individual_ML and 'individual fit' in sub_model:
                    continue  # Skip plotting Individual (ML) if option is False
                
                nl_values = aggregated_data[group][sub_model]
                mean = np.mean(nl_values)
                error = np.std(nl_values) / np.sqrt(len(nl_values))
            
                color = (
                    color_mapping['common fit'] if 'common fit' in sub_model else
                    color_mapping['individual fit'] if 'individual fit' in sub_model else
                    color_mapping['MAP'] if 'MAP' in sub_model else
                    color_mapping[group]  # GRU or other models
                )
            
                # Perform paired t-test against GRU
                t_stat, p_value = ttest_rel(nl_values, gru_values)
                print(f"{sub_model} vs. GRU: t = {t_stat:.3f}, p = {p_value:.3f}")

                # If the model is significantly worse (p < 0.05), mark it
                if p_value < 0.05 and mean < np.mean(gru_values):
                    significant_models.append((current_position, mean + 1.1 * error))

                # Plot individual sub-model
                ax.errorbar(
                    current_position, mean, yerr=error, fmt='o', 
                    color=color, ecolor=color, elinewidth=1.5, capsize=4, markersize=8
                )
            
                sub_positions.append(current_position)
                current_position += 0.1  # Close spacing for common and individual fits
    
        # Compute the X-axis position for the group label
        positions.append(np.mean(sub_positions))
        labels.append(group)
        current_position += 0.5  # Add spacing between model groups

    ax.set_ylim(ylim[0],ylim[1])
    ax.set_xlim(min(positions) - 0.3, max(positions) + 0.3)

    # Adjust X-axis and labels
    ax.set_xticks(positions)
    # ax.set_xticklabels(labels, rotation=0, fontsize=12)
    ax.set_ylabel("Predictive Accuracy", fontsize=16) #  (Normalized Likelihood)

    ax.set_xticklabels(labels, fontsize=x_tick_fontsize)
    # ax.set_yticklabels(ax.get_yticks(), fontsize=tick_fontsize)

    # Add asterisks for models significantly worse than GRU
    for pos, height in significant_models:
        ax.text(pos, height, '*', fontsize=14, ha='center', color='black')

    ax.text(
        1.0, 1.01,  # X and Y position (slightly outside the top-right)
        "* Lower predictive accuracy than GRU (p < 0.05, paired t-test)",
        fontsize=12, color='black', ha='right', va='bottom', transform=ax.transAxes
    )

    # Add a legend for common and individual fits
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color=color_mapping['common fit'], label='Common fit', markersize=8)
    ]

    # Include Individual (ML) in legend only if it's plotted
    if plot_individual_ML:
        legend_elements.append(plt.Line2D([0], [0], marker='o', color=color_mapping['individual fit'], label='Individual (ML)', markersize=8))
        
    legend_elements.append(plt.Line2D([0], [0], marker='o', color=color_mapping['MAP'], label='Individual fit (MAP)', markersize=8))

    ax.legend(handles=legend_elements, loc='upper left', fontsize=12, frameon=False)

    # Adjust layout and save the figure
    plt.tight_layout()
    file_path = os.path.join(save_dir, f"{file_id}.eps")
    plt.savefig(file_path, format='eps', dpi=300)
    print(f"Figure saved at {file_path}")

    # Show the plot
    plt.show()

    return fig

# def compare_model_performance(model_eval_df, models, file_id='model_comparison', save_dir='./Figs/'):
#     """
#     Compare predictive accuracy (normalized likelihood) across different models and save the figure.

#     Parameters:
#     - model_eval_df (DataFrame): DataFrame containing model evaluation results with columns:
#         'model' (str): Model name.
#         'session' (int): Session identifier.
#         'normalized_likelihood' (float): Normalized likelihood for each session.
#     - models (list): A list of model names.
#     - file_id (str): Identifier for the saved file.
#     - save_dir (str): Directory where the figure will be saved.

#     Returns:
#     - fig (matplotlib.figure.Figure): The generated figure.
#     """

#     # Ensure save directory exists
#     os.makedirs(save_dir, exist_ok=True)

#     # Define color mapping for fits
#     color_mapping = {
#         'common fit': 'blue',
#         'individual fit': 'orange',
#         'MAP': 'green', 
#         'GRU': 'red',        # GRU in a prominent color
#         'True model': 'gray' # True model in a less prominent color
#     }

#     # Generate model groups
#     model_groups = group_models(models)

#     # Extract session-wise GRU data for comparison
#     if 'GRU' not in model_eval_df['model'].values:
#         print("GRU model not found in model_eval_df.")
#         return None

#     gru_values = model_eval_df[model_eval_df['model'] == 'GRU'].sort_values(by='session')['normalized_likelihood'].values

#     # Aggregate session-wise data
#     aggregated_data = {}
#     for group, sub_models in model_groups.items():
#         for sub_model in sub_models:
#             if sub_model in model_eval_df['model'].values:
#                 nl_values = model_eval_df[model_eval_df['model'] == sub_model].sort_values(by='session')['normalized_likelihood'].values
#                 aggregated_data.setdefault(group, {}).setdefault(sub_model, []).extend(nl_values)

#     # Initialize the plot
#     fig, ax = plt.subplots(figsize=(10, 6))

#     # Plot models and perform t-tests
#     positions = []
#     means = []
#     errors = []
#     labels = []
#     current_position = 0
#     significant_models = []  # Models significantly worse than GRU

#     for group, sub_models in model_groups.items():
#         sub_positions = []
#         for sub_model in sub_models:
#             if sub_model in aggregated_data.get(group, {}):
#                 nl_values = aggregated_data[group][sub_model]
#                 mean = np.mean(nl_values)
#                 error = np.std(nl_values) / np.sqrt(len(nl_values))
            
#                 color = (
#                     color_mapping['common fit'] if 'common fit' in sub_model else
#                     color_mapping['individual fit'] if 'individual fit' in sub_model else
#                     color_mapping['MAP'] if 'MAP' in sub_model else
#                     color_mapping[group]  # GRU or True model
#                 )
            
#                 # Perform paired t-test against GRU
#                 t_stat, p_value = ttest_rel(nl_values, gru_values)
#                 print(f"{sub_model} vs. GRU: t = {t_stat:.3f}, p = {p_value:.3f}")

#                 # If the model is significantly worse (p < 0.05), mark it
#                 if p_value < 0.05 and mean < np.mean(gru_values):
#                     significant_models.append((current_position, mean + 1.1 * error))

#                 # Plot individual sub-model
#                 ax.errorbar(
#                     current_position, mean, yerr=error, fmt='o', 
#                     color=color, ecolor=color, elinewidth=1.5, capsize=4, markersize=8
#                 )
            
#                 sub_positions.append(current_position)
#                 current_position += 0.1  # Close spacing for common and individual fits
    
#         # Compute the X-axis position for the group label
#         positions.append(np.mean(sub_positions))
#         labels.append(group)
#         current_position += 0.5  # Add spacing between model groups

#     # Adjust X-axis and labels
#     ax.set_xticks(positions)
#     ax.set_xticklabels(labels, rotation=0, fontsize=12)
#     ax.set_ylabel("Predictive Accuracy (Normalized Likelihood)", fontsize=14)
#     #ax.set_title("Model Fits", fontsize=16)

#     # Add asterisks for models significantly worse than GRU
#     for pos, height in significant_models:
#         ax.text(pos, height, '*', fontsize=14, ha='center', color='black')

#     ax.text(
#         1.0, 1.01,  # X and Y position (slightly outside the top-right)
#         "* Lower predictive accuracy than GRU (p < 0.05, paired t-test)",
#         fontsize=12, color='black', ha='right', va='bottom', transform=ax.transAxes
#     )


#     # Add a legend for common and individual fits
#     legend_elements = [
#         plt.Line2D([0], [0], marker='o', color=color_mapping['common fit'], label='Common fit', markersize=8),
#         plt.Line2D([0], [0], marker='o', color=color_mapping['individual fit'], label='Individual (ML)', markersize=8),
#         plt.Line2D([0], [0], marker='o', color=color_mapping['MAP'], label='Individual (MAP)', markersize=8) 
#     ]
#     ax.legend(handles=legend_elements, loc='upper left', fontsize=12, frameon=False)

#     # Adjust layout and save the figure
#     plt.tight_layout()
#     file_path = os.path.join(save_dir, f"{file_id}.eps")
#     plt.savefig(file_path, format='eps', dpi=300)
#     print(f"Figure saved at {file_path}")

#     # Show the plot
#     plt.show()

#     return fig  # Return the figure object

# import os
# import matplotlib.pyplot as plt
# import numpy as np
# from scipy.stats import ttest_rel

# def compare_model_performance(model_eval_df, file_id='model_comparison', save_dir='./Figs/',
#                               plot_individual_fit=False, fontsize=14, fig_width=8):
#     """
#     Compare predictive accuracy (normalized likelihood) across different models and save the figure.

#     Parameters:
#     - model_eval_df (DataFrame): DataFrame containing model evaluation results with columns:
#         'model' (str): Model name.
#         'session' (int): Session identifier.
#         'normalized_likelihood' (float): Normalized likelihood for each session.
#     - file_id (str): Identifier for the saved file.
#     - save_dir (str): Directory where the figure will be saved.
#     - plot_individual_fit (bool): Whether to include 'individual fit' models in the plot. Default is False.
#     - fontsize (int): Font size for labels and annotations.
#     - fig_width (float): Width of the figure. Default is 8.

#     Returns:
#     - fig (matplotlib.figure.Figure): The generated figure.
#     """

#     # Ensure save directory exists
#     os.makedirs(save_dir, exist_ok=True)

#     # Define color mapping
#     color_mapping = {
#         'common fit': 'blue',
#         'individual fit': 'orange',
#         'MAP': 'green',
#         'GRU': 'red',        # GRU in a prominent color
#         'True model': 'gray' # True model in a less prominent color
#     }

#     # Process models
#     model_eval_df['model_group'] = model_eval_df['model'].apply(
#         lambda x: "GRU" if "GRU" in x else 
#                   "True model" if "True model" in x else 
#                   x.split(" (")[0]
#     )

#     # Extract session-wise GRU data for comparison
#     if "GRU" not in model_eval_df['model_group'].values:
#         print("GRU model not found in model_eval_df.")
#         return None

#     gru_values = model_eval_df[model_eval_df['model_group'] == "GRU"].sort_values(by='session')['normalized_likelihood'].values

#     # Aggregate session-wise data
#     aggregated_data = model_eval_df.groupby(["model_group", "session"])["normalized_likelihood"].agg(list).to_dict()

#     # Initialize the plot
#     fig, ax = plt.subplots(figsize=(fig_width, 6))

#     # Plot models and perform t-tests
#     positions = []
#     labels = []
#     current_position = 0
#     significant_models = []  # Models significantly worse than GRU

#     for group in model_eval_df['model_group'].unique():
#         if group == "GRU":
#             color = color_mapping["GRU"]
#         elif group == "True model":
#             color = color_mapping["True model"]
#         else:
#             color = (
#                 color_mapping['common fit'] if 'common fit' in group else
#                 color_mapping['individual fit'] if 'individual fit' in group else
#                 color_mapping['MAP'] if 'MAP' in group else 'black'  # Default color
#             )

#         # Skip 'individual fit' models if the option is disabled
#         if not plot_individual_fit and 'individual fit' in group:
#             continue

#         if group in aggregated_data:
#             nl_values = [np.mean(aggregated_data[(group, s)]) for s in range(len(gru_values))]
#             mean = np.mean(nl_values)
#             error = np.std(nl_values) / np.sqrt(len(nl_values))

#             # True model: plot as a horizontal gray line
#             if group == "True model":
#                 ax.axhline(mean, color=color, linestyle="--", linewidth=2, alpha=0.5)
#                 ax.text(current_position, mean + 0.02, "Ground truth", fontsize=fontsize, color=color, ha="right")
#             else:
#                 # Perform paired t-test against GRU
#                 t_stat, p_value = ttest_rel(nl_values, gru_values)
#                 print(f"{group} vs. GRU: t = {t_stat:.3f}, p = {p_value:.3f}")

#                 # If the model is significantly worse (p < 0.05), mark it
#                 if p_value < 0.05 and mean < np.mean(gru_values):
#                     significant_models.append((current_position, mean + 1.1 * error))

#                 # Plot individual model
#                 ax.errorbar(
#                     current_position, mean, yerr=error, fmt='o',
#                     color=color, ecolor=color, elinewidth=1.5, capsize=4, markersize=8
#                 )

#             positions.append(current_position)
#             labels.append(group)
#             current_position += 0.5  # Add spacing between models

#     # Adjust X-axis and labels
#     ax.set_xticks(positions)
#     ax.set_xticklabels(labels, rotation=0, fontsize=fontsize)
#     ax.set_ylabel("Predictive Accuracy (Normalized Likelihood)", fontsize=fontsize)

#     # Add asterisks for models significantly worse than GRU
#     for pos, height in significant_models:
#         ax.text(pos, height, '*', fontsize=fontsize, ha='center', color='black')

#     ax.text(
#         1.0, 1.01,  # X and Y position (slightly outside the top-right)
#         "* Lower predictive accuracy than GRU (p < 0.05, paired t-test)",
#         fontsize=fontsize - 2, color='black', ha='right', va='bottom', transform=ax.transAxes
#     )

#     # Add a legend for common and individual fits
#     legend_elements = [
#         plt.Line2D([0], [0], marker='o', color=color_mapping['common fit'], label='Common fit', markersize=8),
#         plt.Line2D([0], [0], marker='o', color=color_mapping['individual fit'], label='Individual (ML)', markersize=8),
#         plt.Line2D([0], [0], marker='o', color=color_mapping['MAP'], label='Individual (MAP)', markersize=8)
#     ]
#     ax.legend(handles=legend_elements, loc='upper left', fontsize=fontsize - 2, frameon=False)

#     # Adjust layout and save the figure
#     plt.tight_layout()
#     file_path = os.path.join(save_dir, f"{file_id}.eps")
#     plt.savefig(file_path, format='eps', dpi=300)
#     print(f"Figure saved at {file_path}")

#     # Show the plot
#     plt.show()

#     return fig  # Return the figure object


def plot_individual_predictive_accuracy(model_eval_df):
    """
    Visualizes session-wise predictive accuracy across models by plotting individual session data points 
    connected by lines, along with means and standard error bars.

    Parameters:
    - model_eval_df (DataFrame): DataFrame containing model evaluation results with columns:
        'model' (str): Model name.
        'session' (int): Session identifier.
        'normalized_likelihood' (float): Normalized likelihood for each session.

    Returns:
    - None (displays a plot showing session-wise predictive accuracy)
    """
    # Define models for pairwise comparisons
    models = model_eval_df['model'].unique()

    # Create a dictionary to store session-wise nl values for each model
    sessionwise_data = {
        model: model_eval_df[model_eval_df['model'] == model]
        .sort_values(by='session')['normalized_likelihood'].values for model in models
    }

    # Initialize the plot
    plt.figure(figsize=(8, 6))

    # Plot thin gray lines connecting the same sessions across models
    for session in model_eval_df['session'].unique():
        session_data = model_eval_df[model_eval_df['session'] == session]
        plt.plot(
            session_data['model'], session_data['normalized_likelihood'],
            color='gray', alpha=0.5, linewidth=0.5
        )

    # Compute mean and standard error for each model
    positions = range(len(models))
    nl_values = [sessionwise_data[model] for model in models]
    means = [np.mean(values) for values in nl_values]
    errors = [np.std(values) / np.sqrt(len(values)) for values in nl_values]

    # Plot means with error bars
    plt.errorbar(
        positions, means, yerr=errors, fmt='o', color='black', 
        ecolor='black', elinewidth=1.5, capsize=0, markersize=10, label="Mean ± SE"
    )

    # Overlay individual session data points
    for pos, model in zip(positions, models):
        plt.scatter(
            [pos] * len(sessionwise_data[model]),
            sessionwise_data[model],
            color='gray', alpha=0.5, s=20
        )

    # Add labels, title, and adjust the plot
    plt.xticks(positions, models, rotation=45)
    plt.ylabel("Predictive Accuracy (Normalized Likelihood)", fontsize=14)
    plt.title("Model Fits (Session-wise)", fontsize=16)
    plt.legend(loc='upper left', fontsize=12, frameon=False)

    plt.tight_layout()
    plt.show()


#import numpy as np
#import matplotlib.pyplot as plt

def plot_prediction_RNN_RL(p_true, rewards_seq, c, ax, maxTrial=None, title=None, p_RL=None, p_RNN=None):
    """
    Plots the session probabilities along with reward markers.

    Parameters:
    - p_true: True probabilities (must be provided).
    - rewards_seq: Rewards sequence.
    - c: Choices made during sessions.
    - ax: Axes object for plotting.
    - maxTrial: Maximum number of trials to display.
    - title: Title for the plot.
    - p_RL: RL model probabilities (optional).
    - p_RNN: RNN model probabilities (optional).
    """

    # Plot true probabilities with a thick gray line
    ax.plot(p_true[0], label='True model', color='gray', linewidth=1.5, zorder=2)

    # If p_RNN is provided, plot p_RL first with a lighter green and lower zorder
    if p_RNN is not None and p_RL is not None:
        ax.plot(p_RL[0], label='RL (common fit)', color='lightblue', linewidth=2.5, zorder=1)  # Thin green
    elif p_RL is not None:
        ax.plot(p_RL[0], label='RL (common fit)', color='C0', linewidth=1.5, zorder=3)  # Normal C0 color

    # If p_RNN is provided, plot it in the foreground
    if p_RNN is not None:
        ax.plot(p_RNN[0], label='RNN', color='C1', linewidth=1.8, zorder=4)

    # Extract rewards and choices for the first session and set max trials for display
    maxTrialPlot = min(maxTrial, c.shape[1]) if maxTrial else c.shape[1]
    c_session = c[0, :maxTrialPlot]
    
    # Indices for plotting
    indices = np.arange(0, maxTrialPlot)
    
    # Rewards based on choices
    reward_for_choices = rewards_seq[c_session, indices]
    
    # Plot choice markers
    ax.vlines(indices[c_session == 0], ymin=1.01, ymax=1.04, colors='k', linewidths=1, zorder=5)
    ax.vlines(indices[c_session == 1], ymin=-0.04, ymax=-0.01, colors='k', linewidths=1, zorder=5)
    
    # Determine y-values for rewards
    y_values_A = [1.06] * len(indices[c_session == 0][reward_for_choices[c_session == 0] == 1])
    y_values_B = [-0.06] * len(indices[c_session == 1][reward_for_choices[c_session == 1] == 1])
    
    # Plot rewards for choice 0 (Option A)
    ax.scatter(indices[c_session == 0][reward_for_choices[c_session == 0] == 1], 
               y_values_A, label='Reward for A', s=10, marker='v', color="darkblue", zorder=6)
    
    # Plot rewards for choice 1 (Option B)
    ax.scatter(indices[c_session == 1][reward_for_choices[c_session == 1] == 1], 
               y_values_B, label='Reward for B', s=10, marker='^', color="darkred", zorder=6)
    
    
    # Compute KL-divergence if possible
    avg_kl_div = None
    epsilon = 1e-10  # Small value to avoid log(0)

    if p_RNN is not None:
        pooled_p_true = p_true[0] #np.concatenate(p_true)
        pooled_p_RNN = p_RNN[0] #np.concatenate(p_RNN)
        kl_div = pooled_p_true * np.log((pooled_p_true + epsilon) / (pooled_p_RNN + epsilon)) + \
                 (1 - pooled_p_true) * np.log((1 - pooled_p_true + epsilon) / (1 - pooled_p_RNN + epsilon))
        avg_kl_div = np.mean(kl_div)
    elif p_RL is not None:
        pooled_p_true = p_true[0]
        pooled_p_RL = p_RL[0]
        kl_div = pooled_p_true * np.log((pooled_p_true + epsilon) / (pooled_p_RL + epsilon)) + \
                 (1 - pooled_p_true) * np.log((1 - pooled_p_true + epsilon) / (1 - pooled_p_RL + epsilon))
        avg_kl_div = np.mean(kl_div)

    # Set title with KL-divergence information if computed
    if title and avg_kl_div is not None:
        ax.set_title(f"{title} (KL Divergence: {avg_kl_div:.4f})")
    elif avg_kl_div is not None:
        ax.set_title(f"Session Probabilities (KL Divergence: {avg_kl_div:.4f})")
    elif title:
        ax.set_title(title)
    else:
        ax.set_title("Session Probabilities")
    # # Set title
    # if title:
    #     ax.set_title(title)
    # else:
    #     ax.set_title("Session Probabilities")

    # Set labels, limits, and legend
    ax.set_xlabel('Trial number')
    ax.set_ylabel('Prob. choosing A')
    ax.set_xlim(0, maxTrialPlot)

    # Ensure legend order (RNN first if present)
    handles, labels = ax.get_legend_handles_labels()
    if p_RNN is not None and 'RNN' in labels:
        idx = labels.index('RNN')
        handles.insert(0, handles.pop(idx))
        labels.insert(0, labels.pop(idx))
    ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)


def plot_latents(dim_reduction, latent_tensor, avg, n_components, name, df=None, df_train=None, group_col="group", subject_col = "sub_across_groups", latent_tensor_train = None, param_array = None):
    if "model" in name:
        latents = torch.from_numpy(latent_tensor) #just to make this work for code below
        title = "fitted_" + name
        print("heureka")
    elif avg:
        latents = latent_tensor.mean(dim=1)
        title = "average_latents" + name
    else:
        latents = latent_tensor[:,-1,:] #last latent
        title = "last_latent" + name
    if dim_reduction == "tsne":
        # Apply t-SNE
        tsne_avg = TSNE(n_components=n_components, perplexity=10, random_state=42)
        latents_reduced = tsne_avg.fit_transform(latents.cpu().numpy())
    elif dim_reduction == "mds":
        # Apply Multidimensional Scaling (MDS)
        mds = MDS(n_components=n_components, random_state=42, dissimilarity="euclidean", n_init=4, max_iter=300)
        latents_reduced = mds.fit_transform(latents.cpu().numpy())
    elif dim_reduction == "pca":
        pca = PCA(n_components=n_components)
        latents_reduced = pca.fit_transform(latents.cpu().numpy())
    if name == 'sloutsky':
        color_map = {"Coin CollectorV6": 0,"Coin CollectorV5": 0, "Coin Collector": 1} #"Coin CollectorV5": 0
    elif name == 'palminteri':
        color_map = {"adolescent": 0, "adult": 1}


    # Color-code: first 100 = 0, next 100 = 1
    
    if df is not None and group_col in df.columns:
        df_unique = df.drop_duplicates(subset=subject_col)
        group_vals = df_unique[group_col].values
        
        colors = [color_map[val] for val in group_vals]
    
    elif param_array is not None:
        """n_participants = latents_reduced.shape[0]
        assert n_participants >= 200, "Expected at least 200 participants"
        colors = [0 if i < 100 else 1 for i in range(n_participants)]"""
        colors = param_array
    
    if df_train is not None:
        dftrain_unique = df_train.drop_duplicates(subset=subject_col)
        group_vals_train = dftrain_unique[group_col].values
        colors_train = [color_map[val] for val in group_vals_train]

    # Plot
    plt.figure(figsize=(6, 6))
    plt.scatter(latents_reduced[:, 0], latents_reduced[:, 1], c=colors, cmap='coolwarm')
    plt.title(f"{title}")
    plt.xlabel(f"{dim_reduction} 1")
    plt.ylabel(f"{dim_reduction} 2")
    plt.grid(True)
    plt.savefig(f"plots/dim_reduction_latents{title}.png")
    plt.show()
    print(f"plot saved under plots/dim_reduction_latents{title}.png")

def plot_LDA(latent_tensor, latent_tensor_train, df_test, df_train, dataset_tag, group_col = "group"):
    #run LDA cross-validated
    if dataset_tag == "sloutsky":

        if group_col in df_test.columns:
            latent_tensor = latent_tensor.reshape(latent_tensor.shape[0], -1)
            #latent_tensor_train = latent_tensor_train.reshape(latent_tensor_train.shape[0], -1)
            #latent_overall = torch.cat((latent_tensor_train, latent_tensor), dim=0).detach().cpu().numpy()
            latent_overall = latent_tensor.detach().cpu().numpy()
            df_overall = df_test#pd.concat([df_train, df_test]) #df_train
            df_unique = df_overall.drop_duplicates(subset="subid")
            group_vals = df_unique[group_col].values
            #dftrain_unique = df_train.drop_duplicates(subset="sub_across_groups")
            #group_vals_train = dftrain_unique[group_col].values
            color_map = {"Coin CollectorV6": 0, "Coin CollectorV5": 0, "Coin Collector": 1} #Coin CollectorV6 
            y = np.array([color_map[val] for val in group_vals])
            #colors_train = [color_map[val] for val in group_vals_train]
            idx_adol = np.where(y == 0)[0]
            idx_adult = np.where(y == 1)[0]
            n_folds = min(len(idx_adol), len(idx_adult))
            z_all = np.zeros(len(y))
        else:
            print("df seems to have no group structure")

    for i in range(n_folds):
        test_idx = np.array([idx_adol[i], idx_adult[i]])
        train_idx = np.setdiff1d(np.arange(len(y)), test_idx)

        X_train, y_train = latent_overall[train_idx], y[train_idx]
        X_test, y_test = latent_overall[test_idx], y[test_idx]

        # standardize
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)

        # fit LDA
        lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
        lda.fit(X_train_s, y_train)

        # manually project test participants by extracting projection vector
        w = lda.coef_.ravel()
        if np.mean(X_train_s[y_train == 1] @ w) < np.mean(X_train_s[y_train == 0] @ w):
            w = -w
        z_train = X_train_s @ w
        z_test  = X_test_s  @ w
        z_test = z_test - np.mean(z_train)
        z_all[test_idx] = z_test
    y_jitter = 0.05 * np.random.randn(len(y))

    plt.figure(figsize=(7,3))
    plt.scatter(z_all[y==0], y_jitter[y==0], color='dodgerblue', alpha=0.7, label='Adolescents', edgecolor='k', linewidths=0.3)
    plt.scatter(z_all[y==1], y_jitter[y==1]+0.15, color='tomato', alpha=0.7, label='Adults', edgecolor='k', linewidths=0.3)

    plt.yticks([])
    plt.xlabel("Cross-validated LDA projection (age axis)")
    plt.title("Participant positions along latent age axis")
    plt.legend()
    plt.grid(True, axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"plots/dim_reduction_latents_{dataset_tag}_data.png")
    plt.show()
    print(f"plot saved under plots/dim_reduction_latents_{dataset_tag}_data.png")


def latent_euclidian_distance(X, y_class, model_tag, latent_mode):

    # 1️⃣ Pairwise Euclidean distance matrix
    D = squareform(pdist(X, metric='euclidean'))  # shape (N, N)

    # 2️⃣ Boolean masks
    mask_00 = np.outer(y_class==0, y_class==0)   # pairs both in class 0
    mask_11 = np.outer(y_class==1, y_class==1)   # pairs both in class 1
    mask_between = np.outer(y_class==0, y_class==1) | np.outer(y_class==1, y_class==0)

    # Remove diagonal (self-distance = 0)
    np.fill_diagonal(mask_00, False)
    np.fill_diagonal(mask_11, False)

    # 3️⃣ Compute mean distances
    within_0 = D[mask_00].mean()
    within_1 = D[mask_11].mean()
    between  = D[mask_between].mean()

    print(f"model: {model_tag}, latent mode: {latent_mode}")
    print(f"Within-class (α=low):  {within_0:.3f}")
    print(f"Within-class (α=high): {within_1:.3f}")
    print(f"Between-class:         {between:.3f}")

    # 4️⃣ Separation ratio
    sep_ratio = between / ((within_0 + within_1)/2)
    print(f"Separation ratio (between / within): {sep_ratio:.2f}")

    results = pd.DataFrame([{
        "within_low": within_0,
        "within_high": within_1,
        "between": between,
        "sep_ratio": sep_ratio
    }])

    results.to_csv(f"data/latent_distance_metrics_{model_tag}_{latent_mode}.csv", index=False)
    return print(f"\nSaved results to data/latent_distance_metrics_{model_tag}_{latent_mode}.csv")

def fit_gmm_bic(latent_tensor, model_name, mode="avg", max_components=10, random_state=42, plot=True):
    """
    Fit Gaussian Mixture Models to latent RNN representations and compute BIC.

    Parameters
    ----------
    latent_tensor : torch.Tensor or np.ndarray
        Tensor of shape (n_participants, n_trials, z_dim)
    mode : str
        "all"  -> use all hidden states (flatten trials)
        "avg"  -> use average hidden per participant
        "last" -> use last hidden per participant
    max_components : int
        Maximum number of mixture components to test
    random_state : int
        Random seed
    plot : bool
        Whether to plot BIC curve

    Returns
    -------
    best_gmm : GaussianMixture
        Fitted GMM with lowest BIC
    bics : list of float
        BIC scores for all component counts
    best_k : int
        Number of components in best model
    latents_used : np.ndarray
        Latent representations used for clustering (shape depends on mode)
    """
    # Convert to numpy if needed
    if hasattr(latent_tensor, "detach"):
        latent_tensor = latent_tensor.detach().cpu().numpy()

    n_participants, n_trials, z_dim = latent_tensor.shape

    # Select representation mode
    if mode == "all":
        latents_used = latent_tensor.reshape(-1, z_dim)
    elif mode == "avg":
        latents_used = latent_tensor.mean(axis=1)  # shape (n_participants, z_dim)
    elif mode == "final":
        latents_used = latent_tensor[:, -1, :]     # shape (n_participants, z_dim)
    else:
        raise ValueError("mode must be 'all', 'avg', or 'final'")

    # Fit GMMs with varying number of clusters
    bics = []
    gmms = []
    for k in range(1, max_components + 1):
        gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=random_state)
        gmm.fit(latents_used)
        bic = gmm.bic(latents_used)
        bics.append(bic)
        gmms.append(gmm)

    # Select best model
    best_k = np.argmin(bics) + 1
    best_gmm = gmms[best_k - 1]

    # Optionally plot BIC curve
    if plot:
        plt.figure(figsize=(6, 4))
        plt.plot(range(1, max_components + 1), bics, marker='o')
        plt.xlabel('Number of components (k)')
        plt.ylabel('BIC')
        plt.title(f'GMM BIC ({mode}-hidden representation)')
        plt.savefig(f"plots/gmm_bic_{model_name}_{mode}.png")
        plt.show()

    print(f"Mode: {mode}")
    print(f"Best number of components: {best_k}")
    print(f"BIC of best model: {bics[best_k - 1]:.2f}")

    return best_gmm, bics, best_k, latents_used



def compute_bic_over_trials(latent_tensor, max_components=10, random_state=42, latent=True):
    """
    Compute best BIC over trials for one latent tensor.

    Parameters
    ----------
    latent_tensor : torch.Tensor or np.ndarray
        Shape (n_participants, n_trials, z_dim)
    max_components : int
        Maximum number of GMM components to test
    random_state : int
        Random seed

    Returns
    -------
    best_bics : np.ndarray
        Array of best BIC values per trial (shape: n_trials)
    best_ks : np.ndarray
        Corresponding best number of components per trial
    """
    if hasattr(latent_tensor, "detach"):
        latent_tensor = latent_tensor.detach().cpu().numpy()

    n_participants, n_trials, z_dim = latent_tensor.shape
    best_bics = np.zeros(n_trials)
    best_ks = np.zeros(n_trials, dtype=int)

    for t in range(n_trials):
        if latent:
            latents_t = latent_tensor[:, t, :]  # hidden states at this time step
        else:
            if t == 0:
                # cannot take mean over empty slice -> just take first trial
                latents_t = latent_tensor[:, 0, :]
            else:
                latents_t = latent_tensor[:, :t, :].mean(axis=1)
        bics_t = []
        for k in range(1, max_components + 1):
            gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=random_state)
            gmm.fit(latents_t)
            bics_t.append(gmm.bic(latents_t))
        best_idx = np.argmin(bics_t)
        best_bics[t] = bics_t[best_idx]
        best_ks[t] = best_idx + 1

    return best_bics, best_ks


def plot_bic_trajectories(latent1, model1_name, latent2, model2_name, max_components=10):
    """
    Compare BIC trajectories across trials for two models.
    """
    bics1, k1 = compute_bic_over_trials(latent1, max_components=max_components)
    bics2, k2 = compute_bic_over_trials(latent2, max_components=max_components, latent=False)

    plt.figure(figsize=(7,4))
    plt.plot(k1, label=model1_name, marker='o')
    plt.plot(k2, label=model2_name, marker='s')
    plt.xlabel("Trial")
    plt.ylabel("Number of gaussian mixtures")
    plt.title("Number of mixtures over sequence")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"plots/best_k_over_time.png")
    plt.show()

    print("Average best k per model:")
    print(f"Model 1: mean k = {np.mean(k1):.2f}")
    print(f"Model 2: mean k = {np.mean(k2):.2f}")

    return bics1, bics2, k1, k2


