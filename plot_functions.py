import numpy as np
import matplotlib.pyplot as plt

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
        'GRU': 'C1',        # GRU in a prominent color
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

    return fig  # Return the figure object


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
        pooled_p_true = np.concatenate(p_true)
        pooled_p_RL = np.concatenate(p_RL)
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
