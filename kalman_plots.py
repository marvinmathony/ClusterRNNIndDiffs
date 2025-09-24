import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

df = pd.read_csv("data/part_param_recovery_results_lessem_truncated_prior_30EM2.csv")
correlation_beta = df['true_beta'].corr(df['recovered_beta'])
correlation_gamma = df['true_gamma'].corr(df['recovered_gamma'])

print("Correlation beta:", correlation_beta)
print("Correlation gamma:", correlation_gamma)

def plot_param_recovery(true_vals, recovered_vals, param_name):
    plt.figure(figsize=(6, 6))
    plt.scatter(true_vals, recovered_vals, alpha=0.7)
    max_val = max(max(true_vals), max(recovered_vals))
    min_val = min(min(true_vals), min(recovered_vals))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='Perfect correlation')
    plt.xlabel(f'True {param_name}')
    plt.ylabel(f'Recovered {param_name}')
    plt.title(f'{param_name.capitalize()} Parameter Recovery')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    #save plot under plots/kalman_recovery_{param_name}.png
    plt.savefig(f'plots/kalman_recovery_{param_name}.png')

df2 = pd.read_csv("data/eb_all_runs_truncated_prior_30EM2.csv")

# Plot average across runs
df_mean = df2.groupby("iteration")[["var_beta", "var_gamma"]].mean()

plt.figure(figsize=(8, 4))
plt.plot(df_mean["var_beta"], label="Var(beta)", linewidth=2)
plt.plot(df_mean["var_gamma"], label="Var(gamma)", linewidth=2)
plt.xlabel("EM Iteration")
plt.ylabel("Variance")
plt.title("Empirical Bayes Variance Across Runs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig('plots/variance_EM.png')

# Plot for beta
plot_param_recovery(df['true_beta'], df['recovered_beta'], 'beta')

# Plot for gamma
plot_param_recovery(df['true_gamma'], df['recovered_gamma'], 'gamma')



# Load the two uploaded CSV files
file1 = "data/kalman_simulation_results_em30_noID.csv"
file2 = "data/kalman_simulation_results_truncated_prior_30EM2.csv"

df3 = pd.read_csv(file1)
df4 = pd.read_csv(file2)

# Combine the dataframes
df = pd.concat([df3, df4], ignore_index=True)

# Define the ground truth and predictions
# If with_id is True, expected winner is 'sampled'
# If with_id is False, expected winner is 'fixed'
df['expected_winner'] = df['with_id'].map({True: 'sampled', False: 'fixed'})

# Generate confusion matrix
cm = confusion_matrix(df['expected_winner'], df['model_winner'], labels=['sampled', 'fixed'])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['ID', 'no ID'])

# Display confusion matrix

fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(ax=ax)

# Customize axis labels
ax.set_xlabel("Model Comparison Result", fontsize=12)
ax.set_ylabel("Data Generating Process", fontsize=12)
plt.tight_layout()
plt.title("Simulation and Model Comparison", fontsize=14)
plt.show()
plt.savefig('plots/confusion_matrix.png')


