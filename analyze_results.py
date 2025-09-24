import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plot_functions as plf
import os

# Load data
model_eval_df = pd.read_csv("data/model_eval_df.csv")
rnn_df = pd.read_csv("data/rnn_results.csv")
p1_common_dict = dict(np.load("data/p1_common_dict.npz", allow_pickle=True))
df_train = pd.read_csv("data/df_train.csv")

os.makedirs("plots", exist_ok=True)
print(f"p1 common dict: {p1_common_dict}")
print(f"df train: {df_train.head(50)}")
print(f"model eval df: {model_eval_df.head(50)}")
# Plot p1 for a single session
selected_session = 0
plt.figure(figsize=(10, 6))
for model_name, p1_common in p1_common_dict.items():
    session_indices = df_train['session'] == selected_session
    trials = range(session_indices.sum())
    plt.plot(trials, p1_common[session_indices], label=model_name)

plt.xlabel("Trial")
plt.ylabel("p1 (Prob choose option 1)")
plt.title(f"Comparison of p1 across models (Session {selected_session})")
plt.legend()
plt.savefig("plots/p1_comparison.png", dpi=150)
plt.show()

for model_name, p1_common in p1_common_dict.items():
    # Extract the p1 values for the selected session
    session_indices = df_train['session'] == selected_session
    trials = range(session_indices.sum())  # Number of trials in the selected session
    p1_common_selected = p1_common[session_indices]

    plt.plot(trials, p1_common_selected, label=model_name)

plt.xlabel('Trial')
plt.ylabel('Probability of choosing option 1 (p1)')
plt.title(f'Comparison of p1_common across models (Session {selected_session})')
plt.legend()
plt.savefig("cog_models_p0_common.png", dpi=150)
plt.show()
print("saved p1 plot")


# Compare model performance
models = model_eval_df["model"].unique()
fig = plf.compare_model_performance(
    model_eval_df, models, 
    file_id="model_comparison",
    save_dir="plots/",
    ylim=[0.5, 0.64],
    plot_individual_ML=False,
    fig_width=7,
    x_tick_fontsize=18
)
fig.savefig("plots/model_comparison.png")
print("✅ Plots saved to plots/")

#load necessary data 
rewardsTrain = np.load("data/rewards.npy") 
p0_epoch = pA_per_epoch["1321"] # look up best epoch - this has to be handled automatically in the future 
idxSub = 0 
df_sub = df_train[df_train['session'] == idxSub].copy() 
pA = df_sub['p'] 
c = df_sub['c'] 
session_indices = df_train['session'] == idxSub 
p1_common_selected = p1_common_dict['FQ'][session_indices] 
plf.plot_prediction_RNN_RL([pA], rewardsTrain, c.copy().values.reshape(1, -1), 
                           axes[i,0], maxTrial=200, title=f"Iteration 1300, Subject {idxSub+1}", 
                           p_RL=[p1_common_selected], p_RNN = [p0_epoch[idxSub, :, 0]]) 
idxSub = 50 
df_sub = df_train[df_train['session'] == idxSub].copy() 
pA = df_sub['p'] 
c = df_sub['c'] 
session_indices = df_train['session'] == idxSub 
p1_common_selected = p1_common_dict['FQ'][session_indices] 
plf.plot_prediction_RNN_RL([pA], rewardsTrain, c.copy().values.reshape(1, -1), 
                           axes[i,0], maxTrial=200, title=f"Iteration 1300, Subject {idxSub+1}", 
                           p_RL=[p1_common_selected], p_RNN = [p0_epoch[idxSub, :, 0]]) 
# Adjust layout and save the figure 
plt.tight_layout() 
plt.savefig(f'plots/prediction_time_series_for_different_steps.png') 
plt.show()