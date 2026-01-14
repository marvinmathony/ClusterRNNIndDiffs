import sim_Q_data as sim
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

# Parse arguments
parser = argparse.ArgumentParser(description="Generate synthetic data")
parser.add_argument('--dataset_id', type=int, default=0, help="Dataset ID for multi-dataset experiments")
args = parser.parse_args()

DATASET_ID = args.dataset_id
DATA_DIR = f"data_dataset{DATASET_ID}"
PLOT_DIR = f"plots_dataset{DATASET_ID}"

# Create directories
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# parameters and variables
# Use dataset_id to vary the seed base for distinct datasets
seed_base = 1 + DATASET_ID * 1000  # Each dataset gets a distinct seed range
nTrial = 200
nSession = 200


# --- Data Generation ---
rewardsTrain = sim.gen_reward_seq(seed=seed_base, T=nTrial, interval=50, N = nSession)
rewardsTest = sim.gen_reward_seq(seed=seed_base + 1, T=nTrial, interval=50, N=nSession)

#rewardsTrain = sim.generate_drifting_binary_bandit()
#rewardsTest = sim.generate_drifting_binary_bandit()

np.save(f"{DATA_DIR}/rewards_train.npy", rewardsTrain)

true_param = sim.generate_parameter_lists(true_model='FQ', ind_diff_type='uniform', #ind_diff_type="discrete_alpha_only", #, #,
                                          Delta_alpha=0.6, nSession=nSession)

true_param_test = sim.generate_parameter_lists(true_model='FQ', ind_diff_type='uniform',#ind_diff_type="discrete_alpha_only",# ,#ind_diff_type='uniform', #,
                                          Delta_alpha=0.6, nSession=nSession)

true_param_df = pd.DataFrame(true_param)
true_param_df.to_csv(f"{DATA_DIR}/true_parameter_values.csv", index=False)
true_test_param_df = pd.DataFrame(true_param_test)
true_test_param_df.to_csv(f"{DATA_DIR}/true_test_parameter_values.csv", index=False)

plt.hist(true_param_df["alphaP_list"], 10)
plt.savefig(f'{PLOT_DIR}/param_hist.png')

print("simulating training data")
c, r, pA, Q, CT, df_train, xin_train, choice_one_hot_train, _, _, _ = sim.simulate_Qlearning(
    rewards=rewardsTrain, seed=seed_base + 2, n_sessions=nSession, n_trials=nTrial,
    alphaP_list=true_param['alphaP_list'],
    alphaN_list=true_param['alphaN_list'],
    beta=true_param['beta_list'],
    alphaF_list=true_param['alphaF_list'],
    phi_list=true_param['phi_list'],
    tau_list=true_param['tau_list'], static=False
)

print("simulating test data")
c_test, r_test, pA_test, _, _, df_test, xin_test, choice_one_hot_test, normalized_LL_test, session_ll_df_test, _ = sim.simulate_Qlearning(
    rewards=rewardsTest, seed=seed_base + 10, n_sessions=nSession, n_trials=nTrial,
    alphaP_list=true_param_test['alphaP_list'],
    alphaN_list=true_param_test['alphaN_list'],
    beta=true_param_test['beta_list'],
    alphaF_list=true_param_test['alphaF_list'],
    phi_list=true_param_test['phi_list'],
    tau_list=true_param_test['tau_list'], static=False
)
print(f"xin_train shape: {xin_train.shape}")
print(f"choice_one_hot_test shape: {choice_one_hot_test.shape}")
print(f"choice_one_hot_test: {choice_one_hot_test}")

#save data externally
df_train.to_csv(f"{DATA_DIR}/df_train.csv", index=False)
df_test.to_csv(f"{DATA_DIR}/df_test.csv", index=False)
session_ll_df_test.to_csv(f"{DATA_DIR}/session_ll_df_test.csv", index=False)
np.save(f'{DATA_DIR}/xin_train.npy', xin_train)
np.save(f'{DATA_DIR}/xin_test.npy', xin_test)
np.save(f'{DATA_DIR}/choice_one_hot_train.npy', choice_one_hot_train)
np.save(f'{DATA_DIR}/choice_one_hot_test.npy', choice_one_hot_test)
np.save(f'{DATA_DIR}/c_test.npy', c_test)
np.save(f'{DATA_DIR}/pA_train.npy', pA)
np.save(f'{DATA_DIR}/pA_test.npy', pA_test)
np.save(f'{DATA_DIR}/c_train.npy', c)

print(f"\n✅ Dataset {DATASET_ID} generated and saved to {DATA_DIR}/")