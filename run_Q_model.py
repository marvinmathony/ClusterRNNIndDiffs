import sim_Q_data as sim
import RL_fitting_functions as fit
import plot_functions as plf
import random
import numpy as np
import tensorflow as tf
import torch
from compare_models import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed_value = 42
random.seed(seed_value)           
np.random.seed(seed_value)        
tf.random.set_seed(seed_value)    

sim_nametag = "Katahira_setup"

n_fit_iter = 5 # iteration for RL fitting (from random initialization)
flg_on_policy_check = 1 # True # True: do on-policy check
nTrial_sim = 5000 # nTrial for on-policy check
file_id = 'scenario2_Q+A_homo'  

# Parameter settings for Q-learning used in data generation
nTrial = 200
nSession = 100 # number of subjects

# Set seeds for reproducibility
seed_base = 1


os.makedirs("checkpoints", exist_ok=True)
os.makedirs("plots", exist_ok=True)
os.makedirs("data", exist_ok=True)

# --- Data Generation ---
rewardsTrain = sim.gen_reward_seq(seed=seed_base, T=nTrial, pHigh=0.7, pLow=0.3, interval=50)
rewardsTest = sim.gen_reward_seq(seed=seed_base + 1, T=nTrial, pHigh=0.7, pLow=0.3, interval=50)
np.save("data/rewards_train.npy", rewardsTrain)

true_param = sim.generate_parameter_lists(true_model='Q+A', ind_diff_type='None',
                                          Delta_alpha=None, nSession=nSession)

c, r, pA, Q, CT, df_train, xin_train, choice_one_hot_train, _, _, _ = sim.simulate_Qlearning(
    rewards=rewardsTrain, seed=seed_base + 2, n_sessions=nSession, n_trials=nTrial,
    alphaP_list=true_param['alphaP_list'],
    alphaN_list=true_param['alphaN_list'],
    beta=true_param['beta_list'],
    alphaF_list=true_param['alphaF_list'],
    phi_list=true_param['phi_list'],
    tau_list=true_param['tau_list']
)

c_test, r_test, pA_test, _, _, df_test, xin_test, choice_one_hot_test, normalized_LL_test, session_ll_df_test, _ = sim.simulate_Qlearning(
    rewards=rewardsTest, seed=seed_base + 10, n_sessions=nSession, n_trials=nTrial,
    alphaP_list=true_param['alphaP_list'],
    alphaN_list=true_param['alphaN_list'],
    beta=true_param['beta_list'],
    alphaF_list=true_param['alphaF_list'],
    phi_list=true_param['phi_list'],
    tau_list=true_param['tau_list']
)
df_train.to_csv("data/df_train.csv", index=False)

# Convert to tensors
xin_train = torch.from_numpy(xin_train).float().to(device)
choice_one_hot_train = torch.from_numpy(choice_one_hot_train).float().to(device)
pA = torch.from_numpy(pA).float().to(device)

xin_test = torch.from_numpy(xin_test).float().to(device)
c_test = torch.from_numpy(c_test).float().to(device)

# --- Train RNN ---
hidden = 10
m_abl = AblatedRNN(in_dim=4, hid=hidden, block_structure=False).to(device)
m_abl, train_loss, val_loss, kl_loss, pA_rnn_dict = train_ablated_noblocks(
    m_abl, xin_train, choice_one_hot_train, p_target=pA
)
pA_rnn_dict = {k: v.detach().cpu().numpy() for k, v in pA_rnn_dict.items()}
np.savez("data/pA_rnn_dict.npz", **pA_rnn_dict)

# Save checkpoint
torch.save(m_abl.state_dict(), "checkpoints/ablated_rnn.pt")

# --- Compute RNN likelihoods ---
def compute_rnn_likelihoods_torch(model, test_xin, c_test, train_xin):
    rnn_results = []
    B, T = c_test.shape
    with torch.no_grad():
        logits_test, _, _ = model(test_xin)
        probs_test = F.softmax(logits_test, dim=-1)
        logits_train, _, _ = model(train_xin)
        probs_train = F.softmax(logits_train, dim=-1)

        choices = c_test.view(-1)
        probs_flat = probs_test.view(-1, probs_test.size(-1))
        chosen_probs = probs_flat[torch.arange(B*T), choices.long()].view(B, T)

        log_likelihoods = chosen_probs.clamp(min=1e-8).log()
        log_ll_per_session = log_likelihoods.sum(dim=1)
        norm_ll = (log_ll_per_session / T).exp()

        rnn_results.append(pd.DataFrame({
            "session": np.arange(B),
            "normalized_likelihood": norm_ll.cpu().numpy(),
            "model": "GRU"
        }))
    return pd.concat(rnn_results, ignore_index=True)

rnn_df = compute_rnn_likelihoods_torch(m_abl, xin_test, c_test, xin_train)
rnn_df.to_csv("data/rnn_results.csv", index=False)

# --- Fit Cognitive Models ---
model_configs = {
    "Q": {"asymmetric_alpha": False, "forgetting_type": "none", "choice_trace": False},
    "FQ": {"asymmetric_alpha": False, "forgetting_type": "fixed", "choice_trace": True}
}
model_eval_Q_df, params_dict, p1_common_dict, p1_ML_dict, p1_MAP_dict = fit.fit_all_models(
    model_configs, df_train, df_test, n_iter=n_fit_iter, fit_ML=False
)

model_eval_df = pd.concat([model_eval_Q_df, session_ll_df_test, rnn_df], ignore_index=True)
model_eval_df.to_csv("data/model_eval_df.csv", index=False)

# Save p1 values (for later plotting)
np.savez("data/p1_common_dict.npz", **p1_common_dict)

print("✅ Training and fitting done. Results saved to data/ and checkpoints/")

"""# Generate reward sequences for training and testing
rewardsTrain = sim.gen_reward_seq(seed=seed_base, T=nTrial, pHigh=0.7, pLow=0.3, interval=50)
rewardsTest = sim.gen_reward_seq(seed=seed_base + 1, T=nTrial, pHigh=0.7, pLow=0.3, interval=50)

# true_param = sim.generate_parameter_lists(true_model = 'FQ', ind_diff_type = 'discrete_alpha_only', 
#                                       Delta_alpha=0.8, nSession=nSession)
true_param = sim.generate_parameter_lists(true_model = 'Q+A', ind_diff_type = 'None', 
                                      Delta_alpha=None, nSession=nSession)
# ind_diff_type (str): Type of individual difference ('continuous_all', 'continuous_alpha_only', 'None', 'discrete_alpha_only', 'full_range'

# Generate data for training
c, r, pA, Q, CT, df_train, xin_train, choice_one_hot_train, _, _, _ = sim.simulate_Qlearning(
    rewards=rewardsTrain, seed=seed_base + 2, n_sessions=nSession, n_trials=nTrial, 
    alphaP_list=true_param['alphaP_list'],
    alphaN_list=true_param['alphaN_list'], 
    beta=true_param['beta_list'], alphaF_list=true_param['alphaF_list'], 
    phi_list=true_param['phi_list'], tau_list=true_param['tau_list']
)

c_test, r_test, pA_test, _, _, df_test, xin_test, choice_one_hot_test, normalized_LL_test, session_ll_df_test, _ = sim.simulate_Qlearning(
    rewards=rewardsTest, seed=seed_base + 10, n_sessions=nSession, n_trials=nTrial, 
    alphaP_list=true_param['alphaP_list'],
    alphaN_list=true_param['alphaN_list'], 
    beta=true_param['beta_list'], alphaF_list=true_param['alphaF_list'], 
    phi_list=true_param['phi_list'], tau_list=true_param['tau_list']
)

print(f"df_train: {df_train.head(50)}")
print(f"xin_train: {xin_train}")
print(f"xin shape: {xin_train.shape}")
print(f"onehot encoded choices: {choice_one_hot_train}")
print(f"onehot encoded choices shape: {choice_one_hot_train.shape}")
xin_train = torch.from_numpy(xin_train).to(dtype=torch.float32)
choice_one_hot_train = torch.from_numpy(choice_one_hot_train).to(dtype=torch.float32)
pA = torch.from_numpy(pA).to(dtype=torch.float32)

xin_train = xin_train.to(device)
choice_one_hot_train = choice_one_hot_train.to(device)
pA = pA.to(device)


xin_test = torch.from_numpy(xin_test).to(dtype=torch.float32)
c_test = torch.from_numpy(c_test).to(dtype=torch.float32)

xin_test = xin_test.to(device)
c_test = c_test.to(device)


#xin shape: (100, 200, 4)

#### define model ####
#hyperparams
hidden = 10

m_abl = AblatedRNN(in_dim=4, hid=hidden, block_structure = False).to(device)

m_abl, train_loss, val_loss, kl_loss = train_ablated_noblocks(m_abl, xin_train, choice_one_hot_train, p_target=pA)

# plot loss
figure = sim.plot_kl_and_train_loss(train_loss, kl_loss, sim_nametag)


def compute_rnn_likelihoods_torch(model, test_xin, c_test, train_xin, test_mask=None):
    #later it might be useful to reintroduce the initial states argument when using my setup
    rnn_results = []

    # there is going to be a 3D case with changing context, but for now we don't bother
    B, T = c_test.shape  # Assume 2D case: (sessions, trials) 
    device = test_xin.device


    with torch.no_grad():
        # === Test Predictions ===
        logits_test, _, _ = model(test_xin)
        probs_test = F.softmax(logits_test, dim=-1)  # [B, T, A]
        p0_test = probs_test[:,:,0] #should be shape [B,T]

        # === Train Predictions ===
        logits_train, _, _ = model(train_xin)
        probs_train = F.softmax(logits_train, dim=-1)
        p0_train = probs_train[:,:,0] #should be shape [B,T]

        # === Log-likelihoods ===
        choices = c_test.view(-1)  # shape [B*T]
        probs_flat = probs_test.view(-1, probs_test.size(-1))  # [B*T, A]
        chosen_probs = probs_flat[torch.arange(B*T), choices.long()].view(B, T)

        log_likelihoods = chosen_probs.clamp(min=1e-8).log()  # log(p(a_t))

        if test_mask is not None:
            log_likelihoods = log_likelihoods * test_mask  # mask invalid trials
            valid_trials = test_mask.sum(dim=1)
        else:
            valid_trials = torch.tensor([T] * B, dtype=torch.float32, device=device)

        log_ll_per_session = log_likelihoods.sum(dim=1)
        norm_ll = (log_ll_per_session / valid_trials).exp()  # normalized LL

        # Add to dataframe
        rnn_results.append(pd.DataFrame({
            'session': np.arange(B),
            'normalized_likelihood': norm_ll.detach().cpu().numpy(),
            'model': "RNN"
        }))

    rnn_df = pd.concat(rnn_results, ignore_index=True)
    return rnn_df, p0_test, p0_train

rnn_df, p0_test, p0_train = compute_rnn_likelihoods_torch(
    m_abl, xin_test, c_test, xin_train
)

print(f"RNN df: {rnn_df.head(50)}")

# Define model configurations
model_configs = {
    "Q": {"asymmetric_alpha": False, "forgetting_type": "none", "choice_trace": False},
    #"Q+A": {"asymmetric_alpha": True, "forgetting_type": "none", "choice_trace": False},
    #"Q+C": {"asymmetric_alpha": False, "forgetting_type": "none", "choice_trace": True},
    #"Q+CA": {"asymmetric_alpha": True, "forgetting_type": "none", "choice_trace": True},
    "FQ": {"asymmetric_alpha": False, "forgetting_type": "fixed", "choice_trace": True}
}

# Model fitting
model_eval_Q_df, params_dict, p1_common_dict, p1_ML_dict, p1_MAP_dict = fit.fit_all_models(
    model_configs, df_train, df_test, n_iter=n_fit_iter, fit_ML=False
)

# Append to the model_eval_df DataFrame
model_eval_df = pd.concat([model_eval_Q_df, rnn_df], ignore_index=True)

model_eval_df = pd.concat([model_eval_df, session_ll_df_test], ignore_index=True)

print(f"model_eval_df: {model_eval_df.head(50)}")

# Select the session to plot (e.g., session 1)
selected_session = 0

# Plot p1_common for the selected session across models
plt.figure(figsize=(10, 6))

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


# Combine model evaluation results
model_eval_df = pd.concat([model_eval_Q_df, session_ll_df_test], ignore_index=True)    
model_eval_df = pd.concat([model_eval_df, rnn_df], ignore_index=True)
model_eval_df.to_csv('data/model_eval_df.csv')

# Get unique models
models = model_eval_df['model'].unique()

fig = plf.compare_model_performance(
        model_eval_df, models, 
        file_id=f'model_comparison', 
        save_dir='plots/', ylim=[0.5, 0.64], plot_individual_ML=False, 
        fig_width=7, x_tick_fontsize=18
    )
# Add title with epoch information
fig.suptitle(f'Model Comparison', fontsize=16)
fig.savefig(f'plots/model_comparison_{sim_nametag}.png')
print(f"Figure comparing models (RNN and cog) saved.")"""

