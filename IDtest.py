import os, numpy as np, pandas as pd
import torch, seaborn as sns, matplotlib.pyplot as plt

# re-use your imports/utilities
from data_utils import load_preprocess_data_VAE
from kalman_simulation import simulate_data_probit_random_walk_IDtest, evaluate_log_likelihood_agent, evaluate_log_likelihood_agent_marginal, simulate_data_probit_random_walk_individual_participant
from compare_models import *
from sklearn.model_selection import train_test_split
from ind_diffs_sweep_EMstyle import split_train_val_per_participant
from sklearn.cross_decomposition import CCA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# data generation
hidden = 10
epochs = 3000
zdim = 2
N = 100
sim_nametag = "vanillaRNN"


df_withID = simulate_data_probit_random_walk_IDtest(n_participants=300, n_blocks=10, with_id=True) #maybe do more than 100 participants
fname = "data/simulated_data_withID.csv"
df_withID.to_csv(fname, index=False)
Xtrain, Ytrain, Xtest, Ytest, pid = load_preprocess_data_VAE(fname, n_blocks_pp=10)
Xtrain, Ytrain, Xtest, Ytest = Xtrain.to(device), Ytrain.to(device), Xtest.to(device), Ytest.to(device)
Xtrain, Ytrain, Xval, Yval = split_train_val_per_participant(Xtrain, Ytrain)

# extract probabilities for choosing action 0 from training data 
# 2. Get shape info
n_participants = df_withID["participant"].nunique()
n_blocks = df_withID["block"].max() - 1  # assuming 0-indexed
n_trials = df_withID["trial"].max() + 1  # assuming 0-indexed
num_actions = 2

# 3. Initialize tensor [B, Bk, T, A]
split_idx = int(0.8 * n_participants)

p_target = torch.zeros(split_idx, n_blocks, n_trials, num_actions)
df_train_p = df_withID.iloc[:split_idx]
df_train_p = df_train_p[df_train_p["block"] < n_blocks]
#print(df_train_p.tail(50))
# 4. Iterate over DataFrame to fill in probs
for _, row in df_train_p.iterrows():
    p = int(row["participant"])
    b = int(row["block"])
    t = int(row["trial"])
    p0 = row["probability_arm_0"]
    p1 = row["probability_arm_1"]

    # Fill tensor
    p_target[p, b, t, 0] = p0
    p_target[p, b, t, 1] = p1
print(f"shape p_target: {p_target.shape}") # has shape of dataframe
B, Bk, T, A = p_target.shape
p_target = p_target.view(B, Bk*T, A).to(device)


#### train vanilla RNN ####
m_abl = AblatedRNN(hid=hidden).to(device)
m_abl.name = "AblatedRNN"
model_abl, train_loss, val_loss, kl_values  = train_ablated(
    m_abl, Xtrain, Ytrain, Xval, Yval, p_target, epochs=epochs
)

# plot loss
# Ensure same length
assert len(train_loss) == len(val_loss) == len(kl_values)
print(type(train_loss))

epochs = list(range(1, len(train_loss) + 1))
train_loss_values = [loss.detach().cpu().item() for loss in train_loss]
val_loss_values = [loss for loss in val_loss]
kl_loss_values = [loss for loss in kl_values]

fig, ax1 = plt.subplots(figsize=(8, 5))

# === Plot loss (left y-axis) ===
color1 = 'tab:red'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Cross-Entropy Loss', color=color1)
ax1.plot(epochs, train_loss_values, label='Train Loss', color='tab:red', linestyle='-')
ax1.plot(epochs, val_loss_values, label='Val Loss', color='tab:orange', linestyle='--')
ax1.tick_params(axis='y', labelcolor=color1)

# === Create second y-axis for KL ===
ax2 = ax1.twinx()
color2 = 'tab:blue'
ax2.set_ylabel('KL Divergence', color=color2)
ax2.plot(epochs, kl_loss_values, label='KL Divergence', color=color2, linestyle='-.')
ax2.tick_params(axis='y', labelcolor=color2)

# === Add legend and title ===
fig.suptitle("RNN Training: Loss & KL per Epoch", fontsize=14)
fig.tight_layout()
fig.legend(loc='upper right', bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)

plt.grid(True)
plt.show()
plt.savefig(f"plots/vanillaRNNtraining{sim_nametag}.png")
print("training plotted")

"""# === 1. Train RNN on ID or no-ID data ===
### FIRST STEP ###
m_emb = LatentRNNz(LookupEncoderZ(Xtrain.shape[0], zdim), z_dim=zdim, hid=hidden).to(device)
m_emb.name = f"LatentRNN(z={zdim})"
m_emb, z_emb, _, _, _ = train_latentrnn_early_stoppingZ(
    m_emb,
    torch.arange(Xtrain.size(0)).to(device),  # participant indices for LookupEncoder
    Xtrain, Ytrain, Xval, Yval,
    epochs=2000
)
trained_decoder = m_emb.decoder  # Decoder(in_dim=2, z_dim=zdim, hid=hidden, A=2)

mean_ll_emb, _ = test_latentrnnZ(m_emb, z_emb, Xtest, Ytest)

### SECOND STEP ###
m_secondstep = LatentRNN_secondstep(IDRNN(in_dim=2, z_dim=zdim, hid=hidden) ,z_dim=zdim, in_dim=2, hid=hidden, decoder=trained_decoder).to(device)
m_secondstep.name = f"IDRNN(z={zdim})"
for param in m_secondstep.decoder.parameters():
    param.requires_grad = False
m_secondstep.decoder.eval()
m_secondstep, mu_secondstep, lv_secondstep, _, _ = train_latentrnn_IDRNN(m_secondstep, Xtrain, Xtrain, Ytrain, z_emb, Xval, Yval, epochs=epochs)
"""
"""print(f" z's from latent embedding: {z_emb}")
print(f"shape of z's from latent embedding: {z_emb.shape}")
print(f" z's from IDRNN encoder: {mu_secondstep}")
print(f"shape of z's from IDRNN encoder: {mu_secondstep.shape}")

# Assuming z1 and z2 are your [240, 3] tensors
z1 = z_emb  # tensor from model 1
z2 = mu_secondstep  # tensor from model 2

# Convert to numpy if they're tensors
z1_np = z1.cpu().detach().numpy() if torch.is_tensor(z1) else z1
z2_np = z2.cpu().detach().numpy() if torch.is_tensor(z2) else z2

# Perform CCA
cca = CCA(n_components=2)  # 3 components since you have 3 dimensions
cca.fit(z1_np, z2_np)

# Transform the data
z1_c, z2_c = cca.transform(z1_np, z2_np)

# Calculate canonical correlations
canonical_correlations = [np.corrcoef(z1_c[:, i], z2_c[:, i])[0, 1] for i in range(2)]

print("Canonical correlations:")
for i, corr in enumerate(canonical_correlations):
    print(f"  Component {i+1}: {corr:.3f}")

print(f"Mean canonical correlation: {np.mean(canonical_correlations):.3f}")"""

# === 2. Simulate N Kalman agents with different parameters ===
# skip this for now. First try to recover parameters from trained RNN
#N_agents = 100
#alpha_values = np.linspace(0.1, 0.9, N_agents)
#kalman_data = simulate_kalman_agents(alpha_values, n_blocks=3, n_trials=10)

# === 3. For each simulated agent/session: ===
#1. same environment for all agents


all_recovered_betas = []
all_recovered_gammas = []

#for session in range(100):
    #forward simulate RNN behavior for each session
    
    
    #2. simulate behavior using RNN with z from training
    # -----------------------------
    # 6. Forward Simulation 
    # -----------------------------
n_participants = 100
n_blocks_pp = 10      # blocks per participant
n_trials_per_block = 200  # sequence length
innov_variance = 100  
noise_variance = 1
input_size = 2  # (reward for arm 0, reward for arm 1)
num_classes = 2  # number of actions (arms)
tau_sq = 4  # volatility of the drifting arm
tau = np.sqrt(tau_sq)
n_trials = 200
input_threshold = 50  # number of trials to provide ground truth input

#m_secondstep.eval()

simulation_data = []
global_trial = 0
# Optionally, reset index to ensure we can index by global_trial.
#df_behavior.reset_index(drop=True, inplace=True)

with torch.no_grad():
    for participant in range(n_participants):
        print(f"Simulating participant {participant+1}/{n_participants}")
        #simulate a participant to initialize z via first two trials
        #beta = np.random.uniform(0.5, 5.0)
        #gamma = np.random.uniform(0.5, 5.0)

        df_withID = simulate_data_probit_random_walk_individual_participant(with_id=True, n_blocks=10)
        global_participant_trial = 0

        for block in range(n_blocks_pp):
            mu0 = np.random.normal(0, np.sqrt(innov_variance))  # stable arm
            mu1_list = [np.random.normal(0, np.sqrt(innov_variance))]
            for trial in range(1, n_trials):
                mu1_list.append(np.random.normal(mu1_list[-1], tau))
            input_seq = torch.zeros(n_trials_per_block, input_size, device=device) #check which format is needed
            n_states = n_trials_per_block
            V_t = np.zeros(n_states)
            RU = np.zeros(n_states)  
            TU = np.zeros(n_states) 
            post_mean = np.zeros((2, n_states))
            post_variance = np.ones((2, n_states)) * 10
            kalman_gain = np.zeros((2, n_states))

            hidden_v = None
            hidden = None
            
            for trial in range(n_trials_per_block):
            
                """if trial < 2:
                    # Use ground truth action and reward from the training dataset.
                    rnn_action = current_data["Action"]
                    rnn_reward = current_data["Reward"]
                    
                    # For consistency, you could also verify that the chosen action
                    # corresponds to the reward observed in training.
                    # Encode input according to the action.
                    if rnn_action == 0:
                        encoded_input = [rnn_reward, 0]
                    else:
                        encoded_input = [0, rnn_reward]
                    probs_t = None  # No probability output for ground truth trials.
"""             #provide input, get mu and logvar from encoder
                if trial < input_threshold: 
                    current_data = df_withID.iloc[global_participant_trial]
                    rnn_action = current_data["action"]
                    rnn_reward = current_data["reward"]
                    probs_vanilla_t = None
                    if rnn_action == 0:
                        encoded_input = [rnn_reward, 0] #is this also how it was trained? - Yes

                    else:
                        encoded_input = [0, rnn_reward]
                    input_seq[trial] = torch.tensor(encoded_input, device=device)
                    input_tensor = input_seq[trial:trial+1].unsqueeze(0) #[1,1,input_size]
                    logits, hidden_v, _ = model_abl.dec(input_tensor, hidden_v)
                    action_probs = F.softmax(logits[0, -1], dim=-1) if trial == input_threshold - 1 else None
                else:
                    
                    #####
                    #process through RNN to update hidden state
                    input_tensor = input_seq[trial-1:trial].unsqueeze(0) #[1,1,input_size]
                    logits, hidden_v,_ = model_abl.dec(input_tensor, hidden_v)

                    # Choose action
                    action_probs = F.softmax(logits[0, -1], dim=-1)
                    rnn_action = torch.multinomial(action_probs, num_samples=1).item()
                    #print(f"Trial {trial}: action={rnn_action}, reward={rnn_reward:.2f}, hidden={hidden_v is not None}")
                    
                    ######

                """input_until_t = input_seq[trial:trial+1]  # (t, in_dim)
                input_seq_batched = input_until_t.unsqueeze(0).expand(N, -1, -1)
                input_seq_ = input_seq.unsqueeze(0).unsqueeze(0)  # (1, 1, t, in_dim)
                mu_t, logvar_t = m_secondstep.encoder(input_seq_[:,:,:trial], return_per_timestep=True)
                mu_t = mu_t[0, 0, -1]  # (z_dim,)
                logvar_t = logvar_t[0, 0, -1]  # (z_dim,)
                #print(f"mu_t: {mu_t}, logvar_t: {logvar_t}")
                
                # Sample z from the posterior q(z|x_{1:t})
                std_t = torch.exp(0.5 * logvar_t)
                z_samples = mu_t + std_t * torch.randn((N, zdim), device=device)
                # Forward pass through the decoder to get action logits
                  # (N, n_trials, in_dim)
                outputs, hidden = m_secondstep.decoder(input_seq_batched, z_samples, hidden=hidden, ID_test=True)  # (N, t, num_classes)
                logits_t = outputs[:, -1]  
                # Convert to log-probabilities
                log_probs = F.log_softmax(logits_t, dim=-1)  # (N, A)

                # Monte Carlo integration: marginalized log-probability per action
                log_p_a = torch.logsumexp(log_probs, dim=0) - math.log(N)  # (A,)

                # Choose the most likely action
                rnn_action = torch.argmax(log_p_a).item()

                # Optionally convert to probabilities (normalized)
                probs_t = torch.softmax(log_p_a, dim=0).cpu().numpy()  # (A,)"""
                # choose action with maximum probability
                #rnn_action = np.argmax(probs_t)
                # Retrieve the arm-specific means from the training data.
                #mean0 = current_data["mean_reward_arm0"]
                #mean1 = current_data["mean_reward_arm1"]
                #mean_reward_block = np.random.normal(0, np.sqrt(innov_variance), n_actions)
                if trial >= input_threshold:

                    if rnn_action == 0:
                        rnn_reward = np.random.normal(mu0, np.sqrt(noise_variance))
                        encoded_input = [rnn_reward, 0] #is this also how it was trained?

                    else:
                        rnn_reward = np.random.normal(mu1_list[trial], np.sqrt(noise_variance))
                        encoded_input = [0, rnn_reward]

                    input_seq[trial] = torch.tensor(encoded_input, dtype=torch.float32, device=device)
                
                for i in range(2):
                    if rnn_action == i:
                        """if trial == 0:
                            post_variance[i][trial] = 10
                            post_mean[i][trial] = 0"""

                        if trial < n_trials_per_block - 1:
                            #prev_variance = post_variance[i][trial - 1]
                            #prev_mean = post_mean[i][trial - 1]
                            kalman_gain[i][trial] = post_variance[i][trial] / (post_variance[i][trial] + noise_variance)
                            post_variance[i][trial+1] = (1 - kalman_gain[i][trial]) * post_variance[i][trial]
                            post_mean[i][trial+1] = post_mean[i][trial] + kalman_gain[i][trial] * (rnn_reward - post_mean[i][trial])
                    else:
                        """if trial == 0:
                            post_variance[i][trial] = 10
                            post_mean[i][trial] = 0"""
                        if trial < n_trials_per_block - 1:
                            post_variance[i][trial+1] = post_variance[i][trial]
                            post_mean[i][trial+1] = post_mean[i][trial]
                
                V_t[trial] = post_mean[0][trial] - post_mean[1][trial]
                sigma1 = post_variance[0][trial]
                sigma2 = post_variance[1][trial]
                TU[trial] = np.sqrt(np.sqrt(sigma1**2) + np.sqrt(sigma2**2))
                RU[trial] = np.sqrt(sigma1)- np.sqrt(sigma2)
                
                
                simulation_data.append({
                    "participant": participant,
                    "Block": block,
                    "Trial": trial,
                    "Global_Trial": global_trial,
                    "Global_Participant_Trial": global_participant_trial,
                    #"mean_reward_arm0": current_data["mean_reward_arm0"],
                    #"mean_reward_arm1": current_data["mean_reward_arm1"],
                    "Action": rnn_action,
                    "Reward": rnn_reward,
                    "RNN_Prob_Action0": action_probs[0] if action_probs is not None else None,
                    "V_t": V_t[trial],
                    "Kalman_post_mean_0": post_mean[0][trial],
                    "Kalman_post_mean_1": post_mean[1][trial],
                    "Kalman_post_variance_0": post_variance[0][trial],
                    "Kalman_post_variance_1": post_variance[1][trial],
                    "Kalman_kalman_gain_0": kalman_gain[0][trial],
                    "Kalman_kalman_gain_1": kalman_gain[1][trial],
                    "RU": RU[trial],
                    "TU": TU[trial],
                    "beta_true": current_data["beta_true"] if trial < input_threshold else None,
                    "gamma_true": current_data["gamma_true"] if trial < input_threshold else None
                    #"best_penalty_weight": best_incentive_weight,
                    #"num_layers": num_layers,
                    #"batch_size": batch_size,
                    #"average_loss": np.mean(loss_history),
                    #"average_test_loss": np.mean(test_loss_history),
                    #"optimized_l2_weight": learning_rate,
                    #"hidden_size": hidden_size,
                    #"train_log_likelihood": train_avg_ll,
                    #"test_log_likelihood": test_avg_ll,
                    #"test_cum_log_likelihood": test_cum_ll,
                    #"train_cum_log_likelihood": train_cum_ll,
                })
                global_participant_trial += 1
                global_trial += 1
                


    df_simulation = pd.DataFrame(simulation_data)
    #df_simulation["log_lik_per_trial"] = overall_ll_per_trial
    #df_simulation["prob_per_trial"] = overall_probs_per_trial
    #df_simulation["cumulative_log_likelihood"] = cumulative_ll
    #df_simulation["same_choice"] = predictions
    os.makedirs("data", exist_ok=True)
    df_simulation.to_csv(f"data/IDtest{sim_nametag}.csv", index=False)

"""    #3. simulate behavior using RNN with 1s instead of z
    #4. fit kalman to both behaviors

    # === 5. Fit ground-truth model (Kalman) to RNN behavior ===
    recovered_alpha = fit_kalman_to_data(rnn_behavior)
    all_recovered_alphas.append((alpha_values[i], recovered_alpha))

# === 6. Plot histogram of recovered vs. true alphas ===
true, recovered = zip(*all_recovered_alphas)
plt.hist([r for t, r in all_recovered_alphas], bins=20)
plt.title("Recovered learning rates (RNN-generated behavior)")
plt.xlabel("Recovered alpha"); plt.ylabel("Frequency")

# === 7. Repeat using random z instead of z from history ===
# (e.g., z = torch.randn_like(mu)) → compare histogram shifts"""