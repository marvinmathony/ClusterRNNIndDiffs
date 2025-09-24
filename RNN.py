#!/usr/bin/env python

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna
from sklearn.model_selection import KFold

date = "26022042"

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print("CUDA detected")

# -----------------------------
# Data Loading & Preprocessing
# -----------------------------
hidden_size_default = 100  # default value; can be optimized too
num_classes = 2
num_epochs_cv = 10      # reduced epochs for CV
num_epochs_full = 30000  # final training
batch_size = 128
input_size = 2
sequence_length = 10
num_layers = 2

filename = 'data/results_hybrid.csv'
df = pd.read_csv(filename)
if filename == 'human_data.csv':
    df = df.rename(columns={"choice": "Action", "reward": "Reward"})
    df['Action'] = df['Action'].astype(int) - 1

n_trials_per_block = 10
n_blocks_pp = 20
n_participants = len(df) // (n_trials_per_block * n_blocks_pp)
n_blocks = n_participants * n_blocks_pp

choices = df['Action'].to_numpy().reshape(n_participants, n_blocks_pp, n_trials_per_block)
rewards = df['Reward'].to_numpy().reshape(n_participants, n_blocks_pp, n_trials_per_block)
data = np.stack((choices, rewards), axis=-1)  # (n_participants, n_blocks_pp, n_trials_per_block, 2)

data = data.reshape(-1, n_trials_per_block, 2)  # (600, 10, 2)
data = np.transpose(data, (1, 0, 2))  # (10, 600, 2)

data_tensor = torch.tensor(data, dtype=torch.float32).to(device)

# Split into training and test (we use training for CV)
split_idx = int(0.8 * data_tensor.shape[1])
train_tensor = data_tensor[:, :split_idx, :]  # (10, 480, 2)
test_tensor  = data_tensor[:, split_idx:, :]   # (10, 120, 2)

n_sequences = train_tensor.shape[1]

# Prepare training inputs (xs) and labels (ys)
xs = np.zeros((train_tensor.shape[0], train_tensor.shape[1], input_size), dtype=np.float32)
ys = np.zeros((train_tensor.shape[0], train_tensor.shape[1], 1), dtype=np.float32)

train_choices = train_tensor[:, :, 0].cpu().numpy()  # (10, n_sequences)
train_rewards = train_tensor[:, :, 1].cpu().numpy()    # (10, n_sequences)

for sess_i in range(n_sequences):
    prev_vectors = []
    # Dummy input for first trial
    prev_vectors.append([0, 0])
    for t in range(1, train_tensor.shape[0]):
        choice = train_choices[t-1, sess_i]
        reward = train_rewards[t-1, sess_i]
        if choice == 0:
            vector = [reward, 0]
        else:
            vector = [0, reward]
        prev_vectors.append(vector)
    xs[:, sess_i, :] = np.array(prev_vectors)
    ys[:, sess_i, 0] = train_choices[:, sess_i]

xs = torch.from_numpy(xs).to(device)
ys = torch.from_numpy(ys).to(device)

print(f'xs shape: {xs.shape}')  # Expected: (seq_length, n_sequences, 2)
print(f'ys shape: {ys.shape}')  # Expected: (seq_length, n_sequences, 1)

# -----------------------------
# Additional Code: Prepare Test Inputs and Labels
# -----------------------------
xs_test = np.zeros((test_tensor.shape[0], test_tensor.shape[1], input_size), dtype=np.float32)
ys_test = np.zeros((test_tensor.shape[0], test_tensor.shape[1], 1), dtype=np.float32)

test_choices = test_tensor[:, :, 0].cpu().numpy()
test_rewards = test_tensor[:, :, 1].cpu().numpy()

for sess_i in range(test_tensor.shape[1]):
    prev_vectors = []
    prev_vectors.append([0, 0])
    for t in range(1, test_tensor.shape[0]):
        choice = test_choices[t-1, sess_i]
        reward = test_rewards[t-1, sess_i]
        if choice == 0:
            vector = [reward, 0]
        else:
            vector = [0, reward]
        prev_vectors.append(vector)
    xs_test[:, sess_i, :] = np.array(prev_vectors)
    ys_test[:, sess_i, 0] = test_choices[:, sess_i]

xs_test = torch.from_numpy(xs_test).to(device)
ys_test = torch.from_numpy(ys_test).to(device)

print(f'xs_test shape: {xs_test.shape}')  # Expected: (seq_length, test_n_sequences, 2)
print(f'ys_test shape: {ys_test.shape}')  # Expected: (seq_length, test_n_sequences, 1)

# -----------------------------
# 1. Compute Target Repetition Rate from Data
# -----------------------------
def compute_target_repetition(ys):
    """
    Compute the fraction of consecutive trials in which the same action occurred.
    """
    ys_np = ys.squeeze(-1).cpu().numpy().astype(int)  # shape: (seq_length, n_sequences)
    total_same = 0
    total_pairs = 0
    seq_length, n_seq = ys_np.shape
    for seq in range(n_seq):
        for t in range(1, seq_length):
            total_pairs += 1
            if ys_np[t, seq] == ys_np[t-1, seq]:
                total_same += 1
    return total_same / total_pairs if total_pairs > 0 else 0.0

target_repetition = compute_target_repetition(ys)
print(f"Target repetition rate in the data: {target_repetition:.3f}")

# -----------------------------
# 2. Define the Repetition Incentive Loss Function
# -----------------------------
def repetition_incentive_loss(logits, incentive_weight, target_repetition=target_repetition):
    """
    Computes a loss term that encourages the model's repetition (via softmax similarities)
    to match the target repetition rate observed in the data.
    """
    probs = F.softmax(logits, dim=-1)  # (seq_len, batch_size, num_classes)
    seq_len = probs.shape[0]
    rep_sum = 0.0
    for t in range(1, seq_len):
        sim = torch.sum(probs[t] * probs[t-1], dim=-1)  # similarity per sequence
        rep_sum += sim.mean()
    avg_repetition = rep_sum / (seq_len - 1)
    loss_incentive = incentive_weight * (avg_repetition - target_repetition) ** 2
    return loss_incentive

# -----------------------------
# 3. Define the GRU Model
# -----------------------------
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GRUModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=False)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Initialize hidden state for GRU
        h0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size).to(device)
        out, _ = self.gru(x, h0)
        out = self.fc(out)
        return out

criterion = nn.CrossEntropyLoss()

# -----------------------------
# 4. Hyperparameter Optimization with Optuna (with SQLite storage for dashboard)
# -----------------------------
def objective(trial):
    # Suggest hyperparameters
    incentive_weight = trial.suggest_float('incentive_weight', 0.0, 1.0)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    hidden_size = trial.suggest_int('hidden_size', 1, 200)
    num_layers = trial.suggest_int('num_layers', 1, 10)
    
    k = 5  # number of CV folds
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_val_losses = []
    
    # Cross-validation loop
    for train_idx, val_idx in kf.split(range(n_sequences)):
        xs_train = xs[:, train_idx, :]
        ys_train = ys[:, train_idx, :]
        xs_val = xs[:, val_idx, :]
        ys_val = ys[:, val_idx, :]
        
        model_cv = GRUModel(input_size, hidden_size, num_layers, num_classes).to(device)
        optimizer_cv = torch.optim.Adam(model_cv.parameters(), lr=learning_rate, weight_decay=1e-4)
        
        # Training for a reduced number of epochs
        for epoch in range(num_epochs_cv):
            permutation = torch.randperm(xs_train.shape[1])
            xs_shuffled = xs_train[:, permutation, :]
            ys_shuffled = ys_train[:, permutation, :]
            model_cv.train()
            for i in range(0, xs_train.shape[1], batch_size):
                batch_x = xs_shuffled[:, i:i+batch_size, :].to(device)
                batch_y = ys_shuffled[:, i:i+batch_size, :].to(device)
                batch_y = batch_y.squeeze(-1).long()
                
                outputs = model_cv(batch_x)
                loss_ce = criterion(outputs.view(-1, num_classes), batch_y.view(-1))
                loss_incentive = repetition_incentive_loss(outputs, incentive_weight)
                loss = loss_ce + loss_incentive
                
                optimizer_cv.zero_grad()
                loss.backward()
                optimizer_cv.step()
        
        # Evaluation on the validation fold
        model_cv.eval()
        with torch.no_grad():
            outputs_val = model_cv(xs_val)
            val_batch_y = ys_val.squeeze(-1).long()
            loss_ce_val = criterion(outputs_val.view(-1, num_classes), val_batch_y.view(-1))
            loss_incentive_val = repetition_incentive_loss(outputs_val, incentive_weight)
            val_loss = loss_ce_val + loss_incentive_val
            fold_val_losses.append(val_loss.item())
    
    avg_val_loss = np.mean(fold_val_losses)
    return avg_val_loss

# Use SQLite storage to log the study so that the dashboard can display the results.
storage_name = "sqlite:///optuna_study.db"
study = optuna.create_study(storage=storage_name, load_if_exists=True, direction='minimize')
study.optimize(objective, n_trials=50)

print("Best hyperparameters:")
print(study.best_params)
print(f"Best CV Loss: {study.best_value:.4f}")

# -----------------------------
# 5. Final Training on Full Training Set with Best Hyperparameters (with Early Stopping)
# -----------------------------
best_incentive_weight = study.best_params['incentive_weight']
best_learning_rate = study.best_params['learning_rate']
best_hidden_size = study.best_params['hidden_size']
best_num_layers = study.best_params['num_layers']

model_final = GRUModel(input_size, best_hidden_size, best_num_layers, num_classes).to(device)
optimizer_final = torch.optim.Adam(model_final.parameters(), lr=best_learning_rate, weight_decay=1e-4)
loss_history = []
test_loss_history = []  # To store test loss per epoch
n_sequences_full = xs.shape[1]

# Early stopping parameters
patience = 1000  # Number of epochs with no improvement after which training will be stopped
best_test_loss = float('inf')
epochs_no_improve = 0

for epoch in range(num_epochs_full):
    permutation = torch.randperm(n_sequences_full)
    xs_shuffled = xs[:, permutation, :]
    ys_shuffled = ys[:, permutation, :]
    model_final.train()
    epoch_loss = 0.0
    n_batches = 0
    for i in range(0, n_sequences_full, batch_size):
        batch_x = xs_shuffled[:, i:i+batch_size, :].to(device)
        batch_y = ys_shuffled[:, i:i+batch_size, :].to(device)
        batch_y = batch_y.squeeze(-1).long()
        
        outputs = model_final(batch_x)
        loss_ce = criterion(outputs.view(-1, num_classes), batch_y.view(-1))
        loss_incentive = repetition_incentive_loss(outputs, best_incentive_weight)
        loss = loss_ce + loss_incentive
        
        optimizer_final.zero_grad()
        loss.backward()
        optimizer_final.step()
        
        epoch_loss += loss.item()
        n_batches += 1
    avg_epoch_loss = epoch_loss / n_batches
    loss_history.append(avg_epoch_loss)
    
    # -----------------------------
    # Compute Test Loss at each Epoch
    # -----------------------------
    model_final.eval()
    with torch.no_grad():
        outputs_test = model_final(xs_test)
        test_loss_ce = criterion(outputs_test.view(-1, num_classes), ys_test.squeeze(-1).long().view(-1))
        test_loss_incentive = repetition_incentive_loss(outputs_test, best_incentive_weight)
        test_loss = test_loss_ce + test_loss_incentive
    test_loss_history.append(test_loss.item())
    
    # -----------------------------
    # Early Stopping Check
    # -----------------------------
    if test_loss.item() < best_test_loss:
        best_test_loss = test_loss.item()
        epochs_no_improve = 0
        # Optionally, save the best model weights here
        best_model_state = model_final.state_dict()
    else:
        epochs_no_improve += 1
    
    if epoch % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs_full}], Loss: {avg_epoch_loss:.4f}, Test Loss: {test_loss.item():.4f}")
    
    if epochs_no_improve >= patience:
        print(f"Early stopping triggered after {epoch+1} epochs. No improvement in test loss for {patience} consecutive epochs.")
        break

# Optionally, load the best model state after early stopping
model_final.load_state_dict(best_model_state)

# Plot final training and testing loss
plt.figure(figsize=(10, 6))
plt.plot(loss_history, label='Training Loss')
plt.plot(test_loss_history, label='Test Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(f"Training and Test Loss (Best Incentive Weight = {best_incentive_weight:.3f})")
plt.legend()
save_path = os.path.join('plots', f"trainvstestLoss{date}.png")
plt.savefig(save_path)
plt.show()

"""# Sort the keys so that the line plot connects points in order.
penalty_weights = sorted(incentive_results.keys())
avg_losses = [incentive_results[p] for p in penalty_weights]

plt.figure(figsize=(8, 6))
plt.plot(penalty_weights, avg_losses, marker='o', linestyle='-', color='blue')
plt.xlabel("Penalty Weight")
plt.ylabel("Average CV Loss")
plt.title("Cross Validation Loss per Penalty Weight")
plt.grid(True)
plt.show()"""

# -----------------------------
# Compute and Store Log Likelihood per Trial for the RNN
# -----------------------------

df_hybrid_loaded = pd.read_csv("data/results_hybrid.csv")
# Convert the Hybrid log likelihood column to a list
trial_log_likelihoods_hybrid = df_hybrid_loaded["Hybrid_Log_Likelihood"].tolist()
# Get the average log likelihood (assumed to be the same for every row)
hybrid_avg_log_likelihood = df_hybrid_loaded["avglikelihood"].iloc[0]

def compute_log_likelihood_rnn(model, xs_data, ys_data):
    """
    Compute the average log likelihood and cumulative log likelihood per trial
    for an RNN model over the provided dataset, looping over all individual trials.
    
    Parameters:
        model: The trained RNN model.
        xs_data: Input tensor of shape (seq_length, num_sequences, input_size).
        ys_data: Target tensor of shape (seq_length, num_sequences, 1).
        
    Returns:
        avg_log_likelihood: The average log likelihood across all trials.
        cumulative_ll_list: A list of cumulative log likelihoods for each trial.
        trial_ll_list: A list of log likelihoods for each trial.
    """
    model.eval()
    with torch.no_grad():
        outputs = model(xs_data)  # shape: (seq_length, num_sequences, num_classes)
        log_probs = F.log_softmax(outputs, dim=-1)  # convert logits to log probabilities

        # Flatten the first two dimensions (trials x sequences)
        log_probs_flat = log_probs.reshape(-1, log_probs.shape[-1])  # shape: (seq_length*num_sequences, num_classes)
        true_labels_flat = ys_data.squeeze(-1).long().reshape(-1)      # shape: (seq_length*num_sequences,)

        # Compute log likelihood for each individual trial
        trial_ll_tensor = log_probs_flat.gather(1, true_labels_flat.unsqueeze(1)).squeeze(1)
        trial_ll_list = trial_ll_tensor.tolist()
        cumulative_ll_list = np.cumsum(trial_ll_list).tolist()
        avg_log_likelihood = cumulative_ll_list[-1] / len(trial_ll_list)

    return avg_log_likelihood, cumulative_ll_list, trial_ll_list


# Compute RNN log likelihoods on training and test sets
train_avg_ll, train_cum_ll, train_trial_ll = compute_log_likelihood_rnn(model_final, xs, ys)
test_avg_ll, test_cum_ll, test_trial_ll = compute_log_likelihood_rnn(model_final, xs_test, ys_test)

print(f"train likelihood tensor shape: {len(train_cum_ll)}")
print(f"test likelihood tensor shape: {len(test_cum_ll)}")
print(f"RNN Training Average Log Likelihood: {train_avg_ll:.4f}")
print(f"RNN Testing Average Log Likelihood: {test_avg_ll:.4f}")

# Flatten and concatenate train and test likelihood tensors

concatenated_ll = np.concatenate((train_cum_ll, test_cum_ll))



# -----------------------------
# Plotting RNN vs. Hybrid Log Likelihoods
# -----------------------------
# Assume you have already computed the Hybrid model log likelihoods as follows:
# hybrid_avg_log_likelihood, trial_log_likelihoods = compute_log_likelihood(HybridAgent_opt, df_hybrid, hybrid_optimized_params, n_states_sim)
# where 'trial_log_likelihoods' is a list of cumulative log likelihoods per trial from the Hybrid model.
print(f"Hybrid Model Average Log Likelihood: {hybrid_avg_log_likelihood:.4f}")


# Create x-axes for plotting.
# For the RNN, the number of trials is the sequence length.
# For the Hybrid model, we assume one row per trial.
# Suppose testing starts at trial 101 and hybrid at trial 201
train_trials = list(range(1, len(train_cum_ll) + 1))
test_trials = list(range(len(train_cum_ll), len(train_cum_ll) + len(test_cum_ll)))
hybrid_trials = list(range(len(trial_log_likelihoods_hybrid)))

plt.figure(figsize=(10, 6))
plt.plot(train_trials, train_cum_ll, label='RNN Training Cumulative Log Likelihood', marker='o')
plt.plot(test_trials, test_cum_ll, label='RNN Testing Cumulative Log Likelihood', marker='o')
plt.plot(hybrid_trials, trial_log_likelihoods_hybrid, label='Hybrid Model Cumulative Log Likelihood', marker='o')

plt.xlabel('Trial')
plt.ylabel('Cumulative Log Likelihood')
plt.title('Cumulative Log Likelihood per Trial: RNN vs. Hybrid Model')
plt.legend()
plt.grid(True)
save_path = os.path.join('plots', f"RNNHybridlikelihood{date}.png")
plt.savefig(save_path)
plt.show()




# -----------------------------
# 6. Forward Simulation (unchanged)
# -----------------------------
n_participants = 100
n_blocks_pp = 20      # blocks per participant
n_trials_per_block = 10  # sequence length
innov_variance = 100  
noise_variance = 10

model_final.eval()
simulation_data = []
global_trial = 0

with torch.no_grad():
    for participant in range(n_participants):
        for block in range(n_blocks_pp):
            mean_reward_block = np.random.normal(0, np.sqrt(innov_variance), 2)
            input_seq = torch.zeros(n_trials_per_block, 1, input_size, device=device)
            n_states = n_trials_per_block
            V_t = np.zeros(n_states)
            RU = np.zeros(n_states)  
            TU = np.zeros(n_states) 
            post_mean = np.zeros((2, n_states))
            post_variance = np.ones((2, n_states)) * 5
            kalman_gain = np.zeros((2, n_states))
            
            for trial in range(n_trials_per_block):
                outputs = model_final(input_seq)  # (n_trials, 1, num_classes)
                logits_t = outputs[trial, 0, :]  
                probs_t = F.softmax(logits_t, dim=0).detach().cpu().numpy()
                rnn_action = np.random.choice([0, 1], p=probs_t)
                reward_vector = np.random.normal(mean_reward_block, np.sqrt(noise_variance), 2)
                rnn_reward = reward_vector[rnn_action]
                
                if rnn_action == 0:
                    encoded_input = [rnn_reward, 0]
                else:
                    encoded_input = [0, rnn_reward]
                input_seq[trial, 0, :] = torch.tensor(encoded_input, dtype=torch.float32, device=device)
                
                for i in range(2):
                    if rnn_action == i:
                        if trial == 0:
                            prev_variance = 5
                            prev_mean = 0
                        else:
                            prev_variance = post_variance[i][trial - 1]
                            prev_mean = post_mean[i][trial - 1]
                        kalman_gain[i][trial] = prev_variance / (prev_variance + noise_variance)
                        post_variance[i][trial] = (1 - kalman_gain[i][trial]) * prev_variance
                        post_mean[i][trial] = prev_mean + kalman_gain[i][trial] * (rnn_reward - prev_mean)
                    else:
                        if trial == 0:
                            post_variance[i][trial] = 5
                            post_mean[i][trial] = 0
                        else:
                            post_variance[i][trial] = post_variance[i][trial - 1]
                            post_mean[i][trial] = post_mean[i][trial - 1]
                
                V_t[trial] = post_mean[0][trial] - post_mean[1][trial]
                sigma1 = post_variance[0][trial]
                sigma2 = post_variance[1][trial]
                TU[trial] = np.sqrt(sigma1 + sigma2)
                RU[trial] = np.sqrt(sigma1) - np.sqrt(sigma2)
                
                simulation_data.append({
                    "Participant": participant,
                    "Block": block,
                    "Trial": trial,
                    "Global_Trial": global_trial,
                    "Mean_Reward_Arm0": mean_reward_block[0],
                    "Mean_Reward_Arm1": mean_reward_block[1],
                    "Action": rnn_action,
                    "Reward": rnn_reward,
                    "RNN_Prob_Action0": probs_t[0],
                    "V_t": V_t[trial],
                    "Kalman_post_mean_0": post_mean[0][trial],
                    "Kalman_post_mean_1": post_mean[1][trial],
                    "Kalman_post_variance_0": post_variance[0][trial],
                    "Kalman_post_variance_1": post_variance[1][trial],
                    "Kalman_kalman_gain_0": kalman_gain[0][trial],
                    "Kalman_kalman_gain_1": kalman_gain[1][trial],
                    "RU": RU[trial],
                    "TU": TU[trial],
                    "best_penalty_weight": best_incentive_weight,
                    "num_layers": best_num_layers,
                    "batch_size": batch_size,
                    "average_loss": np.mean(loss_history),
                    "average_test_loss": np.mean(test_loss_history),
                    "optimized_l2_weight": best_learning_rate,
                    "hidden_size": best_hidden_size,
                    "train_log_likelihood": train_avg_ll,
                    "test_log_likelihood": test_avg_ll,
                })
                global_trial += 1


df_simulation = pd.DataFrame(simulation_data)
os.makedirs("data", exist_ok=True)
df_simulation.to_csv(f"data/simulation_RNN_hybrid_{date}.csv", index=False)
print("Simulation complete. Data saved")
