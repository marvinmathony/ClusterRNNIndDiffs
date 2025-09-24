import pandas as pd
import numpy as np
import torch

input_size = 2
sequence_length = 10

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print("CUDA detected")

def load_preprocess_data(filename, n_blocks_pp):

    df = pd.read_csv(filename)
    if filename == 'human_data.csv' or filename == 'kalman_human_data.csv':
        df = df.rename(columns={"choice": "Action", "reward": "Reward", 
                                "mu1": "mean_reward_arm0", "mu2": "mean_reward_arm1"})
    if filename == 'human_data.csv':
        df['Action'] = df['Action'].astype(int) - 1
    elif filename == 'data/exp1_bandit_task_scale.csv':
        df  = df.rename(columns={"C": "Action", "reward": "Reward", 
                                 "mu1": "mean_reward_arm0", "mu2": "mean_reward_arm1"})

    



        
    # check number of unique subjects
    n_participants = df['sub'].nunique()
    print(f"Number of unique subjects: {n_participants}")
    n_trials_per_block = 10
    n_blocks_pp = n_blocks_pp
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
    return xs, ys, xs_test, ys_test, df


def load_preprocess_data_VAE(filename, n_blocks_pp):

    df = pd.read_csv(filename)
    if filename == 'human_data.csv' or filename == 'kalman_human_data.csv':
        df = df.rename(columns={"choice": "Action", "reward": "Reward", 
                                "mu1": "mean_reward_arm0", "mu2": "mean_reward_arm1"})
    if filename == 'human_data.csv':
        df['Action'] = df['Action'].astype(int) - 1
    elif filename == 'data/exp1_bandit_task_scale.csv':
        df  = df.rename(columns={"C": "Action", "reward": "Reward", 
                                 "mu1": "mean_reward_arm0", "mu2": "mean_reward_arm1"})
    else: #elif filename == 'data/sim_individual_differences.csv' or filename == 'data/sim_no_individual_differences.csv':
        df = df.rename(columns={"participant": "sub","action": "Action","reward": "Reward", 
                                 "mu0": "mean_reward_arm0", "mu1": "mean_reward_arm1"})

    # xs_batch should have shape [B, 30, 10, 2] and ys_batch should have shape [B, 30, 10]
    # check number of unique subjects
    n_participants = df['sub'].nunique()
    print(f"Number of unique subjects: {n_participants}")
    n_trials_per_block = 200
    n_blocks_pp = n_blocks_pp
    n_participants_counted = len(df) // (n_trials_per_block * n_blocks_pp)
    assert n_participants == n_participants_counted, "Mismatch in number of participants"
    n_blocks = n_participants * n_blocks_pp

    choices = df['Action'].to_numpy().reshape(n_participants, n_blocks_pp, n_trials_per_block)
    rewards = df['Reward'].to_numpy().reshape(n_participants, n_blocks_pp, n_trials_per_block)
    data = np.stack((choices, rewards), axis=-1)  # (n_participants, n_blocks_pp, n_trials_per_block, 2)

    data_tensor = torch.tensor(data, dtype=torch.float32).to(device)

    # Split into training and test 
    split_idx = int(0.8 * data_tensor.shape[0])
    train_tensor = data_tensor[:split_idx, :, :, :] 
    test_tensor  = data_tensor[split_idx:, :, :, :]   

    n_sequences = train_tensor.shape[0] * train_tensor.shape[1]

    # Prepare training inputs (xs) and labels (ys)
    xs = np.zeros((train_tensor.shape[0], train_tensor.shape[1], train_tensor.shape[2], input_size), dtype=np.float32)
    ys = np.zeros((train_tensor.shape[0], train_tensor.shape[1], train_tensor.shape[2], 1), dtype=np.float32)

    train_choices = train_tensor[:, :, :, 0].cpu().numpy()  
    train_rewards = train_tensor[:, :, :, 1].cpu().numpy()   

    for p in range(train_tensor.shape[0]):  # loop over participants
        for b in range(train_tensor.shape[1]):  # loop over blocks
            prev_vectors = [[0, 0]]  # dummy input for t=0

            for t in range(1, train_tensor.shape[2]):  # loop over trials in block
                choice = train_choices[p, b, t - 1]
                reward = train_rewards[p, b, t - 1]
                if choice == 0:
                    vector = [reward, 0]
                else:
                    vector = [0, reward]
                prev_vectors.append(vector)
            xs[p, b, :, :] = np.array(prev_vectors)
            ys[p, b, :, 0] = train_choices[p, b, :]

    xs = torch.from_numpy(xs).to(device)
    ys = torch.from_numpy(ys).to(device)
    # xs should have shape [B, 30, 10, 2] and ys should have shape [B, 30, 10]

    # Prepare test inputs (xs_test) and labels (ys_test)
    xs_test = np.zeros((test_tensor.shape[0], test_tensor.shape[1], test_tensor.shape[2], input_size), dtype=np.float32)
    ys_test = np.zeros((test_tensor.shape[0], test_tensor.shape[1], test_tensor.shape[2], 1), dtype=np.float32)

    test_choices = test_tensor[:, :, :, 0].cpu().numpy()
    test_rewards = test_tensor[:, :, :, 1].cpu().numpy()

    for p in range(test_tensor.shape[0]):  # loop over test participants
        for b in range(test_tensor.shape[1]):  # loop over blocks
            prev_vectors = [[0, 0]]  # dummy input for t=0
            for t in range(1, test_tensor.shape[2]):  # loop over trials
                choice = test_choices[p, b, t - 1]
                reward = test_rewards[p, b, t - 1]
                if choice == 0:
                    vector = [reward, 0]
                else:
                    vector = [0, reward]
                prev_vectors.append(vector)

            xs_test[p, b, :, :] = np.array(prev_vectors)
            ys_test[p, b, :, 0] = test_choices[p, b, :]

    # Convert to tensors
    xs_test = torch.from_numpy(xs_test).to(device)
    ys_test = torch.from_numpy(ys_test).to(device)

    return xs, ys, xs_test, ys_test, df