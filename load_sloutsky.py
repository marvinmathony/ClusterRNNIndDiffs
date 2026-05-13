import os
import pandas as pd
import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import KFold
import torch
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# load data
children_path = "data/Children_Exp2_train.txt"
adults_path = "data/Adults_Exp2_train.txt"

# Read the data
df_children = pd.read_csv(
    children_path,
    sep=r' ',        
    quotechar='"',     
    engine='python'    
)
df_adults = pd.read_csv(
    adults_path,
    sep=r' ',        
    quotechar='"',     
    engine='python'    
)
df = pd.concat([df_children, df_adults])
df["subid_original"] = df["subid"].copy()
df["subid_original"] = df["subid_original"].astype(str)
df["subid"] = df["subid"].astype("category").cat.codes
# 4-year-olds IDs
young_child = [102986, '109913A', 106722, 100557, 108735, 108721, 105243, 105296, 100370, 106241,
                102785, '102137-1', 104995, 108245, 105583, 103098, 103115, 108779, 100333, '104808-1',
                105262, '107177b',103434, 101277, 103843, 107324, 109400, 104779, 108441,100500]
# 5- to 6-year-olds IDs
old_child = [13911, 100021, 100118, 102162, 102823, 103131, 103886, 104337, 104516, 104812, 105727,
              107224, 108515, 109026, 109259, '103117-1', '184794A',  '108154-1', '105199-1', 108548,
              104558, 107707, 109085, 102099, 104816, 100813, 107346, '104349B', 101933, 106875, "104816B"]

young_child_set = set(map(str, young_child))
old_child_set   = set(map(str, old_child))

df["age"] = "adult"
df.loc[df["subid_original"].isin(young_child_set), "age"] = "young_child"
df.loc[df["subid_original"].isin(old_child_set), "age"] = "old_child"
#print(f"df subid coded {df['subid']}")

def generate_xin_with_context(c, r, X_stimuli):
    """
    Generate RNN inputs from structured per-option stimulus vectors.

    Parameters
    ----------
    c : np.ndarray
        Choices (n_subjects, n_trials), 0-3.
    r : np.ndarray
        Received rewards (n_subjects, n_trials).
    X_stimuli : np.ndarray
        Per-option feature vectors of shape (n_subjects, 4, n_trials, 9).
        Each option's 9 dims: [color_one_hot(4), animal_one_hot(3), novelty(1), reward(1)].

    Returns
    -------
    xin : np.ndarray
        Shape (n_subjects, n_trials, 5 + 4*9).
        Slots 0-3: previous choice one-hot; slot 4: previous received reward;
        slots 5-40: flattened per-option stimulus vectors.
    sequence_lengths : list[int]
    choice_one_hot : np.ndarray  (n_subjects, n_trials, 4)
    c : np.ndarray
    """
    n_subjects, n_trials = c.shape
    input_dim = 5  # 4 prev-choice one-hot + 1 prev received reward

    xin = np.zeros((n_subjects, n_trials, input_dim), dtype=np.float32)
    choice_one_hot = np.zeros((n_subjects, n_trials, 4), dtype=np.float32)
    sequence_lengths = []

    for subj_idx in range(n_subjects):
        c_seq = c[subj_idx]
        r_seq = r[subj_idx]

        seq_length = len(c_seq)
        sequence_lengths.append(seq_length)

        cond1 = (c_seq[:-1] == 0)
        cond2 = (c_seq[:-1] == 1)
        cond3 = (c_seq[:-1] == 2)
        cond4 = (c_seq[:-1] == 3)

        if seq_length > 1:
            # previous choice one-hot
            xin[subj_idx, 1:seq_length, 0][cond1] = 1
            xin[subj_idx, 1:seq_length, 1][cond2] = 1
            xin[subj_idx, 1:seq_length, 2][cond3] = 1
            xin[subj_idx, 1:seq_length, 3][cond4] = 1
            # previous received reward
            xin[subj_idx, 1:seq_length, 4] = r_seq[:-1]

        # one-hot choice encoding (target)
        choice_one_hot[subj_idx, :seq_length] = to_categorical(c_seq, num_classes=4)

    # Flatten stimulus vectors: (n_subjects, 4, n_trials, 9) -> (n_subjects, n_trials, 36)
    X_stimuli_flat = X_stimuli.transpose(0, 2, 1, 3).reshape(n_subjects, n_trials, -1)
    xin = np.concatenate([xin, X_stimuli_flat], axis=-1)

    print(f"xin shape: {xin.shape}")

    return xin, sequence_lengths, choice_one_hot, c

def build_vocabulary(df, columns):
    vocab = set()
    for column in columns:
        vocab.update(df[column].dropna().unique())

    vocab = sorted(vocab)
    # Special tokens
    PAD = "<PAD>"
    UNK = "<UNK>"

    #index
    stoi = {PAD: 0, UNK: 1}
    stoi.update({s:i+2 for i,s in enumerate(vocab)}) #maps stimulus to index
    itos = {i:s for i,s in enumerate(stoi.items())} #maps index to stimulus

    vocab_size = (len(stoi))
    print("Vocab size:", vocab_size)
    return stoi, itos

# def encode_row(row, df, columns, stoi):
#     seq = []
#     for column in columns:
#         stim_id = stoi.get(row[df[column]], "<UNK>")
#         seq.append(stim_id)
#     return seq

def random_table(n_tokens, dim, seed):
    #initialize random generator
    rng = np.random.default_rng(seed)
    #draw according to dim (tokens, dim)
    E = rng.standard_normal((n_tokens, dim)).astype(np.float32)
    #normalize so that embeddings are not interpreted with magnitude
    E /= (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    return E

def max_pairwise_sim(E):
    S = E @ E.T
    np.fill_diagonal(S, -np.inf)
    return float(np.max(S))

def choose_dim(n_tokens, dims, seed, maximum_similarity):
    #iterate over dims
    for dim in dims:
        #create random table
        E = random_table(n_tokens, dim, seed)
        #calculate max pairwise similariy
        m = max_pairwise_sim(E)
        if m <= maximum_similarity:
            return dim, m
    E = random_table(n_tokens, dims[-1], seed=seed)
    print(f"no dim in {dims} has lower cosine similarity than maximum allowed, so highest dim was chosen")
    return dims[-1], max_pairwise_sim(E)    
    
def tokenize_stimuli(df, columns):
    #get vocabulary
    stoi, itos = build_vocabulary(df, columns)
    #get embedding dimensions
    n_tokens = len(stoi)
    maximum_similarity = 0.5
    dimensions=(2, 4, 8, 12, 16, 24, 32)
    seed = 0
    D, m = choose_dim(n_tokens, dimensions, seed, maximum_similarity)
    print("chosen D =", D, " max cosine =", m)
    E = random_table(n_tokens, D, seed)

    def encode_row(row):
        ids = [stoi[row[c]] for c in columns]
        return E[ids]

    X = np.stack([encode_row(r) for _, r in df.iterrows()], axis=0)  # (B, T, D)
    X = torch.tensor(X, dtype=torch.float32)
    print(f"shape of tokenized stimuli: {X.shape}")
    print(f"tokenized tensor: {X}")
    return X

def preprocess_for_rnn_new(df):
    options = {1: "left", 2: "center_left", 3: "center_right", 4: "right"}
    df_options = ["1", "2", "3"]
    colors = ["blue", "pink", "green", "yellow"]
    animals = ["Panda", "Mouse", "Alpaca"]

    # --- One-hot encode colors and animals per option position ---
    for option_num, option_name in options.items():
        for color in colors:
            combined = pd.Series(False, index=df.index)
            for j_key in df_options:
                col = f"Option{j_key}"
                combined = combined | (df[col].str.contains(color) & df[col].str.contains(str(option_num)))
            df[f"{option_name}_{color}"] = combined.astype(int)

        for animal in animals:
            combined = pd.Series(False, index=df.index)
            for j_key in df_options:
                col = f"Option{j_key}"
                combined = combined | (df[col].str.contains(animal) & df[col].str.contains(str(option_num)))
            df[f"{option_name}_{animal.lower()}"] = combined.astype(int)

    # --- Extract option number from Option1/2/3 strings ---
    for j_key in df_options:
        col = f"Option{j_key}"
        df[f"{col}_num"] = df[col].str.extract(r'(\d+)').astype(int)

    # --- Assign rewards per option position ---
    # Start from Value columns as fallback, then override from explicitly shown options
    for option_num_key, option_name_val in options.items():
        value_col = f"Value{option_num_key}"
        df[f"{option_name_val}_reward"] = df[value_col] if value_col in df.columns else 0

    for j_key in df_options:
        num_col = f"Option{j_key}_num"
        val_col = f"Value{j_key}"
        for option_num_val, option_name_val in options.items():
            mask = (df[num_col] == option_num_val)
            df.loc[mask, f"{option_name_val}_reward"] = df.loc[mask, val_col]

    # --- Novelty: 1 if option was not explicitly shown this trial ---
    for option_name_val in options.values():
        df[f"novelty_{option_name_val}"] = 1

    for option_num_key, option_name_val in options.items():
        is_present = pd.Series(False, index=df.index)
        for j_key in df_options:
            num_col = f"Option{j_key}_num"
            if num_col in df.columns:
                is_present = is_present | (df[num_col] == option_num_key)
        df.loc[is_present, f"novelty_{option_name_val}"] = 0

    # --- Build per-option stimulus arrays (n_subjects, n_trials, 9) ---
    input_columns = {
        "left":         ["left_blue", "left_pink", "left_green", "left_yellow",
                         "left_panda", "left_mouse", "left_alpaca", "novelty_left", "left_reward"],
        "center_left":  ["center_left_blue", "center_left_pink", "center_left_green", "center_left_yellow",
                         "center_left_panda", "center_left_mouse", "center_left_alpaca", "novelty_center_left", "center_left_reward"],
        "center_right": ["center_right_blue", "center_right_pink", "center_right_green", "center_right_yellow",
                         "center_right_panda", "center_right_mouse", "center_right_alpaca", "novelty_center_right", "center_right_reward"],
        "right":        ["right_blue", "right_pink", "right_green", "right_yellow",
                         "right_panda", "right_mouse", "right_alpaca", "novelty_right", "right_reward"],
    }

    subject_order = df.pivot(index='subid', columns='TrainingTrial', values='c').index.tolist()
    n_subjects = len(subject_order)
    n_trials = df['TrainingTrial'].nunique()

    option_arrays = []
    for option_name, cols in input_columns.items():
        piv = df.pivot(index='subid', columns='TrainingTrial', values=cols).fillna(0)
        # Multi-value pivot: (n_subjects, n_features * n_trials), features vary slowest
        arr = piv.to_numpy().reshape(n_subjects, len(cols), n_trials).transpose(0, 2, 1)  # (n_subjects, n_trials, 9)
        option_arrays.append(arr)

    X_stimuli = np.stack(option_arrays, axis=1)  # (n_subjects, 4, n_trials, 9)
    print(f"X_stimuli shape: {X_stimuli.shape}")

    # --- Extract choice and reward arrays ---
    c_array = df.pivot(index='subid', columns='TrainingTrial', values='c').loc[subject_order].fillna(0).to_numpy().astype(int)
    r_array = df.pivot(index='subid', columns='TrainingTrial', values='r').loc[subject_order].fillna(0).to_numpy().astype(np.float32)

    xin, seq_lengths, choice_one_hot, c_out = generate_xin_with_context(c_array, r_array, X_stimuli)

    # --- Verification print: subject 0, transition from trial t-1 to trial t ---
    s, t = 0, 1
    subid0 = subject_order[s]
    sorted_trials = sorted(df['TrainingTrial'].unique())
    trial_prev = sorted_trials[t - 1]   # previous trial
    trial_curr = sorted_trials[t]        # current trial
    raw_prev = df[(df['subid'] == subid0) & (df['TrainingTrial'] == trial_prev)].iloc[0]
    raw_curr = df[(df['subid'] == subid0) & (df['TrainingTrial'] == trial_curr)].iloc[0]
    option_names = list(input_columns.keys())
    feat_names = ["blue", "pink", "green", "yellow", "panda", "mouse", "alpaca", "novelty", "reward"]

    print(f"\n=== Verification: subject {subid0}, original_sub_id={raw_prev['subid_original']}, ===")
    print(f"  [PREVIOUS trial {trial_prev}] raw c={raw_prev['c']}, r={raw_prev['r']}")
    print(f"  [CURRENT  trial {trial_curr}] raw Option1={raw_curr['Option1']}, Option2={raw_curr['Option2']}, Option3={raw_curr['Option3']}")
    print(f"  [CURRENT  trial {trial_curr}] raw Value1={raw_curr['Value1']}, Value2={raw_curr['Value2']}, Value3={raw_curr['Value3']}, Value4={raw_curr['Value4']}")
    print(f"  ---")
    print(f"  xin[s,t] prev_choice_OH={xin[s, t, :4]}  <- should match one-hot of prev c={raw_prev['c']}")
    print(f"  xin[s,t] prev_reward   ={xin[s, t, 4]:.3f}     <- should match prev r={raw_prev['r']}")
    for i, oname in enumerate(option_names):
        vec = X_stimuli[s, i, t, :]
        print(f"  xin[s,t] X_stimuli[{oname}]: {dict(zip(feat_names, vec.round(3)))}")
    print(f"  xin full shape: {xin.shape}  (expected n_subjects x n_trials x 41)")

    return xin, seq_lengths, choice_one_hot, c_out, subject_order



def preprocess_for_rnn_context_split(df):
    """
    needed arrays:
    - action
    - reward of chosen action
    - all transparent rewards


    """
    
    r_array = df.pivot(index='subid', columns='TrainingTrial', values='r').fillna(0).to_numpy()
    c_array = df.pivot(index='subid', columns='TrainingTrial', values='c').fillna(0).to_numpy()
    #transparent value columns
    value_columns = ['Value1','Value2','Value3','Value4']
    all_values = df.pivot(index='subid', columns='TrainingTrial', values=value_columns).fillna(0)
    n_trials = all_values.columns.levels[1].size
    n_subjects = all_values.shape[0]
    n_vals = len(value_columns)
    X = all_values.to_numpy().reshape(n_subjects, n_vals, n_trials)
    #stimulus strings
    #pseudocode
    # stimulus_columns = ['Option1', 'Option2', 'Option3']
    # X_stimuli = tokenize_stimuli(df, stimulus_columns)

    stimulus_columns = ['Option1', 'Option2', 'Option3']
    stoi, itos = build_vocabulary(df, stimulus_columns)
    n_tokens = len(stoi)
    D, m = choose_dim(n_tokens, dims=(2, 4, 8, 12, 16, 24, 32), seed=0, maximum_similarity=0.5)
    E = random_table(n_tokens, D, seed=0)

    # Pivot Options 1-3
    subject_order = df.pivot(index='subid', columns='TrainingTrial', values='r').index.tolist()
    option_arrays = []
    for col in stimulus_columns:
        pivoted = df.pivot(index='subid', columns='TrainingTrial', values=col).fillna('<PAD>')
        pivoted = pivoted.loc[subject_order]
        ids = pivoted.map(lambda s: stoi.get(s, stoi['<UNK>'])).to_numpy()  # (n_subjects, n_trials)
        option_arrays.append(E[ids])  # (n_subjects, n_trials, D)

    # Option4: sample from E, excluding each participant's own Options 1-3 stimuli
    special_ids = {stoi['<PAD>'], stoi['<UNK>']}
    rng = np.random.default_rng(seed=1)
    option4 = np.zeros((n_subjects, n_trials, D), dtype=np.float32)

    for i, subid in enumerate(subject_order):
        seen = set(df[df['subid'] == subid][stimulus_columns].stack().unique())
        seen_ids = {stoi[s] for s in seen if s in stoi}
        novel_ids = [idx for idx in range(n_tokens) if idx not in seen_ids and idx not in special_ids]
        sampled = rng.choice(novel_ids, size=n_trials, replace=True)
        option4[i] = E[sampled]

    option_arrays.append(option4)
    X_stimuli = np.stack(option_arrays, axis=1)  # (n_subjects, 4, n_trials, D)

    # stimulus_columns = ['Option1', 'Option2', 'Option3']
    # all_stimuli = df.pivot(index='subid', columns='TrainingTrial', values=stimulus_columns).fillna(0)
    # X_stimuli = all_values.to_numpy().reshape(n_subjects, n_vals, n_trials)

    print(f"c_array: {c_array}")
    print(f"c_array shape: {c_array.shape}")
    print(f"r_array: {r_array}")
    print(f"r_array shape: {r_array.shape}")
    print(f"all_values: {X}")
    print(f"all_values shape: {X.shape}")
    print(f"stimulus tensor shape: {X_stimuli.shape}")
    print(f"stimulus tensor: {X_stimuli}")
    xin, seq_lengths, choice_one_hot, c_out = generate_xin_with_context(c_array, r_array, X, X_stimuli)
    return xin, seq_lengths, choice_one_hot, c_out, subject_order

def split_data_by_subject(subject_ids, train_ratio=0.7, val_ratio=0, test_ratio=0.3, seed=42):
    """
    Split subjects into training, validation, and test sets based on predefined ratios.

    Parameters:
    - subject_ids (array-like): List or array of unique subject IDs.
    - train_ratio (float): Proportion of subjects assigned to the training set.
    - val_ratio (float): Proportion of subjects assigned to the validation set.
    - test_ratio (float): Proportion of subjects assigned to the test set.
    - seed (int): Random seed for reproducibility.

    Returns:
    - train_subjects (set): Set of subject IDs for training.
    - val_subjects (set): Set of subject IDs for validation.
    - test_subjects (set): Set of subject IDs for testing.
    """
    np.random.seed(seed)

    # Shuffle subject IDs
    subject_ids = np.array(subject_ids)
    np.random.shuffle(subject_ids)

    # Determine the split indices
    n_total = len(subject_ids)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_subjects = set(subject_ids[:n_train])
    val_subjects = set(subject_ids[n_train:n_train + n_val])
    test_subjects = set(subject_ids[n_train + n_val:])

    return train_subjects, val_subjects, test_subjects
    
def generate_split_dataframes(final_df, train_subjects, val_subjects, test_subjects):
    """
    Split a DataFrame into training, validation, and test sets based on subject IDs.

    Parameters:
    - final_df (DataFrame): The full dataset containing a 'subject' column.
    - train_subjects (set): Set of subject IDs for training.
    - val_subjects (set): Set of subject IDs for validation.
    - test_subjects (set): Set of subject IDs for testing.

    Returns:
    - train_df (DataFrame): DataFrame containing training data.
    - val_df (DataFrame): DataFrame containing validation data.
    - test_df (DataFrame): DataFrame containing test data.
    """
    final_df["Response_which"] = final_df["Response_which"].str.replace("option", "").astype(int)
    final_df["Response_which"] = final_df["Response_which"] - 1
    final_df = final_df.rename(columns={"Response_which": "c", "Response_Value": "r"})
    train_df = final_df[final_df['subid'].isin(train_subjects)].reset_index(drop=True)
    val_df = final_df[final_df['subid'].isin(val_subjects)].reset_index(drop=True)
    test_df = final_df[final_df['subid'].isin(test_subjects)].reset_index(drop=True)
    print(f"train df after column rename {train_df.head(50)}")
    return train_df, val_df, test_df

def split_arrays_by_subject(subject_order, train_subjects, val_subjects, test_subjects, *arrays):
    s2i = {s: i for i, s in enumerate(subject_order)}
    train_idx = [s2i[s] for s in subject_order if s in train_subjects]
    val_idx   = [s2i[s] for s in subject_order if s in val_subjects]
    test_idx  = [s2i[s] for s in subject_order if s in test_subjects]

    splits = []
    for arr in arrays:
        if isinstance(arr, list):
            splits.append(([arr[i] for i in train_idx],
                           [arr[i] for i in val_idx],
                           [arr[i] for i in test_idx]))
        else:
            splits.append((arr[train_idx], arr[val_idx], arr[test_idx]))
    return splits

def generate_whole_df(df):
    """
    takes in df, changes column names and adapts indexing
    """
    df["Response_which"] = df["Response_which"].str.replace("option", "").astype(int)
    df["Response_which"] = df["Response_which"] - 1
    df = df.rename(columns={"Response_which": "c", "Response_Value": "r"})
    return df

df_whole = df.copy()
exp2_train_all_participants = generate_whole_df(df_whole)

exp2_train_all_participants.to_csv("data_sloutsky/exp2_train_all_participants.csv", index=False)

sub_ids = np.unique(df['subid'])
train_subjects, val_subjects, test_subjects = split_data_by_subject(sub_ids)
train_df, val_df, test_df = generate_split_dataframes(df, train_subjects, val_subjects, test_subjects)

# save the dataframes
train_df.to_csv("data/train_df_sloutsky.csv", index=False)
train_df.to_csv("data_sloutsky/df_train.csv", index=False)
# val_df.to_csv("data/val_df_sloutsky.csv", index=False)
# val_df.to_csv("data_sloutsky/df_val.csv", index=False)
test_df.to_csv("data/test_df_sloutsky.csv", index=False)
test_df.to_csv("data_sloutsky/df_test.csv", index=False)


print("Saved train_df, val_df, and test_df to the 'data' folder.")
(whole_xin, whole_seq_lengths, whole_choice_one_hot, c_whole, subject_order) = preprocess_for_rnn_new(exp2_train_all_participants)

splits = split_arrays_by_subject(
    subject_order, train_subjects, val_subjects, test_subjects,
    whole_xin, whole_seq_lengths, whole_choice_one_hot, c_whole
)
(xin_train,       xin_val,       xin_test), \
(seq_len_train,   seq_len_val,   seq_len_test), \
(choice_oh_train, choice_oh_val, choice_oh_test), \
(c_train,         c_val,         c_test) = splits

# ---- save training arrays ----
np.save("data/train_xin_sloutsky.npy", xin_train)
np.save("data/train_choice_one_hot_sloutsky.npy", choice_oh_train)
np.save("data/c_train_sloutsky.npy", c_train)

np.save("data_sloutsky/xin_train.npy", xin_train)
np.save("data_sloutsky/choice_one_hot_train.npy", choice_oh_train)
np.save("data_sloutsky/c_train.npy", c_train)

# ---- save validation arrays ----
np.save("data/val_xin_sloutsky.npy", xin_val)
np.save("data/val_choice_one_hot_sloutsky.npy", choice_oh_val)
np.save("data/c_val_sloutsky.npy", c_val)

np.save("data_sloutsky/xin_val.npy", xin_val)
np.save("data_sloutsky/choice_one_hot_val.npy", choice_oh_val)
np.save("data_sloutsky/c_val.npy", c_val)

# ---- save test arrays ----
np.save("data/test_xin_sloutsky.npy", xin_test)
np.save("data/test_choice_one_hot_sloutsky.npy", choice_oh_test)
np.save("data/c_test_sloutsky.npy", c_test)

np.save("data_sloutsky/xin_test.npy", xin_test)
np.save("data_sloutsky/choice_one_hot_test.npy", choice_oh_test)
np.save("data_sloutsky/c_test.npy", c_test)

# ---- Outer 3-fold CV splits (all participants are test participants once) ----
N_OUTER_FOLDS = 3
all_subids_arr = np.array(sorted(np.unique(df['subid'])))
kf_outer = KFold(n_splits=N_OUTER_FOLDS, shuffle=True, random_state=42)

for fold_idx, (train_idx, test_idx) in enumerate(kf_outer.split(all_subids_arr)):
    fold_train_subjects = set(all_subids_arr[train_idx])
    fold_test_subjects  = set(all_subids_arr[test_idx])

    fold_dir = f"data_sloutsky/fold{fold_idx}"
    os.makedirs(fold_dir, exist_ok=True)

    # DataFrames (already have renamed columns c/r from generate_whole_df)
    fold_train_df = exp2_train_all_participants[
        exp2_train_all_participants['subid'].isin(fold_train_subjects)
    ].reset_index(drop=True)
    fold_test_df = exp2_train_all_participants[
        exp2_train_all_participants['subid'].isin(fold_test_subjects)
    ].reset_index(drop=True)
    fold_train_df.to_csv(f"{fold_dir}/df_train.csv", index=False)
    fold_test_df.to_csv(f"{fold_dir}/df_test.csv", index=False)

    # Arrays: use empty set for val (inner CV handled inside training)
    fold_splits = split_arrays_by_subject(
        subject_order, fold_train_subjects, set(), fold_test_subjects,
        whole_xin, whole_seq_lengths, whole_choice_one_hot, c_whole
    )
    (xin_tr, _, xin_te), _, (oh_tr, _, oh_te), (c_tr, _, c_te) = fold_splits

    np.save(f"{fold_dir}/xin_train.npy",          xin_tr)
    np.save(f"{fold_dir}/xin_test.npy",           xin_te)
    np.save(f"{fold_dir}/choice_one_hot_train.npy", oh_tr)
    np.save(f"{fold_dir}/choice_one_hot_test.npy",  oh_te)
    np.save(f"{fold_dir}/c_train.npy",            c_tr)
    np.save(f"{fold_dir}/c_test.npy",             c_te)

    print(f"Fold {fold_idx}: {len(fold_train_subjects)} train, "
          f"{len(fold_test_subjects)} test subjects → {fold_dir}/")

print("Outer CV fold data generation complete.")