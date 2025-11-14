import pandas as pd
import numpy as np
from keras.utils import to_categorical
# load data
children_path = "data/data_children.txt"
adults_path = "data/data_adults.txt"

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
df["subid"] = df["subid"].astype("category").cat.codes
#print(f"df subid coded {df['subid']}")

def generate_xin_with_context(c, r):
    """
    Generate RNN inputs without context-based sequence splitting.
    Each participant contributes one sequence (n_trials long).
    Context information is added as one-hot encoded features.

    Parameters
    ----------
    c : np.ndarray
        Choices (n_subjects, n_trials), 1-4.
    r : np.ndarray
        Rewards (n_subjects, n_trials), scalar

    Returns
    -------
    xin : np.ndarray
        Input array for RNN of shape (n_subjects, n_trials, input_dim + context_dim).
        input_dim = 4 (choice–reward combinations), context_dim = 4 (one-hot contexts)
    mask : np.ndarray
        Boolean array of shape (n_subjects, n_trials) marking valid (non-missing) trials.
    sequence_lengths : list[int]
        Number of valid trials per participant.
    choice_one_hot : np.ndarray
        One-hot encoded choices of shape (n_subjects, n_trials, 2).
    c : np.ndarray
        Same choice array returned for convenience.
    """
    n_subjects, n_trials = c.shape
    input_dim = 5 # 4 options and one scalar reward

    xin = np.zeros((n_subjects, n_trials, input_dim), dtype=np.float32)
    #mask = np.zeros((n_subjects, n_trials), dtype=bool)
    choice_one_hot = np.zeros((n_subjects, n_trials, 4), dtype=np.float32)
    sequence_lengths = []

    for subj_idx in range(n_subjects):
        c_seq = c[subj_idx]
        r_seq = r[subj_idx]

        seq_length = len(c_seq)
        sequence_lengths.append(seq_length)

        #valid_trials = (c_seq != -1)
        #mask[subj_idx, :seq_length] = valid_trials

        # ----- build binary features for previous (choice, reward) combos -----
        cond1 = (c_seq[:-1] == 0)
        cond2 = (c_seq[:-1] == 1)
        cond3 = (c_seq[:-1] == 2)
        cond4 = (c_seq[:-1] == 3)
        

        if seq_length > 1:
            xin[subj_idx, 1:seq_length, 0][cond1] = 1
            xin[subj_idx, 1:seq_length, 1][cond2] = 1
            xin[subj_idx, 1:seq_length, 2][cond3] = 1
            xin[subj_idx, 1:seq_length, 3][cond4] = 1

        # ----- one-hot context encoding (dim = 4) -----
        xin[subj_idx, 1:seq_length, 4] = r_seq[:-1]

        # ----- one-hot choice encoding (target) -----
        choice_one_hot[subj_idx, :seq_length] = to_categorical(
            c_seq, num_classes=4)

    return xin, sequence_lengths, choice_one_hot, c

def preprocess_for_rnn_context_split(df):
    """
    needed arrays:
    - action
    - reward


    """
    
    r_array = df.pivot(index='subid', columns='TestingTrial', values='r').fillna(0).to_numpy()
    c_array = df.pivot(index='subid', columns='TestingTrial', values='c').fillna(0).to_numpy()

    print(f"c_array: {c_array}")
    print(f"c_array shape: {c_array.shape}")
    print(f"r_array: {r_array}")
    print(f"r_array shape: {r_array.shape}")
    return generate_xin_with_context(c_array, r_array)

def split_data_by_subject(subject_ids, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42):
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


sub_ids = np.unique(df['subid'])
train_subjects, val_subjects, test_subjects = split_data_by_subject(sub_ids)
train_df, val_df, test_df = generate_split_dataframes(df, train_subjects, val_subjects, test_subjects)

# save the dataframes
train_df.to_csv("data/train_df_sloutsky.csv", index=False)
val_df.to_csv("data/val_df_sloutsky.csv", index=False)
test_df.to_csv("data/test_df_sloutsky.csv", index=False)

print("Saved train_df, val_df, and test_df to the 'data' folder.")

(train_xin, train_seq_lengths, train_choice_one_hot, c_train) = preprocess_for_rnn_context_split(train_df)
(test_xin, test_seq_lengths, test_choice_one_hot, c_test) = preprocess_for_rnn_context_split(test_df)
(val_xin, val_seq_lengths, val_choice_one_hot, c_val) = preprocess_for_rnn_context_split(val_df)

print(train_xin)

# ---- save training arrays ----
np.save("data/train_xin_sloutsky.npy", train_xin)
np.save("data/train_choice_one_hot_sloutsky.npy", train_choice_one_hot)
np.save("data/c_train_sloutsky.npy", c_train)

# ---- save validation arrays ----
np.save("data/val_xin_sloutsky.npy", val_xin)
np.save("data/val_choice_one_hot_sloutsky.npy", val_choice_one_hot)
np.save("data/c_val_sloutsky.npy", c_val)

# ---- save test arrays ----
np.save("data/test_xin_sloutsky.npy", test_xin)
np.save("data/test_choice_one_hot_sloutsky.npy", test_choice_one_hot)
np.save("data/c_test_sloutsky.npy", c_test)