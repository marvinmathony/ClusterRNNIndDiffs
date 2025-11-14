import scipy.io as sio
import pandas as pd
import numpy as np
import RL_fitting_functions as fit
import os
from tensorflow.keras.utils import to_categorical


folder_strings =  ["Adolescents", "Adults"]
folder_string = "data/" 
#os.chdir(f"/content/drive/My Drive/RNNLatents/")
#directory = os.fsencode("/Users/marvinmathony/VSCodeProjects/RNNexploration/data/CompDevLearning/Adolescents")
#directory = "."
df_adolescents = pd.DataFrame()
df_adults = pd.DataFrame()
all_groups = pd.DataFrame()


for folder in folder_strings:
  print(folder)
  directory = f"{folder_string}{folder}"
  df = pd.DataFrame()
  for file in os.listdir(directory):
      filename = os.fsdecode(file)
      if filename.endswith(".mat"): 
          print(filename)
          data = sio.loadmat(os.path.join(directory, filename))
          data = data['data']
          colnames = ["sub", "session", "trial", "context",
                      "checktime", "choice_side", "choice_correct", "factual_outcome",
                      "alternative_outcome", "rt"]
          sub_df = pd.DataFrame(data, columns=colnames)
          df = pd.concat([df, sub_df], ignore_index=True)
          if sub_df.shape[0] != 80:
            print(f"length of participant {filename} is {sub_df.shape[0]} but should be 80")
            
          
      else:
          print("no matlab files in this directory")
  
    ### one-hot encode: 1.c_right, 2.c_left, 3.r_chosen_left, 4.r_chosen_right 5.punish_chosen_left, 6.punish_chosen_right
    # 7. r_unchosen_right, 8. r_unchosen_left, 9. punish_unchosen_left, 10. punish_unchosen_right, 11. neutral_unchosen_right,
    # 12. neutral_unchosen_left
    # conditions: Reward/Partial, Reward/Complete, Punishment/Partial, Punishment/Complete
  con = {}
  cho = {}
  factual_out = {}
  counterfactual_out = {}

  df['sub_across_groups'] = np.repeat(np.arange(len(df) // 80), 80)
  for id in np.unique(df['sub_across_groups']):
    #print(id)
    subj_spec_df = df[df['sub_across_groups'] == id]
    con[id] = subj_spec_df["context"]
    cho[id] = ((subj_spec_df["choice_side"] + 1)/2).astype(int)
    factual_out[id] = subj_spec_df["factual_outcome"]
    counterfactual_out[id] = subj_spec_df["alternative_outcome"]

  

  final_data = []
  for id in np.unique(df['sub_across_groups']):
    subject_df = pd.DataFrame({
        "subject": [id] * len(con[id]),
        "trial": range(1, len(con[id]) + 1),
        "context": (con[id]).astype(int),
        "c": (cho[id]).astype(int),
        "r": (factual_out[id]).astype(int),
        "counterfactual_r": (counterfactual_out[id]).astype(int)
    })
    # sort by first context, then trial
    #subject_df = subject_df.sort_values(by=["context", "trial"]).reset_index(drop=True)

    # create new trial number column after sorting
    #subject_df["trial"] = np.arange(1, len(subject_df)+1)

    # append sorted dataframe to final data list
    final_data.append(subject_df)

  if final_data:
    final_df = pd.concat(final_data, ignore_index=True)
    #print(final_df)
  else:
    print("No data to concatenate.")
  #print(final_df.isna().any())
  is_infinite = np.isinf(final_df.values)
  # Check if any infinity values exist
  res = is_infinite.any()
  #print(res)
  # add a new column to indicate whether the context changed (1:changed, 0: not changed)
  #final_df['context_changed'] = (final_df['context'] != final_df.groupby('subject')['context'].shift(1))
  final_df['group'] = "adolescent" if folder == "Adolescents" else "adult"
  if folder == "Adolescents":
    df_adolescents = pd.concat([df_adolescents, final_df], ignore_index=True)
  else:
    df_adults = pd.concat([df_adults, final_df], ignore_index=True)
  #print(final_df)
all_groups = pd.concat([df_adolescents, df_adults], ignore_index=True)
all_groups['sub_across_groups'] = np.repeat(np.arange(len(all_groups) // 80), 80)
print(all_groups)

# helper functions for data split
# here will have to implement leave one out
def split_data_by_subject(subject_ids, train_ratio=0.75, val_ratio=0.05, test_ratio=0.2, seed=42):
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
    train_df = final_df[final_df['subject'].isin(train_subjects)].reset_index(drop=True)
    val_df = final_df[final_df['subject'].isin(val_subjects)].reset_index(drop=True)
    test_df = final_df[final_df['subject'].isin(test_subjects)].reset_index(drop=True)

    return train_df, val_df, test_df

def generate_xin_with_context(c, r, context, counterfactual_r):
    """
    Generate RNN inputs without context-based sequence splitting.
    Each participant contributes one sequence (n_trials long).
    Context information is added as one-hot encoded features.

    Parameters
    ----------
    c : np.ndarray
        Choices (n_subjects, n_trials), values 0 or 1.
    r : np.ndarray
        Rewards (n_subjects, n_trials), values 0 or 1.
    context : np.ndarray
        Context identifiers (n_subjects, n_trials), integer labels 1–4.
    counterfactual_r: np.ndarray
        counterfactual rewards (n_subjects, n_trials), values 0 or 1

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
    input_dim = 8
    context_dim = 4

    xin = np.zeros((n_subjects, n_trials, input_dim + context_dim), dtype=np.float32)
    mask = np.zeros((n_subjects, n_trials), dtype=bool)
    choice_one_hot = np.zeros((n_subjects, n_trials, 2), dtype=np.float32)
    sequence_lengths = []

    for subj_idx in range(n_subjects):
        c_seq = c[subj_idx]
        r_seq = r[subj_idx]
        ctx_seq = context[subj_idx]
        counterfactual_r_seq = counterfactual_r[subj_idx]

        seq_length = len(c_seq)
        sequence_lengths.append(seq_length)

        valid_trials = (c_seq != -1)
        mask[subj_idx, :seq_length] = valid_trials

        # ----- build 4 binary features for previous (choice, reward) combos -----
        cond1 = (c_seq[:-1] == 0) & (r_seq[:-1] == 1) & (counterfactual_r_seq[:-1] == 1)
        cond2 = (c_seq[:-1] == 0) & (r_seq[:-1] == 1) & (counterfactual_r_seq[:-1] == 0)
        cond3 = (c_seq[:-1] == 1) & (r_seq[:-1] == 1) & (counterfactual_r_seq[:-1] == 1)
        cond4 = (c_seq[:-1] == 1) & (r_seq[:-1] == 1) & (counterfactual_r_seq[:-1] == 0)
        cond5 = (c_seq[:-1] == 0) & (r_seq[:-1] == 0) & (counterfactual_r_seq[:-1] == 1)
        cond6 = (c_seq[:-1] == 0) & (r_seq[:-1] == 0) & (counterfactual_r_seq[:-1] == 0)
        cond7 = (c_seq[:-1] == 1) & (r_seq[:-1] == 0) & (counterfactual_r_seq[:-1] == 1)
        cond8 = (c_seq[:-1] == 1) & (r_seq[:-1] == 0) & (counterfactual_r_seq[:-1] == 0)

        if seq_length > 1:
            xin[subj_idx, 1:seq_length, 0][cond1] = 1
            xin[subj_idx, 1:seq_length, 1][cond2] = 1
            xin[subj_idx, 1:seq_length, 2][cond3] = 1
            xin[subj_idx, 1:seq_length, 3][cond4] = 1
            xin[subj_idx, 1:seq_length, 4][cond5] = 1
            xin[subj_idx, 1:seq_length, 5][cond6] = 1
            xin[subj_idx, 1:seq_length, 6][cond7] = 1
            xin[subj_idx, 1:seq_length, 7][cond8] = 1

        # ----- one-hot context encoding (dim = 4) -----
        context_one_hot = np.zeros((seq_length, context_dim), dtype=np.float32)
        valid_ctx = (ctx_seq > 0)
        context_one_hot[np.arange(seq_length)[valid_ctx], ctx_seq[valid_ctx] - 1] = 1
        xin[subj_idx, :seq_length, input_dim:] = context_one_hot

        # ----- one-hot choice encoding (target) -----
        choice_one_hot[subj_idx, :seq_length] = to_categorical(
            np.where(valid_trials, c_seq, 0), num_classes=2
        )
        choice_one_hot[subj_idx, ~valid_trials] = 0

    return xin, mask, sequence_lengths, choice_one_hot, c


def preprocess_for_rnn_context_split(df):
    """
    Preprocess the data for RNN input.
    """

    r_array = df.pivot(index='sub_across_groups', columns='trial', values='r').fillna(0).to_numpy()
    c_array = df.pivot(index='sub_across_groups', columns='trial', values='c').fillna(0).to_numpy()
    context_array = df.pivot(index='sub_across_groups', columns='trial', values='context').fillna(0).to_numpy()
    counterfactual_r = df.pivot(index='sub_across_groups', columns='trial', values='counterfactual_r').fillna(0).to_numpy()

    print(f"c_array: {c_array}")
    print(f"c_array shape: {c_array.shape}")
    print(f"r_array: {r_array}")
    print(f"r_array shape: {r_array.shape}")
    print(f"context_array: {context_array}")
    print(f"context_array shape: {context_array.shape}")
    print(f"counterfactual_r: {counterfactual_r}")
    print(f"counterfactual_r: {counterfactual_r.shape}")

    #n_subjects = len(df['sub_across_groups'].unique())
    #n_trials = df.groupby('sub_across_groups')['trial'].count().max()
    #context_array = np.full((n_subjects, n_trials), -1, dtype=int)

    """for idx, (subject_id, group) in enumerate(df.groupby('sub_across_groups')):
        context_sequence = group['context'].values
        context_array[idx, :len(context_sequence)] = context_sequence"""
    #print(context_array)
    #print(context_array.shape)
    return generate_xin_with_context(c_array, r_array, context_array, counterfactual_r)

pd.options.display.max_rows = 1000
sub_ids = np.unique(all_groups['sub_across_groups'])
train_subjects, val_subjects, test_subjects = split_data_by_subject(sub_ids)
train_df, val_df, test_df = generate_split_dataframes(all_groups, train_subjects, val_subjects, test_subjects)
# save the dataframes
train_df.to_csv("data/train_df_palminteri.csv", index=False)
val_df.to_csv("data/val_df_palminteri.csv", index=False)
test_df.to_csv("data/test_df_palminteri.csv", index=False)

print("Saved train_df, val_df, and test_df to the 'data' folder.")


(train_xin, train_mask, train_seq_lengths, train_choice_one_hot, c_train) = preprocess_for_rnn_context_split(train_df)
(test_xin, test_mask, test_seq_lengths, test_choice_one_hot, c_test) = preprocess_for_rnn_context_split(test_df)
(val_xin, val_mask, val_seq_lengths, val_choice_one_hot, c_val) = preprocess_for_rnn_context_split(val_df)

print(train_xin)

# ---- save training arrays ----
np.save("data/train_xin_palminteri.npy", train_xin)
np.save("data/train_choice_one_hot_palminteri.npy", train_choice_one_hot)
np.save("data/c_train_palminteri.npy", c_train)

# ---- save validation arrays ----
np.save("data/val_xin_palminteri.npy", val_xin)
np.save("data/val_choice_one_hot_palminteri.npy", val_choice_one_hot)
np.save("data/c_val_palminteri.npy", c_val)

# ---- save test arrays ----
np.save("data/test_xin_palminteri.npy", test_xin)
np.save("data/test_choice_one_hot_palminteri.npy", test_choice_one_hot)
np.save("data/c_test_palminteri.npy", c_test)

print("✅ Saved all train, val, and test NumPy arrays to the 'data' folder.")

print(f"train xin shape: {train_xin.shape}")
print(f"train xin shape: {val_xin.shape}")

"""n_iter = 5

train_df = train_df.rename(columns={'sub_across_groups': 'session'})
test_df = test_df.rename(columns={'sub_across_groups': 'session'})

# Define model configurations
model_configs = {
    "Q": {"asymmetric_alpha": False, "forgetting_type": "none", "choice_trace": False},
    "Q+A": {"asymmetric_alpha": True, "forgetting_type": "none", "choice_trace": False},
    "Q+C": {"asymmetric_alpha": False, "forgetting_type": "none", "choice_trace": True},
    "Q+CA": {"asymmetric_alpha": True, "forgetting_type": "none", "choice_trace": True},
    "FQ": {"asymmetric_alpha": False, "forgetting_type": "fixed", "choice_trace": False},
    "FQ+C": {"asymmetric_alpha": False, "forgetting_type": "fixed", "choice_trace": True}
}

# Model fitting
model_eval_df, params_dict, p1_common_dict, p1_ML_dict, p1_MAP_dict = fit.fit_all_models(
    model_configs, train_df, test_df, n_iter, fit_ML = False 
)"""