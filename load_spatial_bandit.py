"""
Load and preprocess the spatial bandit dataset (Schulz & Wu).

Dataset: behavioralData_Schulz_Wu.csv
- `chosen`: the arm that was chosen (flat grid index in the original study)
- `z`:      reward received at the chosen arm
- `previous_reward`: z_{t-1}, already lagged by the dataset (NA at trial 0 of each round)
- `x`, `y`: spatial coordinates of the chosen arm
- `round`:  block index — each round is an independent episode with a new reward function

Block structure
---------------
Each (participant, round) pair is treated as an independent sequence.
The train/val/test split is by *participant* so all rounds of one participant
stay in the same split.

RNN input at time t within a round (predicting the arm chosen at t):
    feat 0: arm_index_{t-1} / (A-1)   — previous arm, normalised to [0,1]
    feat 1: x_{t-1} / x_max            — previous x coordinate
    feat 2: y_{t-1} / y_max            — previous y coordinate
    feat 3: previous_reward_t / norm   — z_{t-1} from `previous_reward` column
All inputs are 0 at t=0 (first trial of each round has no prior within-round history).

Target: contiguous arm_index for chosen_t  (0..A-1)
Age decoding target: age_months (continuous, one value per sequence)

Output files (all in data_spatial_bandit/):
    xin_train/val/test.npy             - RNN inputs  (n_seq, T_max, 4)
    choice_one_hot_train/val/test.npy  - one-hot     (n_seq, T_max, A)
    c_train/val/test.npy               - int choices (n_seq, T_max)
    df_train/val/test.csv              - one row per sequence (subid, round, age_months, …)
    arm_map.json                       - contiguous_idx -> {original_chosen, x, y}
    age_months_train/val/test.npy      - age_months per sequence (n_seq,)
"""

import os
import json
import numpy as np
import pandas as pd

RAW_CSV = "data/behavioralData_Schulz_Wu.csv"
OUT_DIR = "data_spatial_bandit"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load raw data
# ---------------------------------------------------------------------------
df = pd.read_csv(RAW_CSV)

print(f"Raw data shape:         {df.shape}")
print(f"Unique participants:    {df['id'].nunique()}")
print(f"Unique rounds:          {df['round'].nunique()}")
print(f"Unique chosen arms:     {df['chosen'].nunique()}")
print(f"chosen range:           {df['chosen'].min()} – {df['chosen'].max()}")
print(f"age_months range:       {df['age_months'].min():.1f} – {df['age_months'].max():.1f}")
print(f"agegroup counts:\n{df.drop_duplicates('id')['agegroup'].value_counts().to_string()}")
print()

# ---------------------------------------------------------------------------
# Recode participant IDs to consecutive integers
# ---------------------------------------------------------------------------
df["subid_original"] = df["id"].copy()
df["subid"] = df["id"].astype("category").cat.codes  # 0-indexed

# ---------------------------------------------------------------------------
# Arm index map: unique `chosen` values → contiguous 0..A-1
# ---------------------------------------------------------------------------
unique_chosen = sorted(df["chosen"].unique())
A = len(unique_chosen)
chosen_to_idx = {c: i for i, c in enumerate(unique_chosen)}

chosen_to_xy = (df[["chosen", "x", "y"]].drop_duplicates("chosen")
                  .set_index("chosen").to_dict("index"))
arm_map = {
    int(chosen_to_idx[c]): {
        "original_chosen": int(c),
        "x": int(chosen_to_xy[c]["x"]),
        "y": int(chosen_to_xy[c]["y"])
    }
    for c in unique_chosen
}
with open(os.path.join(OUT_DIR, "arm_map.json"), "w") as f:
    json.dump(arm_map, f, indent=2)

df["arm_index"] = df["chosen"].map(chosen_to_idx)
print(f"Number of unique arms A = {A}")

# ---------------------------------------------------------------------------
# Normalisation constants (fit on full data)
# ---------------------------------------------------------------------------
x_max = float(df["x"].max())
y_max = float(df["y"].max())
reward_min = float(min(df["z"].min(), df["previous_reward"].dropna().min()))
reward_max = float(max(df["z"].max(), df["previous_reward"].dropna().max()))
reward_range = reward_max - reward_min if reward_max != reward_min else 1.0

norm_constants = {
    "x_max": x_max, "y_max": y_max, "A": A,
    "reward_min": reward_min, "reward_max": reward_max, "reward_range": reward_range
}
with open(os.path.join(OUT_DIR, "norm_constants.json"), "w") as f:
    json.dump(norm_constants, f, indent=2)

print(f"Normalisation: x_max={x_max}, y_max={y_max}, "
      f"reward=[{reward_min:.1f}, {reward_max:.1f}]")

# ---------------------------------------------------------------------------
# Participant-level demographics (one row per participant)
# ---------------------------------------------------------------------------
df_subj = (df.sort_values("subid")
             .drop_duplicates(subset="subid")
             [["subid", "subid_original", "id", "age_months", "age_years",
               "gender", "agegroup", "experiment", "condition"]]
             .reset_index(drop=True))

# ---------------------------------------------------------------------------
# Build sequence index: one entry per (participant, round)
# Split is by *participant* so all rounds of a participant stay together.
# ---------------------------------------------------------------------------
seq_meta = (df[["subid", "round", "age_months", "age_years",
                "gender", "agegroup", "experiment", "condition"]]
              .drop_duplicates(subset=["subid", "round"])
              .sort_values(["subid", "round"])
              .reset_index(drop=True))

print(f"\nTotal sequences (participant × round): {len(seq_meta)}")
print(f"Rounds per participant (median): "
      f"{seq_meta.groupby('subid').size().median():.0f}")

# ---------------------------------------------------------------------------
# Train / val / test split by participant  (60 / 10 / 30 %)
# ---------------------------------------------------------------------------
def split_subjects(sub_ids, train_ratio=0.6, val_ratio=0.1, seed=42):
    rng = np.random.default_rng(seed)
    sub_ids = np.array(sub_ids)
    rng.shuffle(sub_ids)
    n = len(sub_ids)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    return (set(sub_ids[:n_train]),
            set(sub_ids[n_train:n_train + n_val]),
            set(sub_ids[n_train + n_val:]))

all_subids     = df_subj["subid"].tolist()
train_subs, val_subs, test_subs = split_subjects(all_subids)

seq_train = seq_meta[seq_meta["subid"].isin(train_subs)].reset_index(drop=True)
seq_val   = seq_meta[seq_meta["subid"].isin(val_subs)].reset_index(drop=True)
seq_test  = seq_meta[seq_meta["subid"].isin(test_subs)].reset_index(drop=True)

print(f"Split: train={len(seq_train)} seqs ({len(train_subs)} participants), "
      f"val={len(seq_val)} seqs ({len(val_subs)} participants), "
      f"test={len(seq_test)} seqs ({len(test_subs)} participants)")

# Save sequence-level DataFrames (one row per sequence = one row per (participant, round))
seq_train.to_csv(os.path.join(OUT_DIR, "df_train.csv"), index=False)
seq_val.to_csv(  os.path.join(OUT_DIR, "df_val.csv"),   index=False)
seq_test.to_csv( os.path.join(OUT_DIR, "df_test.csv"),  index=False)
seq_meta.to_csv( os.path.join(OUT_DIR, "df_all_sequences.csv"), index=False)
df_subj.to_csv(  os.path.join(OUT_DIR, "df_all_participants.csv"), index=False)


# ---------------------------------------------------------------------------
# Build RNN arrays — one sequence per (participant, round)
# ---------------------------------------------------------------------------
def build_rnn_arrays(df, seq_index, A, x_max, y_max, reward_min, reward_range):
    """
    Parameters
    ----------
    seq_index : DataFrame with columns [subid, round]
        Ordered list of (participant, round) pairs to include.

    Returns
    -------
    xin          : (n_seq, T_max, 4)
    choice_one_hot : (n_seq, T_max, A)
    c_array      : (n_seq, T_max) int64
    seq_lengths  : list[int]
    """
    n_seq = len(seq_index)

    # Compute max sequence length across all included (participant, round) pairs
    lengths = []
    for _, row in seq_index.iterrows():
        block = df[(df["subid"] == row["subid"]) & (df["round"] == row["round"])]
        lengths.append(len(block))
    T_max = max(lengths)

    xin            = np.zeros((n_seq, T_max, 4), dtype=np.float32)
    c_array        = np.zeros((n_seq, T_max),    dtype=np.int64)
    choice_one_hot = np.zeros((n_seq, T_max, A), dtype=np.float32)

    for i, (_, row) in enumerate(seq_index.iterrows()):
        block = (df[(df["subid"] == row["subid"]) & (df["round"] == row["round"])]
                   .sort_values("trial").reset_index(drop=True))
        T = len(block)

        arm_seq      = block["arm_index"].to_numpy(dtype=np.int64)
        x_seq        = block["x"].to_numpy(dtype=float)
        y_seq        = block["y"].to_numpy(dtype=float)
        prev_rew_seq = block["previous_reward"].to_numpy(dtype=float)  # z_{t-1}; NaN at t=0

        # ---- Inputs: at time t, use previous trial within this round ----
        # t=0 stays all-zero (no prior context at round start)
        if T > 1:
            xin[i, 1:T, 0] = arm_seq[:-1] / float(A - 1)        # previous arm (normed)
            xin[i, 1:T, 1] = x_seq[:-1] / x_max                  # previous x
            xin[i, 1:T, 2] = y_seq[:-1] / y_max                  # previous y
            # previous_reward at t = z_{t-1}; dataset already has NA at trial 0
            prev_rew_norm = np.where(
                np.isnan(prev_rew_seq[1:]),
                0.0,
                (prev_rew_seq[1:] - reward_min) / reward_range
            )
            xin[i, 1:T, 3] = prev_rew_norm

        # ---- Targets ----
        c_array[i, :T] = arm_seq
        for t in range(T):
            choice_one_hot[i, t, arm_seq[t]] = 1.0

    return xin, choice_one_hot, c_array, lengths


print("\nBuilding RNN arrays...")
xin_train, coh_train, c_train, sl_train = build_rnn_arrays(
    df, seq_train, A, x_max, y_max, reward_min, reward_range)
xin_val,   coh_val,   c_val,   sl_val   = build_rnn_arrays(
    df, seq_val,   A, x_max, y_max, reward_min, reward_range)
xin_test,  coh_test,  c_test,  sl_test  = build_rnn_arrays(
    df, seq_test,  A, x_max, y_max, reward_min, reward_range)

print(f"xin_train shape: {xin_train.shape}  (n_seq, T_max, 4)")
print(f"xin_val shape:   {xin_val.shape}")
print(f"xin_test shape:  {xin_test.shape}")

# ---------------------------------------------------------------------------
# age_months arrays (one per sequence, for age decoding)
# ---------------------------------------------------------------------------
age_train = seq_train["age_months"].to_numpy(dtype=np.float32)
age_val   = seq_val["age_months"].to_numpy(dtype=np.float32)
age_test  = seq_test["age_months"].to_numpy(dtype=np.float32)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
np.save(os.path.join(OUT_DIR, "xin_train.npy"),             xin_train)
np.save(os.path.join(OUT_DIR, "choice_one_hot_train.npy"),  coh_train)
np.save(os.path.join(OUT_DIR, "c_train.npy"),               c_train)
np.save(os.path.join(OUT_DIR, "age_months_train.npy"),      age_train)

np.save(os.path.join(OUT_DIR, "xin_val.npy"),               xin_val)
np.save(os.path.join(OUT_DIR, "choice_one_hot_val.npy"),    coh_val)
np.save(os.path.join(OUT_DIR, "c_val.npy"),                 c_val)
np.save(os.path.join(OUT_DIR, "age_months_val.npy"),        age_val)

np.save(os.path.join(OUT_DIR, "xin_test.npy"),              xin_test)
np.save(os.path.join(OUT_DIR, "choice_one_hot_test.npy"),   coh_test)
np.save(os.path.join(OUT_DIR, "c_test.npy"),                c_test)
np.save(os.path.join(OUT_DIR, "age_months_test.npy"),       age_test)

np.save(os.path.join(OUT_DIR, "seq_lengths_train.npy"), np.array(sl_train))
np.save(os.path.join(OUT_DIR, "seq_lengths_val.npy"),   np.array(sl_val))
np.save(os.path.join(OUT_DIR, "seq_lengths_test.npy"),  np.array(sl_test))

print("\nAll arrays saved to", OUT_DIR)
print(f"  A (number of arms):     {A}")
print(f"  Train sequences:        {len(seq_train)}")
print(f"  Val sequences:          {len(seq_val)}")
print(f"  Test sequences:         {len(seq_test)}")
print(f"  age_months range (all): "
      f"{df_subj['age_months'].min():.1f} – {df_subj['age_months'].max():.1f}")
print("\nDone.")
