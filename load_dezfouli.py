import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

DATA_DIR = "data_dezfouli"
os.makedirs(DATA_DIR, exist_ok=True)

# ── Load raw data ─────────────────────────────────────────────────────────────
df = pd.read_csv('data/for_plos.csv')

# Map choices to binary integers: R1→0, R2→1
df['choice_binary'] = df['key'].map({"R1": 0, "R2": 1}).astype(int)

# Integer-code participant IDs for consistent ordering
df["subid"] = df["ID"].astype("category").cat.codes

unique_ids    = sorted(df['subid'].unique())
unique_blocks = sorted(df['block'].unique())
n_subjects    = len(unique_ids)
n_blocks      = len(unique_blocks)

# ── Compute maximum block length ──────────────────────────────────────────────
block_lengths_all = []
for sub in unique_ids:
    sub_df = df[df['subid'] == sub]
    for block in unique_blocks:
        block_df = sub_df[sub_df['block'] == block]
        block_lengths_all.append(len(block_df))

max_block_len = int(np.max(block_lengths_all))
input_dim = 2   # [prev_choice, prev_reward]

print(f"n_subjects={n_subjects}, n_blocks={n_blocks}, "
      f"max_block_len={max_block_len}")

# ── Build arrays ──────────────────────────────────────────────────────────────
# Shape: (n_subjects, n_blocks, max_block_len, ...)
# Each block is stored separately so the model can reset hidden state between
# blocks (reward contingencies change across blocks → independent episodes).
# Padding value -100: cross-entropy ignores targets of -100 by default.
xin           = np.full((n_subjects, n_blocks, max_block_len, input_dim), -100.0, dtype=np.float32)
c             = np.full((n_subjects, n_blocks, max_block_len),            -100.0, dtype=np.float32)
choice_one_hot = np.zeros((n_subjects, n_blocks, max_block_len, 2),               dtype=np.float32)
diagnosis     = np.empty(n_subjects, dtype=object)

for sub_idx, sub in enumerate(unique_ids):
    sub_df = df[df['subid'] == sub]
    diagnosis[sub_idx] = sub_df['diag'].iloc[0]

    for b_idx, block in enumerate(unique_blocks):
        block_df = sub_df[sub_df['block'] == block].reset_index(drop=True)
        if len(block_df) == 0:
            continue

        L       = len(block_df)
        choices = block_df['choice_binary'].values   # int 0/1
        rewards = block_df['reward'].values          # float

        # First trial of the block: no previous history → input zeros
        xin[sub_idx, b_idx, 0, 0] = 0.0
        xin[sub_idx, b_idx, 0, 1] = 0.0

        # Remaining trials: previous trial's choice & reward as input
        if L > 1:
            xin[sub_idx, b_idx, 1:L, 0] = choices[:L - 1].astype(np.float32)
            xin[sub_idx, b_idx, 1:L, 1] = rewards[:L - 1].astype(np.float32)

        # Targets: actual choice at each trial (padding positions stay -100)
        c[sub_idx, b_idx, :L]    = choices.astype(np.float32)
        choice_one_hot[sub_idx, b_idx, :L, 0] = (choices == 0).astype(np.float32)
        choice_one_hot[sub_idx, b_idx, :L, 1] = (choices == 1).astype(np.float32)

print(f"xin shape: {xin.shape}")
print(f"c shape: {c.shape}, values: {np.unique(c)}")

# ── Build per-subject metadata DataFrame ─────────────────────────────────────
df_all = pd.DataFrame({
    'subid': unique_ids,
    'diag':  [diagnosis[i] for i in range(n_subjects)],
})
print(f"Diagnosis groups: {sorted(df_all['diag'].unique())}")

# ── Helper: subject index lookup ─────────────────────────────────────────────
sub_to_row = {s: i for i, s in enumerate(unique_ids)}


def _idx(subjects):
    return [sub_to_row[s] for s in sorted(subjects) if s in sub_to_row]


# ── Train / val / test split (no val → empty arrays) ─────────────────────────
def split_subjects(subject_ids, train_ratio=0.7, test_ratio=0.3, seed=42):
    rng = np.random.default_rng(seed)
    ids = np.array(sorted(subject_ids))
    rng.shuffle(ids)
    n_train = int(len(ids) * train_ratio)
    return set(ids[:n_train]), set(), set(ids[n_train:])


train_subjects, val_subjects, test_subjects = split_subjects(unique_ids)
tr_idx = _idx(train_subjects)
te_idx = _idx(test_subjects)

# DataFrames
df_all[df_all['subid'].isin(train_subjects)].reset_index(drop=True).to_csv(
    f"{DATA_DIR}/df_train.csv", index=False)
df_all[df_all['subid'].isin(test_subjects)].reset_index(drop=True).to_csv(
    f"{DATA_DIR}/df_test.csv", index=False)

# Empty val arrays (pipeline checks for existence; empty arrays signal no val data)
_empty_xin   = np.zeros((0, n_blocks, max_block_len, input_dim), dtype=np.float32)
_empty_oh    = np.zeros((0, n_blocks, max_block_len, 2),         dtype=np.float32)
_empty_c     = np.zeros((0, n_blocks, max_block_len),            dtype=np.float32)
np.save(f"{DATA_DIR}/xin_val.npy",          _empty_xin)
np.save(f"{DATA_DIR}/choice_one_hot_val.npy", _empty_oh)
np.save(f"{DATA_DIR}/c_val.npy",             _empty_c)

# Train / test arrays
np.save(f"{DATA_DIR}/xin_train.npy",            xin[tr_idx])
np.save(f"{DATA_DIR}/choice_one_hot_train.npy", choice_one_hot[tr_idx])
np.save(f"{DATA_DIR}/c_train.npy",              c[tr_idx])

np.save(f"{DATA_DIR}/xin_test.npy",             xin[te_idx])
np.save(f"{DATA_DIR}/choice_one_hot_test.npy",  choice_one_hot[te_idx])
np.save(f"{DATA_DIR}/c_test.npy",               c[te_idx])

print(f"Train: {len(tr_idx)} subjects, Test: {len(te_idx)} subjects")

# ── Outer 3-fold CV splits ────────────────────────────────────────────────────
N_OUTER_FOLDS   = 3
all_subids_arr  = np.array(sorted(unique_ids))
kf_outer        = KFold(n_splits=N_OUTER_FOLDS, shuffle=True, random_state=42)

for fold_idx, (fold_tr_idx, fold_te_idx) in enumerate(kf_outer.split(all_subids_arr)):
    fold_train_subjects = set(all_subids_arr[fold_tr_idx])
    fold_test_subjects  = set(all_subids_arr[fold_te_idx])

    fold_dir = f"{DATA_DIR}/fold{fold_idx}"
    os.makedirs(fold_dir, exist_ok=True)

    df_all[df_all['subid'].isin(fold_train_subjects)].reset_index(drop=True).to_csv(
        f"{fold_dir}/df_train.csv", index=False)
    df_all[df_all['subid'].isin(fold_test_subjects)].reset_index(drop=True).to_csv(
        f"{fold_dir}/df_test.csv", index=False)

    ftr = _idx(fold_train_subjects)
    fte = _idx(fold_test_subjects)

    np.save(f"{fold_dir}/xin_train.npy",            xin[ftr])
    np.save(f"{fold_dir}/xin_test.npy",             xin[fte])
    np.save(f"{fold_dir}/choice_one_hot_train.npy", choice_one_hot[ftr])
    np.save(f"{fold_dir}/choice_one_hot_test.npy",  choice_one_hot[fte])
    np.save(f"{fold_dir}/c_train.npy",              c[ftr])
    np.save(f"{fold_dir}/c_test.npy",               c[fte])

    print(f"Fold {fold_idx}: {len(ftr)} train, {len(fte)} test → {fold_dir}/")

print("Dezfouli data generation complete.")
