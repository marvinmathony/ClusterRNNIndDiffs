"""
load_thalmann.py — Preprocessing for the Thalmann two-task dataset.

Combines two bandit tasks performed by the same 236 participants:
  Task 0  (2-armed bandit):  30 blocks × 10 trials, binary choices (0/1)
  Task 1  (restless bandit):  1 block  × 200 trials, 4-way choices (0/1/2/3)

Both tasks are stored in a unified flat block structure:
  blocks 0-29  → Task 0 (2-armed), 10 valid trials each
  block  30    → Task 1 (restless), 200 valid trials

Output arrays (saved to data_thalmann/):
  xin[sub, block, trial, feat] — input features, shape (N, 31, 200, BASE_IN_DIM=5)
      feat 0-3: previous-trial choice one-hot (4 dims, unified action space)
      feat  4:  previous-trial reward normalised to [0, 1]
  c[sub, block, trial]          — integer choice label,  shape (N, 31, 200)
  choice_one_hot[sub, b, t, a]  — one-hot choice target, shape (N, 31, 200, A=4)
  Padding value: -100 (used by cross-entropy ignore_index)

task_ids_per_block.npy — shape (31,), values 0 or 1, maps each block to its task.

During training the model is told which task each block belongs to via a learned
task-embedding (nn.Embedding(2, task_emb_dim)) that is concatenated to the decoder
input at every timestep — analogous to the per-participant z embedding used for
individual differences.  The effective decoder in_dim = BASE_IN_DIM + task_emb_dim.
The IDRNN encoder sees only raw features (BASE_IN_DIM = 5) so that individual
differences are inferred from raw behaviour, task-agnostically.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

# ── Constants ─────────────────────────────────────────────────────────────────
DATA_DIR    = "data_thalmann"
TASK_2ARMED = 0
TASK_RESTLESS = 1
N_TASKS     = 2
A           = 4          # unified action space (restless=4 arms, 2-armed padded to 4)
BASE_IN_DIM = 5          # [prev_choice_oh(4), prev_reward_normed(1)]
REWARD_MAX  = 100.0      # rewards are in [1, 95]; normalise to [0, 1]

AB_N_BLOCKS  = 30
AB_BLOCK_LEN = 10
RB_N_BLOCKS  = 1
RB_BLOCK_LEN = 200

N_BLOCKS_TOTAL = AB_N_BLOCKS + RB_N_BLOCKS   # 31
MAX_TRIALS     = max(AB_BLOCK_LEN, RB_BLOCK_LEN)  # 200

os.makedirs(DATA_DIR, exist_ok=True)

# ── Load raw CSVs ──────────────────────────────────────────────────────────────
df_ab = pd.read_csv('data/final2armedBanditSession1.csv')   # 2-armed bandit
df_rb = pd.read_csv('data/finalRestlessSession1.csv')       # restless bandit

# ── Participant list: keep only subjects present in both tasks ─────────────────
shared_ids = sorted(set(df_ab['ID'].unique()) & set(df_rb['ID'].unique()))
n_subjects  = len(shared_ids)
id_to_idx   = {pid: i for i, pid in enumerate(shared_ids)}

print(f"n_subjects={n_subjects} (both tasks), n_blocks_total={N_BLOCKS_TOTAL}, "
      f"max_trials={MAX_TRIALS}, BASE_IN_DIM={BASE_IN_DIM}, A={A}")
assert n_subjects == 236, f"Expected 236 shared participants, found {n_subjects}"

# ── Allocate arrays (padding = -100) ──────────────────────────────────────────
xin            = np.full((n_subjects, N_BLOCKS_TOTAL, MAX_TRIALS, BASE_IN_DIM),
                          -100.0, dtype=np.float32)
c              = np.full((n_subjects, N_BLOCKS_TOTAL, MAX_TRIALS),
                          -100.0, dtype=np.float32)
choice_one_hot = np.zeros((n_subjects, N_BLOCKS_TOTAL, MAX_TRIALS, A),
                           dtype=np.float32)

# ── Fill Task 0 – 2-armed bandit (blocks 0..29) ───────────────────────────────
unique_ab_blocks = sorted(df_ab['block'].unique())   # [1 .. 30]

for b_local, block_num in enumerate(unique_ab_blocks):
    block_df = df_ab[df_ab['block'] == block_num]

    # Pivot: shape (n_subjects, AB_BLOCK_LEN) — subjects ordered by shared_ids
    c_piv = (block_df.pivot(index='ID', columns='trial', values='chosen')
                     .reindex(shared_ids).to_numpy().astype(np.float32))
    r_piv = (block_df.pivot(index='ID', columns='trial', values='reward')
                     .reindex(shared_ids).to_numpy().astype(np.float32) / REWARD_MAX)

    L = c_piv.shape[1]   # == AB_BLOCK_LEN == 10

    # Trial 0: zero input (no previous trial in this block)
    xin[:, b_local, 0, :] = 0.0

    # Trials 1..L-1: one-hot of previous choice + previous reward
    prev_choices = c_piv[:, :L-1].astype(int)   # (N, L-1)
    for arm in range(A):
        xin[:, b_local, 1:L, arm] = (prev_choices == arm).astype(np.float32)
    xin[:, b_local, 1:L, 4] = r_piv[:, :L-1]

    # Targets
    c[:, b_local, :L] = c_piv
    for arm in range(A):
        choice_one_hot[:, b_local, :L, arm] = (c_piv == arm).astype(np.float32)

# ── Fill Task 1 – restless bandit (block 30) ──────────────────────────────────
RB_BLOCK_IDX = AB_N_BLOCKS   # == 30

# Restless bandit: trial column 1..200, no block column
c_rb = (df_rb.pivot(index='ID', columns='trial', values='chosen')
              .reindex(shared_ids).to_numpy().astype(np.float32))
r_rb = (df_rb.pivot(index='ID', columns='trial', values='reward')
              .reindex(shared_ids).to_numpy().astype(np.float32) / REWARD_MAX)

L_rb = c_rb.shape[1]   # == RB_BLOCK_LEN == 200

xin[:, RB_BLOCK_IDX, 0, :] = 0.0

prev_choices_rb = c_rb[:, :L_rb-1].astype(int)
for arm in range(A):
    xin[:, RB_BLOCK_IDX, 1:L_rb, arm] = (prev_choices_rb == arm).astype(np.float32)
xin[:, RB_BLOCK_IDX, 1:L_rb, 4] = r_rb[:, :L_rb-1]

c[:, RB_BLOCK_IDX, :L_rb] = c_rb
for arm in range(A):
    choice_one_hot[:, RB_BLOCK_IDX, :L_rb, arm] = (c_rb == arm).astype(np.float32)

# ── Task IDs per block ─────────────────────────────────────────────────────────
task_ids_per_block = np.array(
    [TASK_2ARMED] * AB_N_BLOCKS + [TASK_RESTLESS] * RB_N_BLOCKS,
    dtype=np.int32
)

print(f"xin shape:            {xin.shape}")
print(f"c shape:              {c.shape}, unique (non-pad): {np.unique(c[c > -50]).astype(int)}")
print(f"choice_one_hot shape: {choice_one_hot.shape}")
print(f"task_ids_per_block:   {task_ids_per_block}")

np.save(f"{DATA_DIR}/task_ids_per_block.npy", task_ids_per_block)

# ── Per-subject metadata DataFrame ────────────────────────────────────────────
df_meta = pd.DataFrame({'subid': shared_ids})

# ── Helper: train / test split ────────────────────────────────────────────────
def split_subjects(subject_ids, train_ratio=0.7, seed=42):
    rng = np.random.default_rng(seed)
    ids = np.array(sorted(subject_ids))
    rng.shuffle(ids)
    n_train = int(len(ids) * train_ratio)
    return set(ids[:n_train]), set(), set(ids[n_train:])


sub_to_row = {s: i for i, s in enumerate(shared_ids)}


def _idx(subjects):
    return [sub_to_row[s] for s in sorted(subjects) if s in sub_to_row]


train_subjects, val_subjects, test_subjects = split_subjects(shared_ids)
tr_idx = _idx(train_subjects)
te_idx = _idx(test_subjects)

# ── DataFrames ────────────────────────────────────────────────────────────────
df_meta[df_meta['subid'].isin(train_subjects)].reset_index(drop=True).to_csv(
    f"{DATA_DIR}/df_train.csv", index=False)
df_meta[df_meta['subid'].isin(test_subjects)].reset_index(drop=True).to_csv(
    f"{DATA_DIR}/df_test.csv", index=False)

# Empty val arrays (pipeline expects them; 0 subjects signals no validation data)
_empty_xin   = np.zeros((0, N_BLOCKS_TOTAL, MAX_TRIALS, BASE_IN_DIM), dtype=np.float32)
_empty_oh    = np.zeros((0, N_BLOCKS_TOTAL, MAX_TRIALS, A),           dtype=np.float32)
_empty_c     = np.zeros((0, N_BLOCKS_TOTAL, MAX_TRIALS),              dtype=np.float32)
np.save(f"{DATA_DIR}/xin_val.npy",              _empty_xin)
np.save(f"{DATA_DIR}/choice_one_hot_val.npy",   _empty_oh)
np.save(f"{DATA_DIR}/c_val.npy",                _empty_c)

# Train / test splits
np.save(f"{DATA_DIR}/xin_train.npy",            xin[tr_idx])
np.save(f"{DATA_DIR}/choice_one_hot_train.npy", choice_one_hot[tr_idx])
np.save(f"{DATA_DIR}/c_train.npy",              c[tr_idx])

np.save(f"{DATA_DIR}/xin_test.npy",             xin[te_idx])
np.save(f"{DATA_DIR}/choice_one_hot_test.npy",  choice_one_hot[te_idx])
np.save(f"{DATA_DIR}/c_test.npy",               c[te_idx])

print(f"Train: {len(tr_idx)} subjects, Test: {len(te_idx)} subjects")

# ── Outer 3-fold CV splits ─────────────────────────────────────────────────────
N_OUTER_FOLDS  = 3
all_subids_arr = np.array(sorted(shared_ids))
kf_outer       = KFold(n_splits=N_OUTER_FOLDS, shuffle=True, random_state=42)

for fold_idx, (fold_tr_idx_kf, fold_te_idx_kf) in enumerate(kf_outer.split(all_subids_arr)):
    fold_train_subjects = set(all_subids_arr[fold_tr_idx_kf])
    fold_test_subjects  = set(all_subids_arr[fold_te_idx_kf])

    fold_dir = f"{DATA_DIR}/fold{fold_idx}"
    os.makedirs(fold_dir, exist_ok=True)

    df_meta[df_meta['subid'].isin(fold_train_subjects)].reset_index(drop=True).to_csv(
        f"{fold_dir}/df_train.csv", index=False)
    df_meta[df_meta['subid'].isin(fold_test_subjects)].reset_index(drop=True).to_csv(
        f"{fold_dir}/df_test.csv", index=False)

    # Empty val for fold (inner CV handled inside training)
    np.save(f"{fold_dir}/xin_val.npy",              _empty_xin)
    np.save(f"{fold_dir}/choice_one_hot_val.npy",   _empty_oh)
    np.save(f"{fold_dir}/c_val.npy",                _empty_c)

    ftr = _idx(fold_train_subjects)
    fte = _idx(fold_test_subjects)

    np.save(f"{fold_dir}/xin_train.npy",            xin[ftr])
    np.save(f"{fold_dir}/xin_test.npy",             xin[fte])
    np.save(f"{fold_dir}/choice_one_hot_train.npy", choice_one_hot[ftr])
    np.save(f"{fold_dir}/choice_one_hot_test.npy",  choice_one_hot[fte])
    np.save(f"{fold_dir}/c_train.npy",              c[ftr])
    np.save(f"{fold_dir}/c_test.npy",               c[fte])

    print(f"Fold {fold_idx}: {len(ftr)} train, {len(fte)} test → {fold_dir}/")

print("Thalmann data generation complete.")
print(f"  BASE_IN_DIM = {BASE_IN_DIM}  (raw features, no task code)")
print(f"  N_TASKS     = {N_TASKS}")
print(f"  N_BLOCKS    = {N_BLOCKS_TOTAL}  (task0: blocks 0-{AB_N_BLOCKS-1}, task1: block {RB_BLOCK_IDX})")
print(f"  task_ids_per_block saved to {DATA_DIR}/task_ids_per_block.npy")
print(f"  When training: set --task_emb_dim > 0 to enable learned task embedding.")
print(f"  Effective decoder in_dim = BASE_IN_DIM + task_emb_dim = {BASE_IN_DIM} + task_emb_dim")
