"""
load_thalmann_s2.py — Preprocessing for the Thalmann two-task dataset *with*
the additional Session 2 (S2) trials.

For each task we now have S1 and S2 files. We pool both sessions, treating
S2 trials as additional independent blocks (block_structure resets the
decoder hidden state at block boundaries — which is exactly what we want
across sessions, since they were on different days).

Layout (62 blocks per participant, 200 max trials per block):
  block  0..29 → Task 0 (2-armed),  Session 1, 10 trials each
  block 30..59 → Task 0 (2-armed),  Session 2, 10 trials each
  block 60     → Task 1 (restless), Session 1, 200 trials
  block 61     → Task 1 (restless), Session 2, 200 trials

A participant is included if they have data across all three tasks
(2-armed, restless, horizon) in *at least one* session — the horizon
task is the held-out target for downstream regret prediction; the two
tasks above are the training data. Blocks belonging to a session a
participant didn't do are left fully padded (-100), and the cross-entropy
ignore_index handles them at training time.

Output arrays (saved to data_thalmann_s2/):
  xin[sub, block, trial, feat] — (N, 62, 200, BASE_IN_DIM=5)
  c[sub, block, trial]          — (N, 62, 200), -100 = padding
  choice_one_hot                — (N, 62, 200, A=4)
  task_ids_per_block.npy        — (62,) 0 or 1

The questionnaire and CFA-factor handling is downstream of this script;
this script only writes a per-subject metadata DataFrame with which
sessions are populated (`has_s1`, `has_s2`).
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

# ── Constants ─────────────────────────────────────────────────────────────────
DATA_DIR    = "data_thalmann_s2"
TASK_2ARMED = 0
TASK_RESTLESS = 1
N_TASKS     = 2
A           = 4
BASE_IN_DIM = 5
REWARD_MAX  = 100.0

AB_N_BLOCKS  = 30
AB_BLOCK_LEN = 10
RB_N_BLOCKS  = 1            # per session
RB_BLOCK_LEN = 200

N_SESSIONS   = 2
N_BLOCKS_PER_SESSION = AB_N_BLOCKS + RB_N_BLOCKS   # 31
# layout: AB_S1 (30) + AB_S2 (30) + RB_S1 (1) + RB_S2 (1) = 62
N_BLOCKS_TOTAL = AB_N_BLOCKS * N_SESSIONS + RB_N_BLOCKS * N_SESSIONS
MAX_TRIALS     = max(AB_BLOCK_LEN, RB_BLOCK_LEN)

# Block index ranges
AB_S1_START = 0
AB_S2_START = AB_N_BLOCKS
RB_S1_IDX   = AB_N_BLOCKS * 2
RB_S2_IDX   = RB_S1_IDX + 1

os.makedirs(DATA_DIR, exist_ok=True)

# ── Load raw CSVs ─────────────────────────────────────────────────────────────
df_ab_s1 = pd.read_csv("data/final2armedBanditSession1.csv")
df_ab_s2 = pd.read_csv("data/final2armedBanditSession2.csv")
df_rb_s1 = pd.read_csv("data/finalRestlessSession1.csv")
df_rb_s2 = pd.read_csv("data/finalRestlessSession2.csv")
df_h_s1  = pd.read_csv("data/finalHorizonSession1.csv")
df_h_s2  = pd.read_csv("data/finalHorizonSession2.csv")

# ── Participant inclusion ─────────────────────────────────────────────────────
# A participant is kept if (S1 or S2) trials exist for ALL three tasks.
ids_ab = set(df_ab_s1["ID"].unique()) | set(df_ab_s2["ID"].unique())
ids_rb = set(df_rb_s1["ID"].unique()) | set(df_rb_s2["ID"].unique())
ids_h  = set(df_h_s1["ID"].unique())  | set(df_h_s2["ID"].unique())
shared_ids  = sorted(ids_ab & ids_rb & ids_h)
n_subjects  = len(shared_ids)

ids_ab_s1 = set(df_ab_s1["ID"].unique())
ids_ab_s2 = set(df_ab_s2["ID"].unique())
ids_rb_s1 = set(df_rb_s1["ID"].unique())
ids_rb_s2 = set(df_rb_s2["ID"].unique())

has_s1 = np.array([(s in ids_ab_s1 and s in ids_rb_s1) for s in shared_ids])
has_s2 = np.array([(s in ids_ab_s2 and s in ids_rb_s2) for s in shared_ids])

print(f"n_subjects={n_subjects} (have all 3 tasks across both sessions)")
print(f"  with S1 data: {has_s1.sum()},  with S2 data: {has_s2.sum()},  "
      f"both: {(has_s1 & has_s2).sum()}")
print(f"  S2-only: {((~has_s1) & has_s2).sum()},  "
      f"S1-only: {(has_s1 & (~has_s2)).sum()}")
print(f"  total blocks per subject = {N_BLOCKS_TOTAL}  "
      f"(AB×2 sessions = {AB_N_BLOCKS*2}; RB×2 sessions = {RB_N_BLOCKS*2})")

# ── Allocate arrays ────────────────────────────────────────────────────────────
xin            = np.full((n_subjects, N_BLOCKS_TOTAL, MAX_TRIALS, BASE_IN_DIM),
                          -100.0, dtype=np.float32)
c              = np.full((n_subjects, N_BLOCKS_TOTAL, MAX_TRIALS),
                          -100.0, dtype=np.float32)
choice_one_hot = np.zeros((n_subjects, N_BLOCKS_TOTAL, MAX_TRIALS, A),
                           dtype=np.float32)


def _fill_2armed(df_ab, block_offset):
    """Write 2-armed bandit trials starting at block index `block_offset`.

    Subjects missing from `df_ab` are left padded.
    """
    if df_ab is None or len(df_ab) == 0:
        return
    unique_blocks = sorted(df_ab["block"].unique())
    for b_local, block_num in enumerate(unique_blocks):
        block_df = df_ab[df_ab["block"] == block_num]
        c_piv = (block_df.pivot(index="ID", columns="trial", values="chosen")
                          .reindex(shared_ids).to_numpy().astype(np.float32))
        r_piv = (block_df.pivot(index="ID", columns="trial", values="reward")
                          .reindex(shared_ids).to_numpy().astype(np.float32)
                  / REWARD_MAX)
        L = c_piv.shape[1]
        present = ~np.isnan(c_piv).all(axis=1)

        # Slot index in target arrays
        bi = block_offset + b_local

        # Default everything to padding (-100); only write rows for present subs
        # Trial 0: zero input
        xin[present, bi, 0, :] = 0.0
        # Trials 1..L-1: prev-choice one-hot + prev reward
        for arm in range(A):
            prev_choices = np.where(
                np.isnan(c_piv[:, :L-1]),
                -100,
                c_piv[:, :L-1]).astype(int)
            mask = (prev_choices == arm).astype(np.float32)
            mask[~present] = 0.0
            xin[:, bi, 1:L, arm] = mask
        r_in = np.where(np.isnan(r_piv[:, :L-1]), 0.0, r_piv[:, :L-1])
        r_in[~present] = 0.0
        xin[:, bi, 1:L, 4] = r_in

        # Targets — NaN entries become -100 (padding)
        target = np.where(np.isnan(c_piv), -100.0, c_piv).astype(np.float32)
        # For absent participants the whole row is -100 (correct)
        c[:, bi, :L] = target
        for arm in range(A):
            choice_one_hot[:, bi, :L, arm] = (
                (target == arm).astype(np.float32)
            )


def _fill_restless(df_rb, block_idx):
    """Write a single restless-bandit block into slot `block_idx`."""
    if df_rb is None or len(df_rb) == 0:
        return
    c_rb = (df_rb.pivot(index="ID", columns="trial", values="chosen")
                  .reindex(shared_ids).to_numpy().astype(np.float32))
    r_rb = (df_rb.pivot(index="ID", columns="trial", values="reward")
                  .reindex(shared_ids).to_numpy().astype(np.float32)
            / REWARD_MAX)
    L_rb = c_rb.shape[1]
    present = ~np.isnan(c_rb).all(axis=1)

    xin[present, block_idx, 0, :] = 0.0
    for arm in range(A):
        prev_choices = np.where(
            np.isnan(c_rb[:, :L_rb-1]),
            -100,
            c_rb[:, :L_rb-1]).astype(int)
        mask = (prev_choices == arm).astype(np.float32)
        mask[~present] = 0.0
        xin[:, block_idx, 1:L_rb, arm] = mask
    r_in = np.where(np.isnan(r_rb[:, :L_rb-1]), 0.0, r_rb[:, :L_rb-1])
    r_in[~present] = 0.0
    xin[:, block_idx, 1:L_rb, 4] = r_in

    target = np.where(np.isnan(c_rb), -100.0, c_rb).astype(np.float32)
    c[:, block_idx, :L_rb] = target
    for arm in range(A):
        choice_one_hot[:, block_idx, :L_rb, arm] = (
            (target == arm).astype(np.float32)
        )


# Fill all four sub-blocks
_fill_2armed(df_ab_s1, AB_S1_START)
_fill_2armed(df_ab_s2, AB_S2_START)
_fill_restless(df_rb_s1, RB_S1_IDX)
_fill_restless(df_rb_s2, RB_S2_IDX)

# ── Task IDs per block ─────────────────────────────────────────────────────────
task_ids_per_block = np.array(
    [TASK_2ARMED] * AB_N_BLOCKS * N_SESSIONS
    + [TASK_RESTLESS] * RB_N_BLOCKS * N_SESSIONS,
    dtype=np.int32,
)

# ── Per-block session id (0=S1, 1=S2) — saved for downstream analyses ────────
session_id_per_block = np.array(
    [0] * AB_N_BLOCKS + [1] * AB_N_BLOCKS + [0] + [1],
    dtype=np.int32,
)

print(f"xin shape:            {xin.shape}")
print(f"c shape:              {c.shape}, unique (non-pad): "
      f"{np.unique(c[c > -50]).astype(int)}")
print(f"task_ids_per_block:   {task_ids_per_block.tolist()}")
print(f"session_id_per_block: {session_id_per_block.tolist()}")

np.save(f"{DATA_DIR}/task_ids_per_block.npy", task_ids_per_block)
np.save(f"{DATA_DIR}/session_id_per_block.npy", session_id_per_block)

# ── Per-subject metadata ──────────────────────────────────────────────────────
df_meta = pd.DataFrame({
    "subid":  shared_ids,
    "has_s1": has_s1.astype(int),
    "has_s2": has_s2.astype(int),
})

# Save full arrays + the canonical subid order (downstream scripts use this
# instead of train/test/fold splits — we train on ALL participants).
np.save(f"{DATA_DIR}/xin_all.npy",            xin)
np.save(f"{DATA_DIR}/choice_one_hot_all.npy", choice_one_hot)
np.save(f"{DATA_DIR}/c_all.npy",              c)
df_meta.to_csv(f"{DATA_DIR}/df_all.csv", index=False)

print("Thalmann S1+S2 data generation complete.")
print(f"  BASE_IN_DIM = {BASE_IN_DIM}, A = {A}, N_BLOCKS = {N_BLOCKS_TOTAL}")
print(f"  saved to {DATA_DIR}/")
