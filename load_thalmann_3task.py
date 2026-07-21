"""load_thalmann_3task.py — build a 3-task full-cohort dataset (S1) for a
SEPARATE exploratory model: does adding the horizon task improve the
representational readout?  The 2-task model (data_thalmann) keeps the horizon
HELD-OUT for the generative-transfer panel; this dataset puts it IN training.

Tasks (S1):
  task 0  2-armed bandit   30 blocks x 10 trials   (final2armedBanditSession1)
  task 1  restless bandit   1 block  x 200 trials  (finalRestlessSession1)
  task 2  horizon task     80 games x up to 10     (finalHorizonSession1)
            - 4 forced "instructed" trials per game -> kept as INPUT history but
              MASKED from targets (c=-100); only the free choices are predicted.
            - Horizon=5 games have 1 free choice (trial 5); Horizon=10 have 6
              (trials 5-10).  h5 games pad trials 6-10 (NaN -> -100).

Layout (same 5-feature encoding as load_thalmann): xin=[prev_choice_oh(4),
prev_reward/100], c=int choice (pad -100), choice_one_hot, MAX_TRIALS=200,
n_blocks = 30+1+80 = 111, task_ids_per_block = [0]*30 + [1] + [2]*80.

Writes data_thalmann_3task_full/ as a full cohort (train = all subjects;
test = copy; empty val) for run_thalmann_canonical_full.py --data_dir.
"""
import os
import numpy as np
import pandas as pd

DST = "data_thalmann_3task_full"; os.makedirs(DST, exist_ok=True)
A = 4; BASE_IN_DIM = 5; REWARD_MAX = 100.0
AB_N, AB_LEN = 30, 10
RB_LEN = 200
HZ_N, HZ_LEN = 80, 10          # 80 games, up to 10 trials
HZ_N_FORCED = 4                # first 4 trials of each horizon game are forced
MAX_TRIALS = 200
N_BLOCKS = AB_N + 1 + HZ_N     # 111

df_ab = pd.read_csv("data/final2armedBanditSession1.csv")
df_rb = pd.read_csv("data/finalRestlessSession1.csv")
df_hz = pd.read_csv("data/finalHorizonSession1.csv")
shared = sorted(set(df_ab.ID) & set(df_rb.ID) & set(df_hz.ID))
N = len(shared); print(f"n_subjects (all 3 tasks) = {N}; n_blocks = {N_BLOCKS}")
idx = {pid: i for i, pid in enumerate(shared)}

xin = np.full((N, N_BLOCKS, MAX_TRIALS, BASE_IN_DIM), -100.0, np.float32)
c = np.full((N, N_BLOCKS, MAX_TRIALS), -100.0, np.float32)
oh = np.zeros((N, N_BLOCKS, MAX_TRIALS, A), np.float32)


def _fill(b, c_piv, r_piv, mask_forced=0):
    """Fill block b from pivots (N, L). mask_forced: #leading trials excluded
    from targets (kept as input)."""
    L = c_piv.shape[1]
    xin[:, b, 0, :] = 0.0
    prev = c_piv[:, :L-1]
    for arm in range(A):
        xin[:, b, 1:L, arm] = (prev == arm).astype(np.float32)
    xin[:, b, 1:L, 4] = np.nan_to_num(r_piv[:, :L-1], nan=0.0)
    # targets (mask NaN and forced leading trials)
    ct = c_piv.copy()
    valid = ~np.isnan(ct)
    if mask_forced:
        valid[:, :mask_forced] = False
    for t in range(L):
        col = ct[:, t]
        m = valid[:, t]
        c[m, b, t] = col[m]
        for arm in range(A):
            oh[m, b, t, arm] = (col[m] == arm).astype(np.float32)


# task 0 (2-armed): blocks 0..29
for bl, bn in enumerate(sorted(df_ab.block.unique())):
    g = df_ab[df_ab.block == bn]
    cp = g.pivot(index="ID", columns="trial", values="chosen").reindex(shared).to_numpy().astype(np.float32)
    rp = g.pivot(index="ID", columns="trial", values="reward").reindex(shared).to_numpy().astype(np.float32) / REWARD_MAX
    _fill(bl, cp, rp)

# task 1 (restless): block 30
cp = df_rb.pivot(index="ID", columns="trial", values="chosen").reindex(shared).to_numpy().astype(np.float32)
rp = df_rb.pivot(index="ID", columns="trial", values="reward").reindex(shared).to_numpy().astype(np.float32) / REWARD_MAX
_fill(AB_N, cp, rp)

# task 2 (horizon): blocks 31..110  (mask first 4 forced trials from targets)
for bl, bn in enumerate(sorted(df_hz.block.unique())):
    g = df_hz[df_hz.block == bn]
    cp = g.pivot(index="ID", columns="trial", values="chosen").reindex(shared).to_numpy().astype(np.float32)
    rp = g.pivot(index="ID", columns="trial", values="reward").reindex(shared).to_numpy().astype(np.float32) / REWARD_MAX
    _fill(AB_N + 1 + bl, cp, rp, mask_forced=HZ_N_FORCED)

task_ids = np.array([0]*AB_N + [1] + [2]*HZ_N, dtype=np.int32)
n_free = int((c[:, AB_N+1:, :] > -50).sum())
print(f"horizon free-choice targets: {n_free}  (forced masked); "
      f"c unique: {np.unique(c[c>-50]).astype(int)}")

np.save(f"{DST}/xin_train.npy", xin);  np.save(f"{DST}/xin_test.npy", xin)
np.save(f"{DST}/c_train.npy", c);      np.save(f"{DST}/c_test.npy", c)
np.save(f"{DST}/choice_one_hot_train.npy", oh); np.save(f"{DST}/choice_one_hot_test.npy", oh)
pd.DataFrame({"subid": shared}).to_csv(f"{DST}/df_train.csv", index=False)
pd.DataFrame({"subid": shared}).to_csv(f"{DST}/df_test.csv", index=False)
np.save(f"{DST}/xin_val.npy", np.zeros((0, N_BLOCKS, MAX_TRIALS, BASE_IN_DIM), np.float32))
np.save(f"{DST}/c_val.npy", np.zeros((0, N_BLOCKS, MAX_TRIALS), np.float32))
np.save(f"{DST}/choice_one_hot_val.npy", np.zeros((0, N_BLOCKS, MAX_TRIALS, A), np.float32))
np.save(f"{DST}/task_ids_per_block.npy", task_ids)
np.save(f"{DST}/subids_full.npy", np.array(shared, dtype=int))
print(f"Wrote {DST}/  xin{xin.shape}  task_ids: 0x{AB_N},1x1,2x{HZ_N}")
