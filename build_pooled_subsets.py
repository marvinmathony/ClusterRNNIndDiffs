#!/usr/bin/env python3
"""Build per-fold pooled fold dirs for ALL 7 non-empty task subsets, for the
data-scaling ('more tasks -> better IDRNN predictive accuracy') analysis.

Reuses the data_thalmann_s2 cohort (238 subjects — already requires all 3 tasks)
and its 3-fold subject split, appending the horizon task (task 2) as 160 blocks
(80 games x S1 + 80 x S2; first 4 forced trials masked from targets, NaN/absent
-> padding).  Each subset gets contiguous task ids (0..k-1) so n_tasks is correct;
the data-scaling trains with --block_weight_mode uniform so every task contributes.

Block layout of the full 3-task array (222 blocks):
  0..59   task 0  (2-armed, S1 30 + S2 30)
  60..61  task 1  (restless, S1 + S2)
  62..221 task 2  (horizon, S1 80 + S2 80)

Subsets -> data_sub_{name}/fold{F}/ :  t0 t1 t2 t01 t02 t12 t012
Each fold dir: {xin,c,choice_one_hot}_{train,test}.npy, *_val.npy (empty),
task_ids_per_block.npy, df_train/test.csv.
"""
import os
import numpy as np
import pandas as pd

A = 4; REWARD_MAX = 100.0; MAX_TRIALS = 200; BASE_IN_DIM = 5; HZ_N_FORCED = 4
S2 = "data_thalmann_s2"

df_all = pd.read_csv(f"{S2}/df_all.csv")
subids = df_all["subid"].values
N = len(subids)
row_of = {int(s): i for i, s in enumerate(subids)}

xin_2t = np.load(f"{S2}/xin_all.npy")           # (N,62,200,5)
c_2t   = np.load(f"{S2}/c_all.npy")             # (N,62,200)
tids2  = np.load(f"{S2}/task_ids_per_block.npy")
T0 = np.where(tids2 == 0)[0]                     # 60 two-armed blocks
T1 = np.where(tids2 == 1)[0]                     # 2 restless blocks
print(f"cohort {N}; task0 blocks={len(T0)} task1 blocks={len(T1)}")


def fill_horizon(df, b0, xin_h, c_h):
    """Fill horizon games from df starting at block b0 (present-masked)."""
    blocks = sorted(df["block"].unique())
    for j, bn in enumerate(blocks):
        g = df[df["block"] == bn]
        cp = g.pivot(index="ID", columns="trial", values="chosen").reindex(subids).to_numpy().astype(np.float32)
        rp = g.pivot(index="ID", columns="trial", values="reward").reindex(subids).to_numpy().astype(np.float32) / REWARD_MAX
        L = cp.shape[1]; bi = b0 + j
        present = ~np.isnan(cp).all(axis=1)
        xin_h[present, bi, 0, :] = 0.0
        prev = cp[:, :L-1]
        for arm in range(A):
            m = (np.where(np.isnan(prev), -100, prev).astype(int) == arm).astype(np.float32)
            m[~present] = 0.0
            xin_h[:, bi, 1:L, arm] = m
        rin = np.where(np.isnan(rp[:, :L-1]), 0.0, rp[:, :L-1]); rin[~present] = 0.0
        xin_h[:, bi, 1:L, 4] = rin
        valid = ~np.isnan(cp); valid[:, :HZ_N_FORCED] = False     # mask forced trials
        tgt = np.where(valid, cp, -100.0).astype(np.float32)
        c_h[:, bi, :L] = tgt
    return len(blocks)


# ── build horizon arrays (S1 then S2) ───────────────────────────────────────────
h1 = pd.read_csv("data/finalHorizonSession1.csv")
h2 = pd.read_csv("data/finalHorizonSession2.csv")
HZ_N = h1["block"].nunique() + h2["block"].nunique()
xin_h = np.full((N, HZ_N, MAX_TRIALS, BASE_IN_DIM), -100.0, np.float32)
c_h   = np.full((N, HZ_N, MAX_TRIALS), -100.0, np.float32)
n1 = fill_horizon(h1, 0, xin_h, c_h)
fill_horizon(h2, n1, xin_h, c_h)
n_free = int((c_h > -50).sum())
print(f"horizon blocks={HZ_N} (S1 {n1}+S2 {HZ_N-n1}); free-choice targets={n_free}; "
      f"choices={np.unique(c_h[c_h>-50]).astype(int)}")

# Full 3-task views (task2 = horizon).  xin/c only; one-hot derived per subset.
XIN = {0: xin_2t[:, T0], 1: xin_2t[:, T1], 2: xin_h}
C   = {0: c_2t[:, T0],   1: c_2t[:, T1],   2: c_h}

SUBSETS = {"t0": [0], "t1": [1], "t2": [2], "t01": [0, 1],
           "t02": [0, 2], "t12": [1, 2], "t012": [0, 1, 2]}


def onehot(c):
    oh = np.zeros(c.shape + (A,), np.float32)
    for arm in range(A):
        oh[..., arm] = (c == arm)
    return oh


for name, tasks in SUBSETS.items():
    xin = np.concatenate([XIN[t] for t in tasks], axis=1)
    c   = np.concatenate([C[t]   for t in tasks], axis=1)
    task_ids = np.concatenate([np.full(XIN[t].shape[1], k, np.int32)
                               for k, t in enumerate(tasks)])
    for f in range(3):
        fd = f"data_sub_{name}/fold{f}"; os.makedirs(fd, exist_ok=True)
        for split in ("train", "test"):
            ids = pd.read_csv(f"{S2}/fold{f}/df_{split}.csv")["subid"].values
            rows = np.array([row_of[int(s)] for s in ids])
            np.save(f"{fd}/xin_{split}.npy", xin[rows])
            np.save(f"{fd}/c_{split}.npy", c[rows])
            np.save(f"{fd}/choice_one_hot_{split}.npy", onehot(c[rows]))
            pd.DataFrame({"subid": ids}).to_csv(f"{fd}/df_{split}.csv", index=False)
        np.save(f"{fd}/xin_val.npy", np.zeros((0, len(task_ids), MAX_TRIALS, BASE_IN_DIM), np.float32))
        np.save(f"{fd}/c_val.npy", np.zeros((0, len(task_ids), MAX_TRIALS), np.float32))
        np.save(f"{fd}/choice_one_hot_val.npy", np.zeros((0, len(task_ids), MAX_TRIALS, A), np.float32))
        np.save(f"{fd}/task_ids_per_block.npy", task_ids)
    print(f"  {name:5s} blocks={len(task_ids):3d} task_ids={sorted(set(task_ids.tolist()))} "
          f"-> data_sub_{name}/fold{{0,1,2}}")
print("done.")
