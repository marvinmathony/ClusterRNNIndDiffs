#!/usr/bin/env python3
"""Full-cohort (no train/test split) pooled fold dirs for the 7 task subsets, for
the REPRESENTATIONAL data-scaling (decode WM/openness from the IDRNN latent vs
dim-matched vanilla, as #training-tasks grows).  Mirrors build_pooled_subsets.py
but writes data_sub_{name}_full/ with xin_train==xin_test==all 238 subjects
(same convention as data_thalmann_full / _3task_full), plus subids_full.npy so
seed_averaged_representation.py can extract z for every subject.
"""
import os
import numpy as np
import pandas as pd

A = 4; REWARD_MAX = 100.0; MAX_TRIALS = 200; BASE_IN_DIM = 5; HZ_N_FORCED = 4
S2 = "data_thalmann_s2"
df_all = pd.read_csv(f"{S2}/df_all.csv"); subids = df_all["subid"].values; N = len(subids)
xin_2t = np.load(f"{S2}/xin_all.npy"); c_2t = np.load(f"{S2}/c_all.npy")
tids2 = np.load(f"{S2}/task_ids_per_block.npy")
T0 = np.where(tids2 == 0)[0]; T1 = np.where(tids2 == 1)[0]


def fill_horizon(df, b0, xin_h, c_h):
    for j, bn in enumerate(sorted(df["block"].unique())):
        g = df[df["block"] == bn]
        cp = g.pivot(index="ID", columns="trial", values="chosen").reindex(subids).to_numpy().astype(np.float32)
        rp = g.pivot(index="ID", columns="trial", values="reward").reindex(subids).to_numpy().astype(np.float32) / REWARD_MAX
        L = cp.shape[1]; bi = b0 + j; present = ~np.isnan(cp).all(axis=1)
        xin_h[present, bi, 0, :] = 0.0; prev = cp[:, :L-1]
        for arm in range(A):
            m = (np.where(np.isnan(prev), -100, prev).astype(int) == arm).astype(np.float32); m[~present] = 0.0
            xin_h[:, bi, 1:L, arm] = m
        rin = np.where(np.isnan(rp[:, :L-1]), 0.0, rp[:, :L-1]); rin[~present] = 0.0
        xin_h[:, bi, 1:L, 4] = rin
        valid = ~np.isnan(cp); valid[:, :HZ_N_FORCED] = False
        c_h[:, bi, :L] = np.where(valid, cp, -100.0).astype(np.float32)
    return len(sorted(df["block"].unique()))


h1 = pd.read_csv("data/finalHorizonSession1.csv"); h2 = pd.read_csv("data/finalHorizonSession2.csv")
HZ_N = h1["block"].nunique() + h2["block"].nunique()
xin_h = np.full((N, HZ_N, MAX_TRIALS, BASE_IN_DIM), -100.0, np.float32)
c_h = np.full((N, HZ_N, MAX_TRIALS), -100.0, np.float32)
n1 = fill_horizon(h1, 0, xin_h, c_h); fill_horizon(h2, n1, xin_h, c_h)
print(f"horizon blocks={HZ_N}")

XIN = {0: xin_2t[:, T0], 1: xin_2t[:, T1], 2: xin_h}
C = {0: c_2t[:, T0], 1: c_2t[:, T1], 2: c_h}
SUBSETS = {"t0": [0], "t1": [1], "t2": [2], "t01": [0, 1], "t02": [0, 2], "t12": [1, 2], "t012": [0, 1, 2]}


def onehot(c):
    oh = np.zeros(c.shape + (A,), np.float32)
    for arm in range(A):
        oh[..., arm] = (c == arm)
    return oh


for name, tasks in SUBSETS.items():
    xin = np.concatenate([XIN[t] for t in tasks], axis=1)
    c = np.concatenate([C[t] for t in tasks], axis=1)
    task_ids = np.concatenate([np.full(XIN[t].shape[1], k, np.int32) for k, t in enumerate(tasks)])
    oh = onehot(c)
    fd = f"data_sub_{name}_full"; os.makedirs(fd, exist_ok=True)
    for split in ("train", "test"):           # train == test == all 238 (full-cohort)
        np.save(f"{fd}/xin_{split}.npy", xin); np.save(f"{fd}/c_{split}.npy", c)
        np.save(f"{fd}/choice_one_hot_{split}.npy", oh)
        pd.DataFrame({"subid": subids}).to_csv(f"{fd}/df_{split}.csv", index=False)
    np.save(f"{fd}/xin_val.npy", np.zeros((0, len(task_ids), MAX_TRIALS, BASE_IN_DIM), np.float32))
    np.save(f"{fd}/c_val.npy", np.zeros((0, len(task_ids), MAX_TRIALS), np.float32))
    np.save(f"{fd}/choice_one_hot_val.npy", np.zeros((0, len(task_ids), MAX_TRIALS, A), np.float32))
    np.save(f"{fd}/task_ids_per_block.npy", task_ids)
    np.save(f"{fd}/subids_full.npy", subids.astype(int))
    print(f"  {name:5s} n_tasks={len(tasks)} blocks={len(task_ids):3d} -> {fd}")
print("done.")
