#!/usr/bin/env python3
"""Build data_thalmann_full/ — the full-cohort (train+test concatenated)
dataset for the canonical representational retrain.

The nested-CV pipeline trains on the 70/30 split (data_thalmann/: 165 train,
71 test).  For the decoding (panel b/c) and three-regression (panel e) panels
we need ONE step-1 latent per subject for ALL 236 subjects, so we retrain the
canonical model on the concatenation of train+test (REPLICATION §2 "retrain on
ALL participants").

This writes a sibling data dir that run_Q_model.py reads via --data_dir, so the
original data_thalmann/ (used by every nested-CV stage) is never mutated.

Layout written:
  data_thalmann_full/
    xin_train.npy            (236, 31, 200, 5)   = concat(train 165, test 71)
    c_train.npy              (236, 31, 200)
    choice_one_hot_train.npy (236, 31, 200, 4)
    df_train.csv             236 rows (subid, in concat order)
    xin_test.npy / c_test.npy / choice_one_hot_test.npy  = copy of the 236
    df_test.csv              = copy of df_train
    xin_val.npy / c_val.npy / choice_one_hot_val.npy     = empty (has_val=False)
    task_ids_per_block.npy   = copy (block structure unchanged)
    subids_full.npy          = subid order of the 236 rows (for z->subid mapping)
"""
import os
import numpy as np
import pandas as pd

SRC = "data_thalmann"
DST = "data_thalmann_full"
os.makedirs(DST, exist_ok=True)


def _concat(name):
    tr = np.load(f"{SRC}/{name}_train.npy", allow_pickle=True)
    te = np.load(f"{SRC}/{name}_test.npy",  allow_pickle=True)
    full = np.concatenate([tr, te], axis=0)
    return tr, te, full


def main():
    names = ["xin", "c", "choice_one_hot"]
    fulls = {}
    for n in names:
        tr, te, full = _concat(n)
        fulls[n] = full
        print(f"  {n}: train{tr.shape} + test{te.shape} -> full{full.shape} ({full.dtype})")

    df_tr = pd.read_csv(f"{SRC}/df_train.csv")
    df_te = pd.read_csv(f"{SRC}/df_test.csv")
    df_full = pd.concat([df_tr, df_te], ignore_index=True)
    assert len(df_full) == fulls["xin"].shape[0], \
        f"df rows {len(df_full)} != xin rows {fulls['xin'].shape[0]}"
    print(f"  df: train {len(df_tr)} + test {len(df_te)} -> full {len(df_full)} "
          f"(cols {list(df_full.columns)})")

    # train = full cohort; test = copy of full cohort (ignored downstream, but
    # avoids any empty-array edge cases in the test-eval path); val = empty.
    for n in names:
        np.save(f"{DST}/{n}_train.npy", fulls[n])
        np.save(f"{DST}/{n}_test.npy",  fulls[n])
    df_full.to_csv(f"{DST}/df_train.csv", index=False)
    df_full.to_csv(f"{DST}/df_test.csv",  index=False)

    # empty val (matches root data_thalmann/, forces has_val=False so the inner
    # CV splits internally and the final model trains on all 236).
    for n, suff in [("xin", (0, 31, 200, 5)), ("c", (0, 31, 200)),
                    ("choice_one_hot", (0, 31, 200, 4))]:
        np.save(f"{DST}/{n}_val.npy", np.zeros(suff, dtype=np.float32))

    # block structure unchanged
    tid = np.load(f"{SRC}/task_ids_per_block.npy")
    np.save(f"{DST}/task_ids_per_block.npy", tid)

    # subid order of the 236 rows (z[i] <-> subids_full[i])
    subids_full = df_full["subid"].values.astype(int)
    np.save(f"{DST}/subids_full.npy", subids_full)
    print(f"  subids_full: {subids_full.shape}  e.g. {subids_full[:5]} ... {subids_full[-3:]}")
    print(f"Wrote {DST}/")


if __name__ == "__main__":
    main()
