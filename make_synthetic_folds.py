"""Add 3-fold CV subject splits to selected synthetic datasets.

Each fold lives at data_dataset{ID}/fold{F}/ with the same file layout as
data_dezfouli/foldF/ and data_thalmann/foldF/, so run_Q_model.py --fold can
load it transparently once we extend the path resolver in run_Q_model.py.

Within a fold:
  - xin_train.npy, choice_one_hot_train.npy, c_train.npy
  - xin_test.npy,  choice_one_hot_test.npy,  c_test.npy
  - df_train.csv,  df_test.csv
  - pA_train.npy,  pA_test.npy
  - true_param_train.csv, true_param_test.csv   (so RSA / param-recovery still works)
  - rewards_train.npy  (per-fold reward sequences — subset of original)

The held-out test set in each fold is a subject-disjoint partition of the
ORIGINAL train simulation — NOT the separately-simulated data_dataset{ID}/xin_test.npy.
The original test arrays are left in place at the dataset root for back-compat;
they're a different population (different true_param draws) and shouldn't be
used for nested-CV evaluation.
"""

import argparse
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

DEFAULT_DATASET_IDS = [0, 2, 5, 10, 17]
N_FOLDS = 3
SEED = 42


def _subset_df(df: pd.DataFrame, keep_sessions: np.ndarray) -> pd.DataFrame:
    return df[df["session"].isin(keep_sessions)].reset_index(drop=True)


def _subset_param_csv(src_csv: str, keep_sessions: np.ndarray) -> pd.DataFrame:
    """true_parameter_values.csv is row-indexed by session (one row per session).
    keep_sessions is in original-session ordering (0..N-1)."""
    df = pd.read_csv(src_csv)
    keep = sorted(keep_sessions.tolist())
    return df.iloc[keep].reset_index(drop=True)


def write_folds_for_dataset(dataset_id: int, n_folds: int = N_FOLDS,
                             seed: int = SEED, overwrite: bool = False) -> None:
    src = f"data_dataset{dataset_id}"
    if not os.path.isdir(src):
        print(f"  ⚠ {src}/ missing — skipping.")
        return

    # Load full training arrays
    xin     = np.load(os.path.join(src, "xin_train.npy"))                       # (N, T, in)
    oh      = np.load(os.path.join(src, "choice_one_hot_train.npy"))            # (N, T, A)
    c       = np.load(os.path.join(src, "c_train.npy"))                         # (N, T)
    pA      = np.load(os.path.join(src, "pA_train.npy"))                        # (N, T, A) or (T, A)
    rewards = np.load(os.path.join(src, "rewards_train.npy"))
    df      = pd.read_csv(os.path.join(src, "df_train.csv"))

    n_subj  = xin.shape[0]
    sessions = np.arange(n_subj)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold_idx, (tr, te) in enumerate(kf.split(sessions)):
        fold_dir = os.path.join(src, f"fold{fold_idx}")
        if os.path.isdir(fold_dir) and not overwrite:
            existing = os.listdir(fold_dir)
            # Tolerate fold dirs that have already accumulated run artefacts;
            # don't overwrite them silently.
            if any(f.endswith(".npy") for f in existing):
                print(f"  fold{fold_idx} already populated ({len(existing)} entries) — skip "
                      "(pass --overwrite to redo)")
                continue
        os.makedirs(fold_dir, exist_ok=True)

        np.save(os.path.join(fold_dir, "xin_train.npy"),            xin[tr])
        np.save(os.path.join(fold_dir, "xin_test.npy"),             xin[te])
        np.save(os.path.join(fold_dir, "choice_one_hot_train.npy"), oh[tr])
        np.save(os.path.join(fold_dir, "choice_one_hot_test.npy"),  oh[te])
        np.save(os.path.join(fold_dir, "c_train.npy"),              c[tr])
        np.save(os.path.join(fold_dir, "c_test.npy"),               c[te])

        # pA may be per-session or shared
        if pA.ndim == xin.ndim:           # per-session
            np.save(os.path.join(fold_dir, "pA_train.npy"), pA[tr])
            np.save(os.path.join(fold_dir, "pA_test.npy"),  pA[te])
        else:                              # shared across sessions
            np.save(os.path.join(fold_dir, "pA_train.npy"), pA)
            np.save(os.path.join(fold_dir, "pA_test.npy"),  pA)

        # Reward sequences — may be (N, T) or shared
        if rewards.ndim >= 2 and rewards.shape[0] == n_subj:
            np.save(os.path.join(fold_dir, "rewards_train.npy"), rewards[tr])
            np.save(os.path.join(fold_dir, "rewards_test.npy"),  rewards[te])
        else:
            np.save(os.path.join(fold_dir, "rewards_train.npy"), rewards)
            np.save(os.path.join(fold_dir, "rewards_test.npy"),  rewards)

        _subset_df(df, tr).to_csv(os.path.join(fold_dir, "df_train.csv"), index=False)
        _subset_df(df, te).to_csv(os.path.join(fold_dir, "df_test.csv"),  index=False)

        # True parameters per session (so RSA / param-recovery still works post-split)
        param_csv = os.path.join(src, "true_parameter_values.csv")
        if os.path.exists(param_csv):
            _subset_param_csv(param_csv, tr).to_csv(
                os.path.join(fold_dir, "true_param_train.csv"), index=False)
            _subset_param_csv(param_csv, te).to_csv(
                os.path.join(fold_dir, "true_param_test.csv"), index=False)

        print(f"  fold{fold_idx}: {len(tr)} train + {len(te)} test → {fold_dir}/")

    print(f"  dataset {dataset_id}: done.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_ids", type=int, nargs="+", default=DEFAULT_DATASET_IDS)
    ap.add_argument("--n_folds", type=int, default=N_FOLDS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    print(f"Generating {args.n_folds}-fold subject splits for datasets "
          f"{args.dataset_ids} (seed={args.seed}).")
    for did in args.dataset_ids:
        print(f"\n── dataset {did} ──")
        write_folds_for_dataset(did, n_folds=args.n_folds, seed=args.seed,
                                 overwrite=args.overwrite)


if __name__ == "__main__":
    main()
