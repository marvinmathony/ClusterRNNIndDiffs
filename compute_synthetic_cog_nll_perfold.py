"""Per-fold cognitive-model NLL for the synthetic nested-CV datasets.

Panel-a apples-to-apples fix: the nested-CV RNN NLLs are evaluated on
subject-disjoint folds of the ORIGINAL TRAIN simulation
(make_synthetic_folds.py), while the legacy cognitive-model NLLs in
data_dataset{D}/model_eval_dfvanilla.csv were evaluated on the separate
test simulation (a different population draw).  This script refits the
cognitive models per fold — mirroring the thalmann pooled_nested_nll.py
convention — so every panel-a bar shares the identical evaluation subjects:

  for each (dataset, fold):
    fit Q/FQ common-fit + empirical-Bayes MAP on fold df_train
    evaluate summed per-session NLL on fold df_test      (fit_all_models)
    True-model NLL on fold df_test from the stored generating choice
    probabilities p (identical formula to sim_Q_data.simulate_Qlearning)

Writes ONLY new files (never touches existing results):
  cog_nll_perfold_parts/ds{D}_fold{F}.csv   — one per (dataset, fold)
  cog_nll_perfold_synthetic.csv             — merged (via --merge)
Columns: dataset_id, fold, session, model, nll_summed.

Usage:
  python compute_synthetic_cog_nll_perfold.py                       # all pairs, serial
  python compute_synthetic_cog_nll_perfold.py --dataset_id 5 --fold 1   # one pair (SLURM array)
  python compute_synthetic_cog_nll_perfold.py --merge               # concat parts -> final CSV
Existing part files are skipped, so reruns/resumes are cheap.
"""
import argparse
import glob
import os

# Cap BLAS threads BEFORE numpy import — the login/notebook nodes kill
# sessions above 200% CPU (see project memory).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
import RL_fittingfunctions2 as fit

DATASETS = [0, 1, 2, 3, 5, 7, 10, 12, 15, 17]
N_FOLDS = 3
N_FIT_ITER = 5           # matches testing_script.py n_fit_iter
OUT_CSV = "cog_nll_perfold_synthetic.csv"
PARTS_DIR = "cog_nll_perfold_parts"

# Identical to testing_script.py's model_configs
MODEL_CONFIGS = {
    "Q":  {"asymmetric_alpha": False, "forgetting_type": "none",  "choice_trace": False},
    "FQ": {"asymmetric_alpha": False, "forgetting_type": "fixed", "choice_trace": False},
}


def true_model_nll(df_test: pd.DataFrame) -> pd.DataFrame:
    """Summed NLL of the generative choice probabilities, exactly as
    sim_Q_data.simulate_Qlearning stores it (verified to reproduce
    session_ll_df_test.csv to 4 decimals)."""
    rows = []
    for session, grp in df_test.groupby("session"):
        c, p = grp["c"].values, grp["p"].values
        nll = -float(np.sum(np.log(np.where(c == 0, p, 1 - p))))
        rows.append({"session": session, "normalized_likelihood": nll,
                     "model": "True model"})
    return pd.DataFrame(rows)


def run_pair(D: int, F: int) -> None:
    """Fit + evaluate one (dataset, fold) pair; write its part file."""
    os.makedirs(PARTS_DIR, exist_ok=True)
    part = os.path.join(PARTS_DIR, f"ds{D}_fold{F}.csv")
    if os.path.exists(part):
        print(f"dataset{D} fold{F}: {part} exists — skip")
        return
    fold_dir = f"data_dataset{D}/fold{F}"
    if not os.path.isdir(fold_dir):
        print(f"dataset{D} fold{F}: {fold_dir}/ missing — skip")
        return
    print(f"=== dataset{D} fold{F} ===", flush=True)
    df_train = pd.read_csv(f"{fold_dir}/df_train.csv")
    df_test = pd.read_csv(f"{fold_dir}/df_test.csv")

    model_eval_df, *_ = fit.fit_all_models(
        MODEL_CONFIGS, df_train, df_test, n_iter=N_FIT_ITER, fit_ML=False
    )
    out = pd.concat([model_eval_df[["session", "normalized_likelihood", "model"]],
                     true_model_nll(df_test)], ignore_index=True)
    out = out.rename(columns={"normalized_likelihood": "nll_summed"})
    out.insert(0, "fold", F)
    out.insert(0, "dataset_id", D)

    tmp = part + ".tmp"          # atomic write: no partial part files on kill
    out.to_csv(tmp, index=False)
    os.replace(tmp, part)
    print(f"dataset{D} fold{F}: wrote {len(out)} rows "
          f"({out['model'].nunique()} models × {df_test['session'].nunique()} sessions)",
          flush=True)


def merge() -> None:
    parts = sorted(glob.glob(os.path.join(PARTS_DIR, "ds*_fold*.csv")))
    expected = {(D, F) for D in DATASETS for F in range(N_FOLDS)}
    have = set()
    dfs = []
    for p in parts:
        d = pd.read_csv(p)
        dfs.append(d)
        have |= set(zip(d["dataset_id"], d["fold"]))
    missing = expected - have
    if missing:
        print(f"WARNING: merging with {len(missing)} pairs missing: {sorted(missing)}")
    if not dfs:
        raise SystemExit("No part files found — nothing to merge.")
    merged = pd.concat(dfs, ignore_index=True)
    merged.to_csv(OUT_CSV, index=False)
    print(f"Merged {len(parts)} parts → {OUT_CSV} ({len(merged)} rows, "
          f"{merged['model'].nunique()} models, "
          f"{merged.groupby(['dataset_id','fold']).ngroups} (dataset, fold) pairs)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_id", type=int, default=None)
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()

    if args.merge:
        merge()
        return
    if args.dataset_id is not None and args.fold is not None:
        run_pair(args.dataset_id, args.fold)
        return
    for D in DATASETS:
        for F in range(N_FOLDS):
            run_pair(D, F)
    merge()


if __name__ == "__main__":
    main()
