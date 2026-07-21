"""SLURM worker entry point for Optuna HP search.

Each worker pulls N trials from the shared SQLite study for the given
(DGP, arch, fold, dataset_id) cell.  Workers can run concurrently; Optuna's
TPE sampler with constant_liar=True handles parallel pending trials.
"""

from __future__ import annotations
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nested_cv.config import REGISTRY
from nested_cv.optuna_search import run_trials


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dgp", required=True, choices=list(REGISTRY.keys()))
    ap.add_argument("--arch", required=True, choices=["idrnn", "vanilla"])
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--dataset_id", type=int, default=None)
    ap.add_argument("--hypothesis_z", type=int, default=None,
                    help="Anchor z to this value (skip searching over z; "
                          "sweep only the remaining HPs).")
    ap.add_argument("--metric", default="cv_val_loss",
                    choices=["cv_val_loss", "step1_specificity"],
                    help="HP-selection objective.  'cv_val_loss' minimises "
                          "NLL+KL; 'step1_specificity' maximises lookup-encoder "
                          "specificity (correlates with α-decoding at z=1).")
    ap.add_argument("--n_trials", type=int, default=5,
                    help="Number of trials this worker runs before exiting.")
    ap.add_argument("--sampler_seed", type=int, default=0,
                    help="Per-worker TPE seed (vary across workers).")
    ap.add_argument("--trial_seed", type=int, default=42,
                    help="Seed passed to run_Q_model.py for every trial (so the "
                          "HP-search compares HPs with the same init seed).")
    args = ap.parse_args()

    run_trials(
        dgp=args.dgp, arch=args.arch, fold=args.fold,
        n_trials=args.n_trials, dataset_id=args.dataset_id,
        hypothesis_z=args.hypothesis_z,
        metric=args.metric,
        sampler_seed=args.sampler_seed, trial_seed=args.trial_seed,
    )


if __name__ == "__main__":
    main()
