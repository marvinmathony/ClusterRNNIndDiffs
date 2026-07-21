#!/usr/bin/env python3
"""Full-cohort canonical retrain for Thalmann (representational panels b/c/e).

REPLICATION §2: decoding + rollouts need ONE step-1 latent per subject for ALL
236 subjects, so we retrain the canonical model (the stage_e HP combo) on the
full cohort (data_thalmann_full/, built by build_thalmann_full_data.py) across
the spec's seeds, then pick the best seed (step1_specificity for IDRNN,
cv_val_loss for Vanilla).

This mirrors nested_cv.launch.stage_f exactly (same fixed flags + HP from the
canonical spec, HP_RUN_DIR redirect) but:
  • adds --data_dir data_thalmann_full  (train on all 236, never mutate data_thalmann/)
  • writes to final_plots/thalmann_z3_full/canonical/{arch}/runs/seed_{S}/

Reads the HP combo + seeds from the stage_e spec:
  final_plots/thalmann_z3/canonical/{arch}_spec.json

Usage:
  python run_thalmann_canonical_full.py --submit            # submit retrain jobs
  python run_thalmann_canonical_full.py --submit --dry_run  # preview sbatch cmds
  python run_thalmann_canonical_full.py --submit --n_seeds 15
"""
import argparse
import json
import os
import sys

sys.path.insert(0, ".")
from nested_cv.config import REGISTRY, path_tag
from nested_cv.launch import _build_run_q_model_cmd, _sbatch

DGP = "thalmann"
HYP_Z = 3
METRIC = "cv_val_loss"
# Overridable so the same driver serves S1 (data_thalmann_full ->
# thalmann_z3_full) and pooled S1+S2 (data_thalmann_s2_full -> thalmann_z3_s2_full).
DATA_DIR_FULL = os.environ.get("THAL_DATA", "data_thalmann_full")
OUT_FULL = os.environ.get("THAL_FULL", f"final_plots/{DGP}{path_tag(HYP_Z, METRIC)}_full")


def _full_runs_dir(arch):
    return f"{OUT_FULL}/canonical/{arch}/runs"


def _spec_path(arch):
    tag = path_tag(HYP_Z, METRIC)            # "_z3"
    return f"final_plots/{DGP}{tag}/canonical/{arch}_spec.json"


def submit(arch, n_seeds, dry_run):
    spec_path = _spec_path(arch)
    if not os.path.exists(spec_path):
        raise FileNotFoundError(
            f"Missing canonical spec {spec_path}. Run stage_e first:\n"
            f"  bash submit_nested_cv.sh thalmann stage_e --hypothesis_z 3")
    payload = json.load(open(spec_path))
    hp = dict(payload["hp"])
    seeds = payload["seeds"][:n_seeds] if n_seeds else payload["seeds"]
    fixed = REGISTRY[DGP].fixed_flags[arch]
    runs_dir = _full_runs_dir(arch)
    os.makedirs(runs_dir, exist_ok=True)

    print(f"[{arch}] HP={payload['hp_tag']}  seeds={seeds}")
    print(f"[{arch}] -> {runs_dir}/seed_*  (--data_dir {DATA_DIR_FULL})")
    job_ids = []
    for seed in seeds:
        run_args = {
            "seed": seed, **fixed,
            "dataset_id": 0,
            "data_dir": DATA_DIR_FULL,
            "block_weight_mode": os.environ.get("THAL_BLOCK_WEIGHT_MODE", "task0_zero"),
        }
        run_args.update(hp)
        body = _build_run_q_model_cmd(run_args)
        jid = _sbatch(
            name=f"full_tha_{arch[:3]}_s{seed}",
            log_prefix=f"full_thalmann_{arch}_s{seed}",
            body=body,
            env={"HP_RUN_DIR": os.path.join(runs_dir, f"seed_{seed}")},
            dry_run=dry_run,
        )
        if jid:
            job_ids.append(jid)
    if job_ids:
        print(f"[{arch}] submitted {len(job_ids)} jobs: {' '.join(job_ids)}")
    return job_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["idrnn", "vanilla", "both"], default="both")
    ap.add_argument("--n_seeds", type=int, default=None,
                    help="Cap on number of canonical seeds (default: all in spec).")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    if not (args.submit or args.dry_run):
        ap.error("pass --submit (optionally with --dry_run to preview)")

    archs = ["idrnn", "vanilla"] if args.arch == "both" else [args.arch]
    all_ids = []
    for a in archs:
        all_ids.extend(submit(a, args.n_seeds, args.dry_run))
    if all_ids:
        print(f"\nAll job IDs: {' '.join(all_ids)}")


if __name__ == "__main__":
    main()
