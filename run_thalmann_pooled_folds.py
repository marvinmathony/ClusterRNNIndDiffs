#!/usr/bin/env python3
"""Per-fold RNN training on the POOLED S1+S2 folds (for the pooled panel-a held-out
NLL).  For each outer fold, trains IDRNN + Vanilla on the fold's train split and
leaves the test split held out (each subject is test in exactly one fold).

Mirrors nested_cv stage_c but on data_thalmann_s2/fold{F} via run_Q_model's
--data_dir override (no --fold flag; the fold dir already has train/test split).
Reuses the canonical z=3 HP from the S1 nested-CV spec.

Output: runs_thalmann_s2_foldcv/{idrnn,vanilla}/fold{F}/seed_{S}/
Usage:  python run_thalmann_pooled_folds.py --submit [--n_seeds 15] [--dry_run]
"""
import argparse, json, os, sys
sys.path.insert(0, ".")
from nested_cv.config import REGISTRY, path_tag
from nested_cv.launch import _build_run_q_model_cmd, _sbatch

DGP = "thalmann"; HYP_Z = 3; METRIC = "cv_val_loss"
DATA_FOLD = "data_thalmann_s2/fold{f}"
OUT = "runs_thalmann_s2_foldcv/{arch}/fold{f}/seed_{s}"
N_FOLDS = 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["idrnn", "vanilla", "both"], default="both")
    ap.add_argument("--n_seeds", type=int, default=15)
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    if not (args.submit or args.dry_run):
        ap.error("pass --submit (optionally --dry_run)")
    archs = ["idrnn", "vanilla"] if args.arch == "both" else [args.arch]
    spec = REGISTRY[DGP]
    all_ids = []
    for arch in archs:
        sp = json.load(open(f"final_plots/{DGP}{path_tag(HYP_Z, METRIC)}/canonical/{arch}_spec.json"))
        hp = dict(sp["hp"]); seeds = sp["seeds"][:args.n_seeds]
        fixed = spec.fixed_flags[arch]
        print(f"[{arch}] HP={sp['hp_tag']}  seeds={seeds}  folds={N_FOLDS}")
        for f in range(N_FOLDS):
            for s in seeds:
                run_args = {"seed": s, **fixed, "dataset_id": 0,
                            "data_dir": DATA_FOLD.format(f=f)}
                run_args.update(hp)
                jid = _sbatch(
                    name=f"pf_{arch[:3]}_f{f}_s{s}",
                    log_prefix=f"pf_thalmann_{arch}_f{f}_s{s}",
                    body=_build_run_q_model_cmd(run_args),
                    env={"HP_RUN_DIR": OUT.format(arch=arch, f=f, s=s)},
                    dry_run=args.dry_run)
                if jid:
                    all_ids.append(jid)
        if not args.dry_run:
            print(f"[{arch}] submitted {N_FOLDS*len(seeds)} jobs")
    if all_ids:
        print(f"all: {len(all_ids)} jobs")


if __name__ == "__main__":
    main()
