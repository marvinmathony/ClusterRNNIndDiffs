#!/usr/bin/env python3
"""Per-fold nested-CV held-out NLL on POOLED (pre-split) fold dirs — self-contained.

Each `data_root/fold{F}/` already holds a subject-disjoint train/test split (built
outside the nested_cv `--fold` machinery), so we drive run_Q_model with
`--data_dir data_root/fold{F}` (no `--fold`).  Per fold we do an honest inner-CV
HP search: sample N HP configs (random subset of the z-anchored grid, seeded per
fold → leakage-free), train 1 seed each, read inner cv_val_loss, pick the min as
that fold's winner; then retrain the winner across seeds.  The held-out NLL is
then computed by analyze_thalmann_nested_cv.py (env-parameterised).

Generalises to any task-subset: just point --data_root at that subset's fold dirs
and give it a --tag.  Used for the pooled panel-a (tag=s2_2task) and the
data-scaling 'more tasks helps IDRNN' analysis (one tag per task-subset).

Stages:
  search  : submit HP-eval runs (1 seed/config) -> hp_pooled/{tag}/{arch}/fold{F}/{cfg}/
  pick    : read cv_val_loss, write winners_pooled/{tag}/{arch}/fold{F}.json
  retrain : submit per-fold winner x seeds -> runs_pooled/{tag}/{arch}/fold{F}/seed_{S}/

Usage:
  python pooled_nested_nll.py search  --data_root data_thalmann_s2 --tag s2_2task --arch both
  python pooled_nested_nll.py pick    --tag s2_2task --arch both
  python pooled_nested_nll.py retrain --data_root data_thalmann_s2 --tag s2_2task --arch both
"""
import argparse, glob, itertools, json, os, random, sys
sys.path.insert(0, ".")
from nested_cv.config import REGISTRY, effective_hp_grid
from nested_cv.launch import _build_run_q_model_cmd, _sbatch

DGP = "thalmann"; HYP_Z = 3; N_FOLDS = 3; N_HP_SAMPLES = 16
FINAL_SEEDS = [200, 300, 400, 500, 600, 999, 2021, 2022, 2023, 2024, 2025, 401, 402, 403, 404]

HP_DIR = "hp_pooled/{tag}/{arch}/fold{f}/{cfg}"
WIN_DIR = "winners_pooled/{tag}/{arch}"
RUN_DIR = "runs_pooled/{tag}/{arch}/fold{f}/seed_{s}"


def _cfg_tag(hp):
    return "_".join(f"{k}{str(v).replace('.','p')}" for k, v in sorted(hp.items()))


def _grid_combos(arch):
    grid = effective_hp_grid(REGISTRY[DGP], arch, hypothesis_z=HYP_Z)
    keys = list(grid.keys())
    return [dict(zip(keys, vals)) for vals in itertools.product(*[grid[k] for k in keys])]


def _fold_combos(arch, fold):
    """Per-fold HP configs to evaluate (full grid for vanilla; seeded random
    subset for idrnn so the inner search is honest + cheap)."""
    combos = _grid_combos(arch)
    if arch == "vanilla" or len(combos) <= N_HP_SAMPLES:
        return combos
    rng = random.Random(1000 + fold)          # per-fold, deterministic
    return rng.sample(combos, N_HP_SAMPLES)


def do_search(args):
    archs = ["idrnn", "vanilla"] if args.arch == "both" else [args.arch]
    spec = REGISTRY[DGP]; n = 0
    for arch in archs:
        fixed = spec.fixed_flags[arch]
        for f in range(N_FOLDS):
            for hp in _fold_combos(arch, f):
                rd = HP_DIR.format(tag=args.tag, arch=arch, f=f, cfg=_cfg_tag(hp))
                if os.path.exists(os.path.join(rd, "config.json")) and not args.force:
                    if "cv_val_loss" in json.load(open(os.path.join(rd, "config.json"))):
                        continue
                run_args = {"seed": 42, **fixed, "dataset_id": 0,
                            "data_dir": f"{args.data_root}/fold{f}",
                            "block_weight_mode": args.block_weight_mode, **hp}
                jid = _sbatch(name=f"hp_{arch[:3]}_{args.tag[:6]}_f{f}",
                              log_prefix=f"hp_{args.tag}_{arch}_f{f}_{_cfg_tag(hp)[:20]}",
                              body=_build_run_q_model_cmd(run_args),
                              env={"HP_RUN_DIR": rd}, dry_run=args.dry_run)
                if jid: n += 1
        print(f"[{arch}] search: {N_FOLDS} folds x {len(_fold_combos(arch,0))} configs")
    print(f"submitted {n} HP-eval jobs")


def do_pick(args):
    """Pick per-fold winners by args.metric (cv_val_loss=ELBO, default; or
    cv_val_nll=predictive NLL, leakage-free, avoids KL-collapsed encoders).
    Reads HP search from args.hp_tag (default args.tag); writes winners to args.tag."""
    archs = ["idrnn", "vanilla"] if args.arch == "both" else [args.arch]
    hp_tag = args.hp_tag or args.tag
    for arch in archs:
        wd = WIN_DIR.format(tag=args.tag, arch=arch); os.makedirs(wd, exist_ok=True)
        for f in range(N_FOLDS):
            cands = []
            for rd in glob.glob(HP_DIR.format(tag=hp_tag, arch=arch, f=f, cfg="*")):
                cp = os.path.join(rd, "config.json")
                if not os.path.exists(cp):
                    continue
                cfg = json.load(open(cp))
                if args.metric not in cfg:
                    continue
                hp = {k: cfg[k] for k in cfg if k in effective_hp_grid(REGISTRY[DGP], arch, hypothesis_z=HYP_Z)}
                hp["z"] = HYP_Z if arch == "idrnn" else cfg.get("z", HYP_Z)
                cands.append((float(cfg[args.metric]), hp, os.path.basename(rd)))
            if not cands:
                print(f"  [{arch} fold{f}] NO candidates yet"); continue
            cands.sort(key=lambda x: x[0])
            best = cands[0]
            json.dump({"winner": best[1], args.metric: best[0],
                       "n_candidates": len(cands), "cfg": best[2]},
                      open(os.path.join(wd, f"fold{f}.json"), "w"), indent=2)
            print(f"  [{arch} fold{f}] winner {args.metric}={best[0]:.4f} "
                  f"({len(cands)} cands): {best[1]}")


def do_retrain(args):
    archs = ["idrnn", "vanilla"] if args.arch == "both" else [args.arch]
    spec = REGISTRY[DGP]; seeds = FINAL_SEEDS[:args.n_seeds]; n = 0
    for arch in archs:
        fixed = spec.fixed_flags[arch]
        for f in range(N_FOLDS):
            wp = os.path.join(WIN_DIR.format(tag=args.tag, arch=arch), f"fold{f}.json")
            if not os.path.exists(wp):
                print(f"  [{arch} fold{f}] no winner JSON — run pick first"); continue
            hp = json.load(open(wp))["winner"]
            for s in seeds:
                rd = RUN_DIR.format(tag=args.tag, arch=arch, f=f, s=s)
                if os.path.exists(os.path.join(rd, "config.json")) and not args.force:
                    continue
                run_args = {"seed": s, **fixed, "dataset_id": 0,
                            "data_dir": f"{args.data_root}/fold{f}",
                            "block_weight_mode": args.block_weight_mode, **hp}
                jid = _sbatch(name=f"rt_{arch[:3]}_{args.tag[:6]}_f{f}_s{s}",
                              log_prefix=f"rt_{args.tag}_{arch}_f{f}_s{s}",
                              body=_build_run_q_model_cmd(run_args),
                              env={"HP_RUN_DIR": rd}, dry_run=args.dry_run)
                if jid: n += 1
        print(f"[{arch}] retrain: {N_FOLDS} folds x {len(seeds)} seeds")
    print(f"submitted {n} retrain jobs")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["search", "pick", "retrain"])
    ap.add_argument("--data_root", default="data_thalmann_s2")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--arch", choices=["idrnn", "vanilla", "both"], default="both")
    ap.add_argument("--n_seeds", type=int, default=15)
    ap.add_argument("--metric", default="cv_val_loss",
                    choices=["cv_val_loss", "cv_val_nll"],
                    help="HP-selection objective for `pick` (cv_val_nll avoids KL-collapsed encoders)")
    ap.add_argument("--hp_tag", default=None,
                    help="read HP search from this tag (default: --tag); lets `pick` reuse an "
                         "existing search while writing winners to a new tag")
    ap.add_argument("--block_weight_mode", default="task0_zero",
                    choices=["task0_zero", "uniform"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    {"search": do_search, "pick": do_pick, "retrain": do_retrain}[args.stage](args)
