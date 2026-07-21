"""Nested-CV SLURM orchestrator.

Stages (dispatched via `python -m nested_cv.launch <stage> --dgp <dgp> ...`):

  stage_a   — submit inner HP-search jobs (per combo × fold × seed)
  stage_b   — pick per-fold winners (CPU-only, no SLURM, see select_hp.py)
  stage_c   — submit final per-fold training jobs (per fold × N final seeds)
              with that fold's winning HPs
  stage_d   — submit testing + analyze_outer_cv (per DGP, one job)
  stage_e   — write canonical retrain spec (CPU-only, see select_canonical.py)
  stage_f   — submit canonical retrain (full-train-set, N seeds, no fold)
  stage_g   — pick canonical seed (CPU-only, see select_canonical.py)

Everything respects --dry_run (prints the sbatch commands but doesn't submit).

The script is dataset-agnostic: per-DGP HP grids and templates live in
nested_cv/config.py, so adding/altering a dataset means editing that one file.
"""

from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nested_cv.config import (
    REGISTRY, DGPSpec, combo_tag, thalmann_v3_tag, N_OUTER_FOLDS, INNER_SEEDS,
    N_OPTUNA_TRIALS, N_OPTUNA_WORKERS, trial_budget_for, workers_for, z_tag,
    path_tag, OPTUNA_OBJECTIVES,
)
from nested_cv.select_hp import _hp_run_dir


# ── SLURM defaults (mirrors submit_*_outer_cv.sh) ─────────────────────────────
SLURM_GPU = [
    "--nodes=1", "--gres=gpu:1", "-p", "gpu_p", "--qos", "gpu_normal",
    "--constraint=a100_80gb|h100_80gb", "--nice=10000",
    "--mem=16G", "--cpus-per-task=4", "--time=12:00:00",
    "--open-mode=append", "--mail-type=FAIL",
    "--mail-user=marvin.mathony@helmholtz-munich.de",
]
SLURM_CPU = [
    "--nodes=1", "-p", "gpu_p", "--qos", "gpu_normal",
    "--constraint=a100_80gb|h100_80gb", "--nice=10000",
    "--mem=32G", "--cpus-per-task=16", "--time=06:00:00",
    "--open-mode=append", "--mail-type=ALL",
    "--mail-user=marvin.mathony@helmholtz-munich.de",
]

CONDA_INIT = (
    "export WANDB_MODE=offline && "
    "source /ictstr01/home/hcai/marvin.mathony/tools/apps/mamba/etc/profile.d/conda.sh && "
    "conda activate RNNproject"
)


# ── Helpers ────────────────────────────────────────────────────────────────────
def _flag(k: str, v: Any) -> List[str]:
    """Convert (k, v) -> ['--k', 'v']; bools become 'True'/'False' strings."""
    if isinstance(v, bool):
        v = "True" if v else "False"
    return [f"--{k}", str(v)]


def _build_run_q_model_cmd(args_dict: Dict[str, Any]) -> str:
    parts = ["python", "run_Q_model.py"]
    for k, v in args_dict.items():
        parts.extend(_flag(k, v))
    return " ".join(parts)


def _sbatch(name: str, log_prefix: str, body: str, *,
            dependencies: Optional[List[str]] = None,
            slurm_flags: Optional[List[str]] = None,
            dry_run: bool = False, env: Optional[Dict[str, str]] = None) -> Optional[str]:
    """Submit a sbatch job; returns the job ID (or None in dry-run mode)."""
    os.makedirs("logs", exist_ok=True)
    dep = ([f"--dependency=afterok:{':'.join(dependencies)}"] if dependencies else [])
    flags = list(slurm_flags or SLURM_GPU)
    env_prefix = " ".join(f"{k}={v}" for k, v in (env or {}).items())
    wrap = f"{CONDA_INIT} && cd {os.getcwd()} && {env_prefix} {body}".strip()
    cmd = [
        "sbatch", *flags, *dep,
        f"--job-name={name}",
        f"--output=logs/{log_prefix}_%j.out",
        f"--error=logs/{log_prefix}_%j.err",
        f"--wrap={wrap}",
    ]
    if dry_run:
        print(" ".join(cmd))
        return None
    out = subprocess.check_output(cmd, text=True).strip()
    # sbatch prints 'Submitted batch job <ID>'
    return out.split()[-1]


# ── Stage A: HP search ─────────────────────────────────────────────────────────
def _hp_run_already_done(spec: DGPSpec, arch: str, hp: Dict[str, Any],
                          fold: int, seed: int,
                          dataset_id: Optional[int] = None) -> bool:
    run_dir = _hp_run_dir(spec, arch, hp, fold, seed, dataset_id=dataset_id)
    cfg = os.path.join(run_dir, "config.json")
    if not os.path.exists(cfg):
        return False
    try:
        return "cv_val_loss" in json.load(open(cfg))
    except Exception:
        return False


def stage_a(dgp: str, arch: str, *, dry_run: bool = False,
            dataset_id: Optional[int] = None,
            force: bool = False) -> List[str]:
    """Submit one job per (combo, fold, seed) that's missing cv_val_loss."""
    spec = REGISTRY[dgp]
    if spec.synth_dataset_ids and dataset_id is None:
        # Recurse over all configured synthetic datasets
        job_ids = []
        for did in spec.synth_dataset_ids:
            job_ids.extend(stage_a(dgp, arch, dry_run=dry_run,
                                    dataset_id=did, force=force))
        return job_ids

    job_ids: List[str] = []
    fixed = spec.fixed_flags[arch]

    for hp in spec.combos(arch):
        for fold in range(N_OUTER_FOLDS):
            for seed in spec.inner_seeds:
                if not force and _hp_run_already_done(spec, arch, hp, fold,
                                                       seed, dataset_id=dataset_id):
                    continue
                run_dir = _hp_run_dir(spec, arch, hp, fold, seed,
                                       dataset_id=dataset_id)
                run_args = {
                    "seed": seed, "fold": fold, **fixed,
                    "dataset_id": dataset_id if dataset_id is not None else 0,
                }
                # CLI flags for the HP combo; --z stays --z (alias for z_dim)
                run_args.update(hp)
                body = _build_run_q_model_cmd(run_args)
                tag = combo_tag(hp)
                did_tag = f"d{dataset_id}_" if dataset_id is not None else ""
                jid = _sbatch(
                    name=f"hp_{dgp[:3]}_{arch[:3]}_{did_tag}f{fold}_s{seed}",
                    log_prefix=f"hp_{dgp}_{arch}_{did_tag}{tag}_f{fold}_s{seed}",
                    body=body,
                    env={"HP_RUN_DIR": run_dir},
                    dry_run=dry_run,
                )
                if jid:
                    job_ids.append(jid)
    return job_ids


# ── Stage A (Optuna): adaptive HP search via TPE ──────────────────────────────
def stage_a_optuna(dgp: str, arch: str, *, dry_run: bool = False,
                    dataset_id: Optional[int] = None,
                    n_trials_total: Optional[int] = None,
                    n_workers: Optional[int] = None,
                    hypothesis_z: Optional[int] = None,
                    metric: str = "cv_val_loss") -> List[str]:
    """Submit `n_workers` SLURM jobs per (dataset, fold) that each run
    `n_trials_total / n_workers` Optuna trials.  All workers in a cell share
    one SQLite study (Optuna handles concurrency).

    `n_trials_total=None` (default) defers to nested_cv.config.trial_budget_for
    which uses 80 trials for IDRNN and 6 trials for vanilla — vanilla's
    cartesian space is only 3 combos so a smaller budget suffices.

    `hypothesis_z` anchors z to a specific value (skipping the z dimension in
    the HP search).  Output paths get an `_z{N}` suffix so the anchored sweep
    doesn't clash with the free-z one.
    """
    spec = REGISTRY[dgp]
    if spec.synth_dataset_ids and dataset_id is None:
        job_ids = []
        for did in spec.synth_dataset_ids:
            job_ids.extend(stage_a_optuna(dgp, arch, dry_run=dry_run,
                                           dataset_id=did,
                                           n_trials_total=n_trials_total,
                                           n_workers=n_workers,
                                           hypothesis_z=hypothesis_z,
                                           metric=metric))
        return job_ids

    if n_trials_total is None:
        # When z is anchored, the search space shrinks by the z dimension —
        # cap the budget proportionally so we don't oversample what's left.
        base = trial_budget_for(spec, arch)
        if hypothesis_z is not None and arch == "idrnn":
            z_choices = len(spec.hp_grid[arch].get("z", [1]))
            base = max(6, base // max(1, z_choices))
        n_trials_total = base
    if n_workers is None:
        n_workers = workers_for(arch)
    # If budget is smaller than the requested worker count, cap workers so we
    # don't submit empty jobs.
    effective_workers = min(n_workers, n_trials_total)
    n_per_worker = max(1, n_trials_total // effective_workers)
    job_ids: List[str] = []

    ptag = path_tag(hypothesis_z, metric)
    print(f"  [{dgp}{ptag}/{arch}"
          f"{f'/dataset{dataset_id}' if dataset_id is not None else ''}] "
          f"budget={n_trials_total} trials, workers={effective_workers}, "
          f"per_worker={n_per_worker}, metric={metric}")

    # Pre-create each fold's SQLite study before submitting workers so 16
    # workers don't race on the initial CREATE TABLE.  (Observed bug: 15/16
    # workers died on "table studies already exists" when the DB was created
    # concurrently from scratch.)
    if not dry_run:
        import optuna as _opt
        from nested_cv.optuna_search import study_storage as _ss, study_name as _sn
        _opt.logging.set_verbosity(_opt.logging.WARNING)
        for fold in range(N_OUTER_FOLDS):
            try:
                _opt.create_study(
                    study_name=_sn(dgp, arch, fold, dataset_id=dataset_id,
                                    hypothesis_z=hypothesis_z, objective=metric),
                    storage=_ss(dgp, arch, fold, dataset_id=dataset_id,
                                 hypothesis_z=hypothesis_z, objective=metric),
                    direction="minimize",
                    load_if_exists=True,
                )
            except Exception:
                pass

    metric_short = "spec_" if metric == "step1_specificity" else ""
    ztag_short = f"z{hypothesis_z}_" if hypothesis_z is not None else ""
    for fold in range(N_OUTER_FOLDS):
        for w in range(effective_workers):
            did_tag = f"d{dataset_id}_" if dataset_id is not None else ""
            sampler_seed = w * 100 + fold
            body = (f"python -m nested_cv.optuna_worker "
                    f"--dgp {dgp} --arch {arch} --fold {fold} "
                    f"--n_trials {n_per_worker} "
                    f"--sampler_seed {sampler_seed}")
            if dataset_id is not None:
                body += f" --dataset_id {dataset_id}"
            if hypothesis_z is not None:
                body += f" --hypothesis_z {hypothesis_z}"
            if metric != "cv_val_loss":
                body += f" --metric {metric}"
            jid = _sbatch(
                name=f"opt_{dgp[:3]}_{arch[:3]}_{ztag_short}{metric_short}{did_tag}f{fold}_w{w}",
                log_prefix=f"opt_{dgp}{ptag}_{arch}_{did_tag}f{fold}_w{w}",
                body=body,
                dry_run=dry_run,
            )
            if jid:
                job_ids.append(jid)
    return job_ids


# ── Stage C: final per-fold training with winners ──────────────────────────────
def _load_fold_winner(dgp: str, arch: str, fold: int,
                       dataset_id: Optional[int] = None,
                       hypothesis_z: Optional[int] = None,
                       metric: str = "cv_val_loss") -> Dict[str, Any]:
    """Read winner JSON.  Sweeps live under nested_cv_winners/{dgp}{path_tag}/."""
    base = os.path.join("nested_cv_winners", f"{dgp}{path_tag(hypothesis_z, metric)}")
    if dataset_id is not None:
        base = os.path.join(base, f"dataset{dataset_id}")
    base = os.path.join(base, arch)
    p = os.path.join(base, f"fold{fold}.json")
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"Missing per-fold winner JSON: {p}.  Run stage_b first "
            f"(python -m nested_cv.select_hp ...)")
    winner = json.load(open(p))["winner"]
    # When z was anchored at search time it's missing from `winner.params`;
    # add it back so the downstream `--z N` CLI arg is correct.
    if hypothesis_z is not None and "z" not in winner:
        winner = dict(winner); winner["z"] = hypothesis_z
    return winner


def stage_c(dgp: str, arch: str, *, dry_run: bool = False,
            dataset_id: Optional[int] = None,
            n_seeds: Optional[int] = None,
            hypothesis_z: Optional[int] = None,
            metric: str = "cv_val_loss") -> List[str]:
    """Submit final per-fold runs using each fold's winning HPs.

    Output paths include suffixes for hypothesis_z and metric so different
    selection criteria don't clash.
    """
    spec = REGISTRY[dgp]
    if spec.synth_dataset_ids and dataset_id is None:
        job_ids = []
        for did in spec.synth_dataset_ids:
            job_ids.extend(stage_c(dgp, arch, dry_run=dry_run,
                                    dataset_id=did, n_seeds=n_seeds,
                                    hypothesis_z=hypothesis_z,
                                    metric=metric))
        return job_ids

    seeds = spec.final_seeds if n_seeds is None else spec.final_seeds[:n_seeds]
    fixed = spec.fixed_flags[arch]
    run_suffix = f"nested_cv{path_tag(hypothesis_z, metric)}"

    job_ids: List[str] = []
    for fold in range(N_OUTER_FOLDS):
        hp = _load_fold_winner(dgp, arch, fold, dataset_id=dataset_id,
                                hypothesis_z=hypothesis_z,
                                metric=metric)
        for seed in seeds:
            run_args = {
                "seed": seed, "fold": fold, **fixed,
                "dataset_id": dataset_id if dataset_id is not None else 0,
                "run_suffix": run_suffix,
            }
            run_args.update(hp)
            body = _build_run_q_model_cmd(run_args)
            did_tag = f"d{dataset_id}_" if dataset_id is not None else ""
            jid = _sbatch(
                name=f"nc_{dgp[:3]}_{arch[:3]}_{did_tag}f{fold}_s{seed}",
                log_prefix=f"nc_{dgp}_{arch}_{did_tag}f{fold}_s{seed}",
                body=body,
                dry_run=dry_run,
            )
            if jid:
                job_ids.append(jid)
    return job_ids


# ── Stage D: testing + outer-CV analysis ──────────────────────────────────────
def stage_d(dgp: str, *, dry_run: bool = False,
            dataset_id: Optional[int] = None,
            depend_on: Optional[List[str]] = None) -> List[str]:
    spec = REGISTRY[dgp]
    if spec.synth_dataset_ids and dataset_id is None:
        job_ids = []
        for did in spec.synth_dataset_ids:
            job_ids.extend(stage_d(dgp, dry_run=dry_run, dataset_id=did,
                                    depend_on=depend_on))
        return job_ids

    out_dir = os.path.join("final_plots", dgp)
    if dataset_id is not None:
        out_dir = os.path.join(out_dir, f"dataset{dataset_id}")
    os.makedirs(out_dir, exist_ok=True)

    test_ids = []
    for fold in range(N_OUTER_FOLDS):
        for arch in spec.archs:
            latent = "True" if arch == "idrnn" else "False"
            ds_id = dataset_id if dataset_id is not None else 0
            ds_flag = ("" if spec.synth_dataset_ids is None
                       else f"--dataset_id {ds_id}")
            body = (f"python testing_script.py --latent {latent} "
                    f"--dgp {dgp} --fold {fold} --model_fitting False "
                    f"--run_suffix nested_cv {ds_flag}")
            did_tag = f"d{ds_id}_" if dataset_id is not None else ""
            jid = _sbatch(
                name=f"nc_test_{dgp[:3]}_{arch[:3]}_{did_tag}f{fold}",
                log_prefix=f"nc_test_{dgp}_{arch}_{did_tag}f{fold}",
                body=body,
                dependencies=depend_on,
                dry_run=dry_run,
            )
            if jid:
                test_ids.append(jid)

    # Analysis: run analyze_outer_cv.py with run_suffix=nested_cv
    seeds_csv = ",".join(str(s) for s in spec.final_seeds[:30])
    body = (f"mkdir -p {out_dir} && "
            f"python analyze_outer_cv.py --dgp {dgp} --run_suffix nested_cv "
            f"--top_k_by_specificity 5 --seeds {seeds_csv}")
    jid = _sbatch(
        name=f"nc_an_{dgp[:3]}",
        log_prefix=f"nc_an_{dgp}",
        body=body,
        dependencies=test_ids if test_ids else None,
        slurm_flags=SLURM_CPU,
        dry_run=dry_run,
    )
    return ([j for j in test_ids if j] + ([jid] if jid else []))


# ── Stage F: canonical retrain on full training data ──────────────────────────
def stage_f(dgp: str, arch: str, *, dry_run: bool = False,
            dataset_id: Optional[int] = None,
            hypothesis_z: Optional[int] = None,
            metric: str = "cv_val_loss") -> List[str]:
    spec = REGISTRY[dgp]
    if spec.synth_dataset_ids and dataset_id is None:
        job_ids = []
        for did in spec.synth_dataset_ids:
            job_ids.extend(stage_f(dgp, arch, dry_run=dry_run, dataset_id=did,
                                    hypothesis_z=hypothesis_z,
                                    metric=metric))
        return job_ids

    spec_dir = os.path.join("final_plots", f"{dgp}{path_tag(hypothesis_z, metric)}")
    if dataset_id is not None:
        spec_dir = os.path.join(spec_dir, f"dataset{dataset_id}")
    spec_dir = os.path.join(spec_dir, "canonical")
    spec_path = os.path.join(spec_dir, f"{arch}_spec.json")
    if not os.path.exists(spec_path):
        raise FileNotFoundError(
            f"Missing canonical spec: {spec_path}. "
            f"Run stage_e first (python -m nested_cv.select_canonical --mode spec ...)")
    payload = json.load(open(spec_path))
    hp = payload["hp"]
    seeds = payload["seeds"]

    fixed = spec.fixed_flags[arch]
    runs_dir = payload["canonical_runs_dir"]

    job_ids: List[str] = []
    for seed in seeds:
        run_args = {
            "seed": seed, **fixed,
            "dataset_id": dataset_id if dataset_id is not None else 0,
        }
        run_args.update(hp)
        body = _build_run_q_model_cmd(run_args)
        did_tag = f"d{dataset_id}_" if dataset_id is not None else ""
        jid = _sbatch(
            name=f"canon_{dgp[:3]}_{arch[:3]}_{did_tag}s{seed}",
            log_prefix=f"canon_{dgp}_{arch}_{did_tag}s{seed}",
            body=body,
            env={"HP_RUN_DIR": os.path.join(runs_dir, f"seed_{seed}")},
            dry_run=dry_run,
        )
        if jid:
            job_ids.append(jid)
    return job_ids


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["stage_a", "stage_a_optuna",
                                        "stage_c", "stage_d", "stage_f"])
    ap.add_argument("--dgp", required=True, choices=list(REGISTRY.keys()))
    ap.add_argument("--arch", choices=["idrnn", "vanilla", "both"], default="both")
    ap.add_argument("--dataset_id", type=int, default=None)
    ap.add_argument("--n_seeds", type=int, default=None,
                    help="Override seed count for stage_c (default: all final seeds).")
    ap.add_argument("--n_trials", type=int, default=None,
                    help="Per-fold trial budget override.  If unset, uses "
                          "config.trial_budget_for(spec, arch): 80 IDRNN / 6 vanilla.")
    ap.add_argument("--n_workers", type=int, default=None,
                    help="Per-fold worker count override.  If unset, uses "
                         "config.workers_for(arch): 16 IDRNN / 2 vanilla.")
    ap.add_argument("--hypothesis_z", type=int, default=None,
                    help="Anchor IDRNN z to this value (e.g. the hypothesised "
                          "individual-difference dimensionality).  All outputs "
                          "get an _z{N} suffix so they don't collide with the "
                          "free-z sweep.  Affects stages a_optuna, c, f.")
    ap.add_argument("--metric", default="cv_val_loss",
                    choices=list(OPTUNA_OBJECTIVES),
                    help="HP-selection objective.  cv_val_loss (default) "
                          "minimises NLL+KL.  step1_specificity maximises "
                          "lookup-encoder specificity (correlates with α-"
                          "decoding at z=1).  Outputs get a _spec suffix.")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="Re-submit even if HP-search cv_val_loss is already on disk.")
    args = ap.parse_args()

    archs = ["idrnn", "vanilla"] if args.arch == "both" else [args.arch]
    all_ids = []

    for arch in archs:
        if args.stage == "stage_a":
            ids = stage_a(args.dgp, arch, dry_run=args.dry_run,
                          dataset_id=args.dataset_id, force=args.force)
        elif args.stage == "stage_a_optuna":
            ids = stage_a_optuna(args.dgp, arch, dry_run=args.dry_run,
                                  dataset_id=args.dataset_id,
                                  n_trials_total=args.n_trials,
                                  n_workers=args.n_workers,
                                  hypothesis_z=args.hypothesis_z,
                                  metric=args.metric)
        elif args.stage == "stage_c":
            ids = stage_c(args.dgp, arch, dry_run=args.dry_run,
                          dataset_id=args.dataset_id, n_seeds=args.n_seeds,
                          hypothesis_z=args.hypothesis_z,
                          metric=args.metric)
        elif args.stage == "stage_f":
            ids = stage_f(args.dgp, arch, dry_run=args.dry_run,
                          dataset_id=args.dataset_id,
                          hypothesis_z=args.hypothesis_z,
                          metric=args.metric)
        else:
            continue
        all_ids.extend(ids)
        print(f"[{arch}] submitted {len(ids)} jobs")

    if args.stage == "stage_d":
        # stage_d is arch-agnostic (the testing script handles both archs internally)
        ids = stage_d(args.dgp, dry_run=args.dry_run, dataset_id=args.dataset_id)
        all_ids.extend(ids)
        print(f"submitted {len(ids)} test+analysis jobs")

    if all_ids:
        print("Job IDs:", " ".join(all_ids))


if __name__ == "__main__":
    main()
