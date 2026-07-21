"""Optuna-based HP search for nested CV.

A "trial" trains one (arch, fold, dataset, HP sample) with run_Q_model.py and
reports cv_val_loss (from the resulting config.json) as the objective value.

Each (DGP, arch, dataset_id, fold) gets its own Optuna study, backed by a
SQLite file under optuna_studies/.  SLURM workers share that file via
storage="sqlite:///path" + load_if_exists=True so trials run in parallel.

This module provides:
  - study_storage()         → SQLite URI per study cell
  - study_name()            → human-readable study name
  - make_objective()        → callable usable with study.optimize()
  - run_trials()            → one worker's entry point (used by optuna_worker.py)

Trials write their training output under
  optuna_runs/{dgp}[/dataset{ID}]/{arch}/fold{F}/trial_{N}/
so multiple parallel workers don't clobber each other.  The trial's config.json
contains all sampled HPs + cv_val_loss + step1_specificity.
"""

from __future__ import annotations
import json
import os
import subprocess
import sys
from typing import Any, Dict, Optional

import optuna

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nested_cv.config import (
    REGISTRY, DGPSpec, effective_hp_grid, z_tag, path_tag, OPTUNA_OBJECTIVES,
)


# ── Paths ─────────────────────────────────────────────────────────────────────
def _study_dir(dgp: str, dataset_id: Optional[int],
                hypothesis_z: Optional[int] = None,
                objective: Optional[str] = None) -> str:
    out = f"optuna_studies/{dgp}{path_tag(hypothesis_z, objective)}"
    if dataset_id is not None:
        out = f"{out}/dataset{dataset_id}"
    return out


def study_storage(dgp: str, arch: str, fold: int,
                  dataset_id: Optional[int] = None,
                  hypothesis_z: Optional[int] = None,
                  objective: Optional[str] = None) -> str:
    d = _study_dir(dgp, dataset_id, hypothesis_z=hypothesis_z, objective=objective)
    os.makedirs(d, exist_ok=True)
    return f"sqlite:///{d}/{arch}_fold{fold}.db"


def study_name(dgp: str, arch: str, fold: int,
                dataset_id: Optional[int] = None,
                hypothesis_z: Optional[int] = None,
                objective: Optional[str] = None) -> str:
    suffix = f"_d{dataset_id}" if dataset_id is not None else ""
    return f"{dgp}{path_tag(hypothesis_z, objective)}_{arch}_fold{fold}{suffix}"


def _trial_run_dir(dgp: str, arch: str, fold: int, trial_number: int,
                    dataset_id: Optional[int] = None,
                    hypothesis_z: Optional[int] = None,
                    objective: Optional[str] = None) -> str:
    base = f"optuna_runs/{dgp}{path_tag(hypothesis_z, objective)}"
    if dataset_id is not None:
        base = f"{base}/dataset{dataset_id}"
    return f"{base}/{arch}/fold{fold}/trial_{trial_number}"


# ── Objective ─────────────────────────────────────────────────────────────────
def _flag(k: str, v: Any) -> list:
    if isinstance(v, bool):
        v = "True" if v else "False"
    return [f"--{k}", str(v)]


def _build_cmd(spec: DGPSpec, arch: str, fold: int, hp: Dict[str, Any],
                dataset_id: Optional[int], seed: int) -> list:
    fixed = spec.fixed_flags[arch]
    args: Dict[str, Any] = {"seed": seed, "fold": fold}
    if dataset_id is not None:
        args["dataset_id"] = dataset_id
    else:
        args["dataset_id"] = 0
    args.update(fixed)
    args.update(hp)
    cmd = ["python", "run_Q_model.py"]
    for k, v in args.items():
        cmd.extend(_flag(k, v))
    return cmd


def make_objective(dgp: str, arch: str, fold: int,
                   dataset_id: Optional[int] = None,
                   hypothesis_z: Optional[int] = None,
                   metric: str = "cv_val_loss",
                   subprocess_timeout: int = 5400,
                   trial_seed: int = 42):
    """Returns an Optuna objective function.

    The objective:
      1. samples HPs from the grid in spec.hp_grid[arch] (with z anchored if
         hypothesis_z is set)
      2. trains one (fold, seed) via run_Q_model.py as a subprocess
      3. reads `metric` from the produced config.json
      4. returns it (negated for maximisation metrics) as the value to minimise

    Supported metrics:
      cv_val_loss        — inner-CV NLL (lower is better; default)
      step1_specificity  — lookup-encoder specificity (higher is better; we
                            shown to correlate with α-decoding R² in the z=1
                            anchored regime).  The objective returns
                            -step1_specificity so Optuna's minimisation
                            converges on high-specificity HPs.

    When hypothesis_z is set, z is fixed at that value (not searched) and the
    objective optimises only the remaining HPs.

    A failure (e.g. CUDA OOM, NaN loss) → optuna.TrialPruned so other trials
    proceed; the failed trial is recorded but not counted toward best.
    """
    assert metric in OPTUNA_OBJECTIVES, f"metric must be one of {OPTUNA_OBJECTIVES}"
    spec = REGISTRY[dgp]
    grid = effective_hp_grid(spec, arch, hypothesis_z=hypothesis_z)

    def objective(trial: optuna.Trial) -> float:
        # Sample HPs from the (possibly z-anchored) grid.
        hp: Dict[str, Any] = {}
        for k, choices in grid.items():
            hp[k] = trial.suggest_categorical(k, choices)

        # Dedup: if any prior COMPLETED trial in this study has the same HPs,
        # reuse its objective value instead of re-training.  This is critical
        # for vanilla (only 3 unique combos) and saves ~half the budget in
        # IDRNN once TPE concentrates on a region.
        for prior in trial.study.get_trials(deepcopy=False, states=(
                optuna.trial.TrialState.COMPLETE,)):
            if prior.number == trial.number:
                continue
            if prior.params == hp:
                trial.set_user_attr("dedup_from_trial", prior.number)
                for k, v in prior.user_attrs.items():
                    trial.set_user_attr(k, v)
                return float(prior.value)

        # Each trial gets its own run_dir so parallel trials don't collide.
        run_dir = _trial_run_dir(dgp, arch, fold, trial.number,
                                  dataset_id=dataset_id,
                                  hypothesis_z=hypothesis_z,
                                  objective=metric)
        os.makedirs(run_dir, exist_ok=True)
        cfg_path = os.path.join(run_dir, "config.json")

        # Sign-flip for maximisation metrics so Optuna's minimisation goes the
        # right direction.
        def _objective_value(cfg: Dict[str, Any]) -> Optional[float]:
            v = cfg.get(metric)
            if v is None:
                return None
            v = float(v)
            return -v if metric == "step1_specificity" else v

        # If a prior worker already wrote a config.json (warm restart of a
        # killed trial), we can short-circuit by reading the objective.
        if os.path.exists(cfg_path):
            try:
                cfg = json.load(open(cfg_path))
                v = _objective_value(cfg)
                if v is not None:
                    return v
            except Exception:
                pass

        env = os.environ.copy()
        env["HP_RUN_DIR"] = run_dir
        env["WANDB_MODE"] = "offline"
        cmd = _build_cmd(spec, arch, fold, hp, dataset_id, trial_seed)
        try:
            result = subprocess.run(cmd, env=env,
                                     capture_output=True, text=True,
                                     timeout=subprocess_timeout)
        except subprocess.TimeoutExpired:
            raise optuna.TrialPruned(f"trial {trial.number} timed out")

        if result.returncode != 0:
            # Surface the tail of stderr for debugging
            tail = (result.stderr or "")[-800:]
            raise optuna.TrialPruned(
                f"trial {trial.number} failed (rc={result.returncode}):\n{tail}")

        if not os.path.exists(cfg_path):
            raise optuna.TrialPruned(f"trial {trial.number}: no config.json")
        try:
            cfg = json.load(open(cfg_path))
        except Exception as e:
            raise optuna.TrialPruned(f"trial {trial.number}: bad config.json — {e}")
        v = _objective_value(cfg)
        if v is None:
            raise optuna.TrialPruned(
                f"trial {trial.number}: no {metric} in config")
        # Record metadata Optuna doesn't track natively so later analyses
        # can compare the alternative metric without re-loading checkpoints.
        for k in ("step1_specificity", "cv_val_loss", "cv_selected_epoch", "cv_val_nll"):
            if k in cfg:
                trial.set_user_attr(k, cfg[k])
        return float(v)

    return objective


# ── Worker entrypoint ────────────────────────────────────────────────────────
def run_trials(dgp: str, arch: str, fold: int,
               n_trials: int,
               dataset_id: Optional[int] = None,
               hypothesis_z: Optional[int] = None,
               metric: str = "cv_val_loss",
               sampler_seed: int = 0,
               trial_seed: int = 42) -> None:
    """One SLURM worker.  Runs `n_trials` trials of the (DGP, arch, fold) study.

    All workers share the SQLite study (load_if_exists=True).  Pre-existing
    completed trials are not re-run; new trials are sampled by Optuna's TPE
    based on what other workers have completed so far.

    With hypothesis_z set, the study is keyed by z; with metric != "cv_val_loss"
    the study is keyed by metric — so this worker contributes only to the
    matching sweep and doesn't touch the others.
    """
    storage = study_storage(dgp, arch, fold, dataset_id=dataset_id,
                             hypothesis_z=hypothesis_z, objective=metric)
    name    = study_name(dgp, arch, fold, dataset_id=dataset_id,
                          hypothesis_z=hypothesis_z, objective=metric)
    sampler = optuna.samplers.TPESampler(seed=sampler_seed,
                                          multivariate=True,
                                          group=True,
                                          constant_liar=True)
    study = optuna.create_study(
        study_name=name, storage=storage, direction="minimize",
        sampler=sampler, load_if_exists=True,
    )
    objective = make_objective(dgp, arch, fold,
                                dataset_id=dataset_id,
                                hypothesis_z=hypothesis_z,
                                metric=metric,
                                trial_seed=trial_seed)
    study.optimize(objective, n_trials=n_trials, gc_after_trial=True,
                    catch=(Exception,))
