"""Per-fold HP winner selection from inner-CV val loss.

For each (DGP, arch, fold) tuple:
  - scan every HP combo's inner-search runs
  - read cv_val_loss from each combo×seed config.json
  - aggregate (mean across seeds) per combo
  - pick the combo with the lowest mean cv_val_loss → that is the winner for this fold

Output: nested_cv_winners/{dgp}[/dataset{ID}]/{arch}/fold{F}.json with the chosen
HP combo, its mean cv_val_loss, its per-seed values, and a summary of the
top-K runners-up so we can sanity-check tie-breaks later.
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nested_cv.config import (
    DGPSpec, REGISTRY, combo_tag, thalmann_v3_tag, N_OUTER_FOLDS, z_tag,
    path_tag, OPTUNA_OBJECTIVES,
)
from nested_cv.optuna_search import study_name, study_storage


def _hp_run_dir(spec: DGPSpec, arch: str, hp: Dict[str, Any], fold: int,
                seed: int, dataset_id: Optional[int] = None) -> str:
    """Filesystem path where (combo × fold × seed)'s config.json should live.

    For thalmann IDRNN this resolves to the existing hp_v3 directory layout so
    we can read the already-trained sweep directly.  Everything else lands
    under hp_search_runs_{dgp}_nested_cv/.
    """
    if spec.name == "thalmann" and arch == "idrnn":
        tag = thalmann_v3_tag(hp)
        return os.path.join(f"runs_thalmann_hp_v3_{tag}",
                            f"fold{fold}", f"seed_{seed}")
    if spec.name == "synthetic":
        return os.path.join(
            f"hp_search_runs_synthetic_dataset{dataset_id}",
            combo_tag(hp), f"fold{fold}", f"seed_{seed}",
        )
    # default: hp_search_runs_{dgp}_{arch}/{combo}/fold{F}/seed_S/
    return os.path.join(
        f"hp_search_runs_{spec.name}_{arch}",
        combo_tag(hp), f"fold{fold}", f"seed_{seed}",
    )


def _read_cv_val_loss(run_dir: str) -> Optional[float]:
    cfg_p = os.path.join(run_dir, "config.json")
    if not os.path.exists(cfg_p):
        return None
    try:
        with open(cfg_p) as f:
            cfg = json.load(f)
        v = cfg.get("cv_val_loss")
        return float(v) if v is not None else None
    except Exception:
        return None


def winners_for_fold_optuna(spec: DGPSpec, arch: str, fold: int,
                              dataset_id: Optional[int] = None,
                              hypothesis_z: Optional[int] = None,
                              metric: str = "cv_val_loss",
                              verbose: bool = True) -> Optional[Dict[str, Any]]:
    """Pull the per-fold winner from the Optuna SQLite study, if one exists.

    Returns None if no study is present (caller should fall back to grid scan).
    """
    import optuna
    storage = study_storage(spec.name, arch, fold, dataset_id=dataset_id,
                             hypothesis_z=hypothesis_z, objective=metric)
    # SQLite URI -> path
    db_path = storage.replace("sqlite:///", "")
    if not os.path.exists(db_path):
        return None
    try:
        st = optuna.load_study(
            study_name=study_name(spec.name, arch, fold, dataset_id=dataset_id,
                                   hypothesis_z=hypothesis_z, objective=metric),
            storage=storage)
    except Exception as e:
        if verbose:
            print(f"  [optuna] load_study failed for fold{fold}: {e}")
        return None

    completed = [t for t in st.get_trials(deepcopy=False)
                 if t.state == optuna.trial.TrialState.COMPLETE
                    and t.value is not None]
    if not completed:
        return None
    completed.sort(key=lambda t: t.value)
    best = completed[0]
    top_k = []
    for rank, t in enumerate(completed[:5]):
        top_k.append({
            "rank": rank + 1,
            "trial_number": t.number,
            "hp": dict(t.params),
            "cv_val_loss": float(t.value),
            "user_attrs": dict(t.user_attrs),
        })

    # For step1_specificity metric Optuna minimised -spec; flip back to report.
    best_objective_signed = float(best.value)
    best_objective = (-best_objective_signed if metric == "step1_specificity"
                       else best_objective_signed)

    if verbose:
        tags = []
        if hypothesis_z is not None: tags.append(f"z={hypothesis_z}")
        if metric != "cv_val_loss":   tags.append(f"metric={metric}")
        tag_str = f" ({', '.join(tags)})" if tags else ""
        print(f"  [{spec.name}{path_tag(hypothesis_z, metric)}/{arch}/fold{fold}{tag_str}] "
              f"optuna winner trial={best.number}  "
              f"hp={best.params}  {metric}={best_objective:.4f}  "
              f"(n_trials_completed={len(completed)})")
    # When z is anchored at search time, it's not in best.params; merge it back
    # so downstream code (combo_tag, run_Q_model CLI) sees the full HP dict.
    winner = dict(best.params)
    if hypothesis_z is not None and "z" not in winner:
        winner["z"] = hypothesis_z
    return {
        "dgp":     spec.name,
        "arch":    arch,
        "fold":    fold,
        "dataset_id":   dataset_id,
        "hypothesis_z": hypothesis_z,
        "metric":  metric,
        "source":  "optuna",
        "winner":  winner,
        "winner_combo_tag": combo_tag(winner),
        # Keep "winner_cv_val_loss" name for back-compat — actually holds the
        # selection-metric value (signed so higher-is-better when applicable).
        "winner_cv_val_loss":     best_objective if metric == "cv_val_loss" else None,
        "winner_metric_value":    best_objective,
        "winner_trial_number":    best.number,
        "winner_user_attrs":      dict(best.user_attrs),
        "top_5":                  top_k,
        "n_trials_completed":     len(completed),
        "n_trials_total":         len(st.get_trials(deepcopy=False)),
    }


def winners_for_fold(spec: DGPSpec, arch: str, fold: int,
                     dataset_id: Optional[int] = None,
                     verbose: bool = True) -> Dict[str, Any]:
    """Return the winning HP combo dict + diagnostics for a single fold."""
    combos = spec.combos(arch)
    seeds  = spec.inner_seeds
    rows: List[Tuple[Dict[str, Any], List[float]]] = []

    for hp in combos:
        losses = []
        for s in seeds:
            v = _read_cv_val_loss(
                _hp_run_dir(spec, arch, hp, fold, s, dataset_id=dataset_id))
            if v is not None:
                losses.append(v)
        if losses:
            rows.append((hp, losses))

    if not rows:
        return {"error": "no completed HP runs", "fold": fold,
                "n_combos_evaluated": 0}

    means = np.array([np.mean(losses) for _, losses in rows])
    order = np.argsort(means)

    top_k = []
    for rank, idx in enumerate(order[:5]):
        hp, losses = rows[idx]
        top_k.append({
            "rank": rank + 1,
            "hp": hp,
            "mean_cv_val_loss": float(np.mean(losses)),
            "std_cv_val_loss":  float(np.std(losses)),
            "per_seed":         [float(x) for x in losses],
            "n_seeds":          len(losses),
        })

    winner_hp, winner_losses = rows[order[0]]
    if verbose:
        print(f"  [{spec.name}/{arch}/fold{fold}] "
              f"winner={combo_tag(winner_hp)}  "
              f"mean_cv_val_loss={np.mean(winner_losses):.4f}  "
              f"(n_combos_evaluated={len(rows)})")
    return {
        "dgp":     spec.name,
        "arch":    arch,
        "fold":    fold,
        "dataset_id": dataset_id,
        "winner":  winner_hp,
        "winner_combo_tag":  combo_tag(winner_hp),
        "winner_mean_cv_val_loss": float(np.mean(winner_losses)),
        "winner_per_seed":   [float(x) for x in winner_losses],
        "top_5":              top_k,
        "n_combos_evaluated": len(rows),
        "n_combos_total":     len(combos),
    }


def write_winners(spec: DGPSpec, arch: str,
                  dataset_id: Optional[int] = None,
                  hypothesis_z: Optional[int] = None,
                  metric: str = "cv_val_loss",
                  out_base: str = "nested_cv_winners") -> Dict[int, Dict[str, Any]]:
    """Write per-fold winners.  Anchored/specificity sweeps land under
    nested_cv_winners/{dgp}{path_tag}/... so different selection criteria
    don't clash."""
    out_dir = os.path.join(out_base, f"{spec.name}{path_tag(hypothesis_z, metric)}")
    if dataset_id is not None:
        out_dir = os.path.join(out_dir, f"dataset{dataset_id}")
    out_dir = os.path.join(out_dir, arch)
    os.makedirs(out_dir, exist_ok=True)

    all_winners = {}
    for fold in range(N_OUTER_FOLDS):
        # Prefer Optuna study results if present; fall back to grid scan.
        rec = winners_for_fold_optuna(spec, arch, fold, dataset_id=dataset_id,
                                        hypothesis_z=hypothesis_z,
                                        metric=metric)
        if rec is None:
            rec = winners_for_fold(spec, arch, fold, dataset_id=dataset_id)
        path = os.path.join(out_dir, f"fold{fold}.json")
        with open(path, "w") as f:
            json.dump(rec, f, indent=2)
        all_winners[fold] = rec
    return all_winners


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dgp", required=True, choices=list(REGISTRY.keys()))
    ap.add_argument("--arch", required=True, choices=["idrnn", "vanilla", "both"])
    ap.add_argument("--dataset_id", type=int, default=None,
                    help="Only used for --dgp synthetic.")
    ap.add_argument("--hypothesis_z", type=int, default=None,
                    help="If set, read winners from the z-anchored Optuna study "
                          "(matches the stage_a_optuna --hypothesis_z flag).")
    ap.add_argument("--metric", default="cv_val_loss",
                    choices=list(OPTUNA_OBJECTIVES),
                    help="HP-selection metric (cv_val_loss or step1_specificity).")
    args = ap.parse_args()

    spec = REGISTRY[args.dgp]
    archs = ["idrnn", "vanilla"] if args.arch == "both" else [args.arch]

    tag_str = ""
    if args.hypothesis_z is not None: tag_str += f" (z={args.hypothesis_z})"
    if args.metric != "cv_val_loss":  tag_str += f" (metric={args.metric})"

    if spec.synth_dataset_ids is not None and args.dataset_id is None:
        # Run all configured synthetic datasets
        for did in spec.synth_dataset_ids:
            print(f"\n=== synthetic dataset {did}{tag_str} ===")
            for arch in archs:
                write_winners(spec, arch, dataset_id=did,
                               hypothesis_z=args.hypothesis_z,
                               metric=args.metric)
    else:
        for arch in archs:
            write_winners(spec, arch, dataset_id=args.dataset_id,
                           hypothesis_z=args.hypothesis_z,
                           metric=args.metric)


if __name__ == "__main__":
    main()
