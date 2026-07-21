"""Canonical model selection: modal HP combo across folds → retrain spec.

For each (DGP, arch) pair we:
  1. Read the per-fold winners (output of select_hp.py).
  2. Compute the "modal" winning HP combo:
        - if one combo wins ≥ 2 / 3 folds, use it
        - otherwise: pick the combo that minimises mean cv_val_loss across all
          folds (re-evaluating every combo as a fall-back tie-breaker)
  3. Emit the canonical-retrain specification (HPs + seed list) to a JSON file
     that submit_nested_cv.sh consumes when launching the final stage.

After the canonical-retrain SLURM jobs finish, run `pick_canonical_seed` from
this module (or its CLI) to pick the top-1 seed by step1_specificity and write
final_plots/{dgp}/canonical_choice.json.
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nested_cv.config import (
    REGISTRY, DGPSpec, combo_tag, FINAL_SEEDS, N_OUTER_FOLDS, thalmann_v3_tag,
    z_tag, path_tag, OPTUNA_OBJECTIVES,
)
from nested_cv.select_hp import winners_for_fold


def _winner_dir(dgp: str, arch: str, dataset_id: Optional[int] = None,
                hypothesis_z: Optional[int] = None,
                metric: str = "cv_val_loss",
                base: str = "nested_cv_winners") -> str:
    out = os.path.join(base, f"{dgp}{path_tag(hypothesis_z, metric)}")
    if dataset_id is not None:
        out = os.path.join(out, f"dataset{dataset_id}")
    return os.path.join(out, arch)


def load_fold_winners(dgp: str, arch: str,
                      dataset_id: Optional[int] = None,
                      hypothesis_z: Optional[int] = None,
                      metric: str = "cv_val_loss") -> List[Dict[str, Any]]:
    d = _winner_dir(dgp, arch, dataset_id, hypothesis_z=hypothesis_z, metric=metric)
    rows = []
    for fold in range(N_OUTER_FOLDS):
        p = os.path.join(d, f"fold{fold}.json")
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Missing per-fold winner: {p}. "
                f"Run `python -m nested_cv.select_hp --dgp {dgp} --arch {arch} ...` first."
            )
        rows.append(json.load(open(p)))
    return rows


def modal_hp(fold_winners: List[Dict[str, Any]],
             spec: DGPSpec, arch: str,
             dataset_id: Optional[int] = None) -> Tuple[Dict[str, Any], str]:
    """Pick the canonical HP combo from the per-fold winners.

    Strategy:
      - if exactly one combo wins ≥ 2 / 3 folds, use it (mode)
      - else: across all combos, compute mean cv_val_loss over folds (using each
        fold's recorded winner_mean_cv_val_loss when the combo is the winner,
        else by re-reading inner-CV runs for that combo) and pick the min.

    Returns (hp_dict, reason_str).
    """
    tags = [combo_tag(rec["winner"]) for rec in fold_winners]
    counter = Counter(tags)
    most_common_tag, count = counter.most_common(1)[0]

    if count >= 2:
        for rec in fold_winners:
            if combo_tag(rec["winner"]) == most_common_tag:
                return rec["winner"], f"mode (wins {count}/{N_OUTER_FOLDS} folds)"

    # tie-breaker: cross-fold mean cv_val_loss across all combos
    from nested_cv.select_hp import _hp_run_dir, _read_cv_val_loss
    combos = spec.combos(arch)
    seeds = spec.inner_seeds

    per_combo_means = []
    for hp in combos:
        losses = []
        for fold in range(N_OUTER_FOLDS):
            seed_losses = [
                _read_cv_val_loss(_hp_run_dir(spec, arch, hp, fold, s,
                                              dataset_id=dataset_id))
                for s in seeds
            ]
            seed_losses = [v for v in seed_losses if v is not None]
            if seed_losses:
                losses.append(np.mean(seed_losses))
        if len(losses) == N_OUTER_FOLDS:
            per_combo_means.append((hp, float(np.mean(losses))))

    if not per_combo_means:
        # Last resort: take fold 0's winner
        return fold_winners[0]["winner"], "fallback (fold0 winner)"

    per_combo_means.sort(key=lambda kv: kv[1])
    best_hp, best_mean = per_combo_means[0]
    return best_hp, (f"min cross-fold mean cv_val_loss "
                     f"({best_mean:.4f}; all 3 folds present)")


def write_canonical_spec(dgp: str, arch: str,
                          dataset_id: Optional[int] = None,
                          hypothesis_z: Optional[int] = None,
                          metric: str = "cv_val_loss",
                          n_seeds: int = 30,
                          out_base: str = "final_plots") -> Dict[str, Any]:
    spec = REGISTRY[dgp]
    fold_winners = load_fold_winners(dgp, arch, dataset_id,
                                       hypothesis_z=hypothesis_z,
                                       metric=metric)

    hp, reason = modal_hp(fold_winners, spec, arch, dataset_id)

    dgp_tagged = f"{dgp}{path_tag(hypothesis_z, metric)}"
    out_dir = os.path.join(out_base, dgp_tagged)
    if dataset_id is not None:
        out_dir = os.path.join(out_dir, f"dataset{dataset_id}")
    out_dir = os.path.join(out_dir, "canonical")
    os.makedirs(out_dir, exist_ok=True)

    # When z is anchored at search time it's missing from .params; merge back
    if hypothesis_z is not None and "z" not in hp:
        hp = dict(hp); hp["z"] = hypothesis_z

    seeds = list(spec.final_seeds[:n_seeds])
    # Helper: get cv_val_loss field name (different across optuna/grid winners)
    def _cv(rec):
        return rec.get("winner_mean_cv_val_loss",
                       rec.get("winner_cv_val_loss"))
    payload = {
        "dgp":         dgp,
        "arch":        arch,
        "dataset_id":  dataset_id,
        "hypothesis_z": hypothesis_z,
        "hp":          hp,
        "hp_tag":      combo_tag(hp),
        "selection_reason": reason,
        "per_fold_winner_tags": [combo_tag(r["winner"]) for r in fold_winners],
        "per_fold_winner_cv_val_loss": [_cv(r) for r in fold_winners],
        "seeds":       seeds,
        # Where the canonical run lives on disk (no --fold flag → trained on
        # the full training data of data_{dgp}/).
        "canonical_runs_dir":
            (f"final_plots/{dgp_tagged}/" +
             (f"dataset{dataset_id}/" if dataset_id is not None else "") +
             f"canonical/{arch}/runs"),
    }
    out = os.path.join(out_dir, f"{arch}_spec.json")
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote canonical spec -> {out}")
    print(f"  HPs: {combo_tag(hp)}  ({reason})")
    return payload


def pick_canonical_seed(dgp: str, arch: str,
                        dataset_id: Optional[int] = None,
                        hypothesis_z: Optional[int] = None,
                        metric: str = "cv_val_loss",
                        out_base: str = "final_plots") -> Dict[str, Any]:
    """After canonical-retrain SLURM jobs finish, pick the top-1 seed by
    step1_specificity (IDRNN) or by final NLL (Vanilla) and record the choice."""
    out_dir = os.path.join(out_base, f"{dgp}{path_tag(hypothesis_z, metric)}")
    if dataset_id is not None:
        out_dir = os.path.join(out_dir, f"dataset{dataset_id}")
    out_dir = os.path.join(out_dir, "canonical")
    spec_path = os.path.join(out_dir, f"{arch}_spec.json")
    spec_obj = json.load(open(spec_path))

    runs_dir = spec_obj["canonical_runs_dir"]
    if not os.path.isdir(runs_dir):
        raise FileNotFoundError(f"Expected canonical runs at {runs_dir}/ (not found).")

    scored = []
    used_key = "step1_specificity" if arch == "idrnn" else "cv_val_loss"
    for d in sorted(os.listdir(runs_dir)):
        if not d.startswith("seed_"):
            continue
        cfg_p = os.path.join(runs_dir, d, "config.json")
        if not os.path.exists(cfg_p):
            continue
        cfg = json.load(open(cfg_p))
        primary_key = "step1_specificity" if arch == "idrnn" else "cv_val_loss"
        v = cfg.get(primary_key)
        key_used = primary_key
        if v is None and arch == "vanilla":
            # Non-fold synthetic vanilla doesn't go through palminteri_CV so
            # cv_val_loss is missing; fall back to the min loss recorded by
            # train_ablated_noblocks in loss/epoch_*.npy.
            loss_dir = os.path.join(runs_dir, d, "loss")
            if os.path.isdir(loss_dir):
                import numpy as _np
                losses = []
                for lf in os.listdir(loss_dir):
                    if lf.startswith("epoch_") and lf.endswith(".npy"):
                        try:
                            losses.append(float(_np.load(os.path.join(loss_dir, lf))))
                        except Exception:
                            pass
                if losses:
                    v = min(losses)
                    key_used = "min_train_loss"
                    used_key = "min_train_loss"
        if v is None:
            continue
        # higher specificity = better; lower loss = better
        signed = float(v) if key_used == "step1_specificity" else -float(v)
        scored.append((int(d.split("_")[1]), float(v), signed, d))

    if not scored:
        raise RuntimeError(f"No scored seeds under {runs_dir}/")

    scored.sort(key=lambda t: -t[2])  # descending signed score
    best_seed, best_score, _, best_dir = scored[0]
    payload = dict(spec_obj)
    payload.update({
        "selected_seed":         best_seed,
        "selected_seed_score":   best_score,
        "score_key":             used_key,
        "selected_run_dir":      os.path.join(runs_dir, best_dir),
        "all_seed_scores":       [{"seed": s, "score": v} for s, v, _, _ in scored],
    })
    out = os.path.join(out_dir, f"{arch}_choice.json")
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Canonical {arch} model for {dgp}"
          f"{f' (dataset{dataset_id})' if dataset_id is not None else ''}:")
    print(f"  HPs:  {spec_obj['hp_tag']}")
    print(f"  Seed: {best_seed}  ({payload['score_key']}={best_score:.4f})")
    print(f"  Run:  {payload['selected_run_dir']}")
    print(f"  Choice JSON -> {out}")
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dgp", required=True, choices=list(REGISTRY.keys()))
    ap.add_argument("--arch", required=True, choices=["idrnn", "vanilla", "both"])
    ap.add_argument("--dataset_id", type=int, default=None)
    ap.add_argument("--hypothesis_z", type=int, default=None,
                    help="Use the z-anchored sweep results (read winners from "
                          "nested_cv_winners/{dgp}_z{N}/).")
    ap.add_argument("--metric", default="cv_val_loss",
                    choices=list(OPTUNA_OBJECTIVES),
                    help="Selection metric used during HP search.")
    ap.add_argument("--n_seeds", type=int, default=30)
    ap.add_argument("--mode", choices=["spec", "pick"], default="spec",
                    help="'spec' writes the retrain spec; 'pick' selects the "
                         "best seed after canonical-retrain jobs complete.")
    args = ap.parse_args()

    spec = REGISTRY[args.dgp]
    archs = ["idrnn", "vanilla"] if args.arch == "both" else [args.arch]
    dataset_ids = (spec.synth_dataset_ids if spec.synth_dataset_ids and args.dataset_id is None
                   else [args.dataset_id])

    for did in dataset_ids:
        for arch in archs:
            if args.mode == "spec":
                write_canonical_spec(args.dgp, arch, dataset_id=did,
                                     hypothesis_z=args.hypothesis_z,
                                     metric=args.metric,
                                     n_seeds=args.n_seeds)
            else:
                pick_canonical_seed(args.dgp, arch, dataset_id=did,
                                     hypothesis_z=args.hypothesis_z,
                                     metric=args.metric)


if __name__ == "__main__":
    main()
