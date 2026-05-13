#!/usr/bin/env python3
"""
Evaluation script for the dezfouli vanilla AblatedRNN HP sweep over `hidden`.

Reads per-seed run config.json files written by run_Q_model.py
(under hp_search_runs_dezfouli_vanilla/h_{H}/seed_{S}/) and averages
cv_val_loss across seeds.  Lower cv_val_loss = better generalisation.

Usage:
    python hyperparam_eval_dezfouli_vanilla.py --hidden 5 --seeds "12 50 76" \
        --output hp_search_results_dezfouli_vanilla/h_5/metrics.json
"""

import os
import json
import argparse
import numpy as np


def evaluate_hidden(hidden, seeds):
    results = {
        "hidden":   hidden,
        "dgp":      "dezfouli",
        "seeds":    seeds,
        "per_seed": {},
    }
    cv_val_losses = []
    cv_selected_epochs = []

    for seed in seeds:
        run_dir = f"hp_search_runs_dezfouli_vanilla/h_{hidden}/seed_{seed}"
        cfg_path = os.path.join(run_dir, "config.json")
        if not os.path.exists(cfg_path):
            print(f"Warning: config.json not found: {cfg_path}")
            continue
        with open(cfg_path) as f:
            cfg = json.load(f)
        cv_val_loss   = cfg.get("cv_val_loss")
        cv_sel_epoch  = cfg.get("cv_selected_epoch")
        if cv_val_loss is None or cv_sel_epoch is None:
            print(f"Warning: cv_val_loss/cv_selected_epoch missing in {cfg_path}")
            continue
        results["per_seed"][seed] = {
            "cv_val_loss":       cv_val_loss,
            "cv_selected_epoch": cv_sel_epoch,
        }
        cv_val_losses.append(cv_val_loss)
        cv_selected_epochs.append(cv_sel_epoch)

    if cv_val_losses:
        results["aggregate"] = {
            "mean_cv_val_loss":       float(np.mean(cv_val_losses)),
            "std_cv_val_loss":        float(np.std(cv_val_losses)),
            "mean_cv_selected_epoch": float(np.mean(cv_selected_epochs)),
            "std_cv_selected_epoch":  float(np.std(cv_selected_epochs)),
            "n_seeds_evaluated":      len(cv_val_losses),
        }
    return results


def main():
    ap = argparse.ArgumentParser(
        description="Evaluate dezfouli vanilla HP sweep results.")
    ap.add_argument("--hidden", type=int, required=True)
    ap.add_argument("--seeds",  type=str, default="12 50 76")
    ap.add_argument("--output", type=str, default=None)
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split()]
    results = evaluate_hidden(args.hidden, seeds)

    print("\n" + "=" * 60)
    print(f"HP Eval (Dezfouli vanilla): hidden={args.hidden}")
    print("=" * 60)
    if "aggregate" in results:
        agg = results["aggregate"]
        print(f"Mean CV val loss:       "
              f"{agg['mean_cv_val_loss']:.4f} +/- {agg['std_cv_val_loss']:.4f}")
        print(f"Mean CV selected epoch: "
              f"{agg['mean_cv_selected_epoch']:.1f} +/- {agg['std_cv_selected_epoch']:.1f}")
        print(f"Seeds evaluated: {agg['n_seeds_evaluated']}")
    for seed, r in results["per_seed"].items():
        print(f"  Seed {seed}: cv_val_loss={r['cv_val_loss']:.4f}, "
              f"epoch={r['cv_selected_epoch']}")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
