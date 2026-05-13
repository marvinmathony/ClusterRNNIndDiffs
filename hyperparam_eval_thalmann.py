#!/usr/bin/env python3
"""
Evaluation script for Thalmann hyperparameter search.

Reads cv_val_loss from config.json (written by run_Q_model.py inner CV).
Lower cv_val_loss = better generalisation = preferred hyperparameters.
"""
import os
import json
import argparse
import numpy as np


def evaluate_hyperparam_combo(lmbd, z_dim, seeds, hidden=5, enc_hidden=5, step1_epochs=1000):
    tag = f"lmbd_{lmbd}_z_{z_dim}_h_{hidden}_eh_{enc_hidden}_s1_{step1_epochs}"
    results = {
        "lmbd": lmbd, "z_dim": z_dim, "hidden": hidden,
        "enc_hidden": enc_hidden, "step1_epochs": step1_epochs,
        "dgp": "thalmann", "seeds": seeds, "per_seed": {},
    }
    cv_val_losses, cv_epochs = [], []

    for seed in seeds:
        run_dir     = f"hp_search_runs_thalmann/{tag}/seed_{seed}"
        config_path = os.path.join(run_dir, "config.json")
        if not os.path.exists(config_path):
            print(f"Warning: not found: {config_path}")
            continue
        with open(config_path) as f:
            cfg = json.load(f)
        cv_val_loss = cfg.get("cv_val_loss")
        cv_epoch    = cfg.get("cv_selected_epoch")
        if cv_val_loss is None or cv_epoch is None:
            print(f"Warning: cv_val_loss/cv_selected_epoch missing in {config_path}")
            continue
        results["per_seed"][seed] = {"cv_val_loss": cv_val_loss, "cv_selected_epoch": cv_epoch}
        cv_val_losses.append(cv_val_loss)
        cv_epochs.append(cv_epoch)

    if cv_val_losses:
        results["aggregate"] = {
            "mean_cv_val_loss":      float(np.mean(cv_val_losses)),
            "std_cv_val_loss":       float(np.std(cv_val_losses)),
            "mean_cv_selected_epoch": float(np.mean(cv_epochs)),
            "std_cv_selected_epoch":  float(np.std(cv_epochs)),
            "n_seeds_evaluated":      len(cv_val_losses),
        }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmbd",         type=float, required=True)
    parser.add_argument("--z",            type=int,   required=True)
    parser.add_argument("--hidden",       type=int,   default=5)
    parser.add_argument("--enc_hidden",   type=int,   default=5)
    parser.add_argument("--step1_epochs", type=int,   default=1000)
    parser.add_argument("--seeds",        type=str,   default="12 50 76")
    parser.add_argument("--output",       type=str,   default=None)
    args = parser.parse_args()

    seeds   = [int(s) for s in args.seeds.split()]
    results = evaluate_hyperparam_combo(
        args.lmbd, args.z, seeds,
        hidden=args.hidden, enc_hidden=args.enc_hidden,
        step1_epochs=args.step1_epochs,
    )

    print("\n" + "=" * 60)
    print(f"HP Eval (Thalmann): λ={args.lmbd}, z={args.z}, "
          f"h={args.hidden}, eh={args.enc_hidden}, s1={args.step1_epochs}")
    print("=" * 60)
    if "aggregate" in results:
        agg = results["aggregate"]
        print(f"Mean CV val loss:       {agg['mean_cv_val_loss']:.4f} ± {agg['std_cv_val_loss']:.4f}")
        print(f"Mean CV selected epoch: {agg['mean_cv_selected_epoch']:.1f} ± {agg['std_cv_selected_epoch']:.1f}")
        print(f"Seeds evaluated:        {agg['n_seeds_evaluated']}")
    for seed, r in results["per_seed"].items():
        print(f"  Seed {seed}: cv_val_loss={r['cv_val_loss']:.4f}, epoch={r['cv_selected_epoch']}")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
