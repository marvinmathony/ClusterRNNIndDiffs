#!/usr/bin/env python3
"""
Quick evaluation script for hyperparameter search.
Extracts best RSA correlation from training runs without full causal evaluation.
"""

import os
import json
import argparse
import numpy as np
import glob
from collections import defaultdict


def load_rsa_correlations(run_dir, dataset_id=0, dgp=None):
    """Load RSA correlations from saved .npy files in rsa/ directory."""
    rsa_dir = os.path.join(run_dir, "rsa")
    if not os.path.exists(rsa_dir):
        return None, None

    # Load ground truth RSA for comparison
    if dgp:
        data_dir = f"data_{dgp}_dataset{dataset_id}"
    else:
        data_dir = f"data_dataset{dataset_id}"
    param_file = os.path.join(data_dir, "true_test_parameter_values.csv")
    if not os.path.exists(param_file):
        print(f"Warning: Ground truth file not found: {param_file}")
        return None, None

    import pandas as pd
    from scipy.spatial.distance import pdist, squareform

    params_df = pd.read_csv(param_file)
    params = params_df["alphaP_list"].values.reshape(-1, 1)

    # Compute ground truth RSA vector
    params_dist = squareform(pdist(params, metric="euclidean"))
    triu_idx = np.triu_indices_from(params_dist, k=1)
    vec_params = params_dist[triu_idx]

    # Load all RSA vectors and compute correlations
    rsa_files = sorted(glob.glob(os.path.join(rsa_dir, "epoch_*.npy")))

    correlations = {}
    for rsa_file in rsa_files:
        epoch = int(os.path.basename(rsa_file).replace("epoch_", "").replace(".npy", ""))
        vec_latents = np.load(rsa_file)
        corr = np.corrcoef(vec_params, vec_latents)[0, 1]
        correlations[epoch] = corr

    if not correlations:
        return None, None

    best_epoch = max(correlations, key=correlations.get)
    best_corr = correlations[best_epoch]

    return correlations, (best_epoch, best_corr)


def load_loss_trajectory(run_dir):
    """Load loss values from training."""
    loss_dir = os.path.join(run_dir, "loss")
    if not os.path.exists(loss_dir):
        return None

    loss_files = sorted(glob.glob(os.path.join(loss_dir, "epoch_*.npy")))
    losses = {}
    for loss_file in loss_files:
        epoch = int(os.path.basename(loss_file).replace("epoch_", "").replace(".npy", ""))
        loss_val = np.load(loss_file).item()
        losses[epoch] = loss_val

    return losses


def evaluate_hyperparam_combo(lmbd, z_dim, dataset_id, seeds, dgp=None):
    """Evaluate a single hyperparameter combination across seeds."""
    results = {
        "lmbd": lmbd,
        "z_dim": z_dim,
        "dataset_id": dataset_id,
        "dgp": dgp,
        "seeds": seeds,
        "per_seed": {}
    }

    all_best_corrs = []
    all_final_corrs = []

    for seed in seeds:
        # Check both possible run directory patterns
        run_dir_hp = f"hp_search_runs_{dgp}/lmbd_{lmbd}_z_{z_dim}/seed_{seed}" if dgp else f"hp_search_runs/lmbd_{lmbd}_z_{z_dim}/seed_{seed}"
        if dgp:
            run_dir_std = f"runs_{dgp}_dataset{dataset_id}/seed_{seed}"
        else:
            run_dir_std = f"runs_dataset{dataset_id}/seed_{seed}"

        run_dir = run_dir_hp if os.path.exists(run_dir_hp) else run_dir_std

        if not os.path.exists(run_dir):
            print(f"Warning: Run directory not found: {run_dir}")
            continue

        correlations, best_info = load_rsa_correlations(run_dir, dataset_id, dgp=dgp)
        losses = load_loss_trajectory(run_dir)

        if correlations is None:
            print(f"Warning: No RSA data found for seed {seed}")
            continue

        best_epoch, best_corr = best_info

        # Get final epoch correlation
        final_epoch = max(correlations.keys())
        final_corr = correlations[final_epoch]

        # Get early checkpoint correlation (e.g., epoch 1000)
        early_epochs = [e for e in correlations.keys() if e <= 1000]
        early_corr = correlations[max(early_epochs)] if early_epochs else None

        seed_results = {
            "best_epoch": best_epoch,
            "best_rsa_corr": best_corr,
            "final_epoch": final_epoch,
            "final_rsa_corr": final_corr,
            "early_rsa_corr": early_corr,
            "final_loss": losses.get(final_epoch) if losses else None,
            "n_epochs_evaluated": len(correlations)
        }

        results["per_seed"][seed] = seed_results
        all_best_corrs.append(best_corr)
        all_final_corrs.append(final_corr)

    # Aggregate statistics
    if all_best_corrs:
        results["aggregate"] = {
            "mean_best_rsa": np.mean(all_best_corrs),
            "std_best_rsa": np.std(all_best_corrs),
            "mean_final_rsa": np.mean(all_final_corrs),
            "std_final_rsa": np.std(all_final_corrs),
            "min_best_rsa": np.min(all_best_corrs),
            "max_best_rsa": np.max(all_best_corrs),
            "n_seeds_evaluated": len(all_best_corrs)
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate hyperparameter search results")
    parser.add_argument("--lmbd", type=float, required=True, help="Lambda value")
    parser.add_argument("--z", type=int, required=True, help="Z dimension")
    parser.add_argument("--dataset_id", type=int, default=0, help="Dataset ID")
    parser.add_argument("--seeds", type=str, default="12 50", help="Space-separated seed list")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    parser.add_argument("--dgp", type=str, default=None,
                        help="Data generating process type (e.g., 'bimodal', 'uniform'). Must match data_generation.py --dgp")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split()]

    results = evaluate_hyperparam_combo(args.lmbd, args.z, args.dataset_id, seeds, dgp=args.dgp)

    # Print summary
    print("\n" + "="*60)
    print(f"Hyperparameter Evaluation: lambda={args.lmbd}, z={args.z}")
    print("="*60)

    if "aggregate" in results:
        agg = results["aggregate"]
        print(f"Mean best RSA correlation: {agg['mean_best_rsa']:.4f} +/- {agg['std_best_rsa']:.4f}")
        print(f"Mean final RSA correlation: {agg['mean_final_rsa']:.4f} +/- {agg['std_final_rsa']:.4f}")
        print(f"Range: [{agg['min_best_rsa']:.4f}, {agg['max_best_rsa']:.4f}]")
        print(f"Seeds evaluated: {agg['n_seeds_evaluated']}")

    for seed, seed_res in results["per_seed"].items():
        print(f"\nSeed {seed}:")
        print(f"  Best RSA: {seed_res['best_rsa_corr']:.4f} at epoch {seed_res['best_epoch']}")
        print(f"  Final RSA: {seed_res['final_rsa_corr']:.4f} at epoch {seed_res['final_epoch']}")

    # Save results
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
