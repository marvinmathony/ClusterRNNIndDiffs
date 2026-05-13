#!/usr/bin/env python3
"""
Summarize hyperparameter search results for discrete_alpha_only into a CSV file.
Run this after all hyperparameter search jobs have completed.

Usage:
    python summarize_hp_search_discrete_alpha_only.py
    python summarize_hp_search_discrete_alpha_only.py --output custom_name.csv
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
from datetime import datetime


def load_results_from_json(results_dir="hp_search_results_discrete_alpha_only"):
    """Load pre-computed results from JSON files."""
    results = []

    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        return results

    for dirname in os.listdir(results_dir):
        if not dirname.startswith("lmbd_"):
            continue

        metrics_path = os.path.join(results_dir, dirname, "metrics.json")
        if not os.path.exists(metrics_path):
            print(f"Warning: No metrics.json found in {dirname}")
            continue

        try:
            with open(metrics_path, "r") as f:
                data = json.load(f)

            if "aggregate" not in data:
                print(f"Warning: No aggregate data in {dirname}")
                continue

            agg = data["aggregate"]
            results.append({
                "lmbd": data["lmbd"],
                "z_dim": data["z_dim"],
                "mean_best_rsa": agg.get("mean_best_rsa"),
                "std_best_rsa": agg.get("std_best_rsa"),
                "mean_final_rsa": agg.get("mean_final_rsa"),
                "std_final_rsa": agg.get("std_final_rsa"),
                "min_best_rsa": agg.get("min_best_rsa"),
                "max_best_rsa": agg.get("max_best_rsa"),
                "n_seeds": agg.get("n_seeds_evaluated"),
            })
        except Exception as e:
            print(f"Warning: Failed to load {metrics_path}: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Summarize discrete_alpha_only hyperparameter search results")
    parser.add_argument("--output", type=str, default="hp_search_summary_discrete_alpha_only.csv",
                        help="Output CSV file name")
    args = parser.parse_args()

    print("=" * 60)
    print("Discrete Alpha Only - Hyperparameter Search Summary")
    print("=" * 60)

    print("Loading cached results from hp_search_results_discrete_alpha_only/...")
    results = load_results_from_json()

    if not results:
        print("\nNo results found. Make sure hyperparameter search has completed.")
        print("Run: sbatch hyperparam_search_discrete_alpha_only.sbatch")
        return

    # Create DataFrame
    df = pd.DataFrame(results)

    # Sort by mean_best_rsa (descending - higher is better)
    df = df.sort_values("mean_best_rsa", ascending=False)

    # Add rank column
    df.insert(0, "rank", range(1, len(df) + 1))

    # Save to CSV
    df.to_csv(args.output, index=False)
    print(f"\nResults saved to: {args.output}")

    # Print summary table
    print("\n" + "=" * 80)
    print("HYPERPARAMETER SEARCH RESULTS (sorted by best RSA correlation)")
    print("=" * 80)
    print(f"{'Rank':>4} {'Lambda':>8} {'Z_dim':>6} {'Mean RSA':>12} {'Std':>8} {'N_seeds':>8}")
    print("-" * 80)

    for _, row in df.iterrows():
        print(f"{row['rank']:>4} {row['lmbd']:>8.3f} {row['z_dim']:>6} "
              f"{row['mean_best_rsa']:>12.4f} {row['std_best_rsa']:>8.4f} "
              f"{row['n_seeds']:>8}")

    # Highlight best configuration
    best = df.iloc[0]
    print("\n" + "=" * 80)
    print("BEST CONFIGURATION:")
    print(f"  Lambda: {best['lmbd']}")
    print(f"  Z_dim:  {int(best['z_dim'])}")
    print(f"  Mean Best RSA: {best['mean_best_rsa']:.4f} +/- {best['std_best_rsa']:.4f}")
    print(f"  Range: [{best['min_best_rsa']:.4f}, {best['max_best_rsa']:.4f}]")
    print("=" * 80)

    # Create pivot table for visualization
    print("\n\nPIVOT TABLE (Mean Best RSA Correlation):")
    print("-" * 50)
    pivot = df.pivot_table(
        values="mean_best_rsa",
        index="lmbd",
        columns="z_dim",
        aggfunc="first"
    )
    print(pivot.to_string(float_format="%.4f"))

    # Save timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\nGenerated: {timestamp}")


if __name__ == "__main__":
    main()
