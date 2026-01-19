#!/usr/bin/env python3
"""
Aggregate hyperparameter search results and identify best combinations.
Run after all hyperparam_search.sbatch jobs complete.
"""

import os
import json
import glob
import numpy as np
import pandas as pd


def load_all_results(results_dir="hp_search_results"):
    """Load all hyperparameter search results."""
    results = []

    for metrics_file in glob.glob(os.path.join(results_dir, "**/metrics.json"), recursive=True):
        try:
            with open(metrics_file) as f:
                data = json.load(f)
            results.append(data)
        except Exception as e:
            print(f"Warning: Failed to load {metrics_file}: {e}")

    return results


def create_summary_table(results):
    """Create a summary table of all hyperparameter combinations."""
    rows = []

    for r in results:
        if "aggregate" not in r:
            continue

        agg = r["aggregate"]
        rows.append({
            "lambda": r["lmbd"],
            "z_dim": r["z_dim"],
            "mean_best_rsa": agg["mean_best_rsa"],
            "std_best_rsa": agg["std_best_rsa"],
            "mean_final_rsa": agg["mean_final_rsa"],
            "min_best_rsa": agg["min_best_rsa"],
            "max_best_rsa": agg["max_best_rsa"],
            "n_seeds": agg["n_seeds_evaluated"]
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("mean_best_rsa", ascending=False)

    return df


def create_heatmap_data(df):
    """Create data for lambda x z_dim heatmap."""
    pivot = df.pivot(index="z_dim", columns="lambda", values="mean_best_rsa")
    return pivot


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="hp_search_results", help="Results directory")
    parser.add_argument("--output", default="hp_search_summary.csv", help="Output CSV file")
    args = parser.parse_args()

    print("Loading hyperparameter search results...")
    results = load_all_results(args.results_dir)

    if not results:
        print("No results found. Make sure hyperparam_search.sbatch jobs have completed.")
        print(f"Looking in: {args.results_dir}")
        return

    print(f"Found {len(results)} hyperparameter combinations.\n")

    # Create summary table
    df = create_summary_table(results)

    if df.empty:
        print("No valid results with aggregate statistics found.")
        return

    # Print table
    print("=" * 80)
    print("HYPERPARAMETER SEARCH RESULTS (sorted by mean best RSA correlation)")
    print("=" * 80)

    # Print as formatted DataFrame
    df_display = df.round(4).copy()
    df_display.columns = ["Lambda", "Z_dim", "Mean Best RSA", "Std", "Mean Final RSA", "Min", "Max", "Seeds"]
    print(df_display.to_string(index=False))

    # Best combination
    best = df.iloc[0]
    print(f"\n{'='*80}")
    print("BEST HYPERPARAMETERS:")
    print(f"  Lambda: {best['lambda']}")
    print(f"  Z_dim: {int(best['z_dim'])}")
    print(f"  Mean RSA correlation: {best['mean_best_rsa']:.4f} +/- {best['std_best_rsa']:.4f}")
    print(f"{'='*80}")

    # Heatmap-style view
    print("\nHeatmap: Mean Best RSA (rows=z_dim, cols=lambda)")
    print("-" * 60)
    heatmap = create_heatmap_data(df)
    print(heatmap.round(4).to_string())

    # Top 3 recommendations
    print("\n" + "=" * 80)
    print("TOP 3 RECOMMENDATIONS:")
    print("=" * 80)
    for i, (_, row) in enumerate(df.head(3).iterrows(), 1):
        print(f"{i}. lambda={row['lambda']}, z={int(row['z_dim'])}: "
              f"RSA={row['mean_best_rsa']:.4f} +/- {row['std_best_rsa']:.4f}")

    # Save to CSV
    df.to_csv(args.output, index=False)
    print(f"\nFull results saved to: {args.output}")

    # Advice for next steps
    print("\n" + "=" * 80)
    print("NEXT STEPS:")
    print("=" * 80)
    best_lmbd = best['lambda']
    best_z = int(best['z_dim'])
    print(f"1. Validate best config (lambda={best_lmbd}, z={best_z}) with full 5 seeds:")
    print(f"   python run_Q_model.py --lmbd {best_lmbd} --z {best_z} --latent True \\")
    print(f"          --epochs 10000 --seed <SEED> --dataset_id <0-4>")
    print()
    print("2. Or run full validation with the validation script:")
    print(f"   sbatch validate_best_hp.sbatch {best_lmbd} {best_z}")


if __name__ == "__main__":
    main()
