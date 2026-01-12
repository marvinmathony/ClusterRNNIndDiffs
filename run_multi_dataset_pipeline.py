"""
Master orchestration script to run the entire pipeline across multiple datasets.

This script:
1. Generates 5 distinct datasets
2. Trains models (both latent and vanilla) with 5 random seeds per dataset
3. Selects best epochs via RSA
4. Runs testing scripts
5. Saves all results for aggregated plotting
"""

import os
import subprocess
import sys
import argparse
from pathlib import Path

# Configuration
N_DATASETS = 5
SEEDS = [12, 50, 76, 100, 142]

def run_command(cmd, description):
    """Execute a command and handle errors."""
    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*80)

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"ERROR: {description} failed with return code {result.returncode}")
        sys.exit(1)

    print(f"✅ {description} completed successfully")
    return result

def main():
    parser = argparse.ArgumentParser(description="Run full pipeline across multiple datasets")
    parser.add_argument('--n_datasets', type=int, default=N_DATASETS, help="Number of datasets to generate")
    parser.add_argument('--lmbd', type=float, default=1.0, help="Lambda parameter for latent model training")
    parser.add_argument('--z', type=int, default=1, help="Dimension of latent space")
    parser.add_argument('--skip_data_generation', action='store_true', help="Skip data generation step")
    parser.add_argument('--skip_training', action='store_true', help="Skip training step")
    parser.add_argument('--skip_selection', action='store_true', help="Skip epoch selection step")
    parser.add_argument('--skip_testing', action='store_true', help="Skip testing step")
    parser.add_argument('--only_analysis', action='store_true', help="Only run the aggregated analysis")
    args = parser.parse_args()

    n_datasets = args.n_datasets

    print(f"\n🚀 Starting multi-dataset pipeline with {n_datasets} datasets")
    print(f"Random seeds: {SEEDS}")

    if args.only_analysis:
        print("\n📊 Running only aggregated analysis...")
        run_command(
            ["python", "analyze_synthetic_multi_dataset.py"],
            "Aggregated analysis across datasets"
        )
        print("\n🎉 Analysis complete!")
        return

    # Step 1: Generate datasets
    if not args.skip_data_generation:
        print("\n" + "="*80)
        print("STEP 1: DATA GENERATION")
        print("="*80)

        for dataset_id in range(n_datasets):
            run_command(
                ["python", "data_generation.py", "--dataset_id", str(dataset_id)],
                f"Data generation for dataset {dataset_id}"
            )

    # Step 2: Train models across seeds
    if not args.skip_training:
        print("\n" + "="*80)
        print("STEP 2: MODEL TRAINING")
        print("="*80)

        for dataset_id in range(n_datasets):
            print(f"\n--- Training on Dataset {dataset_id} ---")

            # Train latent models
            for seed in SEEDS:
                run_command(
                    ["python", "run_Q_model.py",
                     "--seed", str(seed),
                     "--latent", "True",
                     "--dataset_id", str(dataset_id),
                     "--lmbd", str(args.lmbd),
                     "--z", str(args.z)],
                    f"Training latent model on dataset {dataset_id} with seed {seed} (lmbd={args.lmbd}, z={args.z})"
                )

            # Train vanilla models
            for seed in SEEDS:
                run_command(
                    ["python", "run_Q_model.py",
                     "--seed", str(seed),
                     "--latent", "False",
                     "--dataset_id", str(dataset_id)],
                    f"Training vanilla model on dataset {dataset_id} with seed {seed}"
                )

    # Step 3: Select best epochs
    if not args.skip_selection:
        print("\n" + "="*80)
        print("STEP 3: BEST EPOCH SELECTION")
        print("="*80)

        for dataset_id in range(n_datasets):
            # Select for latent models
            run_command(
                ["python", "select_best_epoch_by_rsa.py",
                 "--latent", "True",
                 "--dataset_id", str(dataset_id)],
                f"Selecting best epoch for latent models on dataset {dataset_id}"
            )

            # Select for vanilla models
            run_command(
                ["python", "select_best_epoch_by_rsa.py",
                 "--latent", "False",
                 "--dataset_id", str(dataset_id)],
                f"Selecting best epoch for vanilla models on dataset {dataset_id}"
            )

    # Step 4: Run testing
    if not args.skip_testing:
        print("\n" + "="*80)
        print("STEP 4: MODEL TESTING")
        print("="*80)

        for dataset_id in range(n_datasets):
            # Test latent models
            run_command(
                ["python", "testing_script.py",
                 "--latent", "True",
                 "--dataset_id", str(dataset_id),
                 "--model_fitting", "True"],
                f"Testing latent model on dataset {dataset_id}"
            )

            # Test vanilla models
            run_command(
                ["python", "testing_script.py",
                 "--latent", "False",
                 "--dataset_id", str(dataset_id),
                 "--model_fitting", "True"],
                f"Testing vanilla model on dataset {dataset_id}"
            )

    # Step 5: Aggregate and plot
    print("\n" + "="*80)
    print("STEP 5: AGGREGATED ANALYSIS")
    print("="*80)

    run_command(
        ["python", "analyze_synthetic_multi_dataset.py"],
        "Aggregated analysis across all datasets"
    )

    print("\n" + "="*80)
    print("🎉 PIPELINE COMPLETE!")
    print("="*80)
    print(f"\nProcessed {n_datasets} datasets with {len(SEEDS)} seeds each")
    print("Results are saved in:")
    for dataset_id in range(n_datasets):
        print(f"  - data_dataset{dataset_id}/")
        print(f"  - runs_dataset{dataset_id}/")
        print(f"  - runs_vanilla_dataset{dataset_id}/")
    print("\nAggregated plots are in: plots/multi_dataset/")

if __name__ == "__main__":
    main()
