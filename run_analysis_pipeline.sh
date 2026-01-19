#!/bin/bash
# run_analysis_pipeline.sh
#
# Post-training analysis pipeline for all datasets.
# Runs: epoch selection -> testing -> multi-dataset analysis
#
# Usage:
#   ./run_analysis_pipeline.sh              # Run all steps
#   ./run_analysis_pipeline.sh --skip-selection  # Skip epoch selection (use existing)
#   ./run_analysis_pipeline.sh --dataset 0  # Run only dataset 0

set -euo pipefail

# Configuration
N_DATASETS=5
MIN_EPOCH=1000
MAX_EPOCH=3000

# Parse arguments
SKIP_SELECTION=false
SINGLE_DATASET=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-selection)
            SKIP_SELECTION=true
            shift
            ;;
        --dataset)
            SINGLE_DATASET="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "============================================================"
echo "ANALYSIS PIPELINE"
echo "============================================================"
echo "Number of datasets: $N_DATASETS"
echo "Epoch window: [$MIN_EPOCH, $MAX_EPOCH]"
echo "Skip epoch selection: $SKIP_SELECTION"
if [[ -n "$SINGLE_DATASET" ]]; then
    echo "Running only dataset: $SINGLE_DATASET"
fi
echo "Started at: $(date)"
echo "============================================================"

# Determine which datasets to process
if [[ -n "$SINGLE_DATASET" ]]; then
    DATASETS=($SINGLE_DATASET)
else
    DATASETS=($(seq 0 $((N_DATASETS - 1))))
fi

# Step 1: Select best epochs for each dataset
if [[ "$SKIP_SELECTION" == false ]]; then
    echo ""
    echo "============================================================"
    echo "STEP 1: Selecting best epochs for all datasets"
    echo "============================================================"

    for dataset_id in "${DATASETS[@]}"; do
        echo ""
        echo "--- Dataset $dataset_id ---"

        # Select best epoch for IDRNN (using specificity)
        echo "  Selecting best IDRNN epoch..."
        python select_best_epoch_by_specificity.py \
            --latent True \
            --dataset_id $dataset_id \
            --min_epoch $MIN_EPOCH \
            --max_epoch $MAX_EPOCH

        # Select best epoch for Vanilla (using RSA)
        echo "  Selecting best Vanilla epoch..."
        python select_best_epoch_by_rsa.py \
            --latent False \
            --dataset_id $dataset_id \
            --min_epoch $MIN_EPOCH \
            --max_epoch $MAX_EPOCH

        echo "  Dataset $dataset_id epoch selection complete."
    done
else
    echo ""
    echo "Skipping epoch selection (using existing best_epoch*.json files)"
fi

# Step 2: Run testing script for each dataset
echo ""
echo "============================================================"
echo "STEP 2: Running testing script for all datasets"
echo "============================================================"

for dataset_id in "${DATASETS[@]}"; do
    echo ""
    echo "--- Dataset $dataset_id ---"

    # Test IDRNN model
    echo "  Testing IDRNN model..."
    python testing_script.py \
        --latent True \
        --dataset_id $dataset_id \
        --model_fitting False

    # Test Vanilla model
    echo "  Testing Vanilla model..."
    python testing_script.py \
        --latent False \
        --dataset_id $dataset_id \
        --model_fitting False

    echo "  Dataset $dataset_id testing complete."
done

# Step 3: Run multi-dataset analysis
echo ""
echo "============================================================"
echo "STEP 3: Running multi-dataset analysis"
echo "============================================================"

python analyze_synthetic_multi_dataset.py

echo ""
echo "============================================================"
echo "PIPELINE COMPLETE"
echo "============================================================"
echo "Finished at: $(date)"
echo ""
echo "Results saved in:"
echo "  - plots/multi_dataset/aggregated_rsa_correlation.png"
echo "  - plots/multi_dataset/aggregated_model_likelihoods.png"
echo "  - plots/multi_dataset/per_dataset_rsa.png"
echo "  - plots/multi_dataset/per_dataset_likelihoods.png"
echo "  - plots/multi_dataset/alpha_vs_z_dataset*.png"
echo "  - plots/multi_dataset/alpha_vs_z_aggregated.png"
echo "  - plots/multi_dataset/summary_statistics.txt"
