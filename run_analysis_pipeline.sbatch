#!/bin/bash
#SBATCH --job-name=analysis_pipeline
#SBATCH --output=logs/analysis_pipeline_%j.out
#SBATCH --error=logs/analysis_pipeline_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH -p gpu_p
#SBATCH --qos gpu_normal
#SBATCH --constraint=a100_80gb|h100_80gb
#SBATCH --nice=10000

#SBATCH --mail-type=ALL
#SBATCH --mail-user=marvin.mathony@helmholtz-munich.de

# Post-training analysis pipeline for all datasets.
# Runs: epoch selection -> testing -> multi-dataset analysis
#
# Submit with:
#   sbatch run_analysis_pipeline.sbatch

set -euo pipefail

# Create logs directory
mkdir -p logs

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Starting at: $(date)"

# Set environment variables
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

# Configuration
N_DATASETS=5
MIN_EPOCH=1000
MAX_EPOCH=3000

echo "============================================================"
echo "ANALYSIS PIPELINE"
echo "============================================================"
echo "Number of datasets: $N_DATASETS"
echo "Epoch window: [$MIN_EPOCH, $MAX_EPOCH]"
echo "============================================================"

# Step 1: Select best epochs for each dataset
echo ""
echo "============================================================"
echo "STEP 1: Selecting best epochs for all datasets"
echo "============================================================"

for dataset_id in $(seq 0 $((N_DATASETS - 1))); do
    echo ""
    echo "--- Dataset $dataset_id ---"

    # Select best epoch for IDRNN (using specificity)
    echo "  [$(date '+%H:%M:%S')] Selecting best IDRNN epoch..."
    python select_best_epoch_by_specificity.py \
        --latent True \
        --dataset_id $dataset_id \
        --min_epoch $MIN_EPOCH \
        --max_epoch $MAX_EPOCH

    # Select best epoch for Vanilla (using RSA)
    echo "  [$(date '+%H:%M:%S')] Selecting best Vanilla epoch..."
    python select_best_epoch_by_rsa.py \
        --latent False \
        --dataset_id $dataset_id \
        --min_epoch $MIN_EPOCH \
        --max_epoch $MAX_EPOCH

    echo "  Dataset $dataset_id epoch selection complete."
done

# Step 2: Run testing script for each dataset
echo ""
echo "============================================================"
echo "STEP 2: Running testing script for all datasets"
echo "============================================================"

for dataset_id in $(seq 0 $((N_DATASETS - 1))); do
    echo ""
    echo "--- Dataset $dataset_id ---"

    # Test IDRNN model
    echo "  [$(date '+%H:%M:%S')] Testing IDRNN model..."
    python testing_script.py \
        --latent True \
        --dataset_id $dataset_id \
        --model_fitting False

    # Test Vanilla model
    echo "  [$(date '+%H:%M:%S')] Testing Vanilla model..."
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
echo "Results saved in plots/multi_dataset/"
