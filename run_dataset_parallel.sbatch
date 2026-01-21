#!/bin/bash
#SBATCH --job-name=dataset_%a
#SBATCH --output=logs/dataset_parallel_%a_%j.out
#SBATCH --error=logs/dataset_parallel_%a_%j.err
#SBATCH --open-mode=append
#SBATCH --array=0-4
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH -p gpu_p
#SBATCH --qos gpu_normal
#SBATCH --constraint=a100_80gb|h100_80gb
#SBATCH --nice=10000

#SBATCH --mail-type=ALL
#SBATCH --mail-user=marvin.mathony@helmholtz-munich.de

# Activate conda environment
source /ictstr01/home/hcai/marvin.mathony/tools/apps/mamba/etc/profile.d/conda.sh
conda activate RNNproject

# Exit immediately if a command exits with a non-zero status
# Also fail on pipe errors and undefined variables
set -euo pipefail

# Create logs directory
mkdir -p logs

# Get dataset ID from SLURM array task ID
DATASET_ID=$SLURM_ARRAY_TASK_ID

# Data Generating Process (DGP) configuration
# Set to empty string for default (discrete_alpha_only), or specify e.g. "bimodal", "uniform"
# Can be overridden when submitting: sbatch --export=DGP=bimodal run_dataset_parallel.sbatch
DGP="${DGP:-}"

# Build DGP argument string for python scripts
if [ -n "$DGP" ]; then
    DGP_ARG="--dgp $DGP"
    DGP_INFO=" (DGP: $DGP)"
else
    DGP_ARG=""
    DGP_INFO=""
fi

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Dataset ID: $DATASET_ID"
echo "DGP: ${DGP:-default (discrete_alpha_only)}"
echo "Running on node: $(hostname)"
echo "Starting at: $(date)"

# Set environment variables
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

# Configuration
LMBD=0.05
Z_DIM=3
SEEDS=(12 50 76 100 142)

# Step 1: Generate data
echo "Generating dataset $DATASET_ID$DGP_INFO..."
python data_generation.py --dataset_id $DATASET_ID $DGP_ARG

# Step 2: Train latent models
echo "Training latent models for dataset $DATASET_ID$DGP_INFO..."
for seed in "${SEEDS[@]}"; do
    echo "  [$(date '+%Y-%m-%d %H:%M:%S')] Training latent model with seed $seed..."
    python run_Q_model.py \
        --seed $seed \
        --latent True \
        --dataset_id $DATASET_ID \
        --lmbd $LMBD \
        --z $Z_DIM \
        $DGP_ARG
    echo "  [$(date '+%Y-%m-%d %H:%M:%S')] Completed latent model with seed $seed"
done

# Step 3: Train vanilla models
echo "Training vanilla models for dataset $DATASET_ID$DGP_INFO..."
for seed in "${SEEDS[@]}"; do
    echo "  [$(date '+%Y-%m-%d %H:%M:%S')] Training vanilla model with seed $seed..."
    python run_Q_model.py \
        --seed $seed \
        --latent False \
        --dataset_id $DATASET_ID \
        $DGP_ARG
    echo "  [$(date '+%Y-%m-%d %H:%M:%S')] Completed vanilla model with seed $seed"
done

# Step 4: Select best epochs
# Use composite score (RSA reliability + loss + stability) for both model types
echo "Selecting best epochs for dataset $DATASET_ID$DGP_INFO..."
python select_best_epoch_by_specificity.py --latent True --dataset_id $DATASET_ID $DGP_ARG
python select_best_epoch_by_rsa.py --latent False --dataset_id $DATASET_ID $DGP_ARG

# Step 5: Test models
echo "Testing models for dataset $DATASET_ID$DGP_INFO..."
python testing_script.py --latent True --dataset_id $DATASET_ID --model_fitting False $DGP_ARG
python testing_script.py --latent False --dataset_id $DATASET_ID --model_fitting False $DGP_ARG

echo "Dataset $DATASET_ID$DGP_INFO completed at: $(date)"
