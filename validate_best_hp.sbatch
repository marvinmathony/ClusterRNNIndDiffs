#!/bin/bash
#SBATCH --job-name=hp_validate_%a
#SBATCH --output=logs/hp_validate_%a_%j.out
#SBATCH --error=logs/hp_validate_%a_%j.err
#SBATCH --open-mode=append
#SBATCH --array=0-4
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH -p gpu_p
#SBATCH --qos gpu_normal
#SBATCH --constraint=a100_80gb|h100_80gb
#SBATCH --nice=10000

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=marvin.mathony@helmholtz-munich.de

# Validation script for best hyperparameters
# Runs full 5 seeds x 5 datasets with best lambda and z_dim
# Usage: sbatch validate_best_hp.sbatch <LMBD> <Z_DIM>
# Example: sbatch validate_best_hp.sbatch 0.8 3

set -euo pipefail
mkdir -p logs

# Get best hyperparameters from command line arguments
LMBD=${1:-0.8}
Z_DIM=${2:-3}
EPOCHS=${3:-10000}

DATASET_ID=$SLURM_ARRAY_TASK_ID
SEEDS=(12 50 76 100 142)

echo "========================================"
echo "Hyperparameter Validation"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Dataset ID: $DATASET_ID"
echo "Lambda: $LMBD"
echo "Z_dim: $Z_DIM"
echo "Epochs: $EPOCHS"
echo "Seeds: ${SEEDS[*]}"
echo "Starting at: $(date)"
echo "========================================"

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

# Generate data if not exists
if [ ! -d "data_dataset${DATASET_ID}" ]; then
    echo "Generating dataset $DATASET_ID..."
    python data_generation.py --dataset_id $DATASET_ID
fi

# Train IDRNN with all seeds
echo "Training IDRNN models..."
for seed in "${SEEDS[@]}"; do
    echo "[$(date '+%H:%M:%S')] Training with seed $seed..."
    python run_Q_model.py \
        --seed $seed \
        --latent True \
        --dataset_id $DATASET_ID \
        --lmbd $LMBD \
        --z $Z_DIM \
        --epochs $EPOCHS
    echo "[$(date '+%H:%M:%S')] Completed seed $seed"
done

# Train vanilla baselines
echo "Training vanilla models..."
for seed in "${SEEDS[@]}"; do
    echo "[$(date '+%H:%M:%S')] Training vanilla with seed $seed..."
    python run_Q_model.py \
        --seed $seed \
        --latent False \
        --dataset_id $DATASET_ID \
        --epochs $EPOCHS
    echo "[$(date '+%H:%M:%S')] Completed vanilla seed $seed"
done

# Select best epochs
echo "Selecting best epochs..."
python select_best_epoch_by_specificity.py --latent True --dataset_id $DATASET_ID
python select_best_epoch_by_rsa.py --latent False --dataset_id $DATASET_ID

# Test models
echo "Testing models..."
python testing_script.py --latent True --dataset_id $DATASET_ID --model_fitting False
python testing_script.py --latent False --dataset_id $DATASET_ID --model_fitting False

echo "========================================"
echo "Validation completed at: $(date)"
echo "========================================"
