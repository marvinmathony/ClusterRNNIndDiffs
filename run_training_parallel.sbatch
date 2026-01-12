#!/bin/bash
#SBATCH --job-name=train_%a
#SBATCH --output=logs/train_%a_%j.out
#SBATCH --error=logs/train_%a_%j.err
#SBATCH --open-mode=append
#SBATCH --array=0-49  # 5 datasets × 5 seeds × 2 model types = 50 jobs
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH -p gpu_p
#SBATCH --qos gpu_normal
#SBATCH --constraint=a100_80gb|h100_80gb
#SBATCH --nice=10000

# Create logs directory
mkdir -p logs

# Load modules (adjust based on your cluster)
# module load python/3.9
# module load cuda/11.8

# Activate your environment
# source /path/to/your/venv/bin/activate
# or: conda activate your_env

# Configuration
SEEDS=(12 50 76 100 142)
N_DATASETS=5
N_SEEDS=${#SEEDS[@]}
LMBD=0.5
Z_DIM=1

# Calculate dataset_id, seed_idx, and model_type from array task ID
# Array layout: [dataset0_seed0_latent, dataset0_seed0_vanilla, dataset0_seed1_latent, ...]
TASK_ID=$SLURM_ARRAY_TASK_ID

# Each dataset has 5 seeds × 2 model types = 10 jobs
JOBS_PER_DATASET=$((N_SEEDS * 2))
DATASET_ID=$((TASK_ID / JOBS_PER_DATASET))

# Within dataset, calculate seed and model type
WITHIN_DATASET=$((TASK_ID % JOBS_PER_DATASET))
SEED_IDX=$((WITHIN_DATASET / 2))
MODEL_TYPE=$((WITHIN_DATASET % 2))

SEED=${SEEDS[$SEED_IDX]}

if [ $MODEL_TYPE -eq 0 ]; then
    LATENT="True"
    MODEL_NAME="latent"
else
    LATENT="False"
    MODEL_NAME="vanilla"
fi

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Dataset ID: $DATASET_ID"
echo "Seed: $SEED"
echo "Model Type: $MODEL_NAME"
echo "Running on node: $(hostname)"
echo "Starting at: $(date)"

# Set environment variables
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

# Train model
echo "Training $MODEL_NAME model on dataset $DATASET_ID with seed $SEED..."
if [ "$LATENT" = "True" ]; then
    python run_Q_model.py \
        --seed $SEED \
        --latent True \
        --dataset_id $DATASET_ID \
        --lmbd $LMBD \
        --z $Z_DIM
else
    python run_Q_model.py \
        --seed $SEED \
        --latent False \
        --dataset_id $DATASET_ID
fi

echo "Finished at: $(date)"
