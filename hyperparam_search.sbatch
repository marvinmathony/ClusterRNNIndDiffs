#!/bin/bash
#SBATCH --job-name=hp_search_%a
#SBATCH --output=logs/hp_search_%a_%j.out
#SBATCH --error=logs/hp_search_%a_%j.err
#SBATCH --open-mode=append
#SBATCH --array=0-14
#SBATCH --time=06:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH -p gpu_p
#SBATCH --qos gpu_normal
#SBATCH --constraint=a100_80gb|h100_80gb
#SBATCH --nice=10000

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=marvin.mathony@helmholtz-munich.de

# Fast hyperparameter search for IDRNN
# Grid: lambda x z_dim = 5 x 3 = 15 combinations
# Uses reduced epochs (3000) and 2 seeds for speed
# Runtime estimate: ~1-2 hours per job

set -euo pipefail
mkdir -p logs

# Activate conda environment
source /ictstr01/home/hcai/marvin.mathony/tools/apps/mamba/etc/profile.d/conda.sh
conda activate RNNproject

# Get task ID
TASK_ID=$SLURM_ARRAY_TASK_ID

# Parameter grid
LAMBDAS=(0.05 0.08 0.1 0.12 0.15)
Z_DIMS=(3 4 5)
SEEDS=(12 50)  # Only 2 seeds for fast exploration
EPOCHS=3000    # Reduced from 10000
DATASET_ID=0   # Single dataset for fast search

# Calculate lambda and z_dim from task ID
# Task 0-4: lambda[0-4] with z=1
# Task 5-9: lambda[0-4] with z=2
# Task 10-14: lambda[0-4] with z=3
Z_IDX=$((TASK_ID / 5))
LMBD_IDX=$((TASK_ID % 5))

LMBD=${LAMBDAS[$LMBD_IDX]}
Z_DIM=${Z_DIMS[$Z_IDX]}

echo "========================================"
echo "Hyperparameter Search Job"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Task ID: $TASK_ID"
echo "Lambda: $LMBD"
echo "Z_dim: $Z_DIM"
echo "Epochs: $EPOCHS"
echo "Seeds: ${SEEDS[*]}"
echo "Dataset: $DATASET_ID"
echo "Starting at: $(date)"
echo "========================================"

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

# Create results directory for this hyperparameter combo
RESULTS_DIR="hp_search_results/lmbd_${LMBD}_z_${Z_DIM}"
mkdir -p "$RESULTS_DIR"

# Generate data if not exists
if [ ! -d "data_dataset${DATASET_ID}" ]; then
    echo "Generating dataset $DATASET_ID..."
    python data_generation.py --dataset_id $DATASET_ID
fi

# Train with each seed
for seed in "${SEEDS[@]}"; do
    echo "[$(date '+%H:%M:%S')] Training with seed $seed..."

    # Custom run directory for hyperparam search
    export HP_RUN_DIR="hp_search_runs/lmbd_${LMBD}_z_${Z_DIM}/seed_${seed}"

    python run_Q_model.py \
        --seed $seed \
        --latent True \
        --dataset_id $DATASET_ID \
        --lmbd $LMBD \
        --z $Z_DIM \
        --epochs $EPOCHS

    echo "[$(date '+%H:%M:%S')] Completed seed $seed"
done

# Run evaluation to get best RSA correlation
echo "Evaluating results..."
python hyperparam_eval.py \
    --lmbd $LMBD \
    --z $Z_DIM \
    --dataset_id $DATASET_ID \
    --seeds "${SEEDS[*]}" \
    --output "$RESULTS_DIR/metrics.json"

echo "========================================"
echo "Completed at: $(date)"
echo "Results saved to: $RESULTS_DIR"
echo "========================================"
