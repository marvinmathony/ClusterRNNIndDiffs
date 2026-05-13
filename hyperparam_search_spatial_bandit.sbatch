#!/bin/bash
#SBATCH --job-name=hp_spbandit_%a
#SBATCH --output=logs/hp_spatial_bandit_%a_%j.out
#SBATCH --error=logs/hp_spatial_bandit_%a_%j.err
#SBATCH --open-mode=append
#SBATCH --array=0-15
#SBATCH --time=08:00:00
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

# Hyperparameter search for IDRNN on spatial bandit data (Schulz & Wu)
# Grid: lambda x z_dim = 5 x 3 = 15 combinations + 1 vanilla baseline
# Uses reconstruction specificity as metric (no ground truth available)

set -euo pipefail
mkdir -p logs

source /ictstr01/home/hcai/marvin.mathony/tools/apps/mamba/etc/profile.d/conda.sh
conda activate RNNproject

TASK_ID=$SLURM_ARRAY_TASK_ID

LAMBDAS=(0 0.005 0.01 0.02 0.05)
Z_DIMS=(3 5 8)
SEEDS=(12 50 76)
EPOCHS=3000
DGP="spatial_bandit"
DATASET_ID=0

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

if [ ! -d "data_${DGP}" ]; then
    echo "ERROR: data_${DGP} directory not found! Run load_spatial_bandit.py first."
    exit 1
fi

# Task 15: vanilla baseline
if [ "$TASK_ID" -eq 15 ]; then
    echo "========================================"
    echo "Vanilla Baseline Training (spatial_bandit)"
    echo "========================================"
    echo "Job ID: $SLURM_JOB_ID | Task: $TASK_ID | Seeds: ${SEEDS[*]}"
    echo "Starting at: $(date)"
    echo "========================================"

    for seed in "${SEEDS[@]}"; do
        echo "[$(date '+%H:%M:%S')] Training vanilla seed $seed..."
        python run_Q_model.py \
            --seed $seed \
            --latent False \
            --dataset_id $DATASET_ID \
            --dgp $DGP \
            --epochs $EPOCHS
        echo "[$(date '+%H:%M:%S')] Completed vanilla seed $seed"
    done

    echo "========================================"
    echo "Vanilla training completed at: $(date)"
    echo "========================================"
    exit 0
fi

# Tasks 0-14: IDRNN grid search
Z_IDX=$((TASK_ID / 5))
LMBD_IDX=$((TASK_ID % 5))

LMBD=${LAMBDAS[$LMBD_IDX]}
Z_DIM=${Z_DIMS[$Z_IDX]}

echo "========================================"
echo "Hyperparameter Search Job (spatial_bandit)"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID | Task: $TASK_ID"
echo "Lambda: $LMBD | Z_dim: $Z_DIM | Epochs: $EPOCHS"
echo "Seeds: ${SEEDS[*]} | DGP: $DGP"
echo "Starting at: $(date)"
echo "========================================"

RESULTS_DIR="hp_search_results_${DGP}/lmbd_${LMBD}_z_${Z_DIM}"
mkdir -p "$RESULTS_DIR"

for seed in "${SEEDS[@]}"; do
    echo "[$(date '+%H:%M:%S')] Training seed $seed..."
    export HP_RUN_DIR="hp_search_runs_${DGP}/lmbd_${LMBD}_z_${Z_DIM}/seed_${seed}"

    python run_Q_model.py \
        --seed $seed \
        --latent True \
        --dataset_id $DATASET_ID \
        --dgp $DGP \
        --lmbd $LMBD \
        --z $Z_DIM \
        --epochs $EPOCHS \
        --same_enc_dec True
    echo "[$(date '+%H:%M:%S')] Completed seed $seed"
done

echo "Evaluating reconstruction specificity..."
python hyperparam_eval_spatial_bandit.py \
    --lmbd $LMBD \
    --z $Z_DIM \
    --seeds "${SEEDS[*]}" \
    --output "$RESULTS_DIR/metrics.json"

echo "========================================"
echo "Completed at: $(date)"
echo "Results saved to: $RESULTS_DIR"
echo "========================================"
