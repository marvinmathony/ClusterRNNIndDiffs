#!/bin/bash
#SBATCH --job-name=hp_dec_%a
#SBATCH --output=logs/hp_dec_%a_%j.out
#SBATCH --error=logs/hp_dec_%a_%j.err
#SBATCH --open-mode=append
#SBATCH --array=0-29
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

# Hyperparameter search for IDRNN on Sloutsky data
# Goal: find HP combo where IDRNN beats Vanilla in group decoding
#
# Grid: z_dim(3) x lambda(5) x step1_epochs(2) = 30 combos
# Each combo trains with 2 seeds for rough exploration
#
# After all array tasks finish, run:
#   python hp_eval_decoding_sloutsky.py
# to evaluate group decoding and produce a summary table.

set -euo pipefail
mkdir -p logs

# Activate conda environment
source /ictstr01/home/hcai/marvin.mathony/tools/apps/mamba/etc/profile.d/conda.sh
conda activate RNNproject
export PATH="/ictstr01/home/hcai/marvin.mathony/tools/apps/mamba/envs/RNNproject/bin:$PATH"

TASK_ID=$SLURM_ARRAY_TASK_ID

# ── Parameter grid ──────────────────────────────────────────────────────
Z_DIMS=(2 3 5)
LAMBDAS=(0.0 0.1 0.3 0.5 0.7)
STEP1_EPOCHS_LIST=(1000 3000)
SEEDS=(12 50)
STEP2_EPOCHS=3000
DGP="sloutsky"
DATASET_ID=0

# ── Decode task ID → hyperparameters ────────────────────────────────────
# Layout: task = s1_idx * (n_z * n_lmbd) + z_idx * n_lmbd + lmbd_idx
N_Z=${#Z_DIMS[@]}          # 3
N_LMBD=${#LAMBDAS[@]}      # 5
N_PER_S1=$((N_Z * N_LMBD)) # 15

S1_IDX=$((TASK_ID / N_PER_S1))
REMAINDER=$((TASK_ID % N_PER_S1))
Z_IDX=$((REMAINDER / N_LMBD))
LMBD_IDX=$((REMAINDER % N_LMBD))

STEP1_EP=${STEP1_EPOCHS_LIST[$S1_IDX]}
Z_DIM=${Z_DIMS[$Z_IDX]}
LMBD=${LAMBDAS[$LMBD_IDX]}

echo "========================================"
echo "HP Search – Decoding (Sloutsky)"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID  |  Task: $TASK_ID"
echo "z_dim=$Z_DIM  lambda=$LMBD  step1=$STEP1_EP  step2=$STEP2_EPOCHS"
echo "Seeds: ${SEEDS[*]}"
echo "Starting at: $(date)"
echo "========================================"

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

# Verify data exists
if [ ! -d "data_sloutsky" ]; then
    echo "ERROR: data_sloutsky directory not found!"
    exit 1
fi

# ── Train IDRNN for each seed ──────────────────────────────────────────
for seed in "${SEEDS[@]}"; do
    echo "[$(date '+%H:%M:%S')] Training IDRNN seed=$seed z=$Z_DIM lmbd=$LMBD s1=$STEP1_EP ..."

    export HP_RUN_DIR="hp_search_results_sloutsky/lmbd_${LMBD}_z_${Z_DIM}_s1_${STEP1_EP}/seed_${seed}"

    python run_Q_model.py \
        --seed $seed \
        --latent True \
        --dataset_id $DATASET_ID \
        --dgp $DGP \
        --lmbd $LMBD \
        --z $Z_DIM \
        --step1_epochs $STEP1_EP \
        --epochs $STEP2_EPOCHS

    echo "[$(date '+%H:%M:%S')] Completed seed=$seed"
done

echo "========================================"
echo "Training done at: $(date)"
echo "Run 'python hp_eval_decoding_sloutsky.py' to evaluate decoding."
echo "========================================"
