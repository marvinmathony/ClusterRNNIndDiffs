#!/bin/bash
#SBATCH --job-name=hp_sloutsky_%a
#SBATCH --output=logs/hp_sloutsky_%a_%j.out
#SBATCH --error=logs/hp_sloutsky_%a_%j.err
#SBATCH --open-mode=append
#SBATCH --array=0-45
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

# Hyperparameter search for IDRNN on Sloutsky human data
#
# Grid: hidden_pair x z_dim x lambda = 3 x 3 x 5 = 45 IDRNN combinations
#   + 1 vanilla baseline (task 45)
#
# hidden_pairs: (common_process_hidden:enc_hidden)
#   Hypothesis: increasing enc_hidden relative to hidden pushes more individual
#   variance into the IDRNN latent z.
#
# Task ID mapping (0-44):
#   PAIR_IDX  = task_id / 15        → selects hidden pair (0,1,2)
#   remainder = task_id % 15
#   Z_IDX     = remainder / 5       → selects z_dim (0,1,2)
#   LMBD_IDX  = remainder % 5       → selects lambda (0,1,2,3,4)
# Task 45: vanilla baseline

set -euo pipefail
mkdir -p logs

# Activate conda environment
source /ictstr01/home/hcai/marvin.mathony/tools/apps/mamba/etc/profile.d/conda.sh
conda activate RNNproject

# Get task ID
TASK_ID=$SLURM_ARRAY_TASK_ID

# Parameter grid
LAMBDAS=(0 0.005 0.01 0.02 0.05)          # 5 lambda values
Z_DIMS=(3 5 8)                             # 3 z_dim values
# hidden_pairs: "common_process_hidden:enc_hidden"
# enc_hidden >= hidden (common process constrained, encoder can be larger)
HIDDEN_PAIRS=("5:5" "5:10" "5:20")        # 3 pairs → 45 IDRNN tasks total
VANILLA_HIDDEN=5                           # hidden size for vanilla baseline

SEEDS=(12 50 76)  # 3 seeds for more stable estimates
EPOCHS=3000
DGP="sloutsky"
DATASET_ID=0   # Not used for human data, but required by scripts

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

# Check that human data exists
if [ ! -d "data_sloutsky" ]; then
    echo "ERROR: data_sloutsky directory not found!"
    exit 1
fi

# Task 45: train vanilla baseline (no lambda/z_dim sweep) for all seeds
if [ "$TASK_ID" -eq 45 ]; then
    echo "========================================"
    echo "Vanilla Baseline Training (Sloutsky)"
    echo "========================================"
    echo "Job ID: $SLURM_JOB_ID"
    echo "Task ID: $TASK_ID"
    echo "Epochs: $EPOCHS"
    echo "Hidden: $VANILLA_HIDDEN"
    echo "Seeds: ${SEEDS[*]}"
    echo "DGP: $DGP"
    echo "Starting at: $(date)"
    echo "========================================"

    for seed in "${SEEDS[@]}"; do
        echo "[$(date '+%H:%M:%S')] Training vanilla seed $seed..."
        python run_Q_model.py \
            --seed $seed \
            --latent False \
            --dataset_id $DATASET_ID \
            --dgp $DGP \
            --hidden $VANILLA_HIDDEN \
            --epochs $EPOCHS
        echo "[$(date '+%H:%M:%S')] Completed vanilla seed $seed"
    done

    echo "========================================"
    echo "Vanilla training completed at: $(date)"
    echo "Results in: runs_vanilla_${DGP}/"
    echo "========================================"
    exit 0
fi

# Tasks 0-44: IDRNN hyperparameter sweep
# Decode 3D index: pair × (z × lambda)
N_ZL=15   # 3 z_dims × 5 lambdas per pair
PAIR_IDX=$((TASK_ID / N_ZL))
REMAINDER=$((TASK_ID % N_ZL))
Z_IDX=$((REMAINDER / 5))
LMBD_IDX=$((REMAINDER % 5))

PAIR=${HIDDEN_PAIRS[$PAIR_IDX]}
HIDDEN=${PAIR%%:*}
ENC_HIDDEN=${PAIR##*:}
LMBD=${LAMBDAS[$LMBD_IDX]}
Z_DIM=${Z_DIMS[$Z_IDX]}

echo "========================================"
echo "Hyperparameter Search Job (Sloutsky)"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Task ID: $TASK_ID"
echo "Lambda: $LMBD"
echo "Z_dim: $Z_DIM"
echo "Hidden (common process): $HIDDEN"
echo "Enc_hidden (IDRNN encoder): $ENC_HIDDEN"
echo "Epochs: $EPOCHS"
echo "Seeds: ${SEEDS[*]}"
echo "DGP: $DGP"
echo "Starting at: $(date)"
echo "========================================"

# Results and run directories encode all 4 hyperparameters
RESULTS_DIR="hp_search_results_${DGP}/lmbd_${LMBD}_z_${Z_DIM}_h_${HIDDEN}_eh_${ENC_HIDDEN}"
mkdir -p "$RESULTS_DIR"

# Train with each seed
for seed in "${SEEDS[@]}"; do
    echo "[$(date '+%H:%M:%S')] Training with seed $seed..."

    export HP_RUN_DIR="hp_search_runs_${DGP}/lmbd_${LMBD}_z_${Z_DIM}_h_${HIDDEN}_eh_${ENC_HIDDEN}/seed_${seed}"

    python run_Q_model.py \
        --seed $seed \
        --latent True \
        --dataset_id $DATASET_ID \
        --dgp $DGP \
        --lmbd $LMBD \
        --z $Z_DIM \
        --hidden $HIDDEN \
        --enc_hidden $ENC_HIDDEN \
        --epochs $EPOCHS \
        --step1_epochs $EPOCHS \
        --same_enc_dec True

    echo "[$(date '+%H:%M:%S')] Completed seed $seed"
done

# Run evaluation — reads cv_val_loss from each seed's config.json
echo "Evaluating results..."
python hyperparam_eval_sloutsky.py \
    --lmbd $LMBD \
    --z $Z_DIM \
    --hidden $HIDDEN \
    --enc_hidden $ENC_HIDDEN \
    --seeds "${SEEDS[*]}" \
    --output "$RESULTS_DIR/metrics.json"

echo "========================================"
echo "Completed at: $(date)"
echo "Results saved to: $RESULTS_DIR"
echo "========================================"
