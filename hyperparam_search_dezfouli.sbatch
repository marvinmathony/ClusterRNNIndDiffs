#!/bin/bash
#SBATCH --job-name=hp_dezfouli_%a
#SBATCH --output=logs/hp_dezfouli_%a_%j.out
#SBATCH --error=logs/hp_dezfouli_%a_%j.err
#SBATCH --open-mode=append
#SBATCH --array=0-135
#SBATCH --time=10:00:00
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

# Hyperparameter search for IDRNN on Dezfouli human data
#
# Grid: hidden_pair x z_dim x lambda x step1_epochs = 3 x 3 x 5 x 3 = 135 IDRNN combinations
#   + 1 vanilla baseline (task 135)
#
# Task ID mapping (0-134):
#   PAIR_IDX   = task_id // 45       → selects hidden pair (0,1,2)   [3 pairs × 45 each]
#   remainder  = task_id % 45
#   Z_IDX      = remainder // 15     → selects z_dim (0,1,2)         [3 z × 15 each]
#   remainder2 = remainder % 15
#   LMBD_IDX   = remainder2 // 3    → selects lambda (0,1,2,3,4)    [5 lambdas × 3 each]
#   S1_IDX     = remainder2 % 3     → selects step1_epochs (0,1,2)  [3 values]
# Task 135: vanilla baseline

set -euo pipefail
mkdir -p logs

source /ictstr01/home/hcai/marvin.mathony/tools/apps/mamba/etc/profile.d/conda.sh
conda activate RNNproject

TASK_ID=$SLURM_ARRAY_TASK_ID

# Parameter grid
LAMBDAS=(0 0.005 0.01 0.02 0.05)
Z_DIMS=(3 5 8)
HIDDEN_PAIRS=("5:5" "5:10" "5:20")
STEP1_EPOCHS=(500 1500 3000)
VANILLA_HIDDEN=5

SEEDS=(12 50 76)
EPOCHS=3000
DGP="dezfouli"
DATASET_ID=0

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

if [ ! -d "data_dezfouli" ]; then
    echo "ERROR: data_dezfouli directory not found! Run load_dezfouli.py first."
    exit 1
fi

# Task 135: vanilla baseline
if [ "$TASK_ID" -eq 135 ]; then
    echo "========================================"
    echo "Vanilla Baseline Training (Dezfouli)"
    echo "========================================"
    echo "Job ID: $SLURM_JOB_ID  |  Task: $TASK_ID  |  Hidden: $VANILLA_HIDDEN"
    echo "Seeds: ${SEEDS[*]}  |  Start: $(date)"
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

    echo "Vanilla training completed: $(date)"
    echo "Results in: runs_vanilla_${DGP}/"
    exit 0
fi

# Tasks 0-134: IDRNN sweep
PAIR_IDX=$((TASK_ID / 45))
REMAINDER=$((TASK_ID % 45))
Z_IDX=$((REMAINDER / 15))
REMAINDER2=$((REMAINDER % 15))
LMBD_IDX=$((REMAINDER2 / 3))
S1_IDX=$((REMAINDER2 % 3))

PAIR=${HIDDEN_PAIRS[$PAIR_IDX]}
HIDDEN=${PAIR%%:*}
ENC_HIDDEN=${PAIR##*:}
LMBD=${LAMBDAS[$LMBD_IDX]}
Z_DIM=${Z_DIMS[$Z_IDX]}
STEP1=${STEP1_EPOCHS[$S1_IDX]}

echo "========================================"
echo "Hyperparameter Search Job (Dezfouli)"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID  |  Task: $TASK_ID"
echo "Lambda: $LMBD  |  Z_dim: $Z_DIM"
echo "Hidden: $HIDDEN  |  Enc_hidden: $ENC_HIDDEN"
echo "Step1_epochs: $STEP1  |  Total_epochs: $EPOCHS"
echo "Seeds: ${SEEDS[*]}  |  Start: $(date)"
echo "========================================"

RESULTS_DIR="hp_search_results_dezfouli/lmbd_${LMBD}_z_${Z_DIM}_h_${HIDDEN}_eh_${ENC_HIDDEN}_s1_${STEP1}"
mkdir -p "$RESULTS_DIR"

for seed in "${SEEDS[@]}"; do
    echo "[$(date '+%H:%M:%S')] Training seed $seed..."

    export HP_RUN_DIR="hp_search_runs_dezfouli/lmbd_${LMBD}_z_${Z_DIM}_h_${HIDDEN}_eh_${ENC_HIDDEN}_s1_${STEP1}/seed_${seed}"

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
        --step1_epochs $STEP1 \
        --same_enc_dec True

    echo "[$(date '+%H:%M:%S')] Completed seed $seed"
done

echo "Evaluating results..."
python hyperparam_eval_dezfouli.py \
    --lmbd $LMBD \
    --z $Z_DIM \
    --hidden $HIDDEN \
    --enc_hidden $ENC_HIDDEN \
    --step1_epochs $STEP1 \
    --seeds "${SEEDS[*]}" \
    --output "$RESULTS_DIR/metrics.json"

echo "Completed: $(date)  |  Results: $RESULTS_DIR"
