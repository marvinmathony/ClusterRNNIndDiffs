#!/bin/bash
#SBATCH --job-name=hp_thalmann_%a
#SBATCH --output=logs/hp_thalmann_%a_%j.out
#SBATCH --error=logs/hp_thalmann_%a_%j.err
#SBATCH --open-mode=append
#SBATCH --array=0-135
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH -p gpu_p
#SBATCH --qos gpu_normal
#SBATCH --constraint=a100_80gb|h100_80gb
#SBATCH --nice=10000
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=marvin.mathony@helmholtz-munich.de

# Hyperparameter search for IDRNN on Thalmann two-task data.
#
# Motivation: investigate whether posterior collapse is a local-minimum
# artefact of the current hyperparameters rather than a fundamental problem.
# The block_weights fix (restless-only NLL in step-2) is active for DGP=thalmann.
#
# Grid: lambda × z_dim × hidden_pair × step1_epochs = 5 × 3 × 3 × 3 = 135 IDRNN combos
#   + 1 vanilla baseline (task 135)
#
# Task ID mapping (0-134):
#   PAIR_IDX  = task_id // 45      → hidden pair    [3 pairs  × 45 each]
#   Z_IDX     = (task_id % 45) // 15  → z_dim       [3 z_dims × 15 each]
#   LMBD_IDX  = (task_id % 15) // 3   → lambda      [5 lambdas × 3 each]
#   S1_IDX    = task_id % 3           → step1_epochs [3 values]
# Task 135: vanilla baseline

set -euo pipefail
mkdir -p logs

source /ictstr01/home/hcai/marvin.mathony/tools/apps/mamba/etc/profile.d/conda.sh
conda activate RNNproject

TASK_ID=$SLURM_ARRAY_TASK_ID

# ── Parameter grid ─────────────────────────────────────────────────────────────
LAMBDAS=(0.01 0.05 0.1 0.2 0.5)
Z_DIMS=(3 5 10)
# decoder_hidden:encoder_hidden pairs
HIDDEN_PAIRS=("5:5" "10:10" "20:20")
STEP1_EPOCHS=(500 1000 2000)

# Fixed for all thalmann runs
TASK_EMB_DIM=4
EPOCHS=2000
DGP="thalmann"
DATASET_ID=0
VANILLA_HIDDEN=5

SEEDS=(12 50 76)

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

if [ ! -d "data_thalmann" ]; then
    echo "ERROR: data_thalmann/ not found — run load_thalmann.py first."
    exit 1
fi

# ── Task 135: vanilla baseline ────────────────────────────────────────────────
if [ "$TASK_ID" -eq 135 ]; then
    echo "========================================"
    echo "Vanilla Baseline Training (Thalmann)"
    echo "========================================"
    echo "Job ID: $SLURM_JOB_ID  |  Task: $TASK_ID"
    echo "Hidden: $VANILLA_HIDDEN  |  Seeds: ${SEEDS[*]}  |  Start: $(date)"
    echo "========================================"

    for seed in "${SEEDS[@]}"; do
        echo "[$(date '+%H:%M:%S')] Training vanilla seed $seed..."
        export HP_RUN_DIR="hp_search_runs_thalmann/vanilla/seed_${seed}"
        python run_Q_model.py \
            --seed $seed \
            --latent False \
            --dataset_id $DATASET_ID \
            --dgp $DGP \
            --hidden $VANILLA_HIDDEN \
            --epochs $EPOCHS \
            --task_emb_dim $TASK_EMB_DIM
        echo "[$(date '+%H:%M:%S')] Completed vanilla seed $seed"
    done
    echo "Vanilla completed: $(date)"
    exit 0
fi

# ── Tasks 0-134: IDRNN sweep ──────────────────────────────────────────────────
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
echo "Hyperparameter Search (Thalmann IDRNN)"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID  |  Task: $TASK_ID"
echo "Lambda: $LMBD  |  Z_dim: $Z_DIM"
echo "Dec_hidden: $HIDDEN  |  Enc_hidden: $ENC_HIDDEN"
echo "Step1_epochs: $STEP1  |  Total_epochs: $EPOCHS"
echo "Task_emb_dim: $TASK_EMB_DIM"
echo "Seeds: ${SEEDS[*]}  |  Start: $(date)"
echo "Note: block_weights=task_ids (restless-only NLL) active for DGP=thalmann"
echo "========================================"

TAG="lmbd_${LMBD}_z_${Z_DIM}_h_${HIDDEN}_eh_${ENC_HIDDEN}_s1_${STEP1}"
RESULTS_DIR="hp_search_results_thalmann/${TAG}"
mkdir -p "$RESULTS_DIR"

for seed in "${SEEDS[@]}"; do
    echo "[$(date '+%H:%M:%S')] Training seed $seed..."

    export HP_RUN_DIR="hp_search_runs_thalmann/${TAG}/seed_${seed}"

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
        --task_emb_dim $TASK_EMB_DIM \
        --same_enc_dec False

    echo "[$(date '+%H:%M:%S')] Completed seed $seed"
done

echo "Evaluating..."
python hyperparam_eval_thalmann.py \
    --lmbd $LMBD \
    --z $Z_DIM \
    --hidden $HIDDEN \
    --enc_hidden $ENC_HIDDEN \
    --step1_epochs $STEP1 \
    --seeds "${SEEDS[*]}" \
    --output "$RESULTS_DIR/metrics.json"

echo "Completed: $(date)  |  Results: $RESULTS_DIR"
