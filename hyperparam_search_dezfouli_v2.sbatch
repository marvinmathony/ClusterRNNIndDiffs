#!/bin/bash
#SBATCH --job-name=hp_dez_v2_%a
#SBATCH --output=logs/hp_dez_v2_%a_%j.out
#SBATCH --error=logs/hp_dez_v2_%a_%j.err
#SBATCH --open-mode=append
#SBATCH --array=0-48
#SBATCH --time=12:00:00
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

# Extended Dezfouli HP search — higher lambda range.
#
# Rationale: previous search (v1) covered λ∈{0,0.005,0.01,0.02,0.05} and found
# monotonically decreasing CV val loss; 0.05 was the ceiling.  This sweep probes
# λ∈{0.1,0.2,0.5,1.0} while narrowing the grid to the promising region:
#   z∈{5,8} × enc_hidden∈{5,10,20} × step1∈{1500,3000}   (decoder hidden=5 fixed)
#
# Grid: 4 λ × 2 z × 3 enc_hidden × 2 step1 = 48 IDRNN combos
#   + task 48: vanilla baseline (re-run for fresh reference)
#
# Task ID mapping (0-47):
#   S1_IDX   = task_id % 2             → step1_epochs   [2 values]
#   LMBD_IDX = (task_id // 2) % 4      → lambda         [4 values]
#   Z_IDX    = (task_id // 8) % 2      → z_dim          [2 values]
#   EH_IDX   = task_id // 16           → enc_hidden      [3 values]
# Task 48: vanilla baseline

set -euo pipefail
mkdir -p logs

source /ictstr01/home/hcai/marvin.mathony/tools/apps/mamba/etc/profile.d/conda.sh
conda activate RNNproject

TASK_ID=$SLURM_ARRAY_TASK_ID

# ── Parameter grid ──────────────────────────────────────────────────────────────
LAMBDAS=(0.1 0.2 0.5 1.0)
Z_DIMS=(5 8)
ENC_HIDDENS=(5 10 20)
STEP1_EPOCHS=(1500 3000)

# Fixed
HIDDEN=5          # decoder hidden size (best from v1)
EPOCHS=3000
DGP="dezfouli"
DATASET_ID=0
VANILLA_HIDDEN=5

SEEDS=(12 50 76)

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

if [ ! -d "data_dezfouli" ]; then
    echo "ERROR: data_dezfouli/ not found — run load_dezfouli.py first."
    exit 1
fi

# ── Task 48: vanilla baseline ─────────────────────────────────────────────────
if [ "$TASK_ID" -eq 48 ]; then
    echo "========================================"
    echo "Vanilla Baseline Training (Dezfouli v2)"
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
    echo "Vanilla completed: $(date)"
    exit 0
fi

# ── Tasks 0-47: IDRNN sweep ───────────────────────────────────────────────────
S1_IDX=$((TASK_ID % 2))
LMBD_IDX=$(((TASK_ID / 2) % 4))
Z_IDX=$(((TASK_ID / 8) % 2))
EH_IDX=$((TASK_ID / 16))

LMBD=${LAMBDAS[$LMBD_IDX]}
Z_DIM=${Z_DIMS[$Z_IDX]}
ENC_HIDDEN=${ENC_HIDDENS[$EH_IDX]}
STEP1=${STEP1_EPOCHS[$S1_IDX]}

echo "========================================"
echo "Extended HP Search — Dezfouli IDRNN v2"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID  |  Task: $TASK_ID"
echo "Lambda: $LMBD  |  Z_dim: $Z_DIM"
echo "Dec_hidden: $HIDDEN  |  Enc_hidden: $ENC_HIDDEN"
echo "Step1_epochs: $STEP1  |  Total_epochs: $EPOCHS"
echo "Seeds: ${SEEDS[*]}  |  Start: $(date)"
echo "========================================"

TAG="lmbd_${LMBD}_z_${Z_DIM}_h_${HIDDEN}_eh_${ENC_HIDDEN}_s1_${STEP1}"
RESULTS_DIR="hp_search_results_dezfouli/${TAG}"
mkdir -p "$RESULTS_DIR"

for seed in "${SEEDS[@]}"; do
    echo "[$(date '+%H:%M:%S')] Training seed $seed..."

    export HP_RUN_DIR="hp_search_runs_dezfouli/${TAG}/seed_${seed}"

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

echo "Evaluating..."
python hyperparam_eval_dezfouli.py \
    --lmbd $LMBD \
    --z $Z_DIM \
    --hidden $HIDDEN \
    --enc_hidden $ENC_HIDDEN \
    --step1_epochs $STEP1 \
    --seeds "${SEEDS[*]}" \
    --output "$RESULTS_DIR/metrics.json"

echo "Completed: $(date)  |  Results: $RESULTS_DIR"
