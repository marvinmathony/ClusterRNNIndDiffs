#!/bin/bash
#SBATCH --job-name=dezfouli_ocv
#SBATCH --output=logs/dezfouli_ocv_%j.out
#SBATCH --error=logs/dezfouli_ocv_%j.err
#SBATCH --open-mode=append
#SBATCH --time=24:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH -p gpu_p
#SBATCH --qos gpu_normal
#SBATCH --constraint=a100_80gb|h100_80gb
#SBATCH --nice=10000

#SBATCH --mail-type=ALL
#SBATCH --mail-user=marvin.mathony@helmholtz-munich.de

# Outer 3-fold CV pipeline for Dezfouli dataset.
# Every participant is a test participant exactly once.
# Produces: data_dezfouli/fold{k}/seed_{s}/ for latents and NLL;
#           plots_dezfouli/outer_cv_*.png for analysis.

source /ictstr01/home/hcai/marvin.mathony/tools/apps/mamba/etc/profile.d/conda.sh
conda activate RNNproject

set -uo pipefail
mkdir -p logs

DGP="dezfouli"
N_FOLDS=3

# Seeds: override default by passing a space-separated list as $1
if [ -n "${1:-}" ]; then
    read -ra SEEDS <<< "$1"
else
    SEEDS=(200 300 400 500 600 999 2021 2022 2023 2024)
fi

# Hyperparameters
LMBD=0.005
Z_DIM=5
HIDDEN=5
ENC_HIDDEN=20

echo "========================================"
echo "Dezfouli Outer CV Pipeline"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "DGP: $DGP  |  Folds: $N_FOLDS  |  Seeds: ${SEEDS[*]}"
echo "Running on: $(hostname)  |  Start: $(date)"
echo "========================================"

# --- Step 1: Generate fold data ---
echo "Generating outer CV fold data..."
if [ ! -d "data_dezfouli/fold0" ]; then
    python load_dezfouli.py
else
    echo "  Fold directories already exist, skipping data generation."
fi

N_GPUS=${SLURM_GPUS_ON_NODE:-4}
echo "GPUs available: $N_GPUS"

# --- Step 2: Train folds sequentially; seeds × model types in parallel per fold ---
for fold in $(seq 0 $((N_FOLDS - 1))); do
    echo "──────────────────────────────────────"
    echo "Training fold $fold / $((N_FOLDS - 1))  [$(date '+%H:%M:%S')]"
    echo "──────────────────────────────────────"
    PIDS=()
    TRAIN_LOGS=()
    GPU_COUNTER=0

    for seed in "${SEEDS[@]}"; do
        # IDRNN
        GPU_ID=$((GPU_COUNTER % N_GPUS))
        GPU_COUNTER=$((GPU_COUNTER + 1))
        LOG="logs/train_latent_dez_fold${fold}_seed${seed}.log"
        TRAIN_LOGS+=("$LOG")
        echo "  Launching IDRNN  seed=$seed → GPU $GPU_ID → $LOG"
        CUDA_VISIBLE_DEVICES=$GPU_ID python run_Q_model.py \
            --seed $seed \
            --latent True \
            --dgp $DGP \
            --fold $fold \
            --lmbd $LMBD \
            --z $Z_DIM \
            --hidden $HIDDEN \
            --enc_hidden $ENC_HIDDEN \
            --epochs 3000 \
            --step1_epochs 3000 \
            --same_enc_dec True \
            > "$LOG" 2>&1 &
        PIDS+=($!)

        # Vanilla
        GPU_ID=$((GPU_COUNTER % N_GPUS))
        GPU_COUNTER=$((GPU_COUNTER + 1))
        LOG="logs/train_vanilla_dez_fold${fold}_seed${seed}.log"
        TRAIN_LOGS+=("$LOG")
        echo "  Launching Vanilla seed=$seed → GPU $GPU_ID → $LOG"
        CUDA_VISIBLE_DEVICES=$GPU_ID python run_Q_model.py \
            --seed $seed \
            --latent False \
            --dgp $DGP \
            --fold $fold \
            --hidden $HIDDEN \
            --epochs 3000 \
            > "$LOG" 2>&1 &
        PIDS+=($!)
    done

    echo "  Waiting for fold $fold training..."
    TRAIN_FAILED=0
    for i in "${!PIDS[@]}"; do
        pid=${PIDS[$i]}
        log=${TRAIN_LOGS[$i]}
        if ! wait "$pid"; then
            echo "  ERROR: training job (PID $pid, log $log) failed!"
            TRAIN_FAILED=1
        fi
    done
    if [ "$TRAIN_FAILED" -ne 0 ]; then
        echo "One or more training jobs for fold $fold failed. Aborting."
        exit 1
    fi
    echo "  Fold $fold training complete  [$(date '+%H:%M:%S')]"
done
echo "All folds trained successfully."

# --- Step 3: Test — one call per fold per model type ---
echo "Testing models for each fold..."
for fold in $(seq 0 $((N_FOLDS - 1))); do
    echo "  Testing IDRNN   fold=$fold ..."
    python testing_script.py --latent True  --dgp $DGP --fold $fold --model_fitting False
    echo "  Testing Vanilla fold=$fold ..."
    python testing_script.py --latent False --dgp $DGP --fold $fold --model_fitting False
done

# --- Step 4: Outer CV analysis ---
echo "Running outer CV analysis..."
python analyze_outer_cv.py --dgp $DGP

echo "========================================"
echo "Outer CV pipeline completed: $(date)"
echo "Results:"
echo "  Checkpoints : runs_dezfouli/fold{k}/seed_{s}/"
echo "  Latents/NLL : data_dezfouli/fold{k}/seed_{s}/"
echo "  Plots       : plots_dezfouli/outer_cv_*.png"
echo "========================================"
