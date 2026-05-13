#!/bin/bash
set -euo pipefail

mkdir -p logs

# Activate conda environment
source /ictstr01/home/hcai/marvin.mathony/tools/apps/mamba/etc/profile.d/conda.sh
conda activate RNNproject

# Parameter grid
LAMBDAS=(0 0.01 0.02 0.05 0.1)
Z_DIMS=(1 2 3)
SEEDS=(12 50)
EPOCHS=3000
DGP="sloutsky"
DATASET_ID=0

export PYTHONUNBUFFERED=1

# Check that human data exists
if [ ! -d "data_sloutsky" ]; then
  echo "ERROR: data_sloutsky directory not found!"
  exit 1
fi

# Loop over full grid sequentially
for Z_DIM in "${Z_DIMS[@]}"; do
  for LMBD in "${LAMBDAS[@]}"; do
    echo "========================================"
    echo "Hyperparameter Search (Sloutsky) - INTERACTIVE SEQ"
    echo "Lambda: $LMBD | Z_dim: $Z_DIM | Epochs: $EPOCHS | Seeds: ${SEEDS[*]}"
    echo "Starting at: $(date)"
    echo "========================================"

    # If you are inside an interactive GPU allocation, SLURM will set these.
    echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-N/A}"

    # Results dir for this combo
    RESULTS_DIR="hp_search_results_${DGP}/lmbd_${LMBD}_z_${Z_DIM}"
    mkdir -p "$RESULTS_DIR"

    # Optional: set GPU if available; harmless if not.
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

    # Train for each seed
    for seed in "${SEEDS[@]}"; do
      echo "[$(date '+%H:%M:%S')] Training seed $seed (lmbd=$LMBD, z=$Z_DIM)..."

      export HP_RUN_DIR="hp_search_runs_${DGP}/lmbd_${LMBD}_z_${Z_DIM}/seed_${seed}"
      mkdir -p "$HP_RUN_DIR"

      # tee logs per run (much nicer interactively)
      python run_Q_model.py \
        --seed "$seed" \
        --latent True \
        --dataset_id "$DATASET_ID" \
        --dgp "$DGP" \
        --lmbd "$LMBD" \
        --z "$Z_DIM" \
        --epochs "$EPOCHS" \
        2>&1 | tee -a "logs/hp_sloutsky_lmbd_${LMBD}_z_${Z_DIM}_seed_${seed}.log"

      echo "[$(date '+%H:%M:%S')] Completed seed $seed"
    done

    # Evaluate
    echo "[$(date '+%H:%M:%S')] Evaluating (lmbd=$LMBD, z=$Z_DIM)..."
    python hyperparam_eval_sloutsky.py \
      --lmbd "$LMBD" \
      --z "$Z_DIM" \
      --seeds "${SEEDS[*]}" \
      --output "$RESULTS_DIR/metrics.json" \
      2>&1 | tee -a "logs/hp_sloutsky_eval_lmbd_${LMBD}_z_${Z_DIM}.log"

    echo "========================================"
    echo "Completed combo at: $(date)"
    echo "Results saved to: $RESULTS_DIR"
    echo "========================================"
  done
done
