#!/bin/bash
# Interactive hyperparameter search pipeline for Sloutsky data.
# Runs everything sequentially on a single GPU node (no SLURM array).
#
# Usage:
#   ./run_hp_search_interactive.sh                  # Full pipeline
#   ./run_hp_search_interactive.sh --skip-vanilla    # Skip vanilla training (reuse existing)
#   ./run_hp_search_interactive.sh --eval-only       # Skip all training, run eval only
#
# Expected runtime: ~30 combos × 2 seeds × ~10 min each ≈ 10 hours for IDRNN
#                   + 5 seeds × ~5 min = ~25 min for Vanilla

set -euo pipefail

# ── Parse arguments ──────────────────────────────────────────────────────
SKIP_VANILLA=false
EVAL_ONLY=false

for arg in "$@"; do
    case $arg in
        --skip-vanilla)
            SKIP_VANILLA=true
            shift
            ;;
        --eval-only)
            EVAL_ONLY=true
            shift
            ;;
        *)
            echo "Unknown argument: $arg"
            echo "Usage: $0 [--skip-vanilla|--eval-only]"
            exit 1
            ;;
    esac
done

# ── Activate conda ───────────────────────────────────────────────────────
source /ictstr01/home/hcai/marvin.mathony/tools/apps/mamba/etc/profile.d/conda.sh
conda activate RNNproject

mkdir -p logs

export PYTHONUNBUFFERED=1

# ── Fixed parameters ─────────────────────────────────────────────────────
DGP="sloutsky"
DATASET_ID=0

# ── HP search grid (same as hp_search_sloutsky_decoding.sbatch) ──────────
Z_DIMS=(2 3 5)
LAMBDAS=(0.0 0.1 0.3 0.5 0.7)
STEP1_EPOCHS_LIST=(1000 3000)
HP_SEEDS=(12 50)
STEP2_EPOCHS=3000

# Vanilla training config
VANILLA_SEEDS=(12 50 76 100 142)
VANILLA_EPOCHS=3000

echo "========================================"
echo "HP Search Pipeline – Sloutsky (Interactive)"
echo "========================================"
echo "Running on node: $(hostname)"
echo "Starting at: $(date)"
echo "HP grid: z_dim=${Z_DIMS[*]} × lambda=${LAMBDAS[*]} × s1=${STEP1_EPOCHS_LIST[*]}"
echo "HP seeds: ${HP_SEEDS[*]}"
echo "Vanilla seeds: ${VANILLA_SEEDS[*]}"
N_COMBOS=$(( ${#Z_DIMS[@]} * ${#LAMBDAS[@]} * ${#STEP1_EPOCHS_LIST[@]} ))
N_TOTAL=$(( N_COMBOS * ${#HP_SEEDS[@]} ))
echo "Total IDRNN runs: $N_TOTAL ($N_COMBOS combos × ${#HP_SEEDS[@]} seeds)"
echo "========================================"

# Verify data exists
if [ ! -d "data_sloutsky" ]; then
    echo "ERROR: data_sloutsky directory not found!"
    exit 1
fi

if [ "$EVAL_ONLY" = true ]; then
    echo "Skipping all training (--eval-only mode)"
else

    # ═══════════════════════════════════════════════════════════════════
    # STEP 1: Train Vanilla models
    # ═══════════════════════════════════════════════════════════════════
    if [ "$SKIP_VANILLA" = false ]; then
        echo ""
        echo "╔══════════════════════════════════════════╗"
        echo "║  STEP 1: Training Vanilla models         ║"
        echo "╚══════════════════════════════════════════╝"
        for seed in "${VANILLA_SEEDS[@]}"; do
            echo "[$(date '+%H:%M:%S')] Training Vanilla seed=$seed ..."
            python run_Q_model.py \
                --seed $seed \
                --latent False \
                --dataset_id $DATASET_ID \
                --dgp $DGP \
                --epochs $VANILLA_EPOCHS
            echo "[$(date '+%H:%M:%S')] Completed Vanilla seed=$seed"
        done

        # Select best epoch for vanilla
        echo "[$(date '+%H:%M:%S')] Selecting best vanilla epoch by loss ..."
        python select_best_epoch_by_loss.py \
            --latent False --dataset_id $DATASET_ID --dgp $DGP \
            --min_epoch 100 --max_epoch 1000
    else
        echo ""
        echo "Skipping Vanilla training (--skip-vanilla). Using existing runs_vanilla_sloutsky/."
    fi

    # ═══════════════════════════════════════════════════════════════════
    # STEP 2: Train IDRNN models (HP grid)
    # ═══════════════════════════════════════════════════════════════════
    echo ""
    echo "╔══════════════════════════════════════════╗"
    echo "║  STEP 2: Training IDRNN HP grid          ║"
    echo "╚══════════════════════════════════════════╝"

    RUN_IDX=0
    for S1_EP in "${STEP1_EPOCHS_LIST[@]}"; do
        for Z_DIM in "${Z_DIMS[@]}"; do
            for LMBD in "${LAMBDAS[@]}"; do
                for seed in "${HP_SEEDS[@]}"; do
                    RUN_IDX=$((RUN_IDX + 1))

                    RUN_DIR="hp_search_runs_sloutsky_decoding/lmbd_${LMBD}_z_${Z_DIM}_s1_${S1_EP}/seed_${seed}"

                    # Skip if already trained (frozen_decoder + checkpoints exist)
                    if [ -d "$RUN_DIR/checkpoints" ] && [ -d "$RUN_DIR/frozen_decoder" ]; then
                        N_CKPTS=$(ls "$RUN_DIR/checkpoints/"epoch*.pt 2>/dev/null | wc -l)
                        if [ "$N_CKPTS" -ge 5 ]; then
                            echo "[$RUN_IDX/$N_TOTAL] SKIP (already trained): z=$Z_DIM lmbd=$LMBD s1=$S1_EP seed=$seed ($N_CKPTS checkpoints)"
                            continue
                        fi
                    fi

                    echo "[$RUN_IDX/$N_TOTAL] [$(date '+%H:%M:%S')] Training z=$Z_DIM lmbd=$LMBD s1=$S1_EP seed=$seed ..."

                    export HP_RUN_DIR="$RUN_DIR"

                    python run_Q_model.py \
                        --seed $seed \
                        --latent True \
                        --dataset_id $DATASET_ID \
                        --dgp $DGP \
                        --lmbd $LMBD \
                        --z $Z_DIM \
                        --step1_epochs $S1_EP \
                        --epochs $STEP2_EPOCHS

                    echo "[$RUN_IDX/$N_TOTAL] [$(date '+%H:%M:%S')] Completed."
                done
            done
        done
    done

fi  # end of training block

# ═══════════════════════════════════════════════════════════════════════
# STEP 3: Evaluate all HP combos for group decoding
# ═══════════════════════════════════════════════════════════════════════
echo ""
echo "╔══════════════════════════════════════════╗"
echo "║  STEP 3: Evaluating HP combos            ║"
echo "╚══════════════════════════════════════════╝"

python hp_eval_decoding_sloutsky.py \
    --output hp_decoding_results_sloutsky.json

echo ""
echo "========================================"
echo "HP search pipeline completed at: $(date)"
echo "Results: hp_decoding_results_sloutsky.json"
echo "========================================"
