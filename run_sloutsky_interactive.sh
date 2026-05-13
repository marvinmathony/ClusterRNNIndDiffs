#!/bin/bash
# Interactive script for training on human data (Sloutsky dataset)
# Run this on an interactive node with GPU access
#
# Usage:
#   ./run_sloutsky_interactive.sh           # Run full pipeline
#   ./run_sloutsky_interactive.sh --skip-training    # Skip training, run testing/plotting only
#   ./run_sloutsky_interactive.sh --training-only    # Run training only
#   ./run_sloutsky_interactive.sh --test-only        # Run testing only
#   ./run_sloutsky_interactive.sh --plot-only        # Run plotting only

# Parse arguments
SKIP_TRAINING=false
TRAINING_ONLY=false
TEST_ONLY=false
PLOT_ONLY=false

for arg in "$@"; do
    case $arg in
        --skip-training)
            SKIP_TRAINING=true
            shift
            ;;
        --training-only)
            TRAINING_ONLY=true
            shift
            ;;
        --test-only)
            TEST_ONLY=true
            shift
            ;;
        --plot-only)
            PLOT_ONLY=true
            shift
            ;;
        *)
            echo "Unknown argument: $arg"
            echo "Usage: $0 [--skip-training|--training-only|--test-only|--plot-only]"
            exit 1
            ;;
    esac
done

# Activate conda environment
source /ictstr01/home/hcai/marvin.mathony/tools/apps/mamba/etc/profile.d/conda.sh
conda activate RNNproject

# Exit immediately if a command exits with a non-zero status
set -euo pipefail

# Create logs directory
mkdir -p logs

# Data Generating Process - fixed for this script
DGP="sloutsky"
DATASET_ID=0  # Not used for human data, but required by scripts

# Print job info
echo "========================================"
echo "Sloutsky Human Data Pipeline (Interactive)"
echo "========================================"
echo "Running on node: $(hostname)"
echo "Starting at: $(date)"
echo "========================================"

# Set environment variables
export PYTHONUNBUFFERED=1

# Configuration
LMBD=0.01
Z_DIM=3
#SEEDS=(12 50)
SEEDS=(12 50 76 100 142)

# Step 1: Skip data generation - human data already exists in data_sloutsky/
echo "Using pre-existing human data from data_sloutsky/..."
if [ ! -d "data_sloutsky" ]; then
    echo "ERROR: data_sloutsky directory not found!"
    exit 1
fi

# Training steps
if [ "$SKIP_TRAINING" = false ] && [ "$TEST_ONLY" = false ] && [ "$PLOT_ONLY" = false ]; then
    # Step 2: Train latent models
    echo "Training latent models for Sloutsky data..."
    for seed in "${SEEDS[@]}"; do
        echo "  [$(date '+%Y-%m-%d %H:%M:%S')] Training latent model with seed $seed..."
        python run_Q_model.py \
            --seed $seed \
            --latent True \
            --dataset_id $DATASET_ID \
            --dgp $DGP \
            --lmbd $LMBD \
            --z $Z_DIM \
            --epochs 3000 \
            --step1_epochs 3000
        echo "  [$(date '+%Y-%m-%d %H:%M:%S')] Completed latent model with seed $seed"
    done

    # Step 3: Train vanilla models
    echo "Training vanilla models for Sloutsky data..."
    for seed in "${SEEDS[@]}"; do
        echo "  [$(date '+%Y-%m-%d %H:%M:%S')] Training vanilla model with seed $seed..."
        python run_Q_model.py \
            --seed $seed \
            --latent False \
            --dataset_id $DATASET_ID \
            --dgp $DGP \
            --epochs 3000
        echo "  [$(date '+%Y-%m-%d %H:%M:%S')] Completed vanilla model with seed $seed"
    done

fi

if [ "$TRAINING_ONLY" = true ]; then
    # Run epoch selection before exiting
    echo "Selecting best epochs for Sloutsky data..."
    python select_best_epoch_by_specificity.py --latent True --dataset_id $DATASET_ID --dgp $DGP --min_epoch 100 --max_epoch 2000
    python select_best_epoch_by_loss.py --latent False --dataset_id $DATASET_ID --dgp $DGP --min_epoch 100 --max_epoch 2000
    echo "Training complete. Exiting (--training-only mode)."
    exit 0
fi

# Step 4: Select best epochs (run before testing/plotting)
if [ "$PLOT_ONLY" = false ]; then
    echo "Selecting best epochs for Sloutsky data..."
    python select_best_epoch_by_specificity.py --latent True --dataset_id $DATASET_ID --dgp $DGP --min_epoch 100 --max_epoch 2000
    python select_best_epoch_by_loss.py --latent False --dataset_id $DATASET_ID --dgp $DGP --min_epoch 100 --max_epoch 2000
fi

# Testing step
if [ "$PLOT_ONLY" = false ]; then
    # Step 5: Test models
    echo "Testing models for Sloutsky data..."
    python testing_script.py --latent True --dataset_id $DATASET_ID --dgp $DGP --model_fitting False
    python testing_script.py --latent False --dataset_id $DATASET_ID --dgp $DGP --model_fitting False
fi

if [ "$TEST_ONLY" = true ]; then
    echo "Testing complete. Exiting (--test-only mode)."
    exit 0
fi

# Plotting step
# Step 6: Plot latent representations
echo "Plotting latent representations..."
mkdir -p plots_sloutsky

# Plot IDRNN latents with different dimensionality reduction methods
echo "  Plotting IDRNN latents..."
python plot_sloutsky_latents.py --latent True --dim_reduction pca --avg False
python plot_sloutsky_latents.py --latent True --dim_reduction tsne --avg False

# Plot Vanilla latents
echo "  Plotting Vanilla latents..."
python plot_sloutsky_latents.py --latent False --dim_reduction pca --avg True
python plot_sloutsky_latents.py --latent False --dim_reduction tsne --avg True

echo "========================================"
echo "Sloutsky pipeline completed at: $(date)"
echo "Results saved to:"
echo "  - runs_sloutsky/ (IDRNN model checkpoints)"
echo "  - runs_vanilla_sloutsky/ (Vanilla model checkpoints)"
echo "  - data_sloutsky/ (latent tensors, results)"
echo "  - plots_sloutsky/ (latent visualizations)"
echo "========================================"
