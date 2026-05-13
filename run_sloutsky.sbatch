#!/bin/bash
#SBATCH --job-name=sloutsky
#SBATCH --output=logs/sloutsky_%j.out
#SBATCH --error=logs/sloutsky_%j.err
#SBATCH --open-mode=append
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH -p gpu_p
#SBATCH --qos gpu_normal
#SBATCH --constraint=a100_80gb|h100_80gb
#SBATCH --nice=10000

#SBATCH --mail-type=ALL
#SBATCH --mail-user=marvin.mathony@helmholtz-munich.de

# Pipeline for training on human data (Sloutsky dataset)
# No data generation needed - uses pre-existing human behavioral data

# Activate conda environment
source /ictstr01/home/hcai/marvin.mathony/tools/apps/mamba/etc/profile.d/conda.sh
conda activate RNNproject

set -uo pipefail

# Create logs directory
mkdir -p logs

# Data Generating Process - fixed for this script
DGP="sloutsky"
DATASET_ID=0  # Not used for human data, but required by scripts

# Print job info
echo "========================================"
echo "Sloutsky Human Data Pipeline"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "DGP: $DGP"
echo "Running on node: $(hostname)"
echo "Starting at: $(date)"
echo "========================================"

# Set environment variables
export PYTHONUNBUFFERED=1
N_GPUS=${SLURM_GPUS_ON_NODE:-4}
echo "GPUs available: $N_GPUS"

# Configuration
LMBD=0.005
Z_DIM=5
HIDDEN=5        # hidden size of common-process (decoder) GRU
ENC_HIDDEN=20    # hidden size of IDRNN encoder GRU (increase relative to HIDDEN to get more ID variance)

# Seeds: override default by passing a space-separated list as the first argument
# e.g.  sbatch run_sloutsky.sbatch "200 300 400 500 600"
if [ -n "${1:-}" ]; then
    read -ra SEEDS <<< "$1"
else
    SEEDS=(200 300 400 500 600 999 2021 2022 2023 2024 2025)
fi
echo "Seeds: ${SEEDS[*]}"

# Step 1: Skip data generation - human data already exists in data_sloutsky/
echo "Using pre-existing human data from data_sloutsky/..."
if [ ! -d "data_sloutsky" ]; then
    echo "ERROR: data_sloutsky directory not found!"
    exit 1
fi

# Step 2+3: Train all seeds × model types in parallel, round-robin across GPUs
echo "Training all models in parallel (latent + vanilla, all seeds)..."
PIDS=()
TRAIN_LOGS=()
GPU_COUNTER=0

for seed in "${SEEDS[@]}"; do
    GPU_ID=$((GPU_COUNTER % N_GPUS))
    GPU_COUNTER=$((GPU_COUNTER + 1))
    LOG_LATENT="logs/train_latent_seed${seed}.log"
    TRAIN_LOGS+=("$LOG_LATENT")
    echo "  [$(date '+%Y-%m-%d %H:%M:%S')] Launching latent model seed $seed on GPU $GPU_ID -> $LOG_LATENT"
    CUDA_VISIBLE_DEVICES=$GPU_ID python run_Q_model.py \
        --seed $seed \
        --latent True \
        --dataset_id $DATASET_ID \
        --dgp $DGP \
        --lmbd $LMBD \
        --z $Z_DIM \
        --hidden $HIDDEN \
        --enc_hidden $ENC_HIDDEN \
        --epochs 3000 \
        --step1_epochs 3000 \
        --same_enc_dec True \
        > "$LOG_LATENT" 2>&1 &
    PIDS+=($!)
done

for seed in "${SEEDS[@]}"; do
    GPU_ID=$((GPU_COUNTER % N_GPUS))
    GPU_COUNTER=$((GPU_COUNTER + 1))
    LOG_VAN="logs/train_vanilla_seed${seed}.log"
    TRAIN_LOGS+=("$LOG_VAN")
    echo "  [$(date '+%Y-%m-%d %H:%M:%S')] Launching vanilla model seed $seed on GPU $GPU_ID -> $LOG_VAN"
    CUDA_VISIBLE_DEVICES=$GPU_ID python run_Q_model.py \
        --seed $seed \
        --latent False \
        --dataset_id $DATASET_ID \
        --dgp $DGP \
        --hidden $HIDDEN \
        --epochs 3000 \
        > "$LOG_VAN" 2>&1 &
    PIDS+=($!)
done

# Wait for all training jobs and collect exit codes
echo "Waiting for all training jobs to complete..."
TRAIN_FAILED=0
for i in "${!PIDS[@]}"; do
    pid=${PIDS[$i]}
    log=${TRAIN_LOGS[$i]}
    if ! wait "$pid"; then
        echo "ERROR: Training job (PID $pid, log $log) failed!"
        TRAIN_FAILED=1
    fi
done
if [ "$TRAIN_FAILED" -ne 0 ]; then
    echo "One or more training jobs failed. Aborting."
    exit 1
fi
echo "All training jobs completed successfully."

# Step 4: Test models — iterates over all seeds automatically, uses cv_selected_epoch
# from each seed's config.json. Saves latents to data_sloutsky/seed_{seed}/
echo "Testing models for Sloutsky data (CV checkpoints, per-seed output)..."
python testing_script.py --latent True  --dataset_id $DATASET_ID --dgp $DGP --model_fitting False
python testing_script.py --latent False --dataset_id $DATASET_ID --dgp $DGP --model_fitting False

# Step 5: Analysis — RSA, cog-weight correlations, and decoding across seeds
echo "Running combined analysis (RSA + decoding) across seeds..."
python analyze_cv_seeds.py

# Step 6: Plot latent representations (uses pre-saved per-seed latents)
echo "Plotting latent representations..."
mkdir -p plots_sloutsky

echo "  Plotting IDRNN latents..."
python plot_sloutsky_latents.py --latent True  --dim_reduction tsne --avg False

echo "  Plotting Vanilla latents..."
python plot_sloutsky_latents.py --latent False --dim_reduction pca  --avg True

echo "========================================"
echo "Sloutsky pipeline completed at: $(date)"
echo "Results saved to:"
echo "  - runs_sloutsky/seed_*/ (IDRNN model checkpoints, cv_selected_epoch in config.json)"
echo "  - runs_vanilla_sloutsky/seed_*/ (Vanilla model checkpoints)"
echo "  - data_sloutsky/seed_*/ (per-seed latent tensors and NLL results)"
echo "  - plots_sloutsky/ (RSA, decoding, and latent visualizations)"
echo "========================================"
