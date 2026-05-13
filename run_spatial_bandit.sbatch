#!/bin/bash
#SBATCH --job-name=spatial_bandit
#SBATCH --output=logs/spatial_bandit_%j.out
#SBATCH --error=logs/spatial_bandit_%j.err
#SBATCH --open-mode=append
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH -p gpu_p
#SBATCH --qos gpu_normal
#SBATCH --constraint=a100_80gb|h100_80gb
#SBATCH --nice=10000

#SBATCH --mail-type=ALL
#SBATCH --mail-user=marvin.mathony@helmholtz-munich.de

# Full pipeline for spatial bandit (Schulz & Wu) dataset.
# Steps:
#   1. Preprocess data (load_spatial_bandit.py)
#   2. Train IDRNN + Vanilla RNN
#   3. Select best epochs
#   4. Test models and extract latents
#   5. Plot latents coloured by age_months + run age decoding

source /ictstr01/home/hcai/marvin.mathony/tools/apps/mamba/etc/profile.d/conda.sh
conda activate RNNproject

set -euo pipefail
mkdir -p logs

DGP="spatial_bandit"
DATASET_ID=0

echo "========================================"
echo "Spatial Bandit Pipeline"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "Starting at: $(date)"
echo "========================================"

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

# ---- Best hyperparameters (set these after running the HP search) ----
LMBD=0.01
Z_DIM=5
SEEDS=(12 50 76 100 142)

# # ------------------------------------------------------------------
# # Step 1: Preprocess data
# # ------------------------------------------------------------------
# echo "Step 1: Preprocessing spatial bandit data..."
# python load_spatial_bandit.py
# echo "  Done. Data saved to data_${DGP}/"

# # ------------------------------------------------------------------
# # Step 2: Train IDRNN (latent) models
# # ------------------------------------------------------------------
# echo "Step 2: Training IDRNN models..."
# for seed in "${SEEDS[@]}"; do
#     echo "  [$(date '+%H:%M:%S')] Seed $seed..."
#     python run_Q_model.py \
#         --seed $seed \
#         --latent True \
#         --dataset_id $DATASET_ID \
#         --dgp $DGP \
#         --lmbd $LMBD \
#         --z $Z_DIM \
#         --epochs 3000 \
#         --step1_epochs 3000 \
#         --same_enc_dec True
#     echo "  [$(date '+%H:%M:%S')] Seed $seed done"
# done

# # ------------------------------------------------------------------
# # Step 3: Train Vanilla RNN models
# # ------------------------------------------------------------------
# echo "Step 3: Training Vanilla RNN models..."
# for seed in "${SEEDS[@]}"; do
#     echo "  [$(date '+%H:%M:%S')] Seed $seed..."
#     python run_Q_model.py \
#         --seed $seed \
#         --latent False \
#         --dataset_id $DATASET_ID \
#         --dgp $DGP \
#         --epochs 3000
#     echo "  [$(date '+%H:%M:%S')] Seed $seed done"
# done

# ------------------------------------------------------------------
# Step 4: Select best epochs
# ------------------------------------------------------------------
echo "Step 4: Selecting best epochs..."
python select_best_epoch_by_specificity.py \
    --latent True --dataset_id $DATASET_ID --dgp $DGP \
    --min_epoch 100 --max_epoch 1500
python select_best_epoch_by_loss.py \
    --latent False --dataset_id $DATASET_ID --dgp $DGP

# ------------------------------------------------------------------
# Step 5: Test models + extract latents
# ------------------------------------------------------------------
echo "Step 5: Testing models and extracting latents..."
python testing_script.py --latent True  --dataset_id $DATASET_ID --dgp $DGP --model_fitting False
python testing_script.py --latent False --dataset_id $DATASET_ID --dgp $DGP --model_fitting False

# ------------------------------------------------------------------
# Step 6: Plot latents + age decoding
# ------------------------------------------------------------------
echo "Step 6: Plotting latents and decoding age..."
mkdir -p plots_${DGP}

for avg in True False; do
    for dr in pca tsne; do
        python plot_spatial_bandit_latents.py \
            --latent True  --dim_reduction $dr --avg $avg
        python plot_spatial_bandit_latents.py \
            --latent False --dim_reduction $dr --avg $avg
    done
done

echo "========================================"
echo "Pipeline completed at: $(date)"
echo "Results:"
echo "  runs_${DGP}/             — IDRNN checkpoints"
echo "  runs_vanilla_${DGP}/     — Vanilla checkpoints"
echo "  data_${DGP}/             — latent tensors, decoding results"
echo "  plots_${DGP}/            — visualisations"
echo "========================================"
