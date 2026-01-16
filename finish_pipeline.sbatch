#!/bin/bash
#SBATCH --job-name=finish_%a
#SBATCH --output=logs/finish_%a_%j.out
#SBATCH --error=logs/finish_%a_%j.err
#SBATCH --open-mode=append
#SBATCH --array=0-4
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH -p gpu_p
#SBATCH --qos gpu_normal
#SBATCH --constraint=a100_80gb|h100_80gb
#SBATCH --nice=10000

#SBATCH --mail-type=ALL
#SBATCH --mail-user=marvin.mathony@helmholtz-munich.de

set -euo pipefail

mkdir -p logs

# Activate conda environment
# source /ictstr01/home/hcai/marvin.mathony/tools/apps/mamba/etc/profile.d/conda.sh
# conda activate RNNproject

DATASET_ID=$SLURM_ARRAY_TASK_ID

echo "Job ID: $SLURM_JOB_ID"
echo "Dataset ID: $DATASET_ID"
echo "Running on node: $(hostname)"
echo "Starting at: $(date)"

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

# Step 1: Select best epochs using composite score
echo "Selecting best epoch for vanilla model..."
python select_best_epoch_by_composite.py --latent False --dataset_id $DATASET_ID

# Step 2: Test models
echo "Testing models for dataset $DATASET_ID..."
python testing_script.py --latent True --dataset_id $DATASET_ID --model_fitting True
python testing_script.py --latent False --dataset_id $DATASET_ID --model_fitting True

echo "Dataset $DATASET_ID completed at: $(date)"
