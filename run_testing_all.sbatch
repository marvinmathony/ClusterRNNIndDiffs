#!/bin/bash
#SBATCH --job-name=test_all_%a
#SBATCH --output=logs/test_all_%a_%j.out
#SBATCH --error=logs/test_all_%a_%j.err
#SBATCH --array=0-4
#SBATCH --time=01:00:00
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

# Run testing script for z=2, lmbd=0.5 results on all datasets

set -euo pipefail
mkdir -p logs

# Activate conda environment
source /ictstr01/home/hcai/marvin.mathony/tools/apps/mamba/etc/profile.d/conda.sh
conda activate RNNproject

DATASET_ID=$SLURM_ARRAY_TASK_ID

echo "========================================"
echo "Testing Dataset $DATASET_ID"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Starting at: $(date)"
echo "========================================"

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

# Run testing for latent model (IDRNN with z=2, lmbd=0.5)
echo "Testing latent model..."
python testing_script.py --latent True --dataset_id $DATASET_ID --model_fitting True

# Run testing for vanilla model
echo "Testing vanilla model..."
python testing_script.py --latent False --dataset_id $DATASET_ID --model_fitting True

echo "========================================"
echo "Completed at: $(date)"
echo "========================================"
