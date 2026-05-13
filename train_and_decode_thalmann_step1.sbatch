#!/bin/bash
#SBATCH --job-name=thal_step1_vs_van
#SBATCH --output=logs/thal_step1_vs_van_%j.out
#SBATCH --error=logs/thal_step1_vs_van_%j.err
#SBATCH --open-mode=append
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH -p gpu_p
#SBATCH --qos gpu_normal
#SBATCH --constraint=a100_80gb|h100_80gb
#SBATCH --nice=10000

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=marvin.mathony@helmholtz-munich.de

# Retrain IDRNN step-1 (lookup embeddings) and Vanilla on all 236 Thalmann
# subjects, then LOO-decode 6 questionnaire scales.  3 seeds each, inner-CV
# epoch selection, specificity-based seed selection for IDRNN.

set -euo pipefail
mkdir -p logs

source /ictstr01/home/hcai/marvin.mathony/tools/apps/mamba/etc/profile.d/conda.sh
conda activate RNNproject

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

echo "========================================"
echo "Thalmann step-1 vs vanilla decoding"
echo "Job ID: $SLURM_JOB_ID"
echo "Starting at: $(date)"
echo "========================================"

python train_and_decode_thalmann_step1.py

echo "========================================"
echo "Completed at: $(date)"
echo "========================================"
