#!/bin/bash
#SBATCH --job-name=thal_logit_updates
#SBATCH --output=logs/thal_logit_updates_%j.out
#SBATCH --error=logs/thal_logit_updates_%j.err
#SBATCH --open-mode=append
#SBATCH --time=16:00:00
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

# JSD / entropy / compressibility analyses for Thalmann step-1 IDRNN +
# Vanilla model, on simulated 4-armed restless bandit rollouts. The script
# also runs the cross-task regret analysis (R1 human / R2 exact-sim /
# R3 marg-sim → task3 horizon regret, for IDRNN + Vanilla + Vanilla+h).
# GPU-enabled: rollouts run on cuda when available (auto-detected by script).

set -euo pipefail
mkdir -p logs

source /ictstr01/home/hcai/marvin.mathony/tools/apps/mamba/etc/profile.d/conda.sh
conda activate RNNproject

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

echo "========================================"
echo "Thalmann logit-update / compressibility analysis"
echo "Job ID: $SLURM_JOB_ID"
echo "Starting at: $(date)"
nvidia-smi -L 2>/dev/null || true
echo "========================================"

python analyze_thalmann_logit_updates.py

echo "========================================"
echo "Completed at: $(date)"
echo "========================================"
