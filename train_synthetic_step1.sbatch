#!/bin/bash
#SBATCH --job-name=synth_step1
#SBATCH --output=logs/synth_step1_%j.out
#SBATCH --error=logs/synth_step1_%j.err
#SBATCH --open-mode=append
#SBATCH --time=04:00:00
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

# Step-1 IDRNN (LatentRNNz w/ LookupEncoderZ) + Vanilla baseline training on
# a single synthetic uniform-α dataset. Output saved in
# plots_dataset{ID}/step1_vs_vanilla/ (compatible with cross-regression).
#
# Usage: sbatch train_synthetic_step1.sbatch <dataset_id> [epochs]

set -euo pipefail
mkdir -p logs

DATASET_ID="${1:-0}"
EPOCHS="${2:-3000}"

source /ictstr01/home/hcai/marvin.mathony/tools/apps/mamba/etc/profile.d/conda.sh
conda activate RNNproject

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

echo "========================================"
echo "Synthetic step-1 training"
echo "Job ID: $SLURM_JOB_ID"
echo "Dataset: data_dataset${DATASET_ID}"
echo "Epochs:  ${EPOCHS}"
echo "Starting at: $(date)"
nvidia-smi -L 2>/dev/null || true
echo "========================================"

python train_synthetic_step1.py --dataset_id "${DATASET_ID}" --epochs "${EPOCHS}"

echo "========================================"
echo "Completed at: $(date)"
echo "========================================"
