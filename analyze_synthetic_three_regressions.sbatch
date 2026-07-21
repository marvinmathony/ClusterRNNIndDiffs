#!/bin/bash
#SBATCH --job-name=synth_3reg
#SBATCH --output=logs/synth_3reg_%j.out
#SBATCH --error=logs/synth_3reg_%j.err
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

# Three-regression analysis on a synthetic uniform-α dataset, predicting
# true α from R1 (raw mean reward), R2 (exact-env on-policy sim), R3
# (marg-env on-policy sim). Compares IDRNN, Vanilla+h, Vanilla.
#
# Usage: sbatch analyze_synthetic_three_regressions.sbatch <dataset_id>

set -euo pipefail
mkdir -p logs

DATASET_ID="${1:-0}"
shift || true   # remaining args (e.g. --add_vanH_lasttime) forwarded to python

source /ictstr01/home/hcai/marvin.mathony/tools/apps/mamba/etc/profile.d/conda.sh
conda activate RNNproject

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

echo "========================================"
echo "Synthetic three-regression analysis"
echo "Job ID: $SLURM_JOB_ID"
echo "Dataset: data_dataset${DATASET_ID}"
echo "Starting at: $(date)"
nvidia-smi -L 2>/dev/null || true
echo "========================================"

python analyze_synthetic_three_regressions.py --dataset_id "${DATASET_ID}" "$@"

echo "========================================"
echo "Completed at: $(date)"
echo "========================================"
