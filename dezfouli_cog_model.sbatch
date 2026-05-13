#!/bin/bash
#SBATCH --job-name=dez_cog_%a
#SBATCH --output=logs/dez_cog_fold%a_%j.out
#SBATCH --error=logs/dez_cog_fold%a_%j.err
#SBATCH --open-mode=append
#SBATCH --array=0-2
#SBATCH --time=12:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH -p cpu_p
#SBATCH --qos cpu_normal
#SBATCH --nice=10000
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=marvin.mathony@helmholtz-munich.de

# Fit QLP (specified, with perseveration κ) and QL (ill-specified) per
# Dezfouli outer-CV fold via EM (hierarchical) + CP (common process). Writes
# per-fold CSVs that analyze_outer_cv.py picks up automatically.

set -euo pipefail
mkdir -p logs

source /ictstr01/home/hcai/marvin.mathony/tools/apps/mamba/etc/profile.d/conda.sh
conda activate RNNproject

export PYTHONUNBUFFERED=1

echo "========================================"
echo "Dezfouli cog-model fits — fold ${SLURM_ARRAY_TASK_ID}"
echo "Job ID: $SLURM_JOB_ID  Start: $(date)"
echo "========================================"

python dezfouli_cog_model.py --fold "${SLURM_ARRAY_TASK_ID}"

echo "Completed: $(date)"
