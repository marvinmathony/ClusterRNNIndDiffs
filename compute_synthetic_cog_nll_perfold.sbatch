#!/bin/bash
#SBATCH --job-name=cog_perfold
#SBATCH --output=logs/cog_perfold_%A_%a.out
#SBATCH --error=logs/cog_perfold_%A_%a.err
#SBATCH --open-mode=append
#SBATCH --time=04:00:00
#SBATCH --mem=8G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -p cpu_p
#SBATCH --qos cpu_normal
#SBATCH --nice=10000
#SBATCH --array=0-29

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=marvin.mathony@helmholtz-munich.de

# Per-fold cognitive-model NLL refit (panel-a apples-to-apples fix):
# fit Q/FQ (common fit + empirical-Bayes MAP) on each nested-CV fold's train
# subjects, evaluate on the SAME held-out fold subjects as the RNN NLLs.
# 30 array tasks = 10 datasets x 3 folds; each writes only its own part file
# under cog_nll_perfold_parts/ (existing parts are skipped).
# After the array finishes, merge with:
#   python compute_synthetic_cog_nll_perfold.py --merge

set -euo pipefail
mkdir -p logs

source /ictstr01/home/hcai/marvin.mathony/tools/apps/mamba/etc/profile.d/conda.sh
conda activate RNNproject

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

DATASETS=(0 1 2 3 5 7 10 12 15 17)
D=${DATASETS[$((SLURM_ARRAY_TASK_ID / 3))]}
F=$((SLURM_ARRAY_TASK_ID % 3))

echo "array task ${SLURM_ARRAY_TASK_ID}: dataset ${D} fold ${F}"
python -u compute_synthetic_cog_nll_perfold.py --dataset_id "${D}" --fold "${F}"
