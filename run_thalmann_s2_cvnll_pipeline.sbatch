#!/bin/bash
#SBATCH --job-name=thal_s2_cvnll
#SBATCH --output=logs/thal_s2_cvnll_%j.out
#SBATCH --error=logs/thal_s2_cvnll_%j.err
#SBATCH --open-mode=append
#SBATCH --time=20:00:00
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

# Rerun of the S2 Thalmann pipeline with the corrected seed selection
# (inner-CV val NLL instead of specificity/train-NLL), clipped restless
# marg envs, rollout-semantics assertion, and ±SD bootstrap error bars.
#
# All outputs go to *_cvnll-suffixed directories via S2_RUN_SUFFIX, so the
# earlier (specificity-selected) results stay untouched:
#   main : plots_thalmann_s2_cvnll/        runs_thalmann_s2_step1_vs_vanilla_cvnll/
#   h10z3: plots_thalmann_s2_h10z3_cvnll/  runs_thalmann_s2_h10z3_step1_vs_vanilla_cvnll/
#
# Usage:
#   sbatch run_thalmann_s2_cvnll_pipeline.sbatch          # main (z=10, hid=3)
#   sbatch run_thalmann_s2_cvnll_pipeline.sbatch h10z3    # variant (z=3, hid=10)
#
# data_thalmann_s2/ is NOT rebuilt (inputs already exist and are deterministic).

set -euo pipefail
mkdir -p logs

VARIANT="${1:-main}"

source /ictstr01/home/hcai/marvin.mathony/tools/apps/mamba/etc/profile.d/conda.sh
conda activate RNNproject

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export WANDB_MODE=${WANDB_MODE:-disabled}
export S2_RUN_SUFFIX=_cvnll

echo "========================================"
echo "Thalmann S1+S2 cv-val-NLL pipeline — variant: ${VARIANT}"
echo "Job ID: ${SLURM_JOB_ID:-local}   S2_RUN_SUFFIX=${S2_RUN_SUFFIX}"
echo "Starting at: $(date)"
nvidia-smi -L 2>/dev/null || true
echo "========================================"

if [[ "${VARIANT}" == "h10z3" ]]; then
    TRAIN=train_and_decode_thalmann_step1_s2_h10z3.py
    ANALYZE=analyze_cross_task_regret_s2_h10z3.py
    PLOT=plot_thalmann_regret_headline_s2_h10z3.py
else
    TRAIN=train_and_decode_thalmann_step1_s2.py
    ANALYZE=analyze_cross_task_regret_s2.py
    PLOT=plot_thalmann_regret_headline_s2.py
fi

echo "--- Step 1/3: train + decode (${TRAIN}) ---"
python "${TRAIN}"

echo "--- Step 2/3: cross-task regret rollouts (${ANALYZE}) ---"
python "${ANALYZE}"

echo "--- Step 3/3: regret headline plots (${PLOT}) ---"
python "${PLOT}"

echo "========================================"
echo "Completed at: $(date)"
echo "========================================"
