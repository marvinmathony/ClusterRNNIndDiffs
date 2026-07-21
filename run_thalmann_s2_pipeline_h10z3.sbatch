#!/bin/bash
#SBATCH --job-name=thal_s2_h10z3
#SBATCH --output=logs/thal_s2_h10z3_%j.out
#SBATCH --error=logs/thal_s2_h10z3_%j.err
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

# Thalmann S2 pipeline at HP z=3 / hidden=10 (decoder GRU width swap).
# Reuses data_thalmann_s2/ from the first run (HP-independent).
# Writes to plots_thalmann_s2_h10z3/ and runs_thalmann_s2_h10z3_*.

set -euo pipefail
mkdir -p logs

source /ictstr01/home/hcai/marvin.mathony/tools/apps/mamba/etc/profile.d/conda.sh
conda activate RNNproject

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export WANDB_MODE=${WANDB_MODE:-disabled}

echo "========================================"
echo "Thalmann S2 pipeline — HP z=3, hidden=10"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Starting at: $(date)"
nvidia-smi -L 2>/dev/null || true
echo "========================================"

echo "--- Step 1/3: train + decode (IDRNN + Vanilla, z=3, h=10) ---"
python train_and_decode_thalmann_step1_s2_h10z3.py

echo "--- Step 2/3: cross-task regret rollouts ---"
python analyze_cross_task_regret_s2_h10z3.py

echo "--- Step 3/3: regret headline plots ---"
python plot_thalmann_regret_headline_s2_h10z3.py

echo "--- Bonus: regenerate decoding plot with signed LOO bars ---"
python replot_step1_vs_vanilla_decoding_s2_h10z3.py

echo "========================================"
echo "Completed at: $(date)"
echo "========================================"
