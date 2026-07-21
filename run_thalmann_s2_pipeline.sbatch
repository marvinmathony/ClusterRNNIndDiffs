#!/bin/bash
#SBATCH --job-name=thal_s2_pipeline
#SBATCH --output=logs/thal_s2_pipeline_%j.out
#SBATCH --error=logs/thal_s2_pipeline_%j.err
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

# End-to-end S2 (session-1 + session-2 pooled) Thalmann pipeline:
#   1. Build  data_thalmann_s2/   from raw S1+S2 CSVs
#   2. Retrain IDRNN step-1 + Vanilla on the union of 238 participants
#      using the same HP as the S1-only pipeline (z_dim=10, hid=3,
#      task_emb_dim=2)
#   3. Decode questionnaire scales (pooled S1+S2 questionnaires; CFA factors
#      kept S1-only)  →  plots_thalmann_s2/step1_vs_vanilla/step1_vs_vanilla_decoding.png
#   4. Cross-task regret rollouts (R1/R2/R3 against task3 horizon regret)
#   5. Regenerate the headline regret bar plot
#      →  plots_thalmann_s2/step1_cross_task_regret_bars.png

set -euo pipefail
mkdir -p logs

source /ictstr01/home/hcai/marvin.mathony/tools/apps/mamba/etc/profile.d/conda.sh
conda activate RNNproject

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
# wandb's metric logging is not needed here and can hang on offline nodes
export WANDB_MODE=${WANDB_MODE:-disabled}

echo "========================================"
echo "Thalmann S1+S2 pipeline"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Starting at: $(date)"
nvidia-smi -L 2>/dev/null || true
echo "========================================"

echo "--- Step 1/4: build data_thalmann_s2/ ---"
python load_thalmann_s2.py

echo "--- Step 2/4: train + decode (IDRNN + Vanilla) ---"
python train_and_decode_thalmann_step1_s2.py

echo "--- Step 3/4: cross-task regret rollouts ---"
python analyze_cross_task_regret_s2.py

echo "--- Step 4/4: regret headline plots ---"
python plot_thalmann_regret_headline_s2.py

echo "========================================"
echo "Completed at: $(date)"
echo "========================================"
