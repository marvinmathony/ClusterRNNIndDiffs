#!/bin/bash
#SBATCH --job-name=thal_step1_decode_search
#SBATCH --output=logs/thal_step1_decode_search_%j.out
#SBATCH --error=logs/thal_step1_decode_search_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH -p gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --constraint="a100_80gb|h100_80gb"
#SBATCH --nice=10000
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=marvin.mathony@helmholtz-munich.de

# Thalmann step-1 decodability grid search:
#   IDRNN (z_dim ∈ {3,5,10}) × (hidden ∈ {1,2,3,5}) = 12 combos
#   Vanilla (hidden ∈ {1,2,3,5})                   = 4 combos
#   3 seeds each.
#
# Per-combo caches land in runs_thalmann_step1_decodability_search/, so if
# the job times out you can just resubmit — it picks up where it left off.

source /ictstr01/home/hcai/marvin.mathony/tools/apps/mamba/etc/profile.d/conda.sh
conda activate RNNproject

WORKDIR=/ictstr01/home/hcai/marvin.mathony/RNNsandUncertainty
cd "$WORKDIR"
mkdir -p logs

echo "Host:    $(hostname)"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Start:   $(date)"

python -u search_thalmann_step1_decodability.py

echo "End:     $(date)"
