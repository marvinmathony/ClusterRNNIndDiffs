#!/bin/bash
#SBATCH --job-name=van_slout
#SBATCH --output=logs/van_slout_%a_%j.out
#SBATCH --error=logs/van_slout_%a_%j.err
#SBATCH --array=0-4
#SBATCH --time=06:00:00
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

set -euo pipefail
mkdir -p logs

source /ictstr01/home/hcai/marvin.mathony/tools/apps/mamba/etc/profile.d/conda.sh
conda activate RNNproject

python search_thalmann_step1_decodability.py

