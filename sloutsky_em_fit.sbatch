#!/bin/bash
#SBATCH --job-name=sloutsky_em
#SBATCH --output=logs/sloutsky_em_%j.out
#SBATCH --error=logs/sloutsky_em_%j.err
#SBATCH --open-mode=append
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=20
#SBATCH --nodes=1
#SBATCH -p cpu_p
#SBATCH --qos cpu_normal
#SBATCH --nice=10000

#SBATCH --mail-type=ALL
#SBATCH --mail-user=marvin.mathony@helmholtz-munich.de

# Activate conda environment
source /ictstr01/home/hcai/marvin.mathony/tools/apps/mamba/etc/profile.d/conda.sh
conda activate RNNproject

python sloutsky_cog_model.py
python sloutsky_ill_specified_cog_model.py
