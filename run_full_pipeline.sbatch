#!/bin/bash
#SBATCH --job-name=multi_dataset_pipeline
#SBATCH --output=logs/pipeline_%j.out
#SBATCH --error=logs/pipeline_%j.err
#SBATCH --open-mode=append
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH -p gpu_p
#SBATCH --qos gpu_normal
#SBATCH --constraint=a100_80gb|h100_80gb
#SBATCH --nice=10000

# Create logs directory
mkdir -p logs

# Load modules (adjust based on your cluster)
# module load python/3.9
# module load cuda/11.8

# Activate your environment (adjust path)
# source /path/to/your/venv/bin/activate
# or: conda activate your_env

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Starting at: $(date)"
echo "Working directory: $(pwd)"

# Set environment variables
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

# Run the pipeline
python run_multi_dataset_pipeline.py \
    --n_datasets 5 \
    --lmbd 0.5 \
    --z 1

echo "Finished at: $(date)"
