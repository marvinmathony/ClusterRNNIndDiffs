#!/bin/bash
#SBATCH --job-name=complete_%a
#SBATCH --output=logs/complete_%a_%j.out
#SBATCH --error=logs/complete_%a_%j.err
#SBATCH --open-mode=append
#SBATCH --array=0-4
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH -p gpu_p
#SBATCH --qos gpu_normal
#SBATCH --constraint=a100_80gb|h100_80gb
#SBATCH --nice=10000

#SBATCH --mail-type=ALL
#SBATCH --mail-user=marvin.mathony@helmholtz-munich.de

# Create logs directory
mkdir -p logs

# Load modules (adjust based on your cluster)
# module load python/3.9
# module load cuda/11.8

# Activate your environment
# source /path/to/your/venv/bin/activate
# or: conda activate your_env

# Get dataset ID from SLURM array task ID
DATASET_ID=$SLURM_ARRAY_TASK_ID

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Dataset ID: $DATASET_ID"
echo "Running on node: $(hostname)"
echo "Starting at: $(date)"

# Set environment variables
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

# Configuration
SEEDS=(12 50 76 100 142)

# Step 1: Train vanilla models
echo "Training vanilla models for dataset $DATASET_ID..."
for seed in "${SEEDS[@]}"; do
    echo "  Training vanilla model with seed $seed..."
    python run_Q_model.py \
        --seed $seed \
        --latent False \
        --dataset_id $DATASET_ID
done

# Step 2: Select best epoch for vanilla models using composite score
echo "Selecting best epoch for vanilla models..."
python select_best_epoch_by_composite.py --latent False --dataset_id $DATASET_ID

# Step 3: Test models
echo "Testing models for dataset $DATASET_ID..."
python testing_script.py --latent True --dataset_id $DATASET_ID --model_fitting True
python testing_script.py --latent False --dataset_id $DATASET_ID --model_fitting True

echo "Dataset $DATASET_ID completed at: $(date)"
