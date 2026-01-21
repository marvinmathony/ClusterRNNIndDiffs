#!/bin/bash
#SBATCH --job-name=dataset_%a
#SBATCH --output=logs/dataset_%a_%j.out
#SBATCH --error=logs/dataset_%a_%j.err
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

# Get dataset ID from SLURM array task ID
DATASET_ID=$SLURM_ARRAY_TASK_ID



# Step 5: Test models
echo "Testing models for dataset $DATASET_ID..."
python testing_script.py --latent True --dataset_id $DATASET_ID --model_fitting True
python testing_script.py --latent False --dataset_id $DATASET_ID --model_fitting True
echo "Dataset $DATASET_ID completed at: $(date)"