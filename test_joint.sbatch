#!/bin/bash
#SBATCH --job-name=joint_test
#SBATCH --output=/home/hcai/marvin.mathony/RNNsandUncertainty/logs/joint_test_%j.out
#SBATCH --error=/home/hcai/marvin.mathony/RNNsandUncertainty/logs/joint_test_%j.err
#SBATCH --open-mode=append

#SBATCH --mail-type=ALL
#SBATCH --mail-user=marvin.mathony@helmholtz-munich.de

#SBATCH -p gpu_p
#SBATCH --qos gpu_normal

#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --nice=10000

cd /ictstr01/home/hcai/marvin.mathony/RNNsandUncertainty

# Activate conda environment
source /ictstr01/home/hcai/marvin.mathony/tools/apps/mamba/bin/activate RNNproject

# Two-step training with lambda=0.8 (higher weight on matching lookup z)
#python run_Q_model.py --latent True --joint False --lmbd 0.8 --seed 42 --dataset_id 99 --epochs 1000 --z 1

# Run epoch selection
python select_best_epoch_by_specificity.py --latent True --dataset_id 99

# Run testing
python testing_script.py --latent True --dataset_id 99

echo "Test complete! Check data_dataset99/ for results"
