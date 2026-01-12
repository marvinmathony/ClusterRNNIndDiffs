#!/bin/bash
#SBATCH --job-name=GRU4
#SBATCH --output=/home/hcai/marvin.mathony/RNNsandUncertainty/logs/RNNsimulation_%x_%j.out
#SBATCH --error=/home/hcai/marvin.mathony/RNNsandUncertainty/logs/RNNsimulation_%x_%j.err
#SBATCH --open-mode=append

#SBATCH --mail-type=ALL
#SBATCH --mail-user=marvin.mathony@helmholtz-munich.de

#SBATCH -p gpu_p
#SBATCH --qos gpu_normal

#SBATCH --cpus-per-task=20
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=16:00:00
#SBATCH --constraint=a100_80gb|h100_80gb
#SBATCH --nice=10000

cd /ictstr01/home/hcai/marvin.mathony/RNNsandUncertainty

for seed_value in 12 50 76 100 142; do
    python run_Q_model.py --lmbd 0.5 --z 1 --seed $seed_value --latent False
done

#python run_Q_model.py --lmbd 0.5 --z 3

#python testing_script.py
