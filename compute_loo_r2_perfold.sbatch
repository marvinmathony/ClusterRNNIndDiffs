#!/bin/bash
#SBATCH --job-name=loo_perfold
#SBATCH --output=logs/loo_perfold_%j.out
#SBATCH --error=logs/loo_perfold_%j.err
#SBATCH --open-mode=append
#SBATCH --time=03:00:00
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH -p cpu_p
#SBATCH --qos cpu_normal
#SBATCH --nice=10000

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=marvin.mathony@helmholtz-munich.de

# Leakage-free LOO-R2 train/test bootstrap (compute_loo_r2_traintest_bootstrap_perfold.py).
# Per fold the ENTIRE readout is fit on the bootstrap train rows only: PCA dim-match
# (vanilla), standardisation, and ridge-alpha (RidgeCV). CPU-bound (~25M tiny RidgeCV
# fits); the 28 (target x arch) units are parallelised across cores with joblib, so
# each BLAS lib is pinned to 1 thread and joblib gets SLURM_CPUS_PER_TASK workers.

set -euo pipefail
mkdir -p logs

source /ictstr01/home/hcai/marvin.mathony/tools/apps/mamba/etc/profile.d/conda.sh
conda activate RNNproject

export PYTHONUNBUFFERED=1
# 1 thread per BLAS lib; parallelism is joblib across the (target, arch) units.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export N_JOBS=${SLURM_CPUS_PER_TASK:-4}

echo "========================================"
echo "LOO-R2 train/test bootstrap - leakage-free per-fold (PCA + RidgeCV alpha)"
echo "Job ID: ${SLURM_JOB_ID:-} | CPUs: ${SLURM_CPUS_PER_TASK:-?} | N_JOBS=$N_JOBS"
echo "Starting at: $(date)"
echo "========================================"

python compute_loo_r2_traintest_bootstrap_perfold.py

echo "========================================"
echo "Completed at: $(date)"
echo "========================================"
