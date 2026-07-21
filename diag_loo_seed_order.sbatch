#!/bin/bash
#SBATCH --job-name=loo_seedorder
#SBATCH --output=logs/loo_seedorder_%j.out
#SBATCH --error=logs/loo_seedorder_%j.err
#SBATCH --open-mode=append
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH -p cpu_p
#SBATCH --qos cpu_normal
#SBATCH --nice=10000

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=marvin.mathony@helmholtz-munich.de

# Canonical (non-bootstrap) LOO decoding + JZS Bayes factor per (trait, arch).
# Fast (no 500x bootstrap): 28 units x 10 seeds x ~180 LOO fits. joblib across units,
# BLAS pinned to 1 thread. Produces the canonical R2 (bar) and BF10 vs chance; the
# bootstrap SD error bar stays in loo_r2_traintest_bootstrap_perfold.csv.

set -euo pipefail
mkdir -p logs

source /ictstr01/home/hcai/marvin.mathony/tools/apps/mamba/etc/profile.d/conda.sh
conda activate RNNproject

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export N_JOBS=${SLURM_CPUS_PER_TASK:-4}

echo "========================================"
echo "Seed-order diagnostic: R2(mean yhat) vs mean R2 per seed"
echo "Job ID: ${SLURM_JOB_ID:-} | CPUs: ${SLURM_CPUS_PER_TASK:-?} | N_JOBS=$N_JOBS"
echo "Starting at: $(date)"
echo "========================================"

python diag_loo_seed_order.py

echo "========================================"
echo "Completed at: $(date)"
echo "========================================"
