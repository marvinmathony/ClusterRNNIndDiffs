#!/bin/bash
#SBATCH --job-name=ds_perfold
#SBATCH --output=logs/ds_perfold_%j.out
#SBATCH --error=logs/ds_perfold_%j.err
#SBATCH --open-mode=append
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH -p cpu_p
#SBATCH --qos cpu_normal
#SBATCH --nice=10000

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=marvin.mathony@helmholtz-munich.de

# Leakage-free data-scaling decoding panel — same philosophy as the main LOO-R² panel
# (compute_loo_r2_traintest_bootstrap_perfold.py + compute_loo_r2_canonical_bf.py),
# ported to the 7 task constellations:
#   1) compute_datascaling_traintest_bootstrap_perfold.py — train-resample bootstrap;
#      per fold the ENTIRE readout (PCA dim-match on RAW vanilla h, standardisation,
#      RidgeCV alpha) is fit on the bootstrap train rows only -> boot_sd (error bars).
#   2) compute_datascaling_canonical_bf.py — canonical seed-averaged LOO point + one-
#      sided correlation BF10 + permutation p per trait; merges datascaling_panel.csv.
# CPU-bound; joblib across the 56 (group x arch x constellation x trait) units, each
# BLAS lib pinned to 1 thread. The first run also caches the RAW vanilla-h forward
# passes (serial, ~70 passes) to <ds_dir>/decoding/_bootreps_vanilla_rawh.npy.

set -euo pipefail
mkdir -p logs

source /ictstr01/home/hcai/marvin.mathony/tools/apps/mamba/etc/profile.d/conda.sh
conda activate RNNproject

export PYTHONUNBUFFERED=1
# 1 thread per BLAS lib; parallelism is joblib across the (group, arch, sub, trait) units.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export N_JOBS=${SLURM_CPUS_PER_TASK:-4}

echo "========================================"
echo "Data-scaling LOO-R2 - leakage-free per-fold (PCA + RidgeCV alpha) + canonical/BF"
echo "Job ID: ${SLURM_JOB_ID:-} | CPUs: ${SLURM_CPUS_PER_TASK:-?} | N_JOBS=$N_JOBS"
echo "Starting at: $(date)"
echo "========================================"

python compute_datascaling_traintest_bootstrap_perfold.py

echo "---- bootstrap done: $(date) ----"

python compute_datascaling_canonical_bf.py

echo "========================================"
echo "Completed at: $(date)"
echo "========================================"
