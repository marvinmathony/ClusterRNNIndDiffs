#!/bin/bash
#SBATCH --job-name=pool3task
#SBATCH --output=logs/pool3task_%x_%j.out
#SBATCH --error=logs/pool3task_%x_%j.err
#SBATCH --open-mode=append
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -p cpu_p
#SBATCH --qos cpu_normal
#SBATCH --nice=10000

#SBATCH --mail-type=FAIL
#SBATCH --mail-user=marvin.mathony@helmholtz-munich.de

# Pooled S1+S2 recompute of the journal figure's S1-only panels (c, d, e, g),
# using the EXISTING ds_t012 canonical runs (identical HP to the S1 3task
# canonical) via final_plots/thalmann_z3_s2_3task_full (symlinked runs dir).
# Nothing under final_plots/thalmann_z3_3task_full is touched.
#
# Stages (submitted with dependencies by the driver):
#   boot   -> {NEW}/decoding/loo_r2_traintest_bootstrap_perfold.csv  (error bars)
#   canon  -> {NEW}/decoding/loo_r2_canonical_bf.csv + loo_r2_panel.csv (panel c)
#   z2     -> {NEW}/decoding/z2_consensus_pca.npy + z2_seed_matched*.csv (panels d/e/g)
#   report -> {NEW}/CONSISTENCY_REPORT.md (old-vs-new verdict for the morning)
#
# Usage: sbatch [-c N -t H:MM:SS --dependency=...] run_pooled_3task_consistency.sbatch <stage>

set -euo pipefail
mkdir -p logs

STAGE="${1:?usage: sbatch run_pooled_3task_consistency.sbatch <boot|canon|z2|report>}"

source /ictstr01/home/hcai/marvin.mathony/tools/apps/mamba/etc/profile.d/conda.sh
conda activate RNNproject

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export N_JOBS=${SLURM_CPUS_PER_TASK:-4}

export THAL_FULL=final_plots/thalmann_z3_s2_3task_full
export THAL_DATA=data_sub_t012_full
export N_SEEDS=10

echo "========================================"
echo "pooled 3task consistency — stage: ${STAGE}"
echo "Job ID: ${SLURM_JOB_ID:-local} | CPUs: ${SLURM_CPUS_PER_TASK:-?}"
echo "THAL_FULL=${THAL_FULL}  THAL_DATA=${THAL_DATA}"
echo "Starting at: $(date)"
echo "========================================"

case "${STAGE}" in
  boot)   python compute_loo_r2_traintest_bootstrap_perfold.py ;;
  canon)  python compute_loo_r2_canonical_bf.py ;;
  z2)     python analyze_z2_seed_matched.py ;;
  report) python compare_pooled_consistency.py ;;
  *) echo "unknown stage ${STAGE}"; exit 1 ;;
esac

echo "========================================"
echo "Completed at: $(date)"
echo "========================================"
