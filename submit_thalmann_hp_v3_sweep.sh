#!/bin/bash
# Submit the full hp_v3 sweep (162 combos) + chained analysis.
#
# Usage:
#   bash submit_thalmann_hp_v3_sweep.sh
#
# This submits:
#   1. SLURM array job (0-161): trains all 162 combos × 3 folds × 3 seeds
#   2. Analysis job (dependency: all training done): runs analyze_hp_v3_outer_cv.py
#      then analyze_idrnn_vs_vanilla_thalmann.py

set -euo pipefail
WORKDIR="$(cd "$(dirname "$0")" && pwd)"
cd "$WORKDIR"
mkdir -p logs

CONDA_INIT="source /ictstr01/home/hcai/marvin.mathony/tools/apps/mamba/etc/profile.d/conda.sh && conda activate RNNproject"

# ── Step 1: HP sweep ─────────────────────────────────────────────────────────
echo "Submitting hp_v3 sweep (162 array tasks)..."
SWEEP_JID=$(sbatch hyperparam_search_thalmann_v3.sbatch | awk '{print $NF}')
echo "  Sweep array job: ${SWEEP_JID}"

# ── Step 2: Analysis (after all array tasks complete) ─────────────────────────
echo "Submitting analysis (dependency: sweep)..."
ANA_JID=$(sbatch \
    --nodes=1 -p gpu_p --qos gpu_normal \
    "--constraint=a100_80gb|h100_80gb" \
    --nice=10000 --mem=32G --cpus-per-task=8 --time=04:00:00 \
    --open-mode=append \
    --mail-type=ALL \
    --mail-user=marvin.mathony@helmholtz-munich.de \
    --job-name="thal_hp_v3_analyze" \
    --output="logs/thal_hp_v3_analyze_%j.out" \
    --error="logs/thal_hp_v3_analyze_%j.err" \
    --dependency="afterok:${SWEEP_JID}" \
    --wrap="${CONDA_INIT} && cd ${WORKDIR} && python analyze_hp_v3_outer_cv.py && python analyze_idrnn_vs_vanilla_thalmann.py" \
    | awk '{print $NF}')
echo "  Analysis job: ${ANA_JID}"

echo ""
echo "========================================"
echo "hp_v3 sweep + analysis submitted."
echo "  Sweep:    ${SWEEP_JID} (array 0-161)"
echo "  Analysis: ${ANA_JID}"
echo ""
echo "Monitor: squeue -u $USER"
echo "Cancel:  scancel ${SWEEP_JID} ${ANA_JID}"
echo "========================================"
