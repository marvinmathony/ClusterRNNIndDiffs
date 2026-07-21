#!/bin/bash
# submit_nested_cv.sh <dgp> <stage> [...extra args to launch.py...]
#
# Top-level driver for the nested cross-validation pipeline.  Dispatches to
# nested_cv/launch.py for SLURM stages and to nested_cv/select_{hp,canonical}.py
# for the CPU-only selection stages.
#
# Stages:
#   prep_synth   — generate fold directories for synthetic dataset(s)
#                  (only valid for --dgp synthetic)
#   stage_a      — submit inner HP search jobs (per combo × fold × seed)
#   stage_b      — pick per-fold winners (CPU-only Python)
#   stage_c      — submit final per-fold runs with each fold's winning HPs
#   stage_d      — submit test + analyze_outer_cv (depends on stage_c)
#   stage_e      — write canonical retrain spec (CPU-only Python)
#   stage_f      — submit canonical full-data retrain
#   stage_g      — pick canonical seed by step1_specificity (CPU-only)
#
# Typical run:
#   bash submit_nested_cv.sh dezfouli stage_a            # → SLURM HP search
#   # wait for stage_a to complete on cluster
#   bash submit_nested_cv.sh dezfouli stage_b            # → write per-fold winners
#   bash submit_nested_cv.sh dezfouli stage_c            # → SLURM final per-fold
#   bash submit_nested_cv.sh dezfouli stage_d            # → SLURM test + analysis
#   bash submit_nested_cv.sh dezfouli stage_e            # → write canonical spec
#   bash submit_nested_cv.sh dezfouli stage_f            # → SLURM canonical retrain
#   bash submit_nested_cv.sh dezfouli stage_g            # → pick canonical seed
#
# Pass --dry_run to any SLURM stage to print sbatch commands without submitting.

set -euo pipefail
WORKDIR="$(cd "$(dirname "$0")" && pwd)"
cd "$WORKDIR"

DGP="${1:-}"
STAGE="${2:-}"
shift 2 || true
EXTRA=("$@")

if [[ -z "$DGP" || -z "$STAGE" ]]; then
    sed -n '2,30p' "$0"
    exit 1
fi

# Validate DGP
case "$DGP" in
    dezfouli|thalmann|synthetic) ;;
    *) echo "ERROR: unknown DGP '$DGP' (allowed: dezfouli|thalmann|synthetic)"; exit 1 ;;
esac

# Pick up conda env so the CPU-only stages can also import torch/etc.
source /ictstr01/home/hcai/marvin.mathony/tools/apps/mamba/etc/profile.d/conda.sh
conda activate RNNproject

case "$STAGE" in
    prep_synth)
        if [[ "$DGP" != "synthetic" ]]; then
            echo "prep_synth only applies to --dgp synthetic"; exit 1
        fi
        python make_synthetic_folds.py "${EXTRA[@]}"
        ;;
    stage_a)
        python -m nested_cv.launch stage_a --dgp "$DGP" "${EXTRA[@]}"
        ;;
    stage_a_optuna)
        python -m nested_cv.launch stage_a_optuna --dgp "$DGP" "${EXTRA[@]}"
        ;;
    stage_b)
        python -m nested_cv.select_hp --dgp "$DGP" --arch both "${EXTRA[@]}"
        ;;
    stage_c)
        python -m nested_cv.launch stage_c --dgp "$DGP" "${EXTRA[@]}"
        ;;
    stage_d)
        python -m nested_cv.launch stage_d --dgp "$DGP" "${EXTRA[@]}"
        ;;
    stage_e)
        python -m nested_cv.select_canonical --dgp "$DGP" --arch both --mode spec "${EXTRA[@]}"
        ;;
    stage_f)
        python -m nested_cv.launch stage_f --dgp "$DGP" "${EXTRA[@]}"
        ;;
    stage_g)
        python -m nested_cv.select_canonical --dgp "$DGP" --arch both --mode pick "${EXTRA[@]}"
        ;;
    *)
        echo "ERROR: unknown stage '$STAGE'"
        echo "Valid stages: prep_synth stage_a stage_b stage_c stage_d stage_e stage_f stage_g"
        exit 1
        ;;
esac
