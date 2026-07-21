#!/bin/bash
# Pipeline driver for Dezfouli nested-CV stages c → g with --hypothesis_z 2.
# Submitted as a single sbatch CPU job that polls squeue between stages.
# Stage_b was run interactively before this script; stage_a was the optuna sweep.
#
# Logs to logs/dezfouli_z2_pipeline_${SLURM_JOB_ID}.log via sbatch --output.

set -uo pipefail
WORKDIR="$(cd "$(dirname "$0")" && pwd)"
cd "$WORKDIR"

source /ictstr01/home/hcai/marvin.mathony/tools/apps/mamba/etc/profile.d/conda.sh
conda activate RNNproject

POLL_SECS=120

# ── helpers ───────────────────────────────────────────────────────────────────
extract_job_ids() {
    # reads stdin (a captured log) and emits comma-separated job IDs (7+ digits)
    grep -oE '^Job IDs:.*' "$1" 2>/dev/null \
        | sed 's/Job IDs: //' \
        | tr ' ' '\n' \
        | grep -E '^[0-9]{7,}$' \
        | sort -u \
        | tr '\n' ',' \
        | sed 's/,$//'
}

wait_for() {
    local ids="$1"
    if [[ -z "$ids" ]]; then
        echo "[pipeline] no job IDs to wait for; continuing"
        return
    fi
    echo "[pipeline] waiting on: $ids"
    while squeue -j "$ids" -h 2>/dev/null | grep -qE '^[[:space:]]*[0-9]'; do
        sleep "$POLL_SECS"
    done
    echo "[pipeline] queue cleared at $(date -Iseconds)"
}

run_stage() {
    local label="$1"; shift
    local log="logs/dezfouli_z2_${label}_$(date +%Y%m%d_%H%M%S).log"
    echo "──────── $label @ $(date -Iseconds) ────────"
    echo "[pipeline] log: $log"
    echo "[pipeline] cmd: bash submit_nested_cv.sh $*"
    bash submit_nested_cv.sh "$@" 2>&1 | tee "$log"
    echo "$log"   # last line of stdout is the log path so caller can grep it
}

# ── Stage C: SLURM final per-fold (30 seeds × 3 folds × 2 archs) ─────────────
LOG_C=$(run_stage "stage_c" dezfouli stage_c --hypothesis_z 2 --metric cv_val_loss | tail -1)
IDS_C=$(extract_job_ids "$LOG_C")
echo "[pipeline] stage_c IDs: $IDS_C"
wait_for "$IDS_C"

# ── Stage D: SLURM test + analyze_outer_cv ────────────────────────────────────
LOG_D=$(run_stage "stage_d" dezfouli stage_d | tail -1)
IDS_D=$(extract_job_ids "$LOG_D")
echo "[pipeline] stage_d IDs: $IDS_D"
wait_for "$IDS_D"

# ── Stage E: CPU canonical spec (modal HP, --metric step1_specificity) ────────
run_stage "stage_e" dezfouli stage_e --hypothesis_z 2 --metric step1_specificity > /dev/null

# ── Stage F: SLURM canonical retrain ──────────────────────────────────────────
LOG_F=$(run_stage "stage_f" dezfouli stage_f --hypothesis_z 2 --metric step1_specificity | tail -1)
IDS_F=$(extract_job_ids "$LOG_F")
echo "[pipeline] stage_f IDs: $IDS_F"
wait_for "$IDS_F"

# ── Stage G: CPU canonical-seed selection ─────────────────────────────────────
run_stage "stage_g" dezfouli stage_g --hypothesis_z 2 --metric step1_specificity > /dev/null

echo "────────────────────────────────────────────────"
echo "[pipeline] COMPLETE at $(date -Iseconds)"
