#!/bin/bash
# Submit the full Sloutsky HP search pipeline with SLURM dependencies.
#
# Dependency graph:
#   [vanilla training (array 0-4)] ──► [select best vanilla epoch] ──┐
#                                                                      ├──► [eval decoding]
#   [IDRNN hp search  (array 0-29)] ─────────────────────────────────┘
#
# Usage:
#   bash submit_sloutsky_hp_pipeline.sh

set -euo pipefail

WORKDIR="/ictstr01/home/hcai/marvin.mathony/RNNsandUncertainty"
cd "$WORKDIR"
mkdir -p logs

CONDA_SETUP="source /ictstr01/home/hcai/marvin.mathony/tools/apps/mamba/etc/profile.d/conda.sh \
&& conda activate RNNproject \
&& export PATH=/ictstr01/home/hcai/marvin.mathony/tools/apps/mamba/envs/RNNproject/bin:\$PATH \
&& export PYTHONUNBUFFERED=1"

# ── Step 1a: vanilla training (5 seeds, runs immediately) ──────────────
echo "Submitting vanilla training..."
VAN_JID=$(sbatch --parsable train_vanilla_sloutsky.sbatch)
echo "  Vanilla training job: ${VAN_JID} (array)"

# ── Step 1b: IDRNN hp search (30 combos, runs immediately in parallel) ─
echo "Submitting IDRNN hp search..."
IDRNN_JID=$(sbatch --parsable hp_search_sloutsky_decoding.sbatch)
echo "  IDRNN hp search job:  ${IDRNN_JID} (array)"

# ── Step 2: select best vanilla epoch (waits for ALL vanilla tasks) ────
echo "Submitting vanilla epoch selection (depends on ${VAN_JID})..."
SEL_JID=$(sbatch --parsable \
    --dependency=afterok:${VAN_JID} \
    --kill-on-invalid-dep=yes \
    --job-name=van_sel_sloutsky \
    --output=logs/van_sel_%j.out \
    --error=logs/van_sel_%j.err \
    --time=00:15:00 \
    --mem=4G \
    --cpus-per-task=2 \
    --nodes=1 \
    -p gpu_p \
    --qos=gpu_normal \
    --nice=10000 \
    --wrap="${CONDA_SETUP} && python select_best_epoch_by_loss.py \
        --latent False \
        --dgp sloutsky \
        --min_epoch 100 \
        --max_epoch 3000")
echo "  Epoch selection job:  ${SEL_JID}"

# ── Step 3: eval decoding (waits for epoch selection AND IDRNN search) ─
echo "Submitting decoding evaluation (depends on ${SEL_JID} and ${IDRNN_JID})..."
EVAL_JID=$(sbatch --parsable \
    --dependency=afterok:${SEL_JID}:${IDRNN_JID} \
    --kill-on-invalid-dep=yes \
    --job-name=hp_eval_dec_sloutsky \
    --output=logs/hp_eval_dec_%j.out \
    --error=logs/hp_eval_dec_%j.err \
    --time=02:00:00 \
    --mem=16G \
    --cpus-per-task=8 \
    --gres=gpu:1 \
    --nodes=1 \
    -p gpu_p \
    --qos=gpu_normal \
    --constraint='a100_80gb|h100_80gb' \
    --nice=10000 \
    --wrap="${CONDA_SETUP} && python hp_eval_decoding_sloutsky.py \
        --output hp_decoding_results_sloutsky.json")
echo "  Decoding evaluation job: ${EVAL_JID}"

echo ""
echo "========================================"
echo "Pipeline submitted successfully"
echo "========================================"
echo "  1a. Vanilla training:    ${VAN_JID}  (array 0-4)"
echo "  1b. IDRNN hp search:     ${IDRNN_JID}  (array 0-29)"
echo "  2.  Epoch selection:     ${SEL_JID}  (after ${VAN_JID})"
echo "  3.  Decoding evaluation: ${EVAL_JID}  (after ${SEL_JID} + ${IDRNN_JID})"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f logs/hp_eval_dec_${EVAL_JID}.out"
echo "========================================"
