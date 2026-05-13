#!/bin/bash
# submit_thalmann_lambda_sweep.sh — Train thalmann IDRNN across multiple
# lambda values so rate_distortion_thalmann.py can plot the Pareto frontier.
#
# Each lambda is trained independently (3 folds × N_SEEDS × n_lambdas jobs).
# Checkpoints land in runs_thalmann/fold{k}/lambda_{L}/seed_{S}/
#
# NOTE: this requires run_Q_model.py to support --run_subdir so that
#       different lambda runs don't overwrite each other.  The simplest
#       workaround used here is to temporarily redirect via HP_RUN_DIR.
#
# Usage:
#   bash submit_thalmann_lambda_sweep.sh
#   bash submit_thalmann_lambda_sweep.sh "0.01 0.05 0.2 0.5"  # custom lambdas

set -euo pipefail
WORKDIR="$(cd "$(dirname "$0")" && pwd)"
cd "$WORKDIR"
mkdir -p logs

# ── Config ────────────────────────────────────────────────────────────────────
LAMBDAS_STR="${1:-0.01 0.05 0.2 0.5}"   # space-separated; 0.1 already done
read -ra LAMBDAS <<< "$LAMBDAS_STR"

SEEDS=(200 300 400 500 600 999 2021 2022 2023 2024 2025)
DGP="thalmann"
N_FOLDS=3
Z_DIM=10
HIDDEN=5
ENC_HIDDEN=5
EPOCHS=3000
STEP1_EPOCHS=1000
TASK_EMB_DIM=4
DATASET_ID=0

CONDA_INIT="source /ictstr01/home/hcai/marvin.mathony/tools/apps/mamba/etc/profile.d/conda.sh && conda activate RNNproject"

FLAGS=(
    --nodes=1 --gres=gpu:1 -p gpu_p --qos gpu_normal
    "--constraint=a100_80gb|h100_80gb"
    --nice=10000 --mem=16G --cpus-per-task=4 --time=12:00:00
    --open-mode=append
    --mail-type=FAIL
    --mail-user=marvin.mathony@helmholtz-munich.de
)

ALL_TRAIN_JIDS=()

echo "Submitting lambda sweep: ${LAMBDAS[*]}"
echo ""

for lmbd in "${LAMBDAS[@]}"; do
    # Format lambda for directory name: 0.05 → lmbd0p05
    lmbd_tag="lmbd$(echo "$lmbd" | tr '.' 'p')"

    for fold in $(seq 0 $((N_FOLDS - 1))); do
        for seed in "${SEEDS[@]}"; do
            # Store checkpoints under runs_thalmann/fold{k}/seed_{S}/  but
            # with HP_RUN_DIR pointing to a lambda-specific subdir.
            RUN_SUBDIR="runs_thalmann/fold${fold}/${lmbd_tag}/seed_${seed}"

            JID=$(sbatch "${FLAGS[@]}" \
                --job-name="thal_lsweep_${lmbd_tag}_f${fold}_s${seed}" \
                --output="logs/lsweep_thal_${lmbd_tag}_fold${fold}_seed${seed}_%j.out" \
                --error="logs/lsweep_thal_${lmbd_tag}_fold${fold}_seed${seed}_%j.err" \
                --wrap="${CONDA_INIT} && cd ${WORKDIR} && \
                    HP_RUN_DIR=${RUN_SUBDIR} \
                    python run_Q_model.py \
                        --seed ${seed} --latent True \
                        --dataset_id ${DATASET_ID} --dgp ${DGP} --fold ${fold} \
                        --lmbd ${lmbd} --z ${Z_DIM} \
                        --hidden ${HIDDEN} --enc_hidden ${ENC_HIDDEN} \
                        --epochs ${EPOCHS} --step1_epochs ${STEP1_EPOCHS} \
                        --task_emb_dim ${TASK_EMB_DIM} \
                        --same_enc_dec False" \
                | awk '{print $NF}')
            ALL_TRAIN_JIDS+=("$JID")
            echo "  λ=${lmbd}  fold=${fold}  seed=${seed}  → job ${JID}"
        done
    done
done

echo ""
echo "========================================"
echo "Lambda sweep submitted:"
echo "  Lambdas:  ${LAMBDAS[*]}"
echo "  Total training jobs: ${#ALL_TRAIN_JIDS[@]}"
echo ""
echo "Monitor: squeue -u $USER"
echo "Cancel:  scancel ${ALL_TRAIN_JIDS[*]}"
echo ""
echo "After training completes, run the rate-distortion analysis:"
echo "  python rate_distortion_thalmann.py --trajectory"
echo "========================================"
