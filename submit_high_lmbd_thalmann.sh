#!/bin/bash
# Submit high-lmbd IDRNN experiments for Thalmann.
# Tests whether forcing the encoder to match step-1 embeddings (high lmbd)
# avoids posterior collapse.
#
# Architecture fixed: eh=5, h=5, z=10, uw=0.5 (same as best current combo)
# New lmbd values: 0.5, 0.7, 0.9  (vs current max of 0.2)
#
# Results: runs_thalmann_hp_v2_uw05_lmbd{X}_eh5_h5_z10/fold{k}/seed_{s}/

set -euo pipefail

WORKDIR=/ictstr01/home/hcai/marvin.mathony/RNNsandUncertainty
cd "$WORKDIR"
mkdir -p logs

CONDA_INIT="source /ictstr01/home/hcai/marvin.mathony/tools/apps/mamba/etc/profile.d/conda.sh && conda activate RNNproject"

TRAIN_FLAGS=(
    --nodes=1 --gres=gpu:1 -p gpu_p --qos gpu_normal
    "--constraint=a100_80gb|h100_80gb"
    --nice=10000 --mem=16G --cpus-per-task=4 --time=12:00:00
    --open-mode=append --mail-type=FAIL
    --mail-user=marvin.mathony@helmholtz-munich.de
)

FOLDS=(0 1 2)
SEEDS=(42 123 456)
UW=0.5
EH=5; H=5; Z=10

# High lmbd values to test
declare -A LMBD_COMBOS=(
    ["05"]="0.5"
    ["07"]="0.7"
    ["09"]="0.9"
)

ALL_JIDS=()

for LMBD_TAG in "${!LMBD_COMBOS[@]}"; do
    LMBD="${LMBD_COMBOS[$LMBD_TAG]}"
    COMBO="uw05_lmbd${LMBD_TAG}_eh${EH}_h${H}_z${Z}"
    SUFFIX="hp_v2_${COMBO}"

    echo "--- ${COMBO} (lmbd=${LMBD}) ---"

    for FOLD in "${FOLDS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            SEED_DIR="runs_thalmann_${SUFFIX}/fold${FOLD}/seed_${SEED}"

            if [[ -f "${SEED_DIR}/config.json" ]]; then
                echo "  fold=${FOLD} seed=${SEED} already done — skipping"
                continue
            fi

            mkdir -p "${SEED_DIR}/checkpoints"

            JID=$(sbatch "${TRAIN_FLAGS[@]}" \
                --job-name="hl_${LMBD_TAG}_f${FOLD}_s${SEED}" \
                --output="logs/high_lmbd_${COMBO}_f${FOLD}_s${SEED}_%j.out" \
                --error="logs/high_lmbd_${COMBO}_f${FOLD}_s${SEED}_%j.err" \
                --wrap="${CONDA_INIT} && cd ${WORKDIR} && python run_Q_model.py \
                    --dgp thalmann \
                    --latent True \
                    --seed ${SEED} \
                    --fold ${FOLD} \
                    --lmbd ${LMBD} \
                    --z ${Z} \
                    --hidden ${H} \
                    --enc_hidden ${EH} \
                    --epochs 2000 \
                    --step1_epochs 2000 \
                    --task_emb_dim 4 \
                    --same_enc_dec False \
                    --unif_weight ${UW} \
                    --run_suffix ${SUFFIX}" \
                | awk '{print $NF}')

            ALL_JIDS+=("$JID")
            echo "  fold=${FOLD} seed=${SEED} -> job ${JID}"
        done
    done
done

echo ""
echo "Submitted ${#ALL_JIDS[@]} jobs total."
echo "Monitor: squeue -u marvin.mathony"
echo ""
echo "When done, run z_compare_combos.py with these combos to check z_std vs sigma."
