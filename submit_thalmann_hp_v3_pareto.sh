#!/bin/bash
# Submit outer-CV retraining for Pareto-optimal combo with the continuous encoder.
#
# Architecture (hp_v3):
#   IDRNN encoder GRU runs as ONE continuous sequence across all 31 blocks
#   and both tasks (no hidden-state reset at block boundaries).
#
# Combo: uw00_lmbd01_eh5_h5_z5  (Pareto-optimal NLL vs PHQ decoding from hp_v2)
# Output dirs:
#   runs_thalmann_hp_v3_uw00_lmbd01_eh5_h5_z5/fold{0,1,2}/seed_{seed}/
#
# Usage:
#   bash submit_thalmann_hp_v3_pareto.sh

set -euo pipefail
WORKDIR="$(cd "$(dirname "$0")" && pwd)"
cd "$WORKDIR"
mkdir -p logs

# ── HP combo ──────────────────────────────────────────────────────────────────
COMBO="uw00_lmbd01_eh5_h5_z5"
UW=0.0
LMBD=0.1
ENC_HIDDEN=5
HIDDEN=5
Z_DIM=5
TASK_EMB_DIM=4
EPOCHS=2000
STEP1_EPOCHS=3000
N_FOLDS=3
SEEDS=(42 123 456)
RUN_SUFFIX="hp_v3_${COMBO}"

CONDA_INIT="source /ictstr01/home/hcai/marvin.mathony/tools/apps/mamba/etc/profile.d/conda.sh && conda activate RNNproject"

TRAIN_FLAGS=(
    --nodes=1
    --gres=gpu:1
    -p gpu_p
    --qos gpu_normal
    "--constraint=a100_80gb|h100_80gb"
    --nice=10000
    --mem=16G
    --cpus-per-task=4
    --time=12:00:00
    --open-mode=append
    --mail-type=FAIL
    --mail-user=marvin.mathony@helmholtz-munich.de
)

# ── Training ──────────────────────────────────────────────────────────────────
echo "Submitting IDRNN training (${N_FOLDS} folds x ${#SEEDS[@]} seeds)..."
TRAIN_JOB_IDS=()

for fold in $(seq 0 $((N_FOLDS - 1))); do
    for seed in "${SEEDS[@]}"; do
        JID=$(sbatch "${TRAIN_FLAGS[@]}" \
            --job-name="thal_v3p_f${fold}_s${seed}" \
            --output="logs/thal_v3p_train_fold${fold}_seed${seed}_%j.out" \
            --error="logs/thal_v3p_train_fold${fold}_seed${seed}_%j.err" \
            --wrap="${CONDA_INIT} && cd ${WORKDIR} && python run_Q_model.py \
                --seed ${seed} --latent True \
                --dgp thalmann --fold ${fold} \
                --lmbd ${LMBD} --z ${Z_DIM} \
                --hidden ${HIDDEN} --enc_hidden ${ENC_HIDDEN} \
                --epochs ${EPOCHS} --step1_epochs ${STEP1_EPOCHS} \
                --task_emb_dim ${TASK_EMB_DIM} \
                --same_enc_dec False \
                --unif_weight ${UW} \
                --continuous_encoder True \
                --run_suffix ${RUN_SUFFIX}" \
            | awk '{print $NF}')
        TRAIN_JOB_IDS+=("$JID")
        echo "  fold=${fold} seed=${seed} -> job ${JID}"
    done
done

TRAIN_DEP="afterok:$(IFS=:; echo "${TRAIN_JOB_IDS[*]}")"

# ── Analysis ──────────────────────────────────────────────────────────────────
echo ""
echo "Submitting analysis (dependency: all training)..."

ANA_JID=$(sbatch \
    --nodes=1 -p gpu_p --qos gpu_normal \
    "--constraint=a100_80gb|h100_80gb" \
    --nice=10000 --mem=32G --cpus-per-task=8 --time=02:00:00 \
    --open-mode=append \
    --mail-type=ALL \
    --mail-user=marvin.mathony@helmholtz-munich.de \
    --job-name="thal_v3p_analyze" \
    --output="logs/thal_v3p_analyze_%j.out" \
    --error="logs/thal_v3p_analyze_%j.err" \
    --dependency="${TRAIN_DEP}" \
    --wrap="${CONDA_INIT} && cd ${WORKDIR} && python analyze_idrnn_vs_vanilla_thalmann.py" \
    | awk '{print $NF}')
echo "  Analysis -> job ${ANA_JID}"

echo ""
echo "========================================"
echo "hp_v3 Pareto-optimal combo submitted."
echo "  Combo:    ${COMBO}"
echo "  Suffix:   ${RUN_SUFFIX}"
echo "  Training: ${TRAIN_JOB_IDS[*]}"
echo "  Analysis: ${ANA_JID}"
echo ""
echo "Monitor: squeue -u $USER"
echo "Cancel:  scancel ${TRAIN_JOB_IDS[*]} ${ANA_JID}"
echo "========================================"
