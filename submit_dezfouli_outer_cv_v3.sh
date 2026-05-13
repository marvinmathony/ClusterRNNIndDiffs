#!/bin/bash
# Usage: bash submit_dezfouli_outer_cv_v3.sh ["idrnn_seeds..." ["van_seeds..."] [extra_sbatch_flags]]
#
# v3 retrain: 30 seeds for IDRNN and vanilla.  IDRNN bar averages NLL across
# the top-5 seeds per fold (ranked by step1_specificity).  Vanilla bar
# averages across all 30 seeds per fold.
#
# All artifacts go through --run_suffix v3 so the existing v1 + v2 outputs
# stay untouched (runs_dezfouli/, runs_dezfouli_v2/, fold latents, plots).
#
# Outputs:
#   IDRNN runs : runs_dezfouli_v3/foldF/seed_S/...
#   Vanilla    : runs_vanilla_dezfouli_v3/foldF/seed_S/...
#   Test data  : data_dezfouli/foldF/seed_S/v3/{latents_tensor*.pt, rnn_results*.csv}
#   Plots      : plots_dezfouli/outer_cv_*_v3.png

set -euo pipefail

WORKDIR="$(cd "$(dirname "$0")" && pwd)"
cd "$WORKDIR"
mkdir -p logs

# ── 30 seeds (same pool as the original v1 outer-CV; deduped to 30) ──────────
DEFAULT_SEEDS=(200 300 400 500 600 999 \
               2021 2022 2023 2024 2025 \
               401 402 403 404 405 406 407 408 409 \
               410 411 412 413 414 415 416 417 418 419)

if [ -n "${1:-}" ]; then
    read -ra IDRNN_SEEDS <<< "$1"
else
    IDRNN_SEEDS=("${DEFAULT_SEEDS[@]}")
fi
if [ -n "${2:-}" ]; then
    read -ra VANILLA_SEEDS <<< "$2"
else
    VANILLA_SEEDS=("${DEFAULT_SEEDS[@]}")
fi
EXTRA_FLAGS="${3:-}"

# ── Config (HP-search v2 winner: lmbd=0.5, z=8, h=5, eh=10, step1=3000) ──────
DGP="dezfouli"
DATASET_ID=0
N_FOLDS=3
RUN_SUFFIX="v3"
TOP_K_SPEC=5         # NLL averaged across top-5 IDRNN seeds per fold (by spec)

LMBD=0.5
Z_DIM=8
HIDDEN=5             # IDRNN decoder hidden
ENC_HIDDEN=10        # IDRNN encoder hidden
STEP1_EPOCHS=3000
EPOCHS=3000
VANILLA_HIDDEN=8     # match IDRNN z_dim → equal predictor capacity until the
                     # vanilla HP sweep nominates a different winner

CONDA_INIT="export WANDB_MODE=offline && source /ictstr01/home/hcai/marvin.mathony/tools/apps/mamba/etc/profile.d/conda.sh && conda activate RNNproject"

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

# ── Step 0: ensure fold data exists (NEVER regenerate; would clobber v1/v2) ──
if [ ! -d "data_dezfouli/fold0" ]; then
    echo "ERROR: data_dezfouli/fold0/ missing.  Run load_dezfouli.py first."
    exit 1
fi

# ── Step 1: Training ─────────────────────────────────────────────────────────
echo ""
echo "Submitting v3 training jobs"
echo "  IDRNN seeds (${#IDRNN_SEEDS[@]}):  ${IDRNN_SEEDS[*]}"
echo "  Vanilla seeds (${#VANILLA_SEEDS[@]}): ${VANILLA_SEEDS[*]}"
TRAIN_JOB_IDS=()

for fold in $(seq 0 $((N_FOLDS - 1))); do
    for seed in "${IDRNN_SEEDS[@]}"; do
        JID=$(sbatch "${TRAIN_FLAGS[@]}" ${EXTRA_FLAGS} \
            --job-name="dez_v3_lat_f${fold}_s${seed}" \
            --output="logs/ocv_dez_v3_train_latent_fold${fold}_seed${seed}_%j.out" \
            --error="logs/ocv_dez_v3_train_latent_fold${fold}_seed${seed}_%j.err" \
            --wrap="${CONDA_INIT} && cd ${WORKDIR} && python run_Q_model.py \
                --seed ${seed} --latent True \
                --dataset_id ${DATASET_ID} --dgp ${DGP} --fold ${fold} \
                --lmbd ${LMBD} --z ${Z_DIM} \
                --hidden ${HIDDEN} --enc_hidden ${ENC_HIDDEN} \
                --epochs ${EPOCHS} --step1_epochs ${STEP1_EPOCHS} \
                --same_enc_dec True \
                --run_suffix ${RUN_SUFFIX}" \
            | awk '{print $NF}')
        TRAIN_JOB_IDS+=("$JID")
        echo "  IDRNN   fold=${fold} seed=${seed} -> job ${JID}"
    done

    for seed in "${VANILLA_SEEDS[@]}"; do
        JID=$(sbatch "${TRAIN_FLAGS[@]}" ${EXTRA_FLAGS} \
            --job-name="dez_v3_van_f${fold}_s${seed}" \
            --output="logs/ocv_dez_v3_train_vanilla_fold${fold}_seed${seed}_%j.out" \
            --error="logs/ocv_dez_v3_train_vanilla_fold${fold}_seed${seed}_%j.err" \
            --wrap="${CONDA_INIT} && cd ${WORKDIR} && python run_Q_model.py \
                --seed ${seed} --latent False \
                --dataset_id ${DATASET_ID} --dgp ${DGP} --fold ${fold} \
                --hidden ${VANILLA_HIDDEN} --epochs ${EPOCHS} \
                --run_suffix ${RUN_SUFFIX}" \
            | awk '{print $NF}')
        TRAIN_JOB_IDS+=("$JID")
        echo "  Vanilla fold=${fold} seed=${seed} -> job ${JID}"
    done
done

TRAIN_DEP="afterok:$(IFS=:; echo "${TRAIN_JOB_IDS[*]}")"

# ── Step 2: Testing ──────────────────────────────────────────────────────────
echo ""
echo "Submitting testing jobs (dependency: all training)..."
TEST_JOB_IDS=()

for fold in $(seq 0 $((N_FOLDS - 1))); do
    JID=$(sbatch "${TRAIN_FLAGS[@]}" \
        --job-name="dez_v3_test_lat_f${fold}" \
        --output="logs/ocv_dez_v3_test_latent_fold${fold}_%j.out" \
        --error="logs/ocv_dez_v3_test_latent_fold${fold}_%j.err" \
        --dependency="${TRAIN_DEP}" \
        --wrap="${CONDA_INIT} && cd ${WORKDIR} && python testing_script.py \
            --latent True --dataset_id ${DATASET_ID} --dgp ${DGP} \
            --fold ${fold} --model_fitting False \
            --run_suffix ${RUN_SUFFIX}" \
        | awk '{print $NF}')
    TEST_JOB_IDS+=("$JID")
    echo "  Test IDRNN   fold=${fold} -> job ${JID}"

    JID=$(sbatch "${TRAIN_FLAGS[@]}" \
        --job-name="dez_v3_test_van_f${fold}" \
        --output="logs/ocv_dez_v3_test_vanilla_fold${fold}_%j.out" \
        --error="logs/ocv_dez_v3_test_vanilla_fold${fold}_%j.err" \
        --dependency="${TRAIN_DEP}" \
        --wrap="${CONDA_INIT} && cd ${WORKDIR} && python testing_script.py \
            --latent False --dataset_id ${DATASET_ID} --dgp ${DGP} \
            --fold ${fold} --model_fitting False \
            --run_suffix ${RUN_SUFFIX}" \
        | awk '{print $NF}')
    TEST_JOB_IDS+=("$JID")
    echo "  Test Vanilla fold=${fold} -> job ${JID}"
done

TEST_DEP="afterok:$(IFS=:; echo "${TEST_JOB_IDS[*]}")"

# ── Step 3: Analysis ─────────────────────────────────────────────────────────
IDRNN_SEEDS_CSV=$(IFS=,; echo "${IDRNN_SEEDS[*]}")
VANILLA_SEEDS_CSV=$(IFS=,; echo "${VANILLA_SEEDS[*]}")

echo ""
echo "Submitting analysis job (dependency: all testing)..."

ANALYZE_JID=$(sbatch \
    --nodes=1 -p gpu_p --qos gpu_normal \
    "--constraint=a100_80gb|h100_80gb" \
    --nice=10000 --mem=32G --cpus-per-task=16 --time=06:00:00 \
    --open-mode=append --mail-type=ALL \
    --mail-user=marvin.mathony@helmholtz-munich.de \
    --job-name="dez_v3_analyze" \
    --output="logs/ocv_dez_v3_analyze_%j.out" \
    --error="logs/ocv_dez_v3_analyze_%j.err" \
    --dependency="${TEST_DEP}" \
    --wrap="${CONDA_INIT} && cd ${WORKDIR} && mkdir -p plots_dezfouli && \
        python analyze_outer_cv.py --dgp dezfouli \
            --run_suffix ${RUN_SUFFIX} \
            --top_k_by_specificity ${TOP_K_SPEC} \
            --seeds ${IDRNN_SEEDS_CSV} \
            --vanilla_seeds ${VANILLA_SEEDS_CSV}" \
    | awk '{print $NF}')
echo "  Analysis -> job ${ANALYZE_JID}"

ALL_JOB_IDS=("${TRAIN_JOB_IDS[@]}" "${TEST_JOB_IDS[@]}" "${ANALYZE_JID}")
echo ""
echo "========================================"
echo "Dezfouli v3 outer-CV pipeline submitted."
echo "  Training  (x${#TRAIN_JOB_IDS[@]}): first=${TRAIN_JOB_IDS[0]} last=${TRAIN_JOB_IDS[-1]}"
echo "  Testing   (x${#TEST_JOB_IDS[@]}):  ${TEST_JOB_IDS[*]}"
echo "  Analysis:  ${ANALYZE_JID}"
echo ""
echo "Monitor:  squeue -u $USER"
echo "Cancel:   scancel ${ALL_JOB_IDS[*]}"
echo "========================================"
