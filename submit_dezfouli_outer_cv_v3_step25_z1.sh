#!/bin/bash
# v3_step25 with z_dim=1 — tests the "one-axis hypothesis": if dezfouli's
# individual-difference structure is essentially a single perseveration
# scalar, a 1-D latent should be sufficient (and the latent should map
# directly onto stay-prob).
#
# Same as v3_step25 except --z 1; encoder/decoder hidden sizes stay at the
# HP-search-winner values (h=5, eh=10) and vanilla.hidden stays at 8.  Step
# 2.5 is on (400 epochs, lr=5e-4).
#
# Run suffix: v3_step25_z1.  Outputs:
#   runs_dezfouli_v3_step25_z1/foldF/seed_S/...
#   runs_vanilla_dezfouli_v3_step25_z1/foldF/seed_S/...
#   data_dezfouli/foldF/seed_S/v3_step25_z1/...
#   plots_dezfouli/outer_cv_*_v3_step25_z1.png

set -euo pipefail
WORKDIR="$(cd "$(dirname "$0")" && pwd)"
cd "$WORKDIR"
mkdir -p logs

DEFAULT_SEEDS=(200 300 400 500 600 999 \
               2021 2022 2023 2024 2025 \
               401 402 403 404 405 406 407 408 409 \
               410 411 412 413 414 415 416 417 418 419)
if [ -n "${1:-}" ]; then read -ra IDRNN_SEEDS   <<< "$1"; else IDRNN_SEEDS=("${DEFAULT_SEEDS[@]}");   fi
if [ -n "${2:-}" ]; then read -ra VANILLA_SEEDS <<< "$2"; else VANILLA_SEEDS=("${DEFAULT_SEEDS[@]}"); fi
EXTRA_FLAGS="${3:-}"

DGP="dezfouli"; DATASET_ID=0; N_FOLDS=3
RUN_SUFFIX="v3_step25_z1"
TOP_K_SPEC=5
LMBD=0.5
Z_DIM=1                # ← the one-axis test
HIDDEN=5               # IDRNN decoder hidden (HP-winner, normal-sized)
ENC_HIDDEN=10          # IDRNN encoder hidden (HP-winner, normal-sized)
STEP1_EPOCHS=3000
EPOCHS=3000
VANILLA_HIDDEN=8
STEP25_EPOCHS=400
STEP25_LR=5e-4

CONDA_INIT="export WANDB_MODE=offline && source /ictstr01/home/hcai/marvin.mathony/tools/apps/mamba/etc/profile.d/conda.sh && conda activate RNNproject"

TRAIN_FLAGS=(
    --nodes=1 --gres=gpu:1 -p gpu_p --qos gpu_normal
    "--constraint=a100_80gb|h100_80gb" --nice=10000
    --mem=16G --cpus-per-task=4 --time=12:00:00
    --open-mode=append --mail-type=FAIL
    --mail-user=marvin.mathony@helmholtz-munich.de
)

if [ ! -d "data_dezfouli/fold0" ]; then
    echo "ERROR: data_dezfouli/fold0/ missing.  Run load_dezfouli.py first."
    exit 1
fi

echo "Submitting v3_step25_z1 training jobs"
echo "  IDRNN seeds (${#IDRNN_SEEDS[@]}):  ${IDRNN_SEEDS[*]}"
echo "  Vanilla seeds (${#VANILLA_SEEDS[@]}): ${VANILLA_SEEDS[*]}"
echo "  z_dim=${Z_DIM}  IDRNN hidden=${HIDDEN}  enc_hidden=${ENC_HIDDEN}  vanilla.hidden=${VANILLA_HIDDEN}"
echo "  step2_5_epochs=${STEP25_EPOCHS}  lr=${STEP25_LR}"
TRAIN_JOB_IDS=()

for fold in $(seq 0 $((N_FOLDS - 1))); do
    for seed in "${IDRNN_SEEDS[@]}"; do
        JID=$(sbatch "${TRAIN_FLAGS[@]}" ${EXTRA_FLAGS} \
            --job-name="dez_v3s25z1_lat_f${fold}_s${seed}" \
            --output="logs/ocv_dez_v3s25z1_train_lat_fold${fold}_seed${seed}_%j.out" \
            --error="logs/ocv_dez_v3s25z1_train_lat_fold${fold}_seed${seed}_%j.err" \
            --wrap="${CONDA_INIT} && cd ${WORKDIR} && python run_Q_model.py \
                --seed ${seed} --latent True \
                --dataset_id ${DATASET_ID} --dgp ${DGP} --fold ${fold} \
                --lmbd ${LMBD} --z ${Z_DIM} \
                --hidden ${HIDDEN} --enc_hidden ${ENC_HIDDEN} \
                --epochs ${EPOCHS} --step1_epochs ${STEP1_EPOCHS} \
                --same_enc_dec True \
                --step2_5_epochs ${STEP25_EPOCHS} --step2_5_lr ${STEP25_LR} \
                --run_suffix ${RUN_SUFFIX}" \
            | awk '{print $NF}')
        TRAIN_JOB_IDS+=("$JID")
    done
    for seed in "${VANILLA_SEEDS[@]}"; do
        JID=$(sbatch "${TRAIN_FLAGS[@]}" ${EXTRA_FLAGS} \
            --job-name="dez_v3s25z1_van_f${fold}_s${seed}" \
            --output="logs/ocv_dez_v3s25z1_train_van_fold${fold}_seed${seed}_%j.out" \
            --error="logs/ocv_dez_v3s25z1_train_van_fold${fold}_seed${seed}_%j.err" \
            --wrap="${CONDA_INIT} && cd ${WORKDIR} && python run_Q_model.py \
                --seed ${seed} --latent False \
                --dataset_id ${DATASET_ID} --dgp ${DGP} --fold ${fold} \
                --hidden ${VANILLA_HIDDEN} --epochs ${EPOCHS} \
                --run_suffix ${RUN_SUFFIX}" \
            | awk '{print $NF}')
        TRAIN_JOB_IDS+=("$JID")
    done
done
TRAIN_DEP="afterok:$(IFS=:; echo "${TRAIN_JOB_IDS[*]}")"

echo "Submitting testing jobs (chained on training)..."
TEST_JOB_IDS=()
for fold in $(seq 0 $((N_FOLDS - 1))); do
    JID=$(sbatch "${TRAIN_FLAGS[@]}" --dependency="${TRAIN_DEP}" \
        --job-name="dez_v3s25z1_test_lat_f${fold}" \
        --output="logs/ocv_dez_v3s25z1_test_lat_fold${fold}_%j.out" \
        --error="logs/ocv_dez_v3s25z1_test_lat_fold${fold}_%j.err" \
        --wrap="${CONDA_INIT} && cd ${WORKDIR} && python testing_script.py \
            --latent True --dataset_id ${DATASET_ID} --dgp ${DGP} \
            --fold ${fold} --model_fitting False --run_suffix ${RUN_SUFFIX}" \
        | awk '{print $NF}')
    TEST_JOB_IDS+=("$JID")
    JID=$(sbatch "${TRAIN_FLAGS[@]}" --dependency="${TRAIN_DEP}" \
        --job-name="dez_v3s25z1_test_van_f${fold}" \
        --output="logs/ocv_dez_v3s25z1_test_van_fold${fold}_%j.out" \
        --error="logs/ocv_dez_v3s25z1_test_van_fold${fold}_%j.err" \
        --wrap="${CONDA_INIT} && cd ${WORKDIR} && python testing_script.py \
            --latent False --dataset_id ${DATASET_ID} --dgp ${DGP} \
            --fold ${fold} --model_fitting False --run_suffix ${RUN_SUFFIX}" \
        | awk '{print $NF}')
    TEST_JOB_IDS+=("$JID")
done
TEST_DEP="afterok:$(IFS=:; echo "${TEST_JOB_IDS[*]}")"

IDRNN_SEEDS_CSV=$(IFS=,; echo "${IDRNN_SEEDS[*]}")
VANILLA_SEEDS_CSV=$(IFS=,; echo "${VANILLA_SEEDS[*]}")

ANALYZE_JID=$(sbatch \
    --nodes=1 -p gpu_p --qos gpu_normal "--constraint=a100_80gb|h100_80gb" \
    --nice=10000 --mem=32G --cpus-per-task=16 --time=06:00:00 \
    --open-mode=append --mail-type=ALL \
    --mail-user=marvin.mathony@helmholtz-munich.de \
    --dependency="${TEST_DEP}" \
    --job-name="dez_v3s25z1_analyze" \
    --output="logs/ocv_dez_v3s25z1_analyze_%j.out" \
    --error="logs/ocv_dez_v3s25z1_analyze_%j.err" \
    --wrap="${CONDA_INIT} && cd ${WORKDIR} && mkdir -p plots_dezfouli && \
        python analyze_outer_cv.py --dgp dezfouli \
            --run_suffix ${RUN_SUFFIX} --top_k_by_specificity ${TOP_K_SPEC} \
            --seeds ${IDRNN_SEEDS_CSV} --vanilla_seeds ${VANILLA_SEEDS_CSV}" \
    | awk '{print $NF}')

ALL_JOB_IDS=("${TRAIN_JOB_IDS[@]}" "${TEST_JOB_IDS[@]}" "${ANALYZE_JID}")
echo ""
echo "========================================"
echo "Dezfouli v3_step25_z1 pipeline submitted."
echo "  Training (x${#TRAIN_JOB_IDS[@]}): first=${TRAIN_JOB_IDS[0]} last=${TRAIN_JOB_IDS[-1]}"
echo "  Testing  (x${#TEST_JOB_IDS[@]}):  ${TEST_JOB_IDS[*]}"
echo "  Analysis: ${ANALYZE_JID}"
echo "  Cancel:   scancel ${ALL_JOB_IDS[*]}"
echo "========================================"
