#!/bin/bash
# v3 + step 2.5: same HP-winner combo + 30 seeds + top-5-by-spec aggregation,
# but IDRNN runs an extra 400-epoch fine-tune phase (decoder unfrozen,
# encoder set to return_per_timestep=True) at the end of step 2.  Vanilla
# is retrained unchanged so the v3_step25 plot is self-contained.
#
# All artefacts under suffix "v3_step25" so v1, v2, v3, no_persev are intact.
#
# Outputs:
#   IDRNN runs : runs_dezfouli_v3_step25/foldF/seed_S/...
#                pre-step25 ckpt: epoch{N}_pre_step25.pt
#                post-step25 ckpt: epoch{N}.pt
#   Vanilla    : runs_vanilla_dezfouli_v3_step25/foldF/seed_S/...
#   Test data  : data_dezfouli/foldF/seed_S/v3_step25/...
#   Plots      : plots_dezfouli/outer_cv_*_v3_step25.png

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
RUN_SUFFIX="v3_step25"
TOP_K_SPEC=5
LMBD=0.5; Z_DIM=8; HIDDEN=5; ENC_HIDDEN=10; STEP1_EPOCHS=3000; EPOCHS=3000
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

echo "Submitting v3_step25 training jobs"
echo "  IDRNN seeds (${#IDRNN_SEEDS[@]}):  ${IDRNN_SEEDS[*]}"
echo "  Vanilla seeds (${#VANILLA_SEEDS[@]}): ${VANILLA_SEEDS[*]}"
echo "  step2_5_epochs=${STEP25_EPOCHS}  lr=${STEP25_LR}"
TRAIN_JOB_IDS=()

for fold in $(seq 0 $((N_FOLDS - 1))); do
    for seed in "${IDRNN_SEEDS[@]}"; do
        JID=$(sbatch "${TRAIN_FLAGS[@]}" ${EXTRA_FLAGS} \
            --job-name="dez_v3s25_lat_f${fold}_s${seed}" \
            --output="logs/ocv_dez_v3s25_train_lat_fold${fold}_seed${seed}_%j.out" \
            --error="logs/ocv_dez_v3s25_train_lat_fold${fold}_seed${seed}_%j.err" \
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
        echo "  IDRNN   fold=${fold} seed=${seed} -> ${JID}"
    done
    for seed in "${VANILLA_SEEDS[@]}"; do
        JID=$(sbatch "${TRAIN_FLAGS[@]}" ${EXTRA_FLAGS} \
            --job-name="dez_v3s25_van_f${fold}_s${seed}" \
            --output="logs/ocv_dez_v3s25_train_van_fold${fold}_seed${seed}_%j.out" \
            --error="logs/ocv_dez_v3s25_train_van_fold${fold}_seed${seed}_%j.err" \
            --wrap="${CONDA_INIT} && cd ${WORKDIR} && python run_Q_model.py \
                --seed ${seed} --latent False \
                --dataset_id ${DATASET_ID} --dgp ${DGP} --fold ${fold} \
                --hidden ${VANILLA_HIDDEN} --epochs ${EPOCHS} \
                --run_suffix ${RUN_SUFFIX}" \
            | awk '{print $NF}')
        TRAIN_JOB_IDS+=("$JID")
        echo "  Vanilla fold=${fold} seed=${seed} -> ${JID}"
    done
done
TRAIN_DEP="afterok:$(IFS=:; echo "${TRAIN_JOB_IDS[*]}")"

echo ""
echo "Submitting testing jobs..."
TEST_JOB_IDS=()
for fold in $(seq 0 $((N_FOLDS - 1))); do
    JID=$(sbatch "${TRAIN_FLAGS[@]}" --dependency="${TRAIN_DEP}" \
        --job-name="dez_v3s25_test_lat_f${fold}" \
        --output="logs/ocv_dez_v3s25_test_lat_fold${fold}_%j.out" \
        --error="logs/ocv_dez_v3s25_test_lat_fold${fold}_%j.err" \
        --wrap="${CONDA_INIT} && cd ${WORKDIR} && python testing_script.py \
            --latent True --dataset_id ${DATASET_ID} --dgp ${DGP} \
            --fold ${fold} --model_fitting False --run_suffix ${RUN_SUFFIX}" \
        | awk '{print $NF}')
    TEST_JOB_IDS+=("$JID")
    JID=$(sbatch "${TRAIN_FLAGS[@]}" --dependency="${TRAIN_DEP}" \
        --job-name="dez_v3s25_test_van_f${fold}" \
        --output="logs/ocv_dez_v3s25_test_van_fold${fold}_%j.out" \
        --error="logs/ocv_dez_v3s25_test_van_fold${fold}_%j.err" \
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
    --job-name="dez_v3s25_analyze" \
    --output="logs/ocv_dez_v3s25_analyze_%j.out" \
    --error="logs/ocv_dez_v3s25_analyze_%j.err" \
    --wrap="${CONDA_INIT} && cd ${WORKDIR} && mkdir -p plots_dezfouli && \
        python analyze_outer_cv.py --dgp dezfouli \
            --run_suffix ${RUN_SUFFIX} --top_k_by_specificity ${TOP_K_SPEC} \
            --seeds ${IDRNN_SEEDS_CSV} --vanilla_seeds ${VANILLA_SEEDS_CSV}" \
    | awk '{print $NF}')

ALL_JOB_IDS=("${TRAIN_JOB_IDS[@]}" "${TEST_JOB_IDS[@]}" "${ANALYZE_JID}")
echo ""
echo "========================================"
echo "Dezfouli v3_step25 pipeline submitted."
echo "  Training (x${#TRAIN_JOB_IDS[@]}): first=${TRAIN_JOB_IDS[0]} last=${TRAIN_JOB_IDS[-1]}"
echo "  Testing  (x${#TEST_JOB_IDS[@]}):  ${TEST_JOB_IDS[*]}"
echo "  Analysis: ${ANALYZE_JID}"
echo "  Cancel:   scancel ${ALL_JOB_IDS[*]}"
echo "========================================"
