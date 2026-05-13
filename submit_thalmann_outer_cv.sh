#!/bin/bash
# Usage: bash submit_thalmann_outer_cv.sh ["seed1 seed2 ..."] [extra_sbatch_flags]
#
# Submits the outer-CV Thalmann two-task pipeline as SLURM jobs:
#
#   training (fold × seed × model, all parallel)
#     -> testing (per fold × model, parallel)
#       -> analysis (analyze_outer_cv.py)
#
# Thalmann dataset: 2-armed bandit (30 blocks × 10 trials) +
#                   restless bandit  (1 block  × 200 trials)
# Task embedding: nn.Embedding(2, TASK_EMB_DIM) concatenated to decoder input.
# Effective decoder in_dim = BASE_IN_DIM(5) + TASK_EMB_DIM.

set -euo pipefail

WORKDIR="$(cd "$(dirname "$0")" && pwd)"
cd "$WORKDIR"
mkdir -p logs

# ── Seeds ─────────────────────────────────────────────────────────────────────
if [ -n "${1:-}" ]; then
    read -ra SEEDS <<< "$1"
else
    SEEDS=(200 300 400 500 600 999 2021 2022 2023 2024 2025)
fi
EXTRA_FLAGS="${2:-}"

# ── Config ────────────────────────────────────────────────────────────────────
DGP="thalmann"
DATASET_ID=0
N_FOLDS=3
LMBD=0.05
Z_DIM=10
HIDDEN=5
ENC_HIDDEN=5
EPOCHS=3000
TASK_EMB_DIM=4   # dimension of the learned task embedding; 0 to disable

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

CPU_FLAGS=(
    --nodes=1
    -p gpu_p
    --qos gpu_normal
    "--constraint=a100_80gb|h100_80gb"
    --nice=10000
    --mem=16G
    --cpus-per-task=8
    --time=06:00:00
    --open-mode=append
    --mail-type=FAIL
    --mail-user=marvin.mathony@helmholtz-munich.de
)

# ── Step 0: Generate fold data if not yet done ────────────────────────────────
if [ ! -d "data_thalmann/fold0" ]; then
    echo "Generating Thalmann fold data (data_thalmann/fold{0,1,2}/)..."
    DATA_JID=$(sbatch "${CPU_FLAGS[@]}" ${EXTRA_FLAGS} \
        --job-name="thal_ocv_data" \
        --output="logs/ocv_thal_data_%j.out" \
        --error="logs/ocv_thal_data_%j.err" \
        --wrap="${CONDA_INIT} && cd ${WORKDIR} && python load_thalmann.py" \
        | awk '{print $NF}')
    echo "  Data generation -> job ${DATA_JID}"
    DATA_DEP="afterok:${DATA_JID}"
else
    echo "Fold directories already exist, skipping data generation."
    DATA_DEP=""
fi

# ── Step 1: Training — one job per fold × seed × model type ──────────────────
echo ""
echo "Submitting training jobs (${N_FOLDS} folds × ${#SEEDS[@]} seeds × 2 models)..."
TRAIN_JOB_IDS=()

for fold in $(seq 0 $((N_FOLDS - 1))); do
    for seed in "${SEEDS[@]}"; do
        # Latent (IDRNN)
        JID=$(sbatch "${TRAIN_FLAGS[@]}" ${EXTRA_FLAGS} \
            ${DATA_DEP:+--dependency="${DATA_DEP}"} \
            --job-name="thal_ocv_lat_f${fold}_s${seed}" \
            --output="logs/ocv_thal_train_latent_fold${fold}_seed${seed}_%j.out" \
            --error="logs/ocv_thal_train_latent_fold${fold}_seed${seed}_%j.err" \
            --wrap="${CONDA_INIT} && cd ${WORKDIR} && python run_Q_model.py \
                --seed ${seed} --latent True \
                --dataset_id ${DATASET_ID} --dgp ${DGP} --fold ${fold} \
                --lmbd ${LMBD} --z ${Z_DIM} \
                --hidden ${HIDDEN} --enc_hidden ${ENC_HIDDEN} \
                --epochs ${EPOCHS} --step1_epochs 1000 \
                --task_emb_dim ${TASK_EMB_DIM} \
                --same_enc_dec False" \
            | awk '{print $NF}')
        TRAIN_JOB_IDS+=("$JID")
        echo "  IDRNN   fold=${fold} seed=${seed} -> job ${JID}"

        # Vanilla
        JID=$(sbatch "${TRAIN_FLAGS[@]}" ${EXTRA_FLAGS} \
            ${DATA_DEP:+--dependency="${DATA_DEP}"} \
            --job-name="thal_ocv_van_f${fold}_s${seed}" \
            --output="logs/ocv_thal_train_vanilla_fold${fold}_seed${seed}_%j.out" \
            --error="logs/ocv_thal_train_vanilla_fold${fold}_seed${seed}_%j.err" \
            --wrap="${CONDA_INIT} && cd ${WORKDIR} && python run_Q_model.py \
                --seed ${seed} --latent False \
                --dataset_id ${DATASET_ID} --dgp ${DGP} --fold ${fold} \
                --hidden ${HIDDEN} --epochs ${EPOCHS} \
                --task_emb_dim ${TASK_EMB_DIM}" \
            | awk '{print $NF}')
        TRAIN_JOB_IDS+=("$JID")
        echo "  Vanilla fold=${fold} seed=${seed} -> job ${JID}"
    done
done

TRAIN_DEP="afterok:$(IFS=:; echo "${TRAIN_JOB_IDS[*]}")"

# ── Step 2: Testing — one job per fold × model type ───────────────────────────
echo ""
echo "Submitting testing jobs (dependency: all training)..."
TEST_JOB_IDS=()

for fold in $(seq 0 $((N_FOLDS - 1))); do
    JID=$(sbatch "${TRAIN_FLAGS[@]}" \
        --job-name="thal_ocv_test_lat_f${fold}" \
        --output="logs/ocv_thal_test_latent_fold${fold}_%j.out" \
        --error="logs/ocv_thal_test_latent_fold${fold}_%j.err" \
        --dependency="${TRAIN_DEP}" \
        --wrap="${CONDA_INIT} && cd ${WORKDIR} && python testing_script.py \
            --latent True --dataset_id ${DATASET_ID} --dgp ${DGP} \
            --fold ${fold} --model_fitting False" \
        | awk '{print $NF}')
    TEST_JOB_IDS+=("$JID")
    echo "  Test IDRNN   fold=${fold} -> job ${JID}"

    JID=$(sbatch "${TRAIN_FLAGS[@]}" \
        --job-name="thal_ocv_test_van_f${fold}" \
        --output="logs/ocv_thal_test_vanilla_fold${fold}_%j.out" \
        --error="logs/ocv_thal_test_vanilla_fold${fold}_%j.err" \
        --dependency="${TRAIN_DEP}" \
        --wrap="${CONDA_INIT} && cd ${WORKDIR} && python testing_script.py \
            --latent False --dataset_id ${DATASET_ID} --dgp ${DGP} \
            --fold ${fold} --model_fitting False" \
        | awk '{print $NF}')
    TEST_JOB_IDS+=("$JID")
    echo "  Test Vanilla fold=${fold} -> job ${JID}"
done

TEST_DEP="afterok:$(IFS=:; echo "${TEST_JOB_IDS[*]}")"

# ── Step 3: Analysis ──────────────────────────────────────────────────────────
echo ""
echo "Submitting analysis job (dependency: all testing)..."

ANALYZE_JID=$(sbatch \
    --nodes=1 -p gpu_p --qos gpu_normal \
    "--constraint=a100_80gb|h100_80gb" \
    --nice=10000 --mem=32G --cpus-per-task=16 --time=06:00:00 \
    --open-mode=append --mail-type=ALL \
    --mail-user=marvin.mathony@helmholtz-munich.de \
    --job-name="thal_ocv_analyze" \
    --output="logs/ocv_thal_analyze_%j.out" \
    --error="logs/ocv_thal_analyze_%j.err" \
    --dependency="${TEST_DEP}" \
    --wrap="${CONDA_INIT} && cd ${WORKDIR} && mkdir -p plots_thalmann && python analyze_thalmann.py --dgp thalmann" \
    | awk '{print $NF}')
echo "  Analysis -> job ${ANALYZE_JID}"

# ── Summary ───────────────────────────────────────────────────────────────────
ALL_JOB_IDS=("${TRAIN_JOB_IDS[@]}" "${TEST_JOB_IDS[@]}" "${ANALYZE_JID}")
echo ""
echo "========================================"
echo "Thalmann outer CV pipeline submitted."
echo "  DGP:        ${DGP}"
echo "  TASK_EMB_DIM: ${TASK_EMB_DIM}"
echo "  Training  (x${#TRAIN_JOB_IDS[@]}): ${TRAIN_JOB_IDS[*]}"
echo "  Testing   (x${#TEST_JOB_IDS[@]}):  ${TEST_JOB_IDS[*]}"
echo "  Analysis:  ${ANALYZE_JID}"
echo ""
echo "Monitor with:  squeue -u $USER"
echo "Cancel all:    scancel ${ALL_JOB_IDS[*]}"
echo "========================================"
