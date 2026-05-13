#!/bin/bash
# Submit the Thalmann outer-CV pipeline for a sweep of uniformity_loss weights.
#
# Usage:
#   bash submit_thalmann_unif_cv.sh ["seed1 seed2 ..."] [extra_sbatch_flags]
#
# For each unif_weight value, submits:
#   Training (fold × seed, parallel)
#     -> Testing (fold, parallel)
#       -> Analysis (analyze_unif_sweep_thalmann.py)
#
# IDRNN runs are stored in runs_thalmann_unif{w}/ to keep results separate.
# Vanilla is shared across all unif_weight values (uses existing runs_vanilla_thalmann/).

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

# ── Uniformity loss sweep ──────────────────────────────────────────────────────
UNIF_WEIGHTS=(0.0 0.1 0.5 1.0)

# ── Shared hyperparameters ─────────────────────────────────────────────────────
DGP="thalmann"
DATASET_ID=0
N_FOLDS=3
LMBD=0.1
Z_DIM=5
HIDDEN=5
ENC_HIDDEN=5
EPOCHS=2000
STEP1_EPOCHS=1000
TASK_EMB_DIM=4

CONDA_INIT="source /ictstr01/home/hcai/marvin.mathony/tools/apps/mamba/etc/profile.d/conda.sh && conda activate RNNproject"

TRAIN_FLAGS=(
    --nodes=1 --gres=gpu:1
    -p gpu_p --qos gpu_normal
    "--constraint=a100_80gb|h100_80gb"
    --nice=10000 --mem=16G --cpus-per-task=4 --time=12:00:00
    --open-mode=append --mail-type=FAIL
    --mail-user=marvin.mathony@helmholtz-munich.de
)

CPU_FLAGS=(
    --nodes=1 -p gpu_p --qos gpu_normal
    "--constraint=a100_80gb|h100_80gb"
    --nice=10000 --mem=32G --cpus-per-task=16 --time=06:00:00
    --open-mode=append --mail-type=FAIL
    --mail-user=marvin.mathony@helmholtz-munich.de
)

# ── Step 0: Ensure fold data exists ───────────────────────────────────────────
if [ ! -d "data_thalmann/fold0" ]; then
    echo "Generating Thalmann fold data..."
    DATA_JID=$(sbatch "${CPU_FLAGS[@]}" ${EXTRA_FLAGS} \
        --job-name="thal_unif_data" \
        --output="logs/unif_thal_data_%j.out" \
        --error="logs/unif_thal_data_%j.err" \
        --wrap="${CONDA_INIT} && cd ${WORKDIR} && python load_thalmann.py" \
        | awk '{print $NF}')
    DATA_DEP="afterok:${DATA_JID}"
    echo "  Data generation -> job ${DATA_JID}"
else
    DATA_DEP=""
fi

# ── Step 1: Vanilla training (shared across all unif_weight values) ────────────
# Only submit if vanilla runs don't already exist for all folds and seeds.
VAN_JOB_IDS=()
ALL_VAN_EXIST=true
for fold in $(seq 0 $((N_FOLDS - 1))); do
    for seed in "${SEEDS[@]}"; do
        if [ ! -d "runs_vanilla_${DGP}/fold${fold}/seed_${seed}" ]; then
            ALL_VAN_EXIST=false; break 2
        fi
    done
done

if $ALL_VAN_EXIST; then
    echo "All vanilla runs already exist — skipping vanilla training."
else
    echo "Submitting vanilla training jobs..."
    for fold in $(seq 0 $((N_FOLDS - 1))); do
        for seed in "${SEEDS[@]}"; do
            JID=$(sbatch "${TRAIN_FLAGS[@]}" ${EXTRA_FLAGS} \
                ${DATA_DEP:+--dependency="${DATA_DEP}"} \
                --job-name="thal_unif_van_f${fold}_s${seed}" \
                --output="logs/unif_thal_van_f${fold}_s${seed}_%j.out" \
                --error="logs/unif_thal_van_f${fold}_s${seed}_%j.err" \
                --wrap="${CONDA_INIT} && cd ${WORKDIR} && python run_Q_model.py \
                    --seed ${seed} --latent False \
                    --dataset_id ${DATASET_ID} --dgp ${DGP} --fold ${fold} \
                    --hidden ${HIDDEN} --epochs ${EPOCHS} \
                    --task_emb_dim ${TASK_EMB_DIM}" \
                | awk '{print $NF}')
            VAN_JOB_IDS+=("$JID")
            echo "  Vanilla fold=${fold} seed=${seed} -> job ${JID}"
        done
    done
fi

# ── Steps 2–4: Per-unif_weight IDRNN training, testing, analysis ──────────────
ALL_ANALYZE_JIDS=()

for UW in "${UNIF_WEIGHTS[@]}"; do
    # Format suffix (replace . with p for filesystem safety)
    SUFFIX="unif$(echo "$UW" | tr '.' 'p')"
    echo ""
    echo "========================================"
    echo "unif_weight=${UW}  (suffix: ${SUFFIX})"
    echo "========================================"

    # Step 2: IDRNN training
    IDRNN_JOB_IDS=()
    for fold in $(seq 0 $((N_FOLDS - 1))); do
        for seed in "${SEEDS[@]}"; do
            DEPS=""
            [ -n "${DATA_DEP}" ] && DEPS="${DATA_DEP}"
            [ "${#VAN_JOB_IDS[@]}" -gt 0 ] && DEPS="${DEPS:+${DEPS}:}afterok:$(IFS=:; echo "${VAN_JOB_IDS[*]}")"

            JID=$(sbatch "${TRAIN_FLAGS[@]}" ${EXTRA_FLAGS} \
                ${DEPS:+--dependency="${DEPS}"} \
                --job-name="thal_unif_lat_uw${UW}_f${fold}_s${seed}" \
                --output="logs/unif_thal_lat_uw${UW}_f${fold}_s${seed}_%j.out" \
                --error="logs/unif_thal_lat_uw${UW}_f${fold}_s${seed}_%j.err" \
                --wrap="${CONDA_INIT} && cd ${WORKDIR} && python run_Q_model.py \
                    --seed ${seed} --latent True \
                    --dataset_id ${DATASET_ID} --dgp ${DGP} --fold ${fold} \
                    --lmbd ${LMBD} --z ${Z_DIM} \
                    --hidden ${HIDDEN} --enc_hidden ${ENC_HIDDEN} \
                    --epochs ${EPOCHS} --step1_epochs ${STEP1_EPOCHS} \
                    --task_emb_dim ${TASK_EMB_DIM} \
                    --same_enc_dec False \
                    --unif_weight ${UW} \
                    --run_suffix ${SUFFIX}" \
                | awk '{print $NF}')
            IDRNN_JOB_IDS+=("$JID")
            echo "  IDRNN fold=${fold} seed=${seed} -> job ${JID}"
        done
    done

    IDRNN_DEP="afterok:$(IFS=:; echo "${IDRNN_JOB_IDS[*]}")"

    # Step 3: Testing (IDRNN only; vanilla testing uses existing results)
    TEST_JOB_IDS=()
    for fold in $(seq 0 $((N_FOLDS - 1))); do
        JID=$(sbatch "${TRAIN_FLAGS[@]}" \
            --dependency="${IDRNN_DEP}" \
            --job-name="thal_unif_test_uw${UW}_f${fold}" \
            --output="logs/unif_thal_test_uw${UW}_f${fold}_%j.out" \
            --error="logs/unif_thal_test_uw${UW}_f${fold}_%j.err" \
            --wrap="${CONDA_INIT} && cd ${WORKDIR} && python testing_script.py \
                --latent True --dataset_id ${DATASET_ID} --dgp ${DGP} \
                --fold ${fold} --model_fitting False \
                --run_suffix ${SUFFIX}" \
            | awk '{print $NF}')
        TEST_JOB_IDS+=("$JID")
        echo "  Test IDRNN fold=${fold} -> job ${JID}"
    done

    ALL_ANALYZE_JIDS+=("${TEST_JOB_IDS[@]}")
done

# ── Step 4: Joint analysis across all unif_weight values ──────────────────────
echo ""
echo "Submitting joint analysis job..."

ANALYZE_DEP="afterok:$(IFS=:; echo "${ALL_ANALYZE_JIDS[*]}")"
UNIF_STR=$(IFS=' '; echo "${UNIF_WEIGHTS[*]}")

ANA_JID=$(sbatch \
    --nodes=1 -p gpu_p --qos gpu_normal \
    "--constraint=a100_80gb|h100_80gb" \
    --nice=10000 --mem=32G --cpus-per-task=16 --time=04:00:00 \
    --open-mode=append --mail-type=ALL \
    --mail-user=marvin.mathony@helmholtz-munich.de \
    --job-name="thal_unif_analyze" \
    --output="logs/unif_thal_analyze_%j.out" \
    --error="logs/unif_thal_analyze_%j.err" \
    --dependency="${ANALYZE_DEP}" \
    --wrap="${CONDA_INIT} && cd ${WORKDIR} && mkdir -p plots_thalmann && \
        python analyze_unif_sweep_thalmann.py \
            --unif_weights ${UNIF_STR} \
            --dgp ${DGP} --n_folds ${N_FOLDS}" \
    | awk '{print $NF}')
echo "  Analysis -> job ${ANA_JID}"

echo ""
echo "========================================"
echo "Thalmann uniformity sweep pipeline submitted."
echo "  unif_weights: ${UNIF_WEIGHTS[*]}"
echo "  seeds:        ${SEEDS[*]}"
echo "  Monitor:      squeue -u $USER"
echo "========================================"
