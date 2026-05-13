#!/bin/bash
# Usage: bash submit_sloutsky_pipeline.sh ["seed1 seed2 ..."]
# Submits the full Sloutsky pipeline as separate SLURM jobs with dependency chains:
#   training (per seed × model type, parallel) -> testing -> analysis -> plotting

set -euo pipefail

WORKDIR="$(cd "$(dirname "$0")" && pwd)"
cd "$WORKDIR"
mkdir -p logs

# ── Seeds ────────────────────────────────────────────────────────────────────
if [ -n "${1:-}" ]; then
    read -ra SEEDS <<< "$1"
else
    SEEDS=(200 300 400 500 600 999 2021 2022 2023 2024 2025)
fi
echo "Seeds: ${SEEDS[*]}"

# Optional: exclude specific nodes (e.g. bash submit_sloutsky_pipeline.sh "200" "--exclude=gpusrv60")
EXTRA_FLAGS="${2:-}"

# ── Shared config ─────────────────────────────────────────────────────────────
DGP="sloutsky"
DATASET_ID=0
LMBD=0.00
Z_DIM=10
HIDDEN=10
ENC_HIDDEN=20
EPOCHS=3000

CONDA_INIT="source /ictstr01/home/hcai/marvin.mathony/tools/apps/mamba/etc/profile.d/conda.sh && conda activate RNNproject"

# SLURM flags shared across all training jobs
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

# SLURM flags for CPU-only downstream jobs (no GPU requested, but same partition/qos)
CPU_FLAGS=(
    --nodes=1
    -p gpu_p
    --qos gpu_normal
    "--constraint=a100_80gb|h100_80gb"
    --nice=10000
    --mem=16G
    --cpus-per-task=8
    --time=02:00:00
    --open-mode=append
    --mail-type=FAIL
    --mail-user=marvin.mathony@helmholtz-munich.de
)

# ── Step 1: Submit training jobs (all seeds × model types in parallel) ────────
echo ""
echo "Submitting training jobs..."
TRAIN_JOB_IDS=()

for seed in "${SEEDS[@]}"; do
    # Latent model
    JID=$(sbatch "${TRAIN_FLAGS[@]}" ${EXTRA_FLAGS} \
        --job-name="slt_lat_${seed}" \
        --output="logs/train_latent_seed${seed}_%j.out" \
        --error="logs/train_latent_seed${seed}_%j.err" \
        --wrap="${CONDA_INIT} && cd ${WORKDIR} && python run_Q_model.py \
            --seed ${seed} --latent True \
            --dataset_id ${DATASET_ID} --dgp ${DGP} \
            --lmbd ${LMBD} --z ${Z_DIM} \
            --hidden ${HIDDEN} --enc_hidden ${ENC_HIDDEN} \
            --epochs ${EPOCHS} --step1_epochs 200 \
            --same_enc_dec True" \
        | awk '{print $NF}')
    TRAIN_JOB_IDS+=("$JID")
    echo "  Latent  seed ${seed} -> job ${JID}"

    # Vanilla model
    JID=$(sbatch "${TRAIN_FLAGS[@]}" ${EXTRA_FLAGS} \
        --job-name="slt_van_${seed}" \
        --output="logs/train_vanilla_seed${seed}_%j.out" \
        --error="logs/train_vanilla_seed${seed}_%j.err" \
        --wrap="${CONDA_INIT} && cd ${WORKDIR} && python run_Q_model.py \
            --seed ${seed} --latent False \
            --dataset_id ${DATASET_ID} --dgp ${DGP} \
            --hidden ${HIDDEN} --epochs ${EPOCHS}" \
        | awk '{print $NF}')
    TRAIN_JOB_IDS+=("$JID")
    echo "  Vanilla seed ${seed} -> job ${JID}"
done

# Build afterok dependency string from all training job IDs
TRAIN_DEP=$(IFS=:; echo "afterok:${TRAIN_JOB_IDS[*]}")

# ── Step 2: Testing jobs (depend on all training, run in parallel) ────────────
echo ""
echo "Submitting testing jobs (dependency: ${TRAIN_DEP})..."

TEST_LAT_JID=$(sbatch "${TRAIN_FLAGS[@]}" \
    --job-name="slt_test_lat" \
    --output="logs/test_latent_%j.out" \
    --error="logs/test_latent_%j.err" \
    --dependency="${TRAIN_DEP}" \
    --wrap="${CONDA_INIT} && cd ${WORKDIR} && python testing_script.py \
        --latent True --dataset_id ${DATASET_ID} --dgp ${DGP} --model_fitting False" \
    | awk '{print $NF}')
echo "  Test latent  -> job ${TEST_LAT_JID}"

TEST_VAN_JID=$(sbatch "${TRAIN_FLAGS[@]}" \
    --job-name="slt_test_van" \
    --output="logs/test_vanilla_%j.out" \
    --error="logs/test_vanilla_%j.err" \
    --dependency="${TRAIN_DEP}" \
    --wrap="${CONDA_INIT} && cd ${WORKDIR} && python testing_script.py \
        --latent False --dataset_id ${DATASET_ID} --dgp ${DGP} --model_fitting False" \
    | awk '{print $NF}')
echo "  Test vanilla -> job ${TEST_VAN_JID}"

# ── Step 3: Analysis + decoding (both depend on test jobs, run in parallel) ───
echo ""
TEST_DEP="afterok:${TEST_LAT_JID}:${TEST_VAN_JID}"
echo "Submitting analysis + decoding jobs (dependency: ${TEST_DEP})..."

ANALYZE_JID=$(sbatch "${CPU_FLAGS[@]}" \
    --job-name="slt_analyze" \
    --output="logs/analyze_%j.out" \
    --error="logs/analyze_%j.err" \
    --dependency="${TEST_DEP}" \
    --wrap="${CONDA_INIT} && cd ${WORKDIR} && mkdir -p plots_sloutsky && python analyze_cv_seeds.py" \
    | awk '{print $NF}')
echo "  Analysis  -> job ${ANALYZE_JID}"

DECODE_JID=$(sbatch "${CPU_FLAGS[@]}" \
    --job-name="slt_decode" \
    --output="logs/decode_%j.out" \
    --error="logs/decode_%j.err" \
    --dependency="${TEST_DEP}" \
    --wrap="${CONDA_INIT} && cd ${WORKDIR} && mkdir -p plots_sloutsky && python decoding_logistic.py" \
    | awk '{print $NF}')
echo "  Decoding  -> job ${DECODE_JID}"

# ── Step 4: Plotting (depends on analysis) ────────────────────────────────────
echo ""
ANALYZE_DEP="afterok:${ANALYZE_JID}"
echo "Submitting plotting jobs (dependency: ${ANALYZE_DEP})..."

PLOT_LAT_JID=$(sbatch "${CPU_FLAGS[@]}" \
    --job-name="slt_plot_lat" \
    --output="logs/plot_latent_%j.out" \
    --error="logs/plot_latent_%j.err" \
    --dependency="${ANALYZE_DEP}" \
    --wrap="${CONDA_INIT} && cd ${WORKDIR} && mkdir -p plots_sloutsky && \
        python plot_sloutsky_latents.py --latent True  --dim_reduction tsne --avg False" \
    | awk '{print $NF}')
echo "  Plot IDRNN   -> job ${PLOT_LAT_JID}"

PLOT_VAN_JID=$(sbatch "${CPU_FLAGS[@]}" \
    --job-name="slt_plot_van" \
    --output="logs/plot_vanilla_%j.out" \
    --error="logs/plot_vanilla_%j.err" \
    --dependency="${ANALYZE_DEP}" \
    --wrap="${CONDA_INIT} && cd ${WORKDIR} && mkdir -p plots_sloutsky && \
        python plot_sloutsky_latents.py --latent False --dim_reduction pca  --avg True" \
    | awk '{print $NF}')
echo "  Plot Vanilla -> job ${PLOT_VAN_JID}"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "Pipeline submitted. Job graph:"
echo "  Training (x${#TRAIN_JOB_IDS[@]}): ${TRAIN_JOB_IDS[*]}"
echo "  Testing:  ${TEST_LAT_JID} ${TEST_VAN_JID}"
echo "  Analysis: ${ANALYZE_JID}  |  Decoding: ${DECODE_JID}  (parallel)"
echo "  Plotting: ${PLOT_LAT_JID} ${PLOT_VAN_JID}"
echo ""
echo "Monitor with:  squeue -u $USER"
echo "Cancel all:    scancel ${TRAIN_JOB_IDS[*]} ${TEST_LAT_JID} ${TEST_VAN_JID} ${ANALYZE_JID} ${DECODE_JID} ${PLOT_LAT_JID} ${PLOT_VAN_JID}"
echo "========================================"
