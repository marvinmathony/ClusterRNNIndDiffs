#!/bin/bash
#SBATCH --job-name=hp_dez_van_%a
#SBATCH --output=logs/hp_dez_van_%a_%j.out
#SBATCH --error=logs/hp_dez_van_%a_%j.err
#SBATCH --open-mode=append
#SBATCH --array=0-4
#SBATCH --time=06:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH -p gpu_p
#SBATCH --qos gpu_normal
#SBATCH --constraint=a100_80gb|h100_80gb
#SBATCH --nice=10000
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=marvin.mathony@helmholtz-munich.de

# Vanilla AblatedRNN HP search for dezfouli — single-axis sweep over `hidden`.
# Selection criterion mirrors the IDRNN search: each run's inner-3-fold
# cv_val_loss (saved to config.json by run_Q_model.py), averaged across seeds.
#
# Grid: hidden ∈ {3, 5, 8, 10, 16}   (5 array tasks, 3 seeds each)
#
# Outputs (parallel to hp_search_runs_dezfouli/ but vanilla-only):
#   hp_search_runs_dezfouli_vanilla/h_{H}/seed_{S}/...
#   hp_search_results_dezfouli_vanilla/h_{H}/metrics.json

set -euo pipefail
mkdir -p logs

source /ictstr01/home/hcai/marvin.mathony/tools/apps/mamba/etc/profile.d/conda.sh
conda activate RNNproject

TASK_ID=$SLURM_ARRAY_TASK_ID

HIDDENS=(3 5 8 10 16)
SEEDS=(12 50 76)
EPOCHS=3000
DGP="dezfouli"
DATASET_ID=0

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

if [ ! -d "data_dezfouli" ]; then
    echo "ERROR: data_dezfouli/ not found — run load_dezfouli.py first."
    exit 1
fi

HIDDEN=${HIDDENS[$TASK_ID]}

echo "========================================"
echo "Vanilla HP search — Dezfouli  (task ${TASK_ID})"
echo "Job ID: $SLURM_JOB_ID  |  Hidden: $HIDDEN"
echo "Seeds: ${SEEDS[*]}  |  Start: $(date)"
echo "========================================"

TAG="h_${HIDDEN}"
RESULTS_DIR="hp_search_results_dezfouli_vanilla/${TAG}"
mkdir -p "$RESULTS_DIR"

for seed in "${SEEDS[@]}"; do
    echo "[$(date '+%H:%M:%S')] Training vanilla hidden=${HIDDEN} seed=${seed}..."

    export HP_RUN_DIR="hp_search_runs_dezfouli_vanilla/${TAG}/seed_${seed}"

    python run_Q_model.py \
        --seed $seed \
        --latent False \
        --dataset_id $DATASET_ID \
        --dgp $DGP \
        --hidden $HIDDEN \
        --epochs $EPOCHS

    echo "[$(date '+%H:%M:%S')] Completed vanilla hidden=${HIDDEN} seed=${seed}"
done

echo "Evaluating..."
python hyperparam_eval_dezfouli_vanilla.py \
    --hidden $HIDDEN \
    --seeds "${SEEDS[*]}" \
    --output "$RESULTS_DIR/metrics.json"

echo "Completed: $(date)  |  Results: $RESULTS_DIR"
