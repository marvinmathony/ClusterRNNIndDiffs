#!/bin/bash
# Submit outer-CV training for every HP combo in hp_search_results_thalmann_v2/.
# For each combo × fold × seed, trains IDRNN with the fold's data split.
# Results go to runs_thalmann_hp_v2_{combo}/fold{k}/seed_{s}/.
# After all training: submits the analysis job.

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

# Map combo name tags → real HP values
declare -A UW_MAP=( ["00"]="0.0" ["01"]="0.1" ["05"]="0.5" )
declare -A LMBD_MAP=( ["005"]="0.05" ["01"]="0.1" ["02"]="0.2" )

ALL_TRAIN_JIDS=()

for COMBO_DIR in hp_search_results_thalmann_v2/*/; do
    COMBO=$(basename "$COMBO_DIR")

    # Parse HP from combo name: uw{}_lmbd{}_eh{}_h{}_z{}
    IFS='_' read -ra PARTS <<< "$COMBO"
    UW_TAG="${PARTS[0]:2}"       # e.g. 00, 01, 05
    LMBD_TAG="${PARTS[1]:4}"     # e.g. 005, 01, 02
    EH="${PARTS[2]:2}"            # e.g. 5, 10, 20
    H="${PARTS[3]:1}"             # e.g. 5, 10
    Z="${PARTS[4]:1}"             # e.g. 3, 5, 10

    UW="${UW_MAP[$UW_TAG]:-}"
    LMBD="${LMBD_MAP[$LMBD_TAG]:-}"

    if [[ -z "$UW" || -z "$LMBD" ]]; then
        echo "WARNING: could not parse $COMBO — skipping"
        continue
    fi

    SUFFIX="hp_v2_${COMBO}"

    for FOLD in "${FOLDS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            SEED_DIR="runs_thalmann_${SUFFIX}/fold${FOLD}/seed_${SEED}"

            # Skip if already done
            if [[ -f "${SEED_DIR}/config.json" ]]; then
                # collect jid placeholder so analysis dep still works
                continue
            fi

            mkdir -p "${SEED_DIR}/checkpoints"

            JID=$(sbatch "${TRAIN_FLAGS[@]}" \
                --job-name="hp_v2_${COMBO}_f${FOLD}_s${SEED}" \
                --output="logs/hp_v2_${COMBO}_f${FOLD}_s${SEED}_%j.out" \
                --error="logs/hp_v2_${COMBO}_f${FOLD}_s${SEED}_%j.err" \
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

            ALL_TRAIN_JIDS+=("$JID")
            echo "  ${COMBO} fold=${FOLD} seed=${SEED} -> job ${JID}"
        done
    done
done

if [[ ${#ALL_TRAIN_JIDS[@]} -eq 0 ]]; then
    echo "All training jobs already done — submitting analysis directly."
    DEP_FLAG=""
else
    DEP_STR=$(IFS=:; echo "${ALL_TRAIN_JIDS[*]}")
    DEP_FLAG="--dependency=afterok:${DEP_STR}"
    echo ""
    echo "Submitted ${#ALL_TRAIN_JIDS[@]} training jobs."
fi

# Submit analysis job
AJID=$(sbatch \
    --nodes=1 --gres=gpu:0 -p cpu_p --qos cpu_normal \
    --mem=32G --cpus-per-task=8 --time=04:00:00 \
    $DEP_FLAG \
    --job-name="hp_v2_outer_cv_analyze" \
    --output="logs/hp_v2_outer_cv_analyze_%j.out" \
    --error="logs/hp_v2_outer_cv_analyze_%j.err" \
    --wrap="${CONDA_INIT} && cd ${WORKDIR} && python analyze_hp_v2_outer_cv.py" \
    | awk '{print $NF}')

echo "Analysis job: ${AJID}"
echo ""
echo "Monitor: squeue -u marvin.mathony"
