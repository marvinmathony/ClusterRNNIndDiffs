#!/bin/bash
#SBATCH --job-name=thal_hp_v2
#SBATCH --output=logs/thal_hp_v2_%A_%a.out
#SBATCH --error=logs/thal_hp_v2_%A_%a.err
#SBATCH --array=0-161
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH -p gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --constraint="a100_80gb|h100_80gb"
#SBATCH --nice=10000
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=marvin.mathony@helmholtz-munich.de

source /ictstr01/home/hcai/marvin.mathony/tools/apps/mamba/etc/profile.d/conda.sh
conda activate RNNproject

WORKDIR=/ictstr01/home/hcai/marvin.mathony/RNNsandUncertainty
cd $WORKDIR

# ── HP grid ───────────────────────────────────────────────────────────────────
# unif_weight: [0.0, 0.1, 0.5]        (3)
# lmbd:        [0.05, 0.1, 0.2]       (3)
# enc_hidden:  [5, 10, 20]             (3)
# hidden:      [5, 10]                 (2)
# z_dim:       [3, 5, 10]              (3)
# Total: 3*3*3*2*3 = 162

IDX=$SLURM_ARRAY_TASK_ID

UNIF_WEIGHTS=(0.0 0.1 0.5)
LMBDS=(0.05 0.1 0.2)
ENC_HIDDENS=(5 10 20)
HIDDENS=(5 10)
Z_DIMS=(3 5 10)

tmp=$IDX
z_i=$((tmp % 3));      tmp=$((tmp / 3))
h_i=$((tmp % 2));      tmp=$((tmp / 2))
eh_i=$((tmp % 3));     tmp=$((tmp / 3))
lmbd_i=$((tmp % 3));   tmp=$((tmp / 3))
uw_i=$((tmp % 3))

UW=${UNIF_WEIGHTS[$uw_i]}
LMBD=${LMBDS[$lmbd_i]}
EH=${ENC_HIDDENS[$eh_i]}
H=${HIDDENS[$h_i]}
Z=${Z_DIMS[$z_i]}

# Format combo name (replace dots with 'p')
UW_TAG=$(echo $UW | sed 's/\.//g; s/^0$/0p0/; s/^1$/1p0/')
LMBD_TAG=$(echo $LMBD | sed 's/\.//g')
COMBO="uw${UW_TAG}_lmbd${LMBD_TAG}_eh${EH}_h${H}_z${Z}"

echo "Array task $IDX: unif_weight=${UW} lmbd=${LMBD} enc_hidden=${EH} hidden=${H} z_dim=${Z}"
echo "Combo: ${COMBO}"

RESULT_DIR="hp_search_results_thalmann_v2/${COMBO}"
mkdir -p "${RESULT_DIR}"

SEEDS=(42 123 456)

for SEED in "${SEEDS[@]}"; do
    SEED_DIR="${RESULT_DIR}/seed_${SEED}"
    mkdir -p "${SEED_DIR}/checkpoints"

    if [ -f "${SEED_DIR}/config.json" ]; then
        echo "  Seed ${SEED}: already done, skipping."
        continue
    fi

    echo "  Running seed ${SEED}..."
    HP_RUN_DIR="${SEED_DIR}" python run_Q_model.py \
        --dgp thalmann \
        --latent True \
        --seed ${SEED} \
        --lmbd ${LMBD} \
        --z ${Z} \
        --hidden ${H} \
        --enc_hidden ${EH} \
        --epochs 2000 \
        --step1_epochs 2000 \
        --task_emb_dim 4 \
        --same_enc_dec False \
        --unif_weight ${UW} \
        --run_suffix hp_v2_${COMBO}
done

echo "Combo ${COMBO} done."
