#!/bin/bash
#SBATCH --job-name=thal_hp_v3
#SBATCH --output=logs/thal_hp_v3_%A_%a.out
#SBATCH --error=logs/thal_hp_v3_%A_%a.err
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

# hp_v3: same grid as hp_v2 but with --continuous_encoder True.
# The encoder GRU runs as one continuous sequence across all 31 blocks
# (no hidden-state reset at block boundaries).
#
# Grid: 3 × 3 × 3 × 2 × 3 = 162 combos × 3 folds × 3 seeds = 1458 runs
# Each array task handles one combo (all 3 folds × 3 seeds sequentially).

source /ictstr01/home/hcai/marvin.mathony/tools/apps/mamba/etc/profile.d/conda.sh
conda activate RNNproject

WORKDIR=/ictstr01/home/hcai/marvin.mathony/RNNsandUncertainty
cd $WORKDIR

# ── HP grid ───────────────────────────────────────────────────────────────────
# unif_weight: [0.0, 0.1, 0.5]        (3)
# lmbd:        [0.05, 0.1, 0.2]       (3)
# enc_hidden:  [5, 10, 20]            (3)
# hidden:      [5, 10]                (2)
# z_dim:       [3, 5, 10]             (3)
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

# Format combo name
UW_TAG=$(printf "%02.0f" "$(echo "$UW * 100" | bc)" | sed 's/^0\([0-9]\)/0\1/')
# Simpler: map directly
case $UW in
    0.0) UW_TAG="00" ;;
    0.1) UW_TAG="01" ;;
    0.5) UW_TAG="05" ;;
esac
LMBD_TAG=$(echo $LMBD | sed 's/\.//g; s/^0/0/')
case $LMBD in
    0.05) LMBD_TAG="005" ;;
    0.1)  LMBD_TAG="01"  ;;
    0.2)  LMBD_TAG="02"  ;;
esac
COMBO="uw${UW_TAG}_lmbd${LMBD_TAG}_eh${EH}_h${H}_z${Z}"

echo "Array task $IDX: unif_weight=${UW} lmbd=${LMBD} enc_hidden=${EH} hidden=${H} z_dim=${Z}"
echo "Combo: ${COMBO}"

SEEDS=(42 123 456)
N_FOLDS=3
RUN_SUFFIX="hp_v3_${COMBO}"

for FOLD in $(seq 0 $((N_FOLDS - 1))); do
    for SEED in "${SEEDS[@]}"; do
        # Check if already done (config.json written at end of training)
        RUN_DIR="runs_thalmann_${RUN_SUFFIX}/fold${FOLD}/seed_${SEED}"
        if [ -f "${RUN_DIR}/config.json" ]; then
            echo "  fold=${FOLD} seed=${SEED}: already done, skipping."
            continue
        fi

        echo "  Running fold=${FOLD} seed=${SEED}..."
        python run_Q_model.py \
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
            --continuous_encoder True \
            --run_suffix ${RUN_SUFFIX}
    done
done

echo "Combo ${COMBO} done."
