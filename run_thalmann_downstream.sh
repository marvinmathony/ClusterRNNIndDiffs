#!/bin/bash
# Downstream for one canonical model.  Reads env: THAL_FULL, THAL_DATA, RUN_PANELE.
#   extract latents -> decode (b/c) -> latent correlations -> seed-averaged decoding
#   [-> panel-e (4-regression, corrected generators) -> publication figure]  if RUN_PANELE=1
set -uo pipefail
: "${THAL_FULL:?set THAL_FULL}" "${THAL_DATA:?set THAL_DATA}"
RUN_PANELE="${RUN_PANELE:-0}"; N_SEEDS="${N_SEEDS:-10}"
echo "===== downstream: THAL_FULL=$THAL_FULL  THAL_DATA=$THAL_DATA  panele=$RUN_PANELE ====="

export THAL_FULL THAL_DATA N_SEEDS
python extract_canonical_latents_thalmann.py
python decode_thalmann_canonical.py \
  --idrnn_pt   "$THAL_FULL/canonical/idrnn/latents_idrnn_canonical.pt" \
  --vanilla_pt "$THAL_FULL/canonical/vanilla/latents_vanilla_canonical.pt" \
  --out_dir    "$THAL_FULL/decoding"
python correlate_idrnn_latents.py
python seed_averaged_representation.py
if [ "$RUN_PANELE" = "1" ]; then
  python analyze_cross_task_regret_canonical.py
  python plot_thalmann_publication_figure.py
fi
echo "===== downstream done: $THAL_FULL ====="
