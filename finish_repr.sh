#!/bin/bash
# Wait for the 7-subset full-cohort training, decode each (seed-averaged LOO R^2),
# regenerate the representational data-scaling plot, and refresh the notebook cell.
set -u
cd /ictstr01/home/hcai/marvin.mathony/RNNsandUncertainty
source /ictstr01/home/hcai/marvin.mathony/tools/apps/mamba/etc/profile.d/conda.sh
conda activate RNNproject
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

echo "[$(date +%H:%M)] waiting for full-cohort training (full_)..."
for i in $(seq 1 600); do
  [ "$(squeue -u "$USER" -h -o '%j' | grep -c '^full_')" -eq 0 ] && break
  sleep 60
done

echo "[$(date +%H:%M)] decoding each subset (seed-averaged LOO R^2)..."
for sub in t0 t1 t2 t01 t02 t12 t012; do
  ni=$(ls final_plots/thalmann_z3_ds_$sub/canonical/idrnn/runs/seed_*/step1_z_lookup.npy 2>/dev/null | wc -l)
  echo "--- $sub (idrnn z-done $ni/10) ---"
  THAL_FULL=final_plots/thalmann_z3_ds_$sub THAL_DATA=data_sub_${sub}_full N_SEEDS=10 \
    python3 seed_averaged_representation.py 2>&1 | grep -E "Saved|WM_|open" | tail -3
done

echo "[$(date +%H:%M)] regenerating plot + refreshing notebook cell..."
python3 datascaling_repr_plot.py 2>&1 | grep -vE "^\s*$"
python3 insert_repr_cell.py
echo "[$(date +%H:%M)] FINISH_REPR_DONE"
