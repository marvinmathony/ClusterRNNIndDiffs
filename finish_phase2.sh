#!/bin/bash
# Finish Phase-2: the overnight run's 8h HP-wait expired before t02-van/t12/t012
# completed (queue saturation).  This waits out the remaining HP searches, re-picks
# all winners, retrains anything missing (driver skips existing configs), waits for
# all retrains, then runs the data-scaling analyzer.
set -u
cd /ictstr01/home/hcai/marvin.mathony/RNNsandUncertainty
source /ictstr01/home/hcai/marvin.mathony/tools/apps/mamba/etc/profile.d/conda.sh
conda activate RNNproject
SUBS="t0 t1 t2 t01 t02 t12 t012"

wait_clear () {  # $1 = job-name regex, $2 = max minutes
  for i in $(seq 1 "$2"); do
    [ "$(squeue -u "$USER" -h -o '%j' | grep -cE "$1")" -eq 0 ] && return 0
    sleep 60
  done
}

echo "[$(date +%H:%M)] waiting for remaining HP searches (hp_)..."
wait_clear '^hp_' 600
echo "[$(date +%H:%M)] picking winners (all subsets, now complete)..."
for sub in $SUBS; do echo "--- $sub ---"; python3 pooled_nested_nll.py pick --tag $sub --arch both; done

echo "[$(date +%H:%M)] waiting for first-wave retrains (rt_) to finish before filling gaps..."
wait_clear '^rt_' 600
echo "[$(date +%H:%M)] retraining any missing (skips existing configs)..."
for sub in $SUBS; do
  python3 pooled_nested_nll.py retrain --data_root data_sub_$sub --tag $sub \
      --block_weight_mode uniform --arch both 2>&1 | grep -E "submitted"
done

echo "[$(date +%H:%M)] waiting for all retrains (rt_) to finish..."
wait_clear '^rt_' 600
echo "[$(date +%H:%M)] running data-scaling analyzer..."
export OMP_NUM_THREADS=4
python3 analyze_datascaling_nll.py 2>&1 | grep -E "n=|Saved"
echo "[$(date +%H:%M)] FINISH_PHASE2_DONE"
