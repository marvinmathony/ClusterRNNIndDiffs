#!/bin/bash
# Overnight Phase-2 orchestration: gate on smoke -> launch all 7 task-subset
# per-fold HP searches (uniform block weights) -> pick winners -> launch retrains.
set -u
cd /ictstr01/home/hcai/marvin.mathony/RNNsandUncertainty
source /ictstr01/home/hcai/marvin.mathony/tools/apps/mamba/etc/profile.d/conda.sh
conda activate RNNproject
SUBS="t0 t1 t2 t01 t02 t12 t012"

# 1. wait for smoke jobs
echo "[$(date +%H:%M)] waiting for smoke jobs..."
for i in $(seq 1 40); do
  [ "$(squeue -u "$USER" -h -o '%j' | grep -c '^smk_')" -eq 0 ] && break
  sleep 60
done

# 2. gate: both smoke configs must have a finite cv_val_loss
gate=$(python3 - <<'PY'
import json, math
ok=True
for s in ("t1","t2"):
    try:
        v=json.load(open(f"smoke_sub/{s}/config.json")).get("cv_val_loss")
        if v is None or not math.isfinite(float(v)): ok=False; print(f"  smoke {s}: BAD cv_val_loss={v}")
        else: print(f"  smoke {s}: cv_val_loss={v:.4f} OK")
    except Exception as e: ok=False; print(f"  smoke {s}: {e}")
print("GATE_PASS" if ok else "GATE_FAIL")
PY
)
echo "$gate"
echo "$gate" | grep -q GATE_PASS || { echo "[ABORT] smoke gate failed — not launching Phase 2"; exit 1; }

# 3. launch all 7 subset HP searches (idrnn uniform weights; vanilla unaffected)
echo "[$(date +%H:%M)] launching HP searches..."
for sub in $SUBS; do
  python3 pooled_nested_nll.py search --data_root data_sub_$sub --tag $sub \
      --block_weight_mode uniform --arch both 2>&1 | grep -E "submitted"
done

# 4. wait for all HP-eval jobs to clear (up to 8h)
echo "[$(date +%H:%M)] waiting for HP searches (hp_*)..."
for i in $(seq 1 480); do
  [ "$(squeue -u "$USER" -h -o '%j' | grep -c '^hp_')" -eq 0 ] && break
  sleep 60
done

# 5. pick per-fold winners for all subsets
echo "[$(date +%H:%M)] picking winners..."
for sub in $SUBS; do
  echo "--- $sub ---"
  python3 pooled_nested_nll.py pick --tag $sub --arch both
done

# 6. launch retrains
echo "[$(date +%H:%M)] launching retrains..."
for sub in $SUBS; do
  python3 pooled_nested_nll.py retrain --data_root data_sub_$sub --tag $sub \
      --block_weight_mode uniform --arch both 2>&1 | grep -E "submitted"
done
echo "[$(date +%H:%M)] OVERNIGHT_PHASE2_RETRAIN_LAUNCHED"
