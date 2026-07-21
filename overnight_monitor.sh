#!/bin/bash
#SBATCH --job-name=thal_monitor
#SBATCH --output=logs/thal_monitor_%j.out
#SBATCH --error=logs/thal_monitor_%j.err
#SBATCH -p cpu_p
#SBATCH --qos cpu_normal
#SBATCH --time=24:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --nice=10000
# Monitors the overnight Thalmann z=3 training (stage_c per-fold + full-cohort
# retrain), then writes a morning status report. Does NOT run the downstream
# panels (those scripts are finalised by hand against a real checkpoint).

set -uo pipefail
cd /ictstr01/home/hcai/marvin.mathony/RNNsandUncertainty
source /ictstr01/home/hcai/marvin.mathony/tools/apps/mamba/etc/profile.d/conda.sh
conda activate RNNproject

STATUS=final_plots/thalmann_z3/OVERNIGHT_STATUS.log
mkdir -p final_plots/thalmann_z3
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$STATUS"; }

log "===== overnight monitor started ====="
log "watching: stage_c (nc_tha_*) = 180 jobs ; full-retrain (full_tha_*) = 30 jobs"

wait_gone(){  # $1 = job-name grep pattern; require absent for 2 consecutive checks
  local pat="$1" miss=0 n
  while :; do
    n=$(squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -c "$pat")
    if [ "${n:-1}" -eq 0 ]; then miss=$((miss+1)); else miss=0; fi
    [ "$miss" -ge 2 ] && break
    sleep 120
  done
}

# ── Branch 1: stage_c (panel a) ────────────────────────────────────────────────
wait_gone "nc_tha"
log "--- stage_c complete: per-fold seed success counts ---"
for arch in idrnn vanilla; do
  base="runs_thalmann_nested_cv_z3"
  [ "$arch" = vanilla ] && base="runs_vanilla_thalmann_nested_cv_z3"
  for f in 0 1 2; do
    ok=$(grep -l cv_val_loss "$base"/fold$f/seed_*/config.json 2>/dev/null | wc -l)
    log "    $arch fold$f: $ok/30 seeds with cv_val_loss   ($base/fold$f)"
  done
done

# ── Branch 2: full-cohort retrain (panels b/c/e) ───────────────────────────────
wait_gone "full_tha"
log "--- full-cohort retrain complete: step1_z_lookup presence ---"
for arch in idrnn vanilla; do
  d="final_plots/thalmann_z3_full/canonical/$arch/runs"
  ok=$(ls "$d"/seed_*/step1_z_lookup.npy 2>/dev/null | wc -l)
  log "    $arch: $ok/15 seeds have step1_z_lookup.npy   ($d)"
done

log "--- best full-retrain IDRNN seed by step1_specificity (preview for extraction) ---"
python3 - 2>&1 <<'PY' | tee -a "$STATUS"
import json, glob, os
for arch, key, hi in [("idrnn","step1_specificity",True), ("vanilla","cv_val_loss",False)]:
    d=f"final_plots/thalmann_z3_full/canonical/{arch}/runs"
    best=None
    for cfgp in glob.glob(os.path.join(d,"seed_*/config.json")):
        try: c=json.load(open(cfgp))
        except Exception: continue
        v=c.get(key)
        if v is None: continue
        seed=cfgp.split("seed_")[1].split("/")[0]
        if best is None or (v>best[1] if hi else v<best[1]): best=(seed,v)
    print(f"    {arch}: best seed={best[0] if best else None}  {key}={best[1] if best else None}")
PY

log "===== ALL OVERNIGHT TRAINING COMPLETE — ready for extraction + panels ====="
