#!/bin/bash
#SBATCH --job-name=van_rerun_mon
#SBATCH --output=logs/van_rerun_mon_%j.out
#SBATCH --error=logs/van_rerun_mon_%j.err
#SBATCH -p cpu_p
#SBATCH --qos cpu_normal
#SBATCH --time=12:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --nice=10000
set -uo pipefail
cd /ictstr01/home/hcai/marvin.mathony/RNNsandUncertainty
source /ictstr01/home/hcai/marvin.mathony/tools/apps/mamba/etc/profile.d/conda.sh
conda activate RNNproject
STATUS=final_plots/thalmann_z3/VANILLA_RERUN_STATUS.log
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$STATUS"; }
log "===== vanilla rerun monitor (epochs=15000) ====="

miss=0
while :; do
  n=$(squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -cE "nc_tha_van|full_tha_van")
  if [ "${n:-1}" -eq 0 ]; then miss=$((miss+1)); else miss=0; fi
  [ "$miss" -ge 2 ] && break
  sleep 120
done

log "vanilla reruns complete. checking success + cv_selected_epoch (ceiling=15000):"
python3 - 2>&1 <<'PY' | tee -a "$STATUS"
import json, glob
from collections import Counter
def report(name, pat, n_exp, ceil=15000):
    eps=[]; ok=0
    for c in glob.glob(pat):
        try: j=json.load(open(c))
        except Exception: continue
        if 'cv_val_loss' in j: ok+=1
        e=j.get('cv_selected_epoch')
        if e: eps.append(e)
    at=sum(1 for e in eps if e>=ceil-2)
    print(f"  {name}: {ok}/{n_exp} ok; cv_selected_epoch n={len(eps)} "
          f"at-ceiling(>= {ceil-2})={at}; min={min(eps) if eps else None} "
          f"max={max(eps) if eps else None} median={sorted(eps)[len(eps)//2] if eps else None}")
report("stage_c vanilla", "runs_vanilla_thalmann_nested_cv_z3/fold*/seed_*/config.json", 90)
report("full-retrain vanilla", "final_plots/thalmann_z3_full/canonical/vanilla/runs/seed_*/config.json", 15)
PY
log "===== done — if at-ceiling is high, budget still too small ====="
