#!/bin/bash
# When the marginalized-eval jobs (mg_*) finish: aggregate -> regenerate panel-a +
# per-seed PNGs -> refresh notebook cells (embed) -> update panel-a markdown.
set -u
cd /ictstr01/home/hcai/marvin.mathony/RNNsandUncertainty
source /ictstr01/home/hcai/marvin.mathony/tools/apps/mamba/etc/profile.d/conda.sh
conda activate RNNproject
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLBACKEND=Agg

echo "[$(date +%H:%M)] waiting for marginalized-eval jobs (mg_*)..."
for i in $(seq 1 240); do
  [ "$(squeue -u "$USER" -h -o '%j' | grep -c '^mg_')" -eq 0 ] && break
  sleep 60
done
echo "[$(date +%H:%M)] CSVs present: $(ls runs_pooled/s2_2task/marginal_nll/fold*_seed*.csv 2>/dev/null | wc -l)/15"

python3 aggregate_marginal_nll.py || { echo "aggregate failed"; exit 1; }

# regenerate the embedded figures from the marginalized CSVs
python3 -c "import build_panela_cell as A; ns={}; exec(A.SETUP,ns); exec(A.PANELA,ns)"
python3 -c "import build_panela_cell as A, build_perseed_cell as P; ns={}; exec(A.SETUP,ns); exec(P.CELL,ns)"

# refresh the notebook cells (idempotent; re-embeds the new PNGs)
python3 build_panela_cell.py
python3 build_perseed_cell.py

# update the panel-a markdown header to the marginalized story
python3 - <<'PY'
import nbformat
nb=nbformat.read("thalmann_results.ipynb", as_version=4)
for c in nb.cells:
    if c.cell_type=="markdown" and "Panel a — held-out NLL per participant" in "".join(c.source):
        c.source=("## Panel a — held-out NLL per participant\n"
            "Per-fold nested-CV, seed-averaged. Held-out NLL uses the **marginalized + causal** "
            "likelihood (`compute_rnn_likelihoods_torch`: z inferred causally from data up to trial t "
            "and integrated over q(z|x)) — matching the synthetic pipeline, **not** the `model(xin,xin)` "
            "point estimate (which understated the latent). Paired Bayes factors over held-out "
            "participants; CP-RNN (z=0) vs IDRNN tests the held-out latent benefit; RNNs ≫ cog models.")
        break
nbformat.write(nb,"thalmann_results.ipynb"); print("panel-a MD updated")
PY
echo "[$(date +%H:%M)] FINISH_MARGINAL_PANELA_DONE"
