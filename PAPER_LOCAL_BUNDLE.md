# Paper-writing bundle — run the publication notebooks off-cluster

Branch `paper-local-access` contains everything needed to view the final figures and
re-run the figure-generating analysis cells locally, without cluster access.

```bash
git clone -b paper-local-access <repo-url> RNNsandUncertainty
cd RNNsandUncertainty
jupyter lab   # open the three notebooks below
```

## The three final figures

| Figure (in `final_plots/`) | Notebook | Generating cell |
|---|---|---|
| `thalmann_combined_figure.png/.pdf/.svg` | `thalmann_results.ipynb` | cell 88 (code-cell index; cell 87 is the older variant) |
| `synthetic_publication_panel.png/.pdf/.svg` | `synthetic_publication_panels.ipynb` | cell 25 — **note:** the saved notebook writes `final_plots/synthetic_poster_panel.*`; the `synthetic_publication_panel.*` files on disk were produced by a run of this cell with the output name edited (cell 24 is the older bbox-tight variant) |
| `dezfouli_combined_journal_half.png/.pdf/.svg` | `dezfouli_publication_panels.ipynb` | cell 36 |

All already-rendered outputs (`.png/.pdf/.svg`) are committed under `final_plots/`, so the
figures are viewable without running anything.

## What each notebook reads (all included on this branch)

**`thalmann_results.ipynb`**
- `final_plots/thalmann_z3_3task_full/` and `final_plots/thalmann_z3_s2_3task_full/` — canonical decoding CSVs/NPYs (`DEC`/`FULL` in the code)
- `final_plots/thalmann_z3_full/`, `thalmann_z3_1task_full/`, `thalmann_z3/` — fallback subsets
- `final_plots/thalmann_z3_ds_t{0,1,2,01,02,12,012}/` — data-scaling subsets
- `final_plots/thalmann_z3_datascaling*/`, `thalmann_z3_s2_full/`, `thalmann_z3_s2_pooled/`
- `data_thalmann_3task_full/{c_train.npy, task_ids_per_block.npy}`
- `data_sub_t012_full/{c_train.npy, task_ids_per_block.npy}`
- `data_thalmann_s2/{df_all.csv, task_ids_per_block.npy, fold{0,1,2}/c_test.npy}`
- `data/final2armedBanditSession1.csv` (raw; `finalRestlessSession1.csv` also included)

**`synthetic_publication_panels.ipynb`** — the final panel (cell 25) reads only light files:
- `cog_nll_perfold_synthetic.csv`, `cp_rnn_nll_nestedcv.csv`, `env_decoding_canonical.csv` (repo root)
- `final_plots/synthetic/nested_cv_z1/dataset{D}/nested_cv_summary.json` for D in {0,1,2,3,5,7,10,12,15,17}
- `final_plots/synthetic_z1/dataset0/canonical/idrnn/latents_train.npy`
- `data_dataset{D}/{true_parameter_values.csv, model_eval_dfvanilla.csv, fold{0,1,2}/true_param_test.csv}`
- `plots_dataset0/step1_three_regressions.npz`
- Cells 18–22 are heavy seed-walking cells that read `runs_dataset*` checkpoints and
  per-seed `latents_tensor*.pt` — those inputs are **not** in the bundle (cluster-only).
  The final panel does not need them.

**`dezfouli_publication_panels.ipynb`**
- `final_plots/dezfouli_z{1,2,3}_full/` — `per_participant_nll.csv`, `nested_cv_summary.json`,
  `three_regressions_dezfouli.npz`, `canonical/` latents, `publication_figure/`
  (incl. `datascaling.csv`, bootstrap NPZs)

Local modules imported by the notebooks: `nature_plot_style.py`, `panelb_svm_decoding.py`
(both at repo root, included along with all analysis scripts `analyze_*.py`, `compute_*.py`,
`build_*.py`, `nested_cv/`, etc.).

## Environment

Cluster env is conda `RNNproject` (Python 3.9.19). Versions the figures were made with:

```
numpy 1.26.4, pandas 2.3.3, matplotlib 3.9.4, scipy 1.13.1,
scikit-learn 1.6.1, torch 2.6.0 (CPU is fine), pingouin 0.5.5, jupyter
```

`pingouin` supplies `bayesfactor_ttest` / `bayesfactor_pearson`; the notebooks fall back
gracefully in places but the BF annotations need it. No GPU is required for any
figure-generating cell.

Notebook style notes (already encoded in the setup cells):
- First cells cap BLAS threads at 1 (cluster CPU watchdog) — harmless locally.
- Journal figures are saved **without** `bbox_inches="tight"` so the 180 mm figsize stays
  authoritative (NHB max width); fonts Nimbus Sans → falls back to DejaVu Sans if absent.
