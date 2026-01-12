# Multi-Dataset Pipeline Documentation

This document describes the enhanced pipeline that runs your entire experiment across multiple distinct datasets to demonstrate robust trends.

## Overview

The original pipeline ran with 5 random seeds on a single dataset. The enhanced pipeline now:
- Generates **5 distinct datasets** (each with different random seeds for data generation)
- Trains models with **5 random seeds per dataset** (total: 25 model runs per model type)
- Aggregates results across all datasets to show robust trends
- Produces the same plots, now with error bars showing consistency across datasets

## Pipeline Structure

### 1. Data Generation (`data_generation.py`)
- **New parameter**: `--dataset_id` (default: 0)
- Generates distinct datasets by varying the seed base: `seed_base = 1 + dataset_id * 1000`
- Saves data to: `data_dataset{dataset_id}/`
- Saves plots to: `plots_dataset{dataset_id}/`

### 2. Model Training (`run_Q_model.py`)
- **New parameter**: `--dataset_id` (default: 0)
- Trains models on specified dataset
- Saves checkpoints to: `runs_dataset{dataset_id}/` or `runs_vanilla_dataset{dataset_id}/`
- Saves RSA vectors and losses for epoch selection

### 3. Best Epoch Selection (`select_best_epoch_by_rsa.py`)
- **New parameters**:
  - `--dataset_id` (default: 0)
  - `--latent` (default: True)
- Selects best epoch based on cross-seed RSA reliability and loss
- Saves selection to: `{BASE_DIR}/best_epoch_by_rsa.json`

### 4. Model Testing (`testing_script.py`)
- **New parameters**:
  - `--dataset_id` (default: 0)
  - `--latent` (default: True)
  - `--model_fitting` (default: False)
- Tests models and fits cognitive baselines
- Saves results to: `data_dataset{dataset_id}/`

### 5. Aggregated Analysis (`analyze_synthetic_multi_dataset.py`)
- Collects results from all datasets
- Generates aggregated plots with error bars
- Produces per-dataset breakdown plots
- Saves to: `plots/multi_dataset/`

## Usage

### Quick Start: Run Everything

```bash
python run_multi_dataset_pipeline.py
```

This will execute the entire pipeline: data generation → training → selection → testing → analysis

### Run with Custom Number of Datasets

```bash
python run_multi_dataset_pipeline.py --n_datasets 10
```

### Skip Certain Steps

```bash
# Skip data generation (if already generated)
python run_multi_dataset_pipeline.py --skip_data_generation

# Skip training (if already trained)
python run_multi_dataset_pipeline.py --skip_training

# Only run the aggregated analysis
python run_multi_dataset_pipeline.py --only_analysis
```

### Run Individual Components

Generate a specific dataset:
```bash
python data_generation.py --dataset_id 0
```

Train models on a specific dataset:
```bash
# Latent model
python run_Q_model.py --seed 12 --latent True --dataset_id 0 --lmbd 1.0 --z 1

# Vanilla model
python run_Q_model.py --seed 12 --latent False --dataset_id 0
```

Select best epoch:
```bash
# For latent models
python select_best_epoch_by_rsa.py --latent True --dataset_id 0

# For vanilla models
python select_best_epoch_by_rsa.py --latent False --dataset_id 0
```

Test models:
```bash
# Test latent model
python testing_script.py --latent True --dataset_id 0 --model_fitting True

# Test vanilla model
python testing_script.py --latent False --dataset_id 0 --model_fitting True
```

Run aggregated analysis:
```bash
python analyze_synthetic_multi_dataset.py
```

## Output Structure

After running the full pipeline, your directory will contain:

```
RNNsandUncertainty/
├── data_dataset0/           # Data for dataset 0
├── data_dataset1/           # Data for dataset 1
├── ...
├── data_dataset4/           # Data for dataset 4
├── runs_dataset0/           # Latent model runs for dataset 0
│   ├── seed_12/
│   │   ├── checkpoints/
│   │   ├── rsa/
│   │   └── loss/
│   ├── seed_50/
│   ├── ...
│   └── best_epoch_by_rsa.json
├── runs_vanilla_dataset0/   # Vanilla model runs for dataset 0
├── ...
├── plots/
│   └── multi_dataset/       # Aggregated plots
│       ├── aggregated_rsa_correlation.png
│       ├── aggregated_model_likelihoods.png
│       ├── per_dataset_rsa.png
│       ├── per_dataset_likelihoods.png
│       └── summary_statistics.txt
└── plots_dataset0/          # Individual dataset plots
```

## Generated Plots

### Aggregated Plots (in `plots/multi_dataset/`)

1. **aggregated_rsa_correlation.png**
   - Bar plot comparing IDRNN vs Vanilla RNN
   - Error bars show SEM across datasets
   - Individual dataset points overlaid
   - Shows that IDRNN better captures ground truth geometry

2. **aggregated_model_likelihoods.png**
   - Bar plot comparing all models (Q, FQ, RNN variants)
   - Error bars show SEM across datasets
   - Individual dataset points overlaid
   - Statistical significance markers
   - Shows IDRNN outperforms vanilla and common process models

3. **per_dataset_rsa.png**
   - Shows RSA correlation for each dataset individually
   - Demonstrates consistency of IDRNN advantage across datasets

4. **per_dataset_likelihoods.png**
   - Shows RNN model likelihoods for each dataset
   - Demonstrates robust performance trends

### Summary Statistics

A text file (`summary_statistics.txt`) contains:
- Number of datasets processed
- Mean ± SEM for all metrics
- RSA correlations
- Model likelihoods

## Key Features

1. **Robustness**: Results averaged over 5 distinct datasets × 5 random seeds = 25 runs
2. **Backward Compatible**: Original scripts still work with default `dataset_id=0`
3. **Flexible**: Can run any subset of the pipeline
4. **Reproducible**: Each dataset uses a distinct, deterministic seed range
5. **Efficient**: Can skip completed steps to save time

## Computational Considerations

- Full pipeline with 5 datasets: ~25× the computation of single dataset
- Training is the most time-consuming step
- Consider using GPU for faster training
- Can parallelize across datasets if multiple GPUs available

## Configuration

Edit `run_multi_dataset_pipeline.py` to change:
- `N_DATASETS`: Number of datasets (default: 5)
- `SEEDS`: Random seeds per dataset (default: [12, 50, 76, 100, 142])

## Troubleshooting

If a step fails:
1. Check the error message for which dataset/seed failed
2. Re-run that specific step with the appropriate `--dataset_id` and `--seed`
3. Use `--skip_*` flags to continue from where you left off

Example:
```bash
# If training failed on dataset 2, seed 76, re-run:
python run_Q_model.py --seed 76 --latent True --dataset_id 2 --lmbd 1.0 --z 1

# Then continue the pipeline from selection:
python run_multi_dataset_pipeline.py --skip_data_generation --skip_training
```

## Citation

If you use this pipeline, please cite the original work and mention the multi-dataset extension for robustness testing.
