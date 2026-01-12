# Quick Start: Multi-Dataset Pipeline

## TL;DR - Just Run This

```bash
# Run everything (generates 5 datasets, trains all models, analyzes results)
python run_multi_dataset_pipeline.py
```

**Time estimate**: ~5× your single-dataset pipeline time

**Output**: Results in `plots/multi_dataset/`

## What You Get

Four key plots in `plots/multi_dataset/`:

1. **aggregated_rsa_correlation.png** - Main RSA result with error bars
2. **aggregated_model_likelihoods.png** - Main model comparison with error bars
3. **per_dataset_rsa.png** - Shows consistency across datasets
4. **per_dataset_likelihoods.png** - Shows consistency across datasets

Plus: `summary_statistics.txt` with all numbers

## Common Commands

```bash
# Full pipeline
python run_multi_dataset_pipeline.py

# Only run analysis (if data already exists)
python run_multi_dataset_pipeline.py --only_analysis

# Skip data generation (if already generated)
python run_multi_dataset_pipeline.py --skip_data_generation

# Use different number of datasets
python run_multi_dataset_pipeline.py --n_datasets 10
```

## Alternative: Shell Script

```bash
# Make executable (first time only)
chmod +x run_pipeline.sh

# Run everything
./run_pipeline.sh full

# Only analysis
./run_pipeline.sh analysis

# Generate all datasets
./run_pipeline.sh data

# Train specific dataset
./run_pipeline.sh train 0

# Select best epochs for dataset
./run_pipeline.sh select 0

# Test specific dataset
./run_pipeline.sh test 0

# Help
./run_pipeline.sh help
```

## Step-by-Step (Manual Control)

```bash
# 1. Generate 5 datasets
for i in {0..4}; do python data_generation.py --dataset_id $i; done

# 2. Train latent models (all seeds, all datasets)
for dataset in {0..4}; do
  for seed in 12 50 76 100 142; do
    python run_Q_model.py --seed $seed --latent True --dataset_id $dataset --lmbd 1.0 --z 1
  done
done

# 3. Train vanilla models (all seeds, all datasets)
for dataset in {0..4}; do
  for seed in 12 50 76 100 142; do
    python run_Q_model.py --seed $seed --latent False --dataset_id $dataset
  done
done

# 4. Select best epochs
for i in {0..4}; do
  python select_best_epoch_by_rsa.py --latent True --dataset_id $i
  python select_best_epoch_by_rsa.py --latent False --dataset_id $i
done

# 5. Test models
for i in {0..4}; do
  python testing_script.py --latent True --dataset_id $i --model_fitting True
  python testing_script.py --latent False --dataset_id $i --model_fitting True
done

# 6. Aggregate and plot
python analyze_synthetic_multi_dataset.py
```

## Troubleshooting

**Pipeline fails partway through?**
```bash
# Continue from where it stopped
python run_multi_dataset_pipeline.py --skip_data_generation --skip_training
```

**Need to re-run one specific part?**
```bash
# Example: Re-train dataset 2, seed 76 (latent)
python run_Q_model.py --seed 76 --latent True --dataset_id 2 --lmbd 1.0 --z 1

# Then continue pipeline from selection
python run_multi_dataset_pipeline.py --skip_data_generation --skip_training
```

**Just want new plots?**
```bash
python analyze_synthetic_multi_dataset.py
```

## File Locations

**Your results are here:**
- Main plots: `plots/multi_dataset/`
- Dataset 0 data: `data_dataset0/`
- Dataset 0 latent models: `runs_dataset0/`
- Dataset 0 vanilla models: `runs_vanilla_dataset0/`
- (same pattern for datasets 1-4)

## Configuration

Edit these in `run_multi_dataset_pipeline.py`:
```python
N_DATASETS = 5  # Number of datasets to generate
SEEDS = [12, 50, 76, 100, 142]  # Random seeds per dataset
```

## More Information

- **Full documentation**: `README_MULTI_DATASET.md`
- **Summary of changes**: `MULTI_DATASET_SUMMARY.md`
- **Shell script help**: `./run_pipeline.sh help`

## That's It!

```bash
python run_multi_dataset_pipeline.py
```

Then check `plots/multi_dataset/` for your publication-ready figures! 🎉
