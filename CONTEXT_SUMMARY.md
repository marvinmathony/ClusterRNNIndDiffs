# Context Summary: Unsupervised Model Selection for IDRNN

## Project Overview

This project compares two RNN models for learning individual differences in reinforcement learning behavior:
- **IDRNN**: A latent model that learns individual-specific parameters
- **Vanilla RNN**: A baseline without explicit individual difference modeling

The goal is to select the best checkpoint (epoch + seed) for each model using **unsupervised criteria** (no ground truth access), to enable generalization to real human data.

## Problem Statement

When selecting checkpoints:
- **With oracle (ground truth) selection**: IDRNN beats vanilla in 3/4 datasets (by 0.03-0.06 correlation)
- **With unsupervised selection**: The current metrics often favor vanilla because it's more "consistent" across seeds

The challenge: Find unsupervised metrics that reliably identify good IDRNN checkpoints.

## Key Files Modified

### 1. `run_dataset_parallel.sbatch`
- Added `set -euo pipefail` to catch training failures
- Added timestamps to training logs
- Fixed silent failure bug where seed 12 vanilla training would fail without stopping the pipeline

### 2. `run_Q_model.py`
- Added defensive `wandb.finish()` before `wandb.init()` to prevent stale wandb state

### 3. `analyze_synthetic_multi_dataset.py`
- Fixed plotting bug where x-tick labels didn't match data when datasets were missing
- Now tracks `valid_rsa_datasets` and `valid_ll_datasets` separately

### 4. `select_best_epoch_unsupervised.py` (NEW)
The main selection script with unsupervised metrics.

## Empirical Findings on Metrics

Correlation of metrics with ground truth RSA correlation:

| Metric | Correlation with GT |
|--------|---------------------|
| Distance Entropy | **-0.67** (low = better) |
| Coefficient of Variation (CV) | **+0.52** (high = better) |
| Behavioral Alignment | ~0.09 (weak signal) |
| Cross-seed Consistency | Useful for filtering outliers, not primary selection |

## Current Selection Algorithm (`select_best_epoch_unsupervised.py`)

```python
# Weighted composite score
composite = (
    0.30 * scores['entropy'] +               # KEY: low entropy = structured
    0.20 * scores['cv'] +                    # KEY: high CV = discriminative
    0.15 * scores['behavioral_alignment'] +  # Behavior alignment
    0.10 * scores['pearson_consistency'] +   # Cross-seed reproducibility
    0.10 * scores['stability'] +             # Temporal stability
    0.05 * scores['spearman_consistency'] +
    0.05 * scores['smoothness'] +
    0.05 * scores['loss']
)
```

### Seed Selection Strategy
1. Filter outlier seeds using centrality threshold (median - 1 std, min 0.3)
2. Among good seeds, select based on behavioral alignment + centrality + loss

## Results Summary

### Best Achievable (Oracle Selection)
| Dataset | IDRNN Best | Vanilla Best | Difference |
|---------|------------|--------------|------------|
| 1 | 0.9427 | 0.9135 | +0.029 |
| 2 | 0.9505 | 0.9046 | +0.046 |
| 3 | 0.9590 | 0.9019 | +0.057 |
| 4 | 0.9177 | 0.9299 | -0.012 |

### With Current Unsupervised Selection (Dataset 1)
- IDRNN: 0.869 (selected epoch 6400, seed 76)
- Vanilla: 0.902 (selected epoch 9200, seed 100)
- Gap to oracle for IDRNN: 0.943 - 0.869 = 0.074

## Key Insights

1. **IDRNN has higher variance** across epochs/seeds than vanilla - it can reach better peaks but also worse valleys
2. **Vanilla is artificially more consistent** because it doesn't try to model individual differences
3. **Entropy is the strongest unsupervised predictor** of good geometry
4. **Behavioral alignment is weak** because the ceiling (behavior vs ground truth) is only ~0.09

## Remaining Challenges

1. Unsupervised metrics don't perfectly correlate with ground truth
2. The best epochs for IDRNN (high ground truth correlation) don't always have the lowest entropy
3. Need better ways to identify when IDRNN's individual difference learning has "locked in"

## NEW: Reconstruction Specificity Metric

### Concept
A new unsupervised metric based on the idea that if latent representations are input-specific:
- **Matched reconstruction**: Using participant i's latent to decode participant i's sequence should work well
- **Mismatched reconstruction**: Using participant j's latent to decode participant i's sequence should perform worse
- **Specificity** = mean(loss_mismatched) - mean(loss_matched)

Higher specificity means the model has learned more participant-specific representations.

### Implementation Files
- `compute_reconstruction_specificity.py` - Compute and analyze specificity over epochs
- `select_best_epoch_by_specificity.py` - Select best epoch using specificity

### Results: Correlation with Ground Truth RSA

**This metric shows MUCH stronger correlation with ground truth than previous metrics:**

| Dataset | Specificity-GT Correlation | p-value |
|---------|----------------------------|---------|
| 0 | **0.915** | 0.0002 |
| 1 | **0.839** | 0.0024 |
| 2 | **0.889** | 0.0006 |
| 3 | **0.917** | 0.0002 |
| 4 | **0.933** | 0.0001 |

Compare to previous metrics: Entropy had -0.67, CV had +0.52.

### Key Observations

1. **Specificity increases over training** - positive Spearman correlation (r~0.6-0.9) with epoch number
2. **Per-checkpoint correlation is ~0.77** (Pearson) when looking at individual epoch-seed combinations
3. **The metric is computed without ground truth** - uses only the model and test data

### Limitations

- Selecting the checkpoint with highest specificity doesn't always get the absolute best GT correlation
- There's variance across seeds that the metric captures imperfectly
- Example on Dataset 0: Selected epoch 7400/seed 50 (GT=0.58) vs oracle epoch 5800/seed 76 (GT=0.90)

### Example Usage

```bash
# Analyze specificity over epochs
python compute_reconstruction_specificity.py --dataset_id 0 --min_epoch 1000 --step 10 --latent True --analyze_correlation

# Select best epoch by specificity
python select_best_epoch_by_specificity.py --dataset_id 0 --min_epoch 3000 --latent True
```

## Possible Next Steps

1. **Late-stage stability analysis**: Epochs that remain stable longer might be better
2. **Cross-validation on behavioral prediction**: Test if latents predict held-out behavior
3. **Ensemble selection**: Select multiple good epochs and average predictions
4. **Loss trajectory analysis**: Look at rate of loss improvement, not just absolute loss
5. **Per-seed selection**: Instead of one epoch for all seeds, select best epoch per seed
6. **Combine specificity with other metrics**: Use specificity as primary filter, then apply entropy/CV

## Directory Structure

```
runs_dataset{0-4}/           # IDRNN runs
  seed_{12,50,76,100,142}/
    rsa/epoch_XXXX.npy       # RSA vectors (19900-dim, upper triangle of 200x200)
    loss/epoch_XXXX.npy      # Training loss
    checkpoints/             # Model weights

runs_vanilla_dataset{0-4}/   # Vanilla runs (same structure)

data_dataset{0-4}/
  c_test.npy                 # Choice data (200 participants x 200 trials)
  true_test_parameter_values.csv  # Ground truth parameters
```

## Commands to Run

```bash
# Run unsupervised selection
python select_best_epoch_unsupervised.py --latent True --dataset_id 1 --min_epoch 3000
python select_best_epoch_unsupervised.py --latent False --dataset_id 1 --min_epoch 3000

# Run full pipeline (on SLURM)
sbatch run_dataset_parallel.sbatch

# Analyze results across datasets
python analyze_synthetic_multi_dataset.py
```

## Key Functions in `select_best_epoch_unsupervised.py`

- `compute_distance_entropy(rsa_vector)`: Lower = more structured
- `compute_distance_variance_ratio(rsa_vector)`: Higher CV = more discriminative
- `compute_behavioral_alignment(rsa_vector, behavioral_rsa)`: Spearman correlation
- `compute_behavioral_rsa()`: Pairwise choice pattern similarity
- `mean_pairwise_corr(vectors)`: Cross-seed consistency
- `compute_temporal_stability(...)`: Correlation with nearby epochs
