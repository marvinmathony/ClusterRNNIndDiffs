# SBATCH Quick Reference

## TL;DR - Just Do This

1. **Edit configuration** in `submit_pipeline.sh` (lines with `LMBD` and `Z_DIM`)
2. **Submit**: `./submit_pipeline.sh`
3. **Monitor**: `watch -n 2 squeue -u $USER`
4. **Check progress**: `./check_progress.sh`

That's it! The script handles everything with proper dependencies.

---

## Available Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `submit_pipeline.sh` | **Recommended**: Full pipeline with dependencies | `./submit_pipeline.sh` |
| `run_full_pipeline.sbatch` | Single job running everything | `sbatch run_full_pipeline.sbatch` |
| `run_dataset_parallel.sbatch` | One job per dataset (5 parallel) | `sbatch run_dataset_parallel.sbatch` |
| `run_training_parallel.sbatch` | Maximum parallelism (50 jobs) | `sbatch run_training_parallel.sbatch` |
| `check_progress.sh` | Check what's completed | `./check_progress.sh` |

---

## Common Commands

```bash
# Submit the pipeline
./submit_pipeline.sh

# Check your jobs
squeue -u $USER

# Check progress
./check_progress.sh

# Cancel all your jobs
scancel -u $USER

# Watch logs in real-time
tail -f logs/train_0_*.out

# Check disk usage
du -sh data_dataset* runs* plots*
```

---

## Customization

### Change Lambda or Z dimension

**Option 1** (Easy): Edit `submit_pipeline.sh`
```bash
LMBD=0.3    # Change this line
Z_DIM=2     # Change this line
```

**Option 2** (Quick): Modify and run directly
```bash
# For lmbd=0.3, z=2
for dataset in {0..4}; do
    for seed in 12 50 76 100 142; do
        sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=train_d${dataset}_s${seed}
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
python run_Q_model.py --seed $seed --latent True --dataset_id $dataset --lmbd 0.3 --z 2
EOF
    done
done
```

### Run fewer datasets

Edit array size in scripts:
```bash
# In submit_pipeline.sh or run_dataset_parallel.sbatch
#SBATCH --array=0-2  # Only 3 datasets instead of 5
```

### Request different resources

Edit SBATCH directives:
```bash
#SBATCH --time=24:00:00    # More time
#SBATCH --mem=32G          # More memory
#SBATCH --gres=gpu:2       # More GPUs
#SBATCH --cpus-per-task=8  # More CPUs
```

---

## Before First Run

1. **Test your environment**:
   ```bash
   # Request interactive node
   srun --pty --gres=gpu:1 --mem=16G bash

   # Load your environment
   # source activate myenv  # or whatever you use

   # Test one command
   python data_generation.py --dataset_id 0
   ```

2. **Edit scripts** to uncomment/adjust:
   - Module loading (e.g., `module load python/3.9`)
   - Environment activation (e.g., `source activate myenv`)
   - Partition names (e.g., `#SBATCH --partition=gpu`)

3. **Create logs directory**:
   ```bash
   mkdir -p logs
   ```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Jobs stay pending | Check partition exists: `sinfo` |
| "python not found" | Uncomment environment activation in scripts |
| Out of memory | Increase `--mem=` in SBATCH directives |
| GPU not found | Check `--partition=` and `--gres=gpu:1` |
| Jobs fail quickly | Check logs: `cat logs/*.err` |

---

## Job Dependencies in submit_pipeline.sh

The script automatically chains jobs:

```
Data Gen (5 jobs)
    ↓ (waits for completion)
Training (50 jobs)
    ↓ (waits for completion)
Epoch Selection (10 jobs)
    ↓ (waits for completion)
Testing (10 jobs)
    ↓ (waits for completion)
Analysis (1 job)
```

This ensures no resources are wasted on jobs that will fail due to missing inputs.

---

## Output Files

After completion:
- **Main results**: `plots/multi_dataset/*.png`
- **Summary stats**: `plots/multi_dataset/summary_statistics.txt`
- **Logs**: `logs/*.out` and `logs/*.err`
- **Datasets**: `data_dataset0/` through `data_dataset4/`
- **Checkpoints**: `runs_dataset*/` and `runs_vanilla_dataset*/`

---

## Estimated Time & Cost

For 5 datasets with 5 seeds each:

| Step | Jobs | Time/Job | Total GPU-hours |
|------|------|----------|-----------------|
| Data Gen | 5 | 1h | 0 |
| Training | 50 | 10h | 500 |
| Selection | 10 | 0.5h | 0 |
| Testing | 10 | 3h | 30 |
| Analysis | 1 | 0.5h | 0 |
| **Total** | **76** | - | **~530** |

With parallelization: ~12-15 hours wall time (if enough GPUs available)

Without parallelization: ~500+ hours wall time

---

## Examples

### Run full pipeline with lmbd=0.3
```bash
# Edit submit_pipeline.sh to set LMBD=0.3
./submit_pipeline.sh
```

### Run just one dataset manually
```bash
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=dataset0
#SBATCH --output=logs/dataset0_%j.out
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

DATASET_ID=0
LMBD=0.5
SEEDS=(12 50 76 100 142)

python data_generation.py --dataset_id \$DATASET_ID

for seed in "\${SEEDS[@]}"; do
    python run_Q_model.py --seed \$seed --latent True --dataset_id \$DATASET_ID --lmbd \$LMBD --z 1
    python run_Q_model.py --seed \$seed --latent False --dataset_id \$DATASET_ID
done

python select_best_epoch_by_rsa.py --latent True --dataset_id \$DATASET_ID
python select_best_epoch_by_rsa.py --latent False --dataset_id \$DATASET_ID

python testing_script.py --latent True --dataset_id \$DATASET_ID --model_fitting True
python testing_script.py --latent False --dataset_id \$DATASET_ID --model_fitting True
EOF
```

### Restart failed training for specific dataset/seed
```bash
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=retry
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
python run_Q_model.py --seed 76 --latent True --dataset_id 2 --lmbd 0.5 --z 1
EOF
```

---

## Quick Checks

```bash
# How many jobs running?
squeue -u $USER | wc -l

# Check specific job
squeue -j <job_id>

# Is data generated?
ls data_dataset*/df_train.csv

# How many seeds trained?
ls runs_dataset0/seed_*/checkpoints

# Are results ready?
ls plots/multi_dataset/*.png

# Full progress check
./check_progress.sh
```

---

## Getting Help

- Full documentation: `SBATCH_USAGE.md`
- Pipeline overview: `README_MULTI_DATASET.md`
- Quick start: `QUICK_START.md`
- Check cluster docs: `man sbatch` or your cluster's documentation
