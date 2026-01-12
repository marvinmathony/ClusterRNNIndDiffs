# SLURM/SBATCH Usage Guide

## Quick Start

### Option 1: Automatic Pipeline with Dependencies (Recommended)
```bash
# Edit configuration in submit_pipeline.sh first (LMBD, Z_DIM)
./submit_pipeline.sh
```

This submits all jobs with proper dependencies. Steps run automatically in order:
1. Data generation (5 parallel jobs)
2. Training (50 parallel jobs)
3. Epoch selection (10 parallel jobs)
4. Testing (10 parallel jobs)
5. Analysis (1 job)

### Option 2: Full Pipeline in Single Job
```bash
# Edit parameters in run_full_pipeline.sbatch first
sbatch run_full_pipeline.sbatch
```

### Option 3: Parallel by Dataset
```bash
# Runs 5 jobs, one per dataset
sbatch run_dataset_parallel.sbatch
```

### Option 4: Maximum Parallelization
```bash
# First generate data
sbatch --array=0-4 <<EOF
#!/bin/bash
#SBATCH --job-name=gen_data
#SBATCH --output=logs/gen_%a.out
#SBATCH --time=02:00:00
#SBATCH --mem=8G
python data_generation.py --dataset_id \$SLURM_ARRAY_TASK_ID
EOF

# Then submit 50 parallel training jobs
sbatch run_training_parallel.sbatch
```

## Configuration

### Before Submitting, Edit These Lines in Scripts:

**In all .sbatch files:**
```bash
# Adjust these based on your cluster setup:
#SBATCH --partition=gpu          # Your GPU partition name
#SBATCH --time=12:00:00          # Wall time
#SBATCH --mem=16G                # Memory
#SBATCH --gres=gpu:1             # GPU request

# Module loading (uncomment and adjust):
# module load python/3.9
# module load cuda/11.8

# Environment activation (uncomment and adjust):
# source /path/to/your/venv/bin/activate
# or: conda activate your_env
```

**In submit_pipeline.sh and other scripts:**
```bash
# Set your hyperparameters:
LMBD=0.5        # Lambda weight
Z_DIM=1         # Latent dimension
```

## Monitoring Jobs

```bash
# Check all your jobs
squeue -u $USER

# Check specific job
squeue -j <job_id>

# Watch job queue (updates every 2 seconds)
watch -n 2 squeue -u $USER

# Check job details
scontrol show job <job_id>

# Check completed jobs
sacct -u $USER --format=JobID,JobName,State,ExitCode,Elapsed

# Check specific job output
tail -f logs/train_0_<job_id>.out
```

## Canceling Jobs

```bash
# Cancel specific job
scancel <job_id>

# Cancel all your jobs
scancel -u $USER

# Cancel specific job array
scancel <job_id>

# Cancel specific tasks in array
scancel <job_id>_[0-5]  # Cancel tasks 0-5
```

## Resource Recommendations

### Data Generation
- Time: 1-2 hours per dataset
- Memory: 8GB
- CPUs: 1
- GPU: Not needed

### Training (per seed/dataset)
- Time: 8-12 hours
- Memory: 16GB
- CPUs: 2-4
- GPU: 1 (required)

### Epoch Selection
- Time: 30 minutes - 1 hour
- Memory: 8GB
- CPUs: 1
- GPU: Not needed

### Testing
- Time: 2-4 hours
- Memory: 16GB
- CPUs: 2
- GPU: 1 (required for model evaluation)

### Analysis
- Time: 30 minutes - 1 hour
- Memory: 16GB
- CPUs: 2
- GPU: Not needed

## Job Array Layouts

### run_dataset_parallel.sbatch
```
Array ID 0-4: One job per dataset
```

### run_training_parallel.sbatch
```
Array ID 0-49: 5 datasets × 5 seeds × 2 models
Layout: [D0S0L, D0S0V, D0S1L, D0S1V, ..., D4S4L, D4S4V]
where D=dataset, S=seed, L=latent, V=vanilla
```

### submit_pipeline.sh arrays
- Data generation: Array 0-4 (5 datasets)
- Training: Array 0-49 (50 training runs)
- Selection: Array 0-9 (5 datasets × 2 models)
- Testing: Array 0-9 (5 datasets × 2 models)

## Customization Examples

### Run with different lambda
Edit in script or override:
```bash
# Option 1: Edit LMBD in submit_pipeline.sh
LMBD=0.3

# Option 2: Manual submission
sbatch --export=ALL,LMBD=0.3 run_full_pipeline.sbatch
```

### Run with 10 datasets instead of 5
Edit submit_pipeline.sh:
```bash
# Change array sizes:
--array=0-9        # for data generation (was 0-4)
--array=0-99       # for training (was 0-49)
--array=0-19       # for selection and testing (was 0-9)

# Update N_DATASETS calculation
N_DATASETS=10
```

### Run only specific datasets
```bash
# Only datasets 0, 2, and 4
sbatch --array=0,2,4 run_dataset_parallel.sbatch
```

### Request more resources for large models
```bash
sbatch --mem=32G --gres=gpu:2 run_full_pipeline.sbatch
```

## Output Files

Logs are saved in `logs/` directory:
- `gen_data_<array>_<jobid>.out` - Data generation
- `train_<array>_<jobid>.out` - Training
- `select_<array>_<jobid>.out` - Epoch selection
- `test_<array>_<jobid>.out` - Testing
- `analyze_<jobid>.out` - Analysis

Error messages go to `.err` files with same naming.

## Troubleshooting

### "command not found: python"
Make sure to uncomment and adjust the environment activation lines:
```bash
# source /path/to/your/venv/bin/activate
# or: conda activate your_env
```

### "Out of memory"
Increase memory request:
```bash
#SBATCH --mem=32G  # or higher
```

### Jobs pending forever
Check your partition and time limits match cluster policies:
```bash
sinfo  # See available partitions and time limits
```

### GPU not found
Make sure GPU partition and request are correct:
```bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
```

### Jobs fail during training
Check the logs in `logs/` directory. Common issues:
- Missing dependencies (check environment activation)
- Data not generated yet (use job dependencies)
- Insufficient resources (increase --mem or --time)

## Best Practices

1. **Test first**: Run one job manually before submitting arrays
   ```bash
   python data_generation.py --dataset_id 0
   python run_Q_model.py --seed 12 --latent True --dataset_id 0 --lmbd 0.5
   ```

2. **Start small**: Test with 1 dataset before running all 5
   ```bash
   sbatch --array=0 run_dataset_parallel.sbatch
   ```

3. **Monitor disk space**: Pipeline generates lots of data
   ```bash
   du -sh data_dataset* runs*
   ```

4. **Use dependencies**: Prevents wasted resources from failed jobs
   ```bash
   ./submit_pipeline.sh  # Automatically sets up dependencies
   ```

5. **Keep logs organized**: Create dated subdirectories
   ```bash
   mkdir -p logs/$(date +%Y%m%d)
   # Then update --output paths in scripts
   ```

## Interactive Testing

Before batch submission, test interactively:
```bash
# Request interactive GPU node
srun --pty --gres=gpu:1 --mem=16G --time=2:00:00 bash

# Load environment
# source activate your_env

# Test commands
python data_generation.py --dataset_id 0
python run_Q_model.py --seed 12 --latent True --dataset_id 0 --lmbd 0.5

# Exit when done
exit
```

## Restarting Failed Jobs

If some jobs fail, restart just those:
```bash
# Example: Training failed for dataset 2, seed 76
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=retry_train
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
python run_Q_model.py --seed 76 --latent True --dataset_id 2 --lmbd 0.5 --z 1
EOF
```

## Summary

**Easiest approach:**
```bash
./submit_pipeline.sh
```

**Most control:**
Manually submit jobs with dependencies using `sbatch --dependency=afterok:<job_id>`

**Monitor:**
```bash
watch -n 2 squeue -u $USER
```

**Results:**
Check `plots/multi_dataset/` when all jobs complete
