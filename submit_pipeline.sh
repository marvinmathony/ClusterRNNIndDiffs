#!/bin/bash
# Master submission script with job dependencies
# This ensures steps run in the correct order

mkdir -p logs

echo "Submitting multi-dataset pipeline with dependencies..."

# Configuration
LMBD=0.5
Z_DIM=1

# Step 1: Submit data generation jobs (parallel across datasets)
echo "Step 1: Submitting data generation jobs..."
JOB1=$(sbatch --parsable --array=0-4 <<EOF
#!/bin/bash
#SBATCH --job-name=gen_data_%a
#SBATCH --output=logs/gen_data_%a_%j.out
#SBATCH --error=logs/gen_data_%a_%j.err
#SBATCH --open-mode=append
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH -p gpu_p
#SBATCH --qos gpu_normal
#SBATCH --nice=10000

DATASET_ID=\$SLURM_ARRAY_TASK_ID
echo "Generating dataset \$DATASET_ID at \$(date)"
python data_generation.py --dataset_id \$DATASET_ID
echo "Done at \$(date)"
EOF
)
echo "  Submitted data generation job array: $JOB1"

# Step 2: Submit training jobs (depends on data generation)
echo "Step 2: Submitting training jobs..."
JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 --array=0-49 <<EOF
#!/bin/bash
#SBATCH --job-name=train_%a
#SBATCH --output=logs/train_%a_%j.out
#SBATCH --error=logs/train_%a_%j.err
#SBATCH --open-mode=append
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH -p gpu_p
#SBATCH --qos gpu_normal
#SBATCH --constraint=a100_80gb|h100_80gb
#SBATCH --nice=10000

SEEDS=(12 50 76 100 142)
N_SEEDS=\${#SEEDS[@]}
TASK_ID=\$SLURM_ARRAY_TASK_ID

JOBS_PER_DATASET=\$((N_SEEDS * 2))
DATASET_ID=\$((TASK_ID / JOBS_PER_DATASET))
WITHIN_DATASET=\$((TASK_ID % JOBS_PER_DATASET))
SEED_IDX=\$((WITHIN_DATASET / 2))
MODEL_TYPE=\$((WITHIN_DATASET % 2))
SEED=\${SEEDS[\$SEED_IDX]}

if [ \$MODEL_TYPE -eq 0 ]; then
    python run_Q_model.py --seed \$SEED --latent True --dataset_id \$DATASET_ID --lmbd $LMBD --z $Z_DIM
else
    python run_Q_model.py --seed \$SEED --latent False --dataset_id \$DATASET_ID
fi
EOF
)
echo "  Submitted training job array: $JOB2"

# Step 3: Submit epoch selection jobs (depends on training)
echo "Step 3: Submitting epoch selection jobs..."
JOB3=$(sbatch --parsable --dependency=afterok:$JOB2 --array=0-9 <<EOF
#!/bin/bash
#SBATCH --job-name=select_%a
#SBATCH --output=logs/select_%a_%j.out
#SBATCH --error=logs/select_%a_%j.err
#SBATCH --open-mode=append
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH -p gpu_p
#SBATCH --qos gpu_normal
#SBATCH --nice=10000

TASK_ID=\$SLURM_ARRAY_TASK_ID
DATASET_ID=\$((TASK_ID / 2))
MODEL_TYPE=\$((TASK_ID % 2))

if [ \$MODEL_TYPE -eq 0 ]; then
    python select_best_epoch_by_rsa.py --latent True --dataset_id \$DATASET_ID
else
    python select_best_epoch_by_rsa.py --latent False --dataset_id \$DATASET_ID
fi
EOF
)
echo "  Submitted epoch selection job array: $JOB3"

# Step 4: Submit testing jobs (depends on epoch selection)
echo "Step 4: Submitting testing jobs..."
JOB4=$(sbatch --parsable --dependency=afterok:$JOB3 --array=0-9 <<EOF
#!/bin/bash
#SBATCH --job-name=test_%a
#SBATCH --output=logs/test_%a_%j.out
#SBATCH --error=logs/test_%a_%j.err
#SBATCH --open-mode=append
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH -p gpu_p
#SBATCH --qos gpu_normal
#SBATCH --constraint=a100_80gb|h100_80gb
#SBATCH --nice=10000

TASK_ID=\$SLURM_ARRAY_TASK_ID
DATASET_ID=\$((TASK_ID / 2))
MODEL_TYPE=\$((TASK_ID % 2))

if [ \$MODEL_TYPE -eq 0 ]; then
    python testing_script.py --latent True --dataset_id \$DATASET_ID --model_fitting True
else
    python testing_script.py --latent False --dataset_id \$DATASET_ID --model_fitting True
fi
EOF
)
echo "  Submitted testing job array: $JOB4"

# Step 5: Submit analysis job (depends on testing)
echo "Step 5: Submitting analysis job..."
JOB5=$(sbatch --parsable --dependency=afterok:$JOB4 <<EOF
#!/bin/bash
#SBATCH --job-name=analyze
#SBATCH --output=logs/analyze_%j.out
#SBATCH --error=logs/analyze_%j.err
#SBATCH --open-mode=append
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH -p gpu_p
#SBATCH --qos gpu_normal
#SBATCH --nice=10000

echo "Running aggregated analysis at \$(date)"
python analyze_synthetic_multi_dataset.py
echo "Pipeline complete at \$(date)"
EOF
)
echo "  Submitted analysis job: $JOB5"

echo ""
echo "Pipeline submitted successfully!"
echo "Job dependency chain: $JOB1 → $JOB2 → $JOB3 → $JOB4 → $JOB5"
echo ""
echo "Monitor progress with: squeue -u \$USER"
echo "Cancel all jobs with: scancel $JOB1 $JOB2 $JOB3 $JOB4 $JOB5"
