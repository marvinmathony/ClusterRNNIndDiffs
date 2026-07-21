#!/bin/bash
# submit_gpu_job.sh NAME "python ... command ..."
# Submit a downstream compute step to gpu_p so it runs on a COMPUTE node, never
# the login node (hpc-submit02 enforces a 200% CPU cap and kills offenders).
# Threads are capped to the allocated cpus. Prints the job id (--parsable).
set -euo pipefail
NAME="$1"; shift
CMD="$*"
mkdir -p logs
sbatch --parsable \
  --nodes=1 --gres=gpu:1 -p gpu_p --qos gpu_normal \
  --constraint='a100_80gb|h100_80gb' --nice=10000 \
  --mem=16G --cpus-per-task=4 --time=03:00:00 \
  --job-name="$NAME" \
  --output="logs/${NAME}_%j.out" --error="logs/${NAME}_%j.err" \
  --wrap="export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 WANDB_MODE=disabled && source /ictstr01/home/hcai/marvin.mathony/tools/apps/mamba/etc/profile.d/conda.sh && conda activate RNNproject && cd $(pwd) && $CMD"
