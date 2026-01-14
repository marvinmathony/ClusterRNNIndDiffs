#!/bin/bash

# Batch script to run testing_script.py for all datasets
# Usage: ./run_all_datasets.sh [--latent true|false] [--model_fitting true|false]

# Default values
LATENT=true
MODEL_FITTING=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --latent)
            LATENT="$2"
            shift 2
            ;;
        --model_fitting)
            MODEL_FITTING="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--latent true|false] [--model_fitting true|false]"
            exit 1
            ;;
    esac
done

echo "Running testing_script.py for all datasets"
echo "Parameters: latent=$LATENT, model_fitting=$MODEL_FITTING"
echo "================================================"

# Loop through datasets 0-4
for dataset_id in {0..4}; do
    echo ""
    echo "Processing Dataset $dataset_id..."
    echo "----------------------------------------"

    python testing_script.py \
        --dataset_id $dataset_id \
        --latent $LATENT \
        --model_fitting $MODEL_FITTING

    exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo "✓ Dataset $dataset_id completed successfully"
    else
        echo "✗ Dataset $dataset_id failed with exit code $exit_code"
        # Optionally, uncomment the next line to stop on first failure
        # exit $exit_code
    fi

    echo "----------------------------------------"
done

echo ""
echo "================================================"
echo "All datasets processed!"
