#!/bin/bash

# Script to complete the remaining steps after fixing the bug
# This will:
# 1. Train vanilla models for all datasets
# 2. Select best epochs for vanilla models
# 3. Run testing script for both latent and vanilla models

DATASET_ID=${1:-0}  # Default to dataset 0 if no argument provided

echo "================================================"
echo "Completing pipeline for Dataset $DATASET_ID"
echo "================================================"

# Configuration
SEEDS=(12 50 76 100 142)

# Step 1: Train vanilla models
echo ""
echo "Step 1: Training vanilla models for dataset $DATASET_ID..."
echo "----------------------------------------"
for seed in "${SEEDS[@]}"; do
    echo "  Training vanilla model with seed $seed..."
    python run_Q_model.py \
        --seed $seed \
        --latent False \
        --dataset_id $DATASET_ID

    if [ $? -eq 0 ]; then
        echo "  ✓ Seed $seed completed"
    else
        echo "  ✗ Seed $seed failed"
        exit 1
    fi
done

# Step 2: Select best epoch for vanilla models
echo ""
echo "Step 2: Selecting best epoch for vanilla models..."
echo "----------------------------------------"
python select_best_epoch_by_rsa.py --latent False --dataset_id $DATASET_ID

if [ $? -eq 0 ]; then
    echo "✓ Best epoch selection completed"
else
    echo "✗ Best epoch selection failed"
    exit 1
fi

# Step 3: Test models
echo ""
echo "Step 3: Testing models for dataset $DATASET_ID..."
echo "----------------------------------------"

echo "  Testing latent model..."
python testing_script.py --latent True --dataset_id $DATASET_ID --model_fitting True

if [ $? -eq 0 ]; then
    echo "  ✓ Latent model testing completed"
else
    echo "  ✗ Latent model testing failed"
    exit 1
fi

echo "  Testing vanilla model..."
python testing_script.py --latent False --dataset_id $DATASET_ID --model_fitting True

if [ $? -eq 0 ]; then
    echo "  ✓ Vanilla model testing completed"
else
    echo "  ✗ Vanilla model testing failed"
    exit 1
fi

echo ""
echo "================================================"
echo "Dataset $DATASET_ID pipeline completed successfully!"
echo "================================================"
