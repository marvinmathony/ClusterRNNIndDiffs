#!/bin/bash
# Quick reference script for running the multi-dataset pipeline
# This is a helper script - you can also run commands directly

# Configuration
N_DATASETS=5
SEEDS=(12 50 76 100 142)

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Function to run full pipeline
run_full_pipeline() {
    print_header "Running Full Multi-Dataset Pipeline"
    python run_multi_dataset_pipeline.py "$@"
    if [ $? -eq 0 ]; then
        print_success "Pipeline completed successfully!"
    else
        print_error "Pipeline failed. Check error messages above."
        exit 1
    fi
}

# Function to run only analysis
run_analysis_only() {
    print_header "Running Aggregated Analysis Only"
    python analyze_synthetic_multi_dataset.py
    if [ $? -eq 0 ]; then
        print_success "Analysis completed!"
    else
        print_error "Analysis failed."
        exit 1
    fi
}

# Function to generate data for all datasets
generate_all_data() {
    print_header "Generating All Datasets"
    for dataset_id in $(seq 0 $((N_DATASETS-1))); do
        echo -e "${BLUE}Generating dataset $dataset_id...${NC}"
        python data_generation.py --dataset_id $dataset_id
        if [ $? -ne 0 ]; then
            print_error "Failed on dataset $dataset_id"
            exit 1
        fi
    done
    print_success "All datasets generated!"
}

# Function to train models for a specific dataset
train_dataset() {
    local dataset_id=$1
    print_header "Training Models on Dataset $dataset_id"

    # Train latent models
    echo "Training latent models..."
    for seed in "${SEEDS[@]}"; do
        echo "  Seed $seed..."
        python run_Q_model.py --seed $seed --latent True --dataset_id $dataset_id --lmbd 1.0 --z 1
        if [ $? -ne 0 ]; then
            print_error "Latent model training failed for seed $seed"
            exit 1
        fi
    done

    # Train vanilla models
    echo "Training vanilla models..."
    for seed in "${SEEDS[@]}"; do
        echo "  Seed $seed..."
        python run_Q_model.py --seed $seed --latent False --dataset_id $dataset_id
        if [ $? -ne 0 ]; then
            print_error "Vanilla model training failed for seed $seed"
            exit 1
        fi
    done

    print_success "Training completed for dataset $dataset_id"
}

# Function to select best epochs for a specific dataset
select_best_epoch() {
    local dataset_id=$1
    print_header "Selecting Best Epochs for Dataset $dataset_id"

    echo "Selecting for latent models..."
    python select_best_epoch_by_rsa.py --latent True --dataset_id $dataset_id

    echo "Selecting for vanilla models..."
    python select_best_epoch_by_rsa.py --latent False --dataset_id $dataset_id

    print_success "Epoch selection completed for dataset $dataset_id"
}

# Function to test models for a specific dataset
test_dataset() {
    local dataset_id=$1
    print_header "Testing Models on Dataset $dataset_id"

    echo "Testing latent models..."
    python testing_script.py --latent True --dataset_id $dataset_id --model_fitting True

    echo "Testing vanilla models..."
    python testing_script.py --latent False --dataset_id $dataset_id --model_fitting True

    print_success "Testing completed for dataset $dataset_id"
}

# Parse command line arguments
case "$1" in
    full|"")
        # Run full pipeline
        shift
        run_full_pipeline "$@"
        ;;
    analysis|analyze)
        # Run only analysis
        run_analysis_only
        ;;
    data|generate)
        # Generate all datasets
        generate_all_data
        ;;
    train)
        # Train models for a specific dataset
        if [ -z "$2" ]; then
            echo "Usage: $0 train <dataset_id>"
            exit 1
        fi
        train_dataset $2
        ;;
    select)
        # Select best epochs for a specific dataset
        if [ -z "$2" ]; then
            echo "Usage: $0 select <dataset_id>"
            exit 1
        fi
        select_best_epoch $2
        ;;
    test)
        # Test models for a specific dataset
        if [ -z "$2" ]; then
            echo "Usage: $0 test <dataset_id>"
            exit 1
        fi
        test_dataset $2
        ;;
    help|-h|--help)
        cat <<EOF
Multi-Dataset Pipeline Helper Script

Usage: $0 [command] [options]

Commands:
    full, (default)      Run the complete pipeline
    analysis, analyze    Run only the aggregated analysis
    data, generate       Generate all datasets
    train <dataset_id>   Train models for a specific dataset
    select <dataset_id>  Select best epochs for a specific dataset
    test <dataset_id>    Test models for a specific dataset
    help                 Show this help message

Examples:
    $0 full                  # Run everything
    $0 analysis              # Only run analysis
    $0 data                  # Generate all 5 datasets
    $0 train 0               # Train models on dataset 0
    $0 select 2              # Select best epochs for dataset 2
    $0 test 4                # Test models on dataset 4

    # Run with custom number of datasets
    $0 full --n_datasets 10

    # Skip certain steps
    $0 full --skip_data_generation
    $0 full --skip_training

For more details, see README_MULTI_DATASET.md
EOF
        ;;
    *)
        echo "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac
