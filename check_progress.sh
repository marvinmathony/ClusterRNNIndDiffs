#!/bin/bash
# Script to check progress of multi-dataset pipeline

echo "======================================"
echo "Multi-Dataset Pipeline Progress Check"
echo "======================================"
echo ""

N_DATASETS=5
SEEDS=(12 50 76 100 142)

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

check_file() {
    if [ -f "$1" ]; then
        echo -e "${GREEN}✓${NC}"
        return 0
    else
        echo -e "${RED}✗${NC}"
        return 1
    fi
}

check_dir() {
    if [ -d "$1" ]; then
        echo -e "${GREEN}✓${NC}"
        return 0
    else
        echo -e "${RED}✗${NC}"
        return 1
    fi
}

# Check data generation
echo "1. Data Generation"
echo "-------------------"
data_complete=0
for dataset_id in $(seq 0 $((N_DATASETS-1))); do
    printf "  Dataset $dataset_id: "
    if [ -f "data_dataset${dataset_id}/df_train.csv" ] && \
       [ -f "data_dataset${dataset_id}/df_test.csv" ]; then
        echo -e "${GREEN}✓ Complete${NC}"
        ((data_complete++))
    else
        echo -e "${RED}✗ Missing${NC}"
    fi
done
echo "  Progress: $data_complete/$N_DATASETS datasets"
echo ""

# Check training
echo "2. Model Training"
echo "------------------"
latent_complete=0
vanilla_complete=0
for dataset_id in $(seq 0 $((N_DATASETS-1))); do
    printf "  Dataset $dataset_id:\n"

    # Check latent models
    printf "    Latent models: "
    latent_count=0
    for seed in "${SEEDS[@]}"; do
        if [ -d "runs_dataset${dataset_id}/seed_${seed}/checkpoints" ]; then
            ((latent_count++))
        fi
    done
    printf "$latent_count/${#SEEDS[@]} seeds "
    if [ $latent_count -eq ${#SEEDS[@]} ]; then
        echo -e "${GREEN}✓${NC}"
        ((latent_complete++))
    elif [ $latent_count -gt 0 ]; then
        echo -e "${YELLOW}⚠ Partial${NC}"
    else
        echo -e "${RED}✗${NC}"
    fi

    # Check vanilla models
    printf "    Vanilla models: "
    vanilla_count=0
    for seed in "${SEEDS[@]}"; do
        if [ -d "runs_vanilla_dataset${dataset_id}/seed_${seed}/checkpoints" ]; then
            ((vanilla_count++))
        fi
    done
    printf "$vanilla_count/${#SEEDS[@]} seeds "
    if [ $vanilla_count -eq ${#SEEDS[@]} ]; then
        echo -e "${GREEN}✓${NC}"
        ((vanilla_complete++))
    elif [ $vanilla_count -gt 0 ]; then
        echo -e "${YELLOW}⚠ Partial${NC}"
    else
        echo -e "${RED}✗${NC}"
    fi
done
echo "  Latent models complete: $latent_complete/$N_DATASETS datasets"
echo "  Vanilla models complete: $vanilla_complete/$N_DATASETS datasets"
echo ""

# Check epoch selection
echo "3. Epoch Selection"
echo "-------------------"
selection_complete=0
for dataset_id in $(seq 0 $((N_DATASETS-1))); do
    printf "  Dataset $dataset_id: "
    latent_selected=false
    vanilla_selected=false

    if [ -f "runs_dataset${dataset_id}/best_epoch_by_rsa.json" ]; then
        latent_selected=true
    fi
    if [ -f "runs_vanilla_dataset${dataset_id}/best_epoch_by_rsa.json" ]; then
        vanilla_selected=true
    fi

    if $latent_selected && $vanilla_selected; then
        echo -e "${GREEN}✓ Complete${NC}"
        ((selection_complete++))
    elif $latent_selected || $vanilla_selected; then
        echo -e "${YELLOW}⚠ Partial${NC}"
    else
        echo -e "${RED}✗ Missing${NC}"
    fi
done
echo "  Progress: $selection_complete/$N_DATASETS datasets"
echo ""

# Check testing
echo "4. Model Testing"
echo "-----------------"
testing_complete=0
for dataset_id in $(seq 0 $((N_DATASETS-1))); do
    printf "  Dataset $dataset_id: "
    latent_tested=false
    vanilla_tested=false

    if [ -f "data_dataset${dataset_id}/rnn_resultslatentmodel.csv" ]; then
        latent_tested=true
    fi
    if [ -f "data_dataset${dataset_id}/rnn_resultsvanilla.csv" ]; then
        vanilla_tested=true
    fi

    if $latent_tested && $vanilla_tested; then
        echo -e "${GREEN}✓ Complete${NC}"
        ((testing_complete++))
    elif $latent_tested || $vanilla_tested; then
        echo -e "${YELLOW}⚠ Partial${NC}"
    else
        echo -e "${RED}✗ Missing${NC}"
    fi
done
echo "  Progress: $testing_complete/$N_DATASETS datasets"
echo ""

# Check analysis
echo "5. Aggregated Analysis"
echo "-----------------------"
printf "  Final plots: "
if [ -d "plots/multi_dataset" ] && \
   [ -f "plots/multi_dataset/aggregated_rsa_correlation.png" ] && \
   [ -f "plots/multi_dataset/aggregated_model_likelihoods.png" ]; then
    echo -e "${GREEN}✓ Complete${NC}"
    analysis_complete=true
else
    echo -e "${RED}✗ Missing${NC}"
    analysis_complete=false
fi
echo ""

# Overall progress
echo "======================================"
echo "Overall Progress Summary"
echo "======================================"

total_steps=$((N_DATASETS * 4 + 1))  # 4 steps per dataset + 1 analysis
completed_steps=$((data_complete + latent_complete + vanilla_complete + selection_complete + testing_complete))
if $analysis_complete; then
    ((completed_steps++))
fi

percentage=$((completed_steps * 100 / total_steps))

echo "Steps completed: $completed_steps/$total_steps ($percentage%)"
echo ""

if [ $completed_steps -eq $total_steps ]; then
    echo -e "${GREEN}✓ Pipeline Complete!${NC}"
    echo "Results available in: plots/multi_dataset/"
else
    echo -e "${YELLOW}Pipeline in progress...${NC}"
    echo ""
    echo "Next steps:"
    if [ $data_complete -lt $N_DATASETS ]; then
        echo "  - Generate remaining datasets"
    elif [ $latent_complete -lt $N_DATASETS ] || [ $vanilla_complete -lt $N_DATASETS ]; then
        echo "  - Complete model training"
    elif [ $selection_complete -lt $N_DATASETS ]; then
        echo "  - Run epoch selection"
    elif [ $testing_complete -lt $N_DATASETS ]; then
        echo "  - Run model testing"
    else
        echo "  - Run aggregated analysis: python analyze_synthetic_multi_dataset.py"
    fi
fi

echo ""
echo "======================================"

# Check for running SLURM jobs
if command -v squeue &> /dev/null; then
    echo ""
    echo "SLURM Jobs Status:"
    echo "-------------------"
    job_count=$(squeue -u $USER 2>/dev/null | tail -n +2 | wc -l)
    if [ $job_count -gt 0 ]; then
        echo "Running jobs: $job_count"
        squeue -u $USER --format="%.8i %.9P %.30j %.8T %.10M %.6D"
    else
        echo "No jobs currently running"
    fi
fi

# Disk usage
echo ""
echo "Disk Usage:"
echo "------------"
if command -v du &> /dev/null; then
    total_size=$(du -sh data_dataset* runs* 2>/dev/null | awk '{sum+=$1} END {print sum}')
    echo "Data and checkpoints: $(du -sh data_dataset* runs* 2>/dev/null | awk '{sum+=$1} END {print sum"M"}')"
    echo "Plots: $(du -sh plots* 2>/dev/null | tail -1 | cut -f1)"
fi
