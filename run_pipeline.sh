#!/bin/bash

# Master pipeline script to run all analysis steps in order
# 
# PREREQUISITE: The following cleaned datasets must be present:
#   - final_combined_attn_results/combined_attn_results_cleaned.csv
#   - final_combined_ss_results/combined_ss_results_cleaned.csv

echo "Starting tokenization analysis pipeline..."
echo "========================================"

# Check for prerequisite files
if [ ! -f "final_combined_attn_results/combined_attn_results_cleaned.csv" ]; then
    echo "Error: Required file final_combined_attn_results/combined_attn_results_cleaned.csv not found!"
    exit 1
fi

if [ ! -f "final_combined_ss_results/combined_ss_results_cleaned.csv" ]; then
    echo "Error: Required file final_combined_ss_results/combined_ss_results_cleaned.csv not found!"
    exit 1
fi

echo "✓ Prerequisite files found"

# Create output directories if they don't exist
mkdir -p final_combined_attn_results
mkdir -p final_combined_ss_results
mkdir -p ss_tables
mkdir -p ss_figures
mkdir -p results_tables

# Step 1: Find best hyperparameters for attention models
echo "Step 1: Finding best hyperparameters for attention models..."
cd attn_scripts
python find_best_hyperparameters.py > best_hyperparameters.log 2>&1
if [ $? -eq 0 ]; then
    echo "✓ Found best hyperparameters for attention models"
else
    echo "✗ Error finding best hyperparameters. Check best_hyperparameters.log"
    exit 1
fi
cd ..

# Step 2: Find best hyperparameters for state space models
echo "Step 2: Finding best hyperparameters for state space models..."
cd ss_scripts
python find_best_hyperparameters_ss.py > best_hyperparameters_ss.log 2>&1
if [ $? -eq 0 ]; then
    echo "✓ Found best hyperparameters for state space models"
else
    echo "✗ Error finding best hyperparameters. Check best_hyperparameters_ss.log"
    exit 1
fi
cd ..

# Step 3: Create hyperparameter LaTeX tables
echo "Step 3: Creating hyperparameter LaTeX tables..."
cd ss_scripts
python create_hp_latex_tables.py > create_hp_latex_tables.log 2>&1
if [ $? -eq 0 ]; then
    echo "✓ Created hyperparameter LaTeX tables"
else
    echo "✗ Error creating LaTeX tables. Check create_hp_latex_tables.log"
    exit 1
fi
cd ..

# Step 4: Create NaN heatmap
echo "Step 4: Creating NaN heatmap..."
cd ss_scripts
python create_nan_heatmap.py > create_nan_heatmap.log 2>&1
if [ $? -eq 0 ]; then
    echo "✓ Created NaN heatmap"
else
    echo "✗ Error creating heatmap. Check create_nan_heatmap.log"
    exit 1
fi
cd ..

# Step 5: Create results LaTeX tables
echo "Step 5: Creating results LaTeX tables..."
cd results_scripts
python create_results_latex_tables.py > create_results_latex_tables.log 2>&1
if [ $? -eq 0 ]; then
    echo "✓ Created results LaTeX tables"
else
    echo "✗ Error creating results tables. Check create_results_latex_tables.log"
    exit 1
fi
cd ..

# Step 6: Create category summary table
echo "Step 6: Creating category summary table..."
cd results_scripts
python create_category_summary_table.py > create_category_summary_table.log 2>&1
if [ $? -eq 0 ]; then
    echo "✓ Created category summary table"
else
    echo "✗ Error creating category summary table. Check create_category_summary_table.log"
    exit 1
fi
cd ..

# Step 7: Create MCC heatmap
echo "Step 7: Creating MCC heatmap..."
cd results_scripts
python create_mcc_heatmap.py > create_mcc_heatmap.log 2>&1
if [ $? -eq 0 ]; then
    echo "✓ Created MCC heatmap"
else
    echo "✗ Error creating MCC heatmap. Check create_mcc_heatmap.log"
    exit 1
fi
cd ..

# Step 8: Create BPE vs Char scatter plot
echo "Step 8: Creating BPE vs Char scatter plot..."
cd results_scripts
python create_bpe_vs_char_scatter.py > create_bpe_vs_char_scatter.log 2>&1
if [ $? -eq 0 ]; then
    echo "✓ Created BPE vs Char scatter plot"
else
    echo "✗ Error creating scatter plot. Check create_bpe_vs_char_scatter.log"
    exit 1
fi
cd ..

# Step 9: Calculate Mamba paired significance tests
echo "Step 9: Calculating paired statistical significance for Mamba models (matched by seed)..."
cd results_scripts
python calculate_mamba_significance_paired.py > calculate_mamba_significance_paired.log 2>&1
if [ $? -eq 0 ]; then
    echo "✓ Completed paired significance tests"
else
    echo "✗ Error calculating paired significance. Check calculate_mamba_significance_paired.log"
    exit 1
fi
cd ..

# Step 10: Calculate Mamba paired significance tests by individual task
echo "Step 10: Calculating paired statistical significance for Mamba models by individual task..."
cd results_scripts
python calculate_mamba_significance_paired_by_task.py > calculate_mamba_significance_paired_by_task.log 2>&1
if [ $? -eq 0 ]; then
    echo "✓ Completed paired significance tests by task"
else
    echo "✗ Error calculating paired significance by task. Check calculate_mamba_significance_paired_by_task.log"
    exit 1
fi
cd ..

echo "========================================"
echo "Pipeline completed successfully!"
echo "Results can be found in:"
echo "  - final_combined_attn_results/"
echo "  - final_combined_ss_results/"
echo "  - ss_tables/"
echo "  - ss_figures/"
echo "  - results_tables/"
echo "  - results_figures/"