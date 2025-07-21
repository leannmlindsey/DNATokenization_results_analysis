# Data Processing Pipeline Documentation

## Overview
This repository contains the data processing pipeline to reproduce all of the figures, tables and results in the paper:

[![bioRxiv](https://img.shields.io/badge/bioRxiv-2024.09.09.612081-red)](https://doi.org/10.1101/2024.09.09.612081)

## Pipeline Execution
**Main Script**: `run_pipeline.sh`

This pipeline assumes the cleaned datasets are already available in:
- `final_combined_attn_results/combined_attn_results_cleaned.csv`
- `final_combined_ss_results/combined_ss_results_cleaned.csv`

Step 1: Finds best hyperparameters for attention models
Step 2: Finds best hyperparameters for state space models
Step 3: Creates hyperparameter LaTeX tables for the Supplementary Material (Tables 2-6)
Step 4: Creates NaN heatmap for the Supplementary Material (Figure 2)
Step 5: Creates results LaTeX tables for the Supplementary Material (Tables 7-11)
Step 6: Creates category summary table (Table 2)
Step 7: Creates MCC heatmap (Figure 2)
Step 8: Creates BPE vs Char scatter plot (Figure 3)
Step 9: Calculates paired statistical significance for Mamba models (matched by seed) (Table 4)
Step 10: Calculates paired statistical significance for Mamba models by individual task (Table 3)

## Detailed Pipeline Steps

### Phase 1: Attention Model Processing

#### Step 1: Find Best Hyperparameters for Attention Models
**Script**: `attn_scripts/find_best_hyperparameters.py`

**Input Files**:
- `final_combined_attn_results/combined_attn_results_cleaned.csv`

**Processing**:
- Groups by task, model, learning_rate, and epoch
- Calculates mean and std for metrics (MCC, accuracy, F1, precision, recall)
- Requires minimum 3 replicates per hyperparameter combination
- Selects best combination based on highest MCC
- Handles mixed-source data by selecting best performing source

**Output Files**:
- `final_combined_attn_results/best_hyperparameters_by_task.csv` - Summary of best hyperparameters
- `final_combined_attn_results/best_replicates.csv` - Individual replicate data for best combinations

---

### Phase 2: State Space Model Processing

#### Step 2: Find Best Hyperparameters for State Space Models
**Script**: `ss_scripts/find_best_hyperparameters_ss.py`

**Input Files**:
- `final_combined_ss_results/combined_ss_results_cleaned.csv`

**Processing**:
- Groups by task, model, learning_rate, batch_size, and rc_aug
- Prefers combinations with ≥10 replicates, falls back to ≥3 if needed
- Logs warnings when using fewer than 10 replicates
- Selects best based on highest MCC

**Output Files**:
- `final_combined_ss_results/best_hyperparameters_by_task_ss.csv` - Summary of best hyperparameters
- `final_combined_ss_results/best_replicates_ss.csv` - Individual replicate data
- `final_combined_ss_results/ss_nan_analysis.csv` - Analysis of failed runs
- `final_combined_ss_results/all_hyperparameters_analysis_ss.csv` - All hyperparameter combinations
- `final_combined_ss_results/hyperparameter_selection_log_[timestamp].log` - Detailed log with warnings

---

#### Step 3: Create Hyperparameter LaTeX Tables
**Script**: `ss_scripts/create_hp_latex_tables.py`

**Input Files**:
- `final_combined_ss_results/all_hyperparameters_analysis_ss.csv`

**Processing**:
- Creates LaTeX formatted tables for hyperparameter results
- Highlights best combinations

**Output Files**:
- `ss_tables/[various].tex` - LaTeX table files

---

#### Step 4: Create NaN Heatmap
**Script**: `ss_scripts/create_nan_heatmap.py`

**Input Files**:
- `final_combined_ss_results/ss_nan_analysis.csv`

**Processing**:
- Creates visualization of missing data patterns
- Generates heatmap showing NaN distribution

**Output Files**:
- `ss_figures/ntv2_nan_heatmap.pdf`
- `ss_figures/ntv2_nan_heatmap.png`

---

### Phase 3: Combined Results Analysis

#### Step 5: Create Results LaTeX Tables
**Script**: `results_scripts/create_results_latex_tables.py`

**Input Files**:
- `final_combined_attn_results/best_hyperparameters_by_task.csv`
- `final_combined_ss_results/best_hyperparameters_by_task_ss.csv`

**Processing**:
- Combines attention and state space model results
- Creates separate tables for each benchmark (GB, GUE, NTv2)
- Formats results with mean ± std
- Highlights best and second-best performers

**Output Files**:
- `results_tables/latex_table_gb_accuracy.tex`
- `results_tables/latex_table_gb_mcc.tex`
- `results_tables/latex_table_gue_accuracy.tex`
- `results_tables/latex_table_gue_mcc.tex`
- `results_tables/latex_table_ntv2_accuracy.tex`
- `results_tables/latex_table_ntv2_mcc.tex`

---

#### Step 6: Create Category Summary Table
**Script**: `results_scripts/create_category_summary_table.py`

**Input Files**:
- `final_combined_attn_results/best_hyperparameters_by_task.csv`
- `final_combined_ss_results/best_hyperparameters_by_task_ss.csv`

**Processing**:
- Groups tasks by category
- Calculates average performance per category
- Creates summary statistics table

**Output Files**:
- `results_tables/category_summary_table.tex`

---

#### Step 7: Create MCC Heatmap
**Script**: `results_scripts/create_mcc_heatmap.py`

**Input Files**:
- `final_combined_attn_results/best_hyperparameters_by_task.csv`
- `final_combined_ss_results/best_hyperparameters_by_task_ss.csv`

**Processing**:
- Creates heatmap visualization of MCC scores
- Organizes by model and task

**Output Files**:
- `results_figures/mcc_heatmap.pdf`
- `results_figures/mcc_heatmap.png`

---

#### Step 8: Create BPE vs Character Scatter Plot
**Script**: `results_scripts/create_bpe_vs_char_scatter.py`

**Input Files**:
- `final_combined_ss_results/best_hyperparameters_by_task_ss.csv`

**Processing**:
- Compares Mamba-BPE vs Mamba-char performance
- Creates scatter plot with regression line

**Output Files**:
- `results_figures/bpevschar.pdf`
- `results_figures/bpevschar.png`

---

#### Step 9: Calculate Mamba Significance (Overall)
**Script**: `results_scripts/calculate_mamba_significance_paired.py`

**Input Files**:
- `final_combined_attn_results/best_replicates.csv`
- `final_combined_ss_results/best_replicates_ss.csv`

**Processing**:
- Performs paired t-tests between Mamba models and baselines
- Matches samples by seed for proper paired testing
- Calculates overall significance across all tasks

**Output Files**:
- `results_tables/mamba_paired_significance_overall.csv`
- `results_tables/mamba_paired_significance_summary.txt`

---

#### Step 10: Calculate Mamba Significance (By Task)
**Script**: `results_scripts/calculate_mamba_significance_paired_by_task.py`

**Input Files**:
- `final_combined_attn_results/best_replicates.csv`
- `final_combined_ss_results/best_replicates_ss.csv`

**Processing**:
- Similar to Step 11 but analyzes each task separately
- Provides task-specific significance results

**Output Files**:
- `results_tables/mamba_paired_significance_by_task.csv`
- `results_tables/mamba_paired_significance_by_task_summary.txt`

---

## Directory Structure

### Input Data Location
- Root directory contains source CSV files

### Output Directory Structure
```
final_combined_attn_results/
├── combined_all_results_06.11.2025.csv
├── best_hyperparameters_by_task.csv
└── best_replicates.csv

final_combined_ss_results/
├── combined_ss_results_cleaned.csv
├── best_hyperparameters_by_task_ss.csv
├── best_replicates_ss.csv
├── ss_nan_analysis.csv
├── all_hyperparameters_analysis_ss.csv
└── hyperparameter_selection_log_*.log

ss_tables/
└── [various LaTeX table files]

ss_figures/
├── ntv2_nan_heatmap.pdf
└── ntv2_nan_heatmap.png

results_tables/
├── latex_table_*.tex
├── category_summary_table.tex
└── mamba_paired_significance_*.csv/txt

results_figures/
├── mcc_heatmap.pdf/png
└── bpevschar.pdf/png
```

## Key Data Transformations

### Model Name Standardization
- DNABERT2_repeated → DNABERT2
- DNABERT2-finetuned → DNABERT2
- hyenadna_repeated → HyenaDNA
- nt_repeated → NT-500M-human
- nt_v2_repeated → NT-v2-500M

### Benchmark Categories
- **GB**: Genomic Benchmarks
- **GUE**: Genomic Understanding Evaluation
- **NTv2**: Nucleotide Transformer v2 tasks
