#!/usr/bin/env python3
"""
Calculate statistical significance of differences between Mamba-char and Mamba-bpe by category.
Uses paired t-tests matching replicates by seed for more accurate comparison.
"""

import pandas as pd
import numpy as np
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

def calculate_mamba_significance_paired():
    """
    Perform paired t-tests comparing Mamba-char vs Mamba-bpe for each category.
    Matches replicates by seed for true paired comparison.
    """
    
    # Read the best replicates file (with individual replicate data)
    replicates_file = '../final_combined_ss_results/best_replicates_ss.csv'
    
    if not os.path.exists(replicates_file):
        print(f"Error: File {replicates_file} not found!")
        return
    
    print(f"Reading file: {replicates_file}")
    df = pd.read_csv(replicates_file)
    
    # Check if seed column exists
    if 'seed' not in df.columns:
        print("Error: No 'seed' column found. Cannot perform paired analysis.")
        return
    
    # Define valid tasks (same as other analyses)
    valid_tasks = {
        # GB tasks
        'demo_coding_vs_intergenomic', 'demo_coding_vs_intergenomic_seqs', 'demo_human_or_worm',
        'human_enhancers_cohn', 'human_enhancers_ensembl', 'human_ensembl_regulatory',
        'human_nontata_promoters', 'human_ocr_ensembl', 'dummy_mouse_enhancers',
        # GUE tasks
        'prom_core_all', 'prom_core_notata', 'prom_core_tata', 'prom_300_all', 'prom_300_notata',
        'prom_300_tata', 'tf_0', 'tf_1', 'tf_2', 'tf_3', 'tf_4', 'splice_sites_all',
        'splice_sites_acceptors', 'splice_sites_donors', 'enhancers', 'enhancers_types',
        'mouse_0', 'mouse_1', 'mouse_2', 'mouse_3', 'mouse_4', 'reconstructed', 'covid',
        # NTv2 tasks
        'H2AFZ', 'H3K27ac', 'H3K27me3', 'H3K36me3', 'H3K4me1', 'H3K4me2', 'H3K4me3',
        'H3K9ac', 'H3K9me3', 'H4K20me1', 'promoter_all', 'promoter_no_tata', 'promoter_tata'
    }
    
    # Task to category mapping
    task_to_category = {
        # Regulatory Elements
        "human_ensembl_regulatory": "regulatory",
        "human_ocr_ensembl": "regulatory",
        "human_nontata_promoters": "regulatory",
        # Promoter Detection
        "prom_core_all": "promoters",
        "prom_core_tata": "promoters",
        "prom_core_notata": "promoters",
        "prom_300_all": "promoters",
        "prom_300_tata": "promoters",
        "prom_300_notata": "promoters",
        "promoter_all": "promoters",
        "promoter_no_tata": "promoters",
        "promoter_tata": "promoters",
        # Enhancer Detection
        "dummy_mouse_enhancers": "enhancers",
        "human_enhancers_cohn": "enhancers",
        "human_enhancers_ensembl": "enhancers",
        "enhancers": "enhancers",
        "enhancers_types": "enhancers",
        # Transcription Factor Binding
        "tf_0": "transcription factors",
        "tf_1": "transcription factors",
        "tf_2": "transcription factors",
        "tf_3": "transcription factors",
        "tf_4": "transcription factors",
        "mouse_0": "transcription factors",
        "mouse_1": "transcription factors",
        "mouse_2": "transcription factors",
        "mouse_3": "transcription factors",
        "mouse_4": "transcription factors",
        # Epigenetic Marks
        "H2AFZ": "epigenetic marks",
        "H3K27ac": "epigenetic marks",
        "H3K27me3": "epigenetic marks",
        "H3K36me3": "epigenetic marks",
        "H3K4me1": "epigenetic marks",
        "H3K4me2": "epigenetic marks",
        "H3K4me3": "epigenetic marks",
        "H3K9ac": "epigenetic marks",
        "H3K9me3": "epigenetic marks",
        "H4K20me1": "epigenetic marks",
        # Splice Sites
        "splice_sites_all": "splice sites",
        "splice_sites_acceptors": "splice sites",
        "splice_sites_donors": "splice sites",
        "reconstructed": "splice sites",
        # Coding vs Non-coding
        "demo_coding_vs_intergenomic": "coding",
        "demo_coding_vs_intergenomic_seqs": "coding",
        # Taxonomic Classification
        "demo_human_or_worm": "taxonomic",
        # Virus Variant Detection
        "covid": "virus variant detection"
    }
    
    # Filter dataframes to only include valid tasks and Mamba models
    mamba_df = df[(df['task'].isin(valid_tasks)) & 
                  (df['model'].isin(['Mamba-bpe', 'Mamba-char'])) &
                  (df['mcc'].notna())]
    
    print(f"\nTotal valid Mamba rows: {len(mamba_df)}")
    print(f"Tasks with data: {sorted(mamba_df['task'].unique())}")
    
    # Process each task to create paired data
    paired_results = []
    task_details = []
    
    for task in sorted(valid_tasks):
        task_data = mamba_df[mamba_df['task'] == task]
        if len(task_data) == 0:
            continue
        
        # Get data for each model
        bpe_data = task_data[task_data['model'] == 'Mamba-bpe']
        char_data = task_data[task_data['model'] == 'Mamba-char']
        
        if len(bpe_data) == 0 or len(char_data) == 0:
            continue
        
        # Find common seeds
        bpe_seeds = set(bpe_data['seed'].unique())
        char_seeds = set(char_data['seed'].unique())
        common_seeds = bpe_seeds.intersection(char_seeds)
        
        if len(common_seeds) < 2:  # Need at least 2 paired observations
            print(f"Task {task}: Only {len(common_seeds)} paired observations - skipping")
            continue
        
        # Create paired observations
        paired_mcc_bpe = []
        paired_mcc_char = []
        
        for seed in sorted(common_seeds):
            # Take the first value if multiple runs with same seed (shouldn't happen with best replicates)
            bpe_value = bpe_data[bpe_data['seed'] == seed]['mcc'].iloc[0]
            char_value = char_data[char_data['seed'] == seed]['mcc'].iloc[0]
            
            paired_mcc_bpe.append(bpe_value)
            paired_mcc_char.append(char_value)
        
        # Convert to numpy arrays
        paired_mcc_bpe = np.array(paired_mcc_bpe)
        paired_mcc_char = np.array(paired_mcc_char)
        
        # Store paired data for this task
        category = task_to_category.get(task, 'unknown')
        
        task_result = {
            'task': task,
            'category': category,
            'n_paired': len(common_seeds),
            'n_unpaired_bpe': len(bpe_seeds - common_seeds),
            'n_unpaired_char': len(char_seeds - common_seeds),
            'mcc_bpe_mean': np.mean(paired_mcc_bpe),
            'mcc_char_mean': np.mean(paired_mcc_char),
            'paired_mcc_bpe': paired_mcc_bpe,
            'paired_mcc_char': paired_mcc_char
        }
        
        paired_results.append(task_result)
        
        # Store detailed info
        task_details.append({
            'task': task,
            'category': category,
            'n_paired': len(common_seeds),
            'paired_seeds': sorted(common_seeds),
            'unpaired_bpe_seeds': sorted(bpe_seeds - common_seeds),
            'unpaired_char_seeds': sorted(char_seeds - common_seeds)
        })
    
    print(f"\nTasks with paired data: {len(paired_results)}")
    
    # Aggregate by category and perform t-tests
    category_results = {}
    
    for category in sorted(set(r['category'] for r in paired_results)):
        category_tasks = [r for r in paired_results if r['category'] == category]
        
        if len(category_tasks) == 0:
            continue
        
        # Combine all paired observations for this category
        all_bpe = np.concatenate([r['paired_mcc_bpe'] for r in category_tasks])
        all_char = np.concatenate([r['paired_mcc_char'] for r in category_tasks])
        
        # Perform paired t-test
        if len(all_bpe) >= 2:
            t_stat, p_value = stats.ttest_rel(all_char, all_bpe)
            
            mean_diff = np.mean(all_char - all_bpe)
            std_diff = np.std(all_char - all_bpe, ddof=1)
            
            category_results[category] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'mean_diff': mean_diff,
                'std_diff': std_diff,
                'n_tasks': len(category_tasks),
                'n_paired_obs': len(all_bpe),
                'mean_char': np.mean(all_char),
                'mean_bpe': np.mean(all_bpe)
            }
    
    # Calculate Bonferroni alpha
    n_categories = len(category_results)
    bonferroni_alpha = 0.05 / n_categories
    
    # Print results
    print("\n" + "=" * 80)
    print("Paired T-test Results by Category (Matched by Seed)")
    print(f"Bonferroni correction: α = 0.05 / {n_categories} = {bonferroni_alpha:.4f}")
    print("=" * 80)
    
    for category, results in sorted(category_results.items()):
        print(f"\nCategory: {category}")
        print(f"  Number of tasks: {results['n_tasks']}")
        print(f"  Total paired observations: {results['n_paired_obs']}")
        print(f"  Mean MCC (Mamba-char): {results['mean_char']:.4f}")
        print(f"  Mean MCC (Mamba-bpe): {results['mean_bpe']:.4f}")
        print(f"  Mean difference (char - bpe): {results['mean_diff']:.4f} ± {results['std_diff']:.4f}")
        print(f"  t-statistic: {results['t_statistic']:.4f}")
        print(f"  p-value: {results['p_value']:.4f}", end="")
        
        # Add significance markers
        if results['p_value'] < 0.001:
            print(" ***", end="")
        elif results['p_value'] < 0.01:
            print(" **", end="")
        elif results['p_value'] < 0.05:
            print(" *", end="")
        else:
            print(" (ns)", end="")
        
        # Add Bonferroni marker
        if results['p_value'] < bonferroni_alpha:
            print(" †")
        else:
            print()
        
        # Interpretation
        if results['p_value'] < 0.05:
            if results['mean_diff'] > 0:
                print(f"  → Mamba-char significantly outperforms Mamba-bpe", end="")
            else:
                print(f"  → Mamba-bpe significantly outperforms Mamba-char", end="")
            if results['p_value'] < bonferroni_alpha:
                print(" (Bonferroni significant)")
            else:
                print()
    
    # Overall comparison
    print("\n" + "=" * 80)
    print("Overall Paired Comparison (All Tasks)")
    print("=" * 80)
    
    all_bpe_overall = np.concatenate([r['paired_mcc_bpe'] for r in paired_results])
    all_char_overall = np.concatenate([r['paired_mcc_char'] for r in paired_results])
    
    overall_t_stat, overall_p_value = stats.ttest_rel(all_char_overall, all_bpe_overall)
    overall_mean_diff = np.mean(all_char_overall - all_bpe_overall)
    overall_std_diff = np.std(all_char_overall - all_bpe_overall, ddof=1)
    
    print(f"Total paired observations: {len(all_bpe_overall)}")
    print(f"Mean MCC (Mamba-char): {np.mean(all_char_overall):.4f}")
    print(f"Mean MCC (Mamba-bpe): {np.mean(all_bpe_overall):.4f}")
    print(f"Mean difference (char - bpe): {overall_mean_diff:.4f} ± {overall_std_diff:.4f}")
    print(f"t-statistic: {overall_t_stat:.4f}")
    print(f"p-value: {overall_p_value:.4f}", end="")
    
    if overall_p_value < 0.001:
        print(" ***")
    elif overall_p_value < 0.01:
        print(" **")
    elif overall_p_value < 0.05:
        print(" *")
    else:
        print(" (ns)")
    
    # Save results
    results_df = []
    for category, results in category_results.items():
        results_df.append({
            'Category': category,
            'N_Tasks': results['n_tasks'],
            'N_Paired_Observations': results['n_paired_obs'],
            'Mean_Mamba_Char': results['mean_char'],
            'Mean_Mamba_BPE': results['mean_bpe'],
            'Mean_Difference': results['mean_diff'],
            'Std_Difference': results['std_diff'],
            'T_Statistic': results['t_statistic'],
            'P_Value': results['p_value'],
            'Significant': 'Yes' if results['p_value'] < 0.05 else 'No',
            'Bonferroni_Significant': 'Yes' if results['p_value'] < bonferroni_alpha else 'No',
            'Superior': 'CHAR' if results['p_value'] < 0.05 and results['mean_diff'] > 0 else 'BPE' if results['p_value'] < 0.05 else 'ns'
        })
    
    # Add overall row
    results_df.append({
        'Category': 'OVERALL',
        'N_Tasks': len(paired_results),
        'N_Paired_Observations': len(all_bpe_overall),
        'Mean_Mamba_Char': np.mean(all_char_overall),
        'Mean_Mamba_BPE': np.mean(all_bpe_overall),
        'Mean_Difference': overall_mean_diff,
        'Std_Difference': overall_std_diff,
        'T_Statistic': overall_t_stat,
        'P_Value': overall_p_value,
        'Significant': 'Yes' if overall_p_value < 0.05 else 'No',
        'Bonferroni_Significant': 'No',  # Bonferroni doesn't apply to overall
        'Superior': 'CHAR' if overall_p_value < 0.05 and overall_mean_diff > 0 else 'BPE' if overall_p_value < 0.05 else 'ns'
    })
    
    results_df = pd.DataFrame(results_df)
    os.makedirs('../results_tables', exist_ok=True)
    results_df.to_csv('../results_tables/mamba_paired_significance_tests.csv', index=False)
    print(f"\nResults saved to: ../results_tables/mamba_paired_significance_tests.csv")
    
    # Save task-level details
    task_details_df = []
    for r in paired_results:
        for i, seed in enumerate(sorted(set(zip(r['paired_mcc_bpe'], r['paired_mcc_char'])))):
            task_details_df.append({
                'task': r['task'],
                'category': r['category'],
                'mcc_bpe': seed[0],
                'mcc_char': seed[1],
                'difference': seed[1] - seed[0]
            })
    
    task_details_df = pd.DataFrame(task_details_df)
    task_details_df.to_csv('../results_tables/mamba_paired_task_details.csv', index=False)
    print(f"Task details saved to: ../results_tables/mamba_paired_task_details.csv")
    
    # Create LaTeX table
    print("\n" + "=" * 80)
    print("LaTeX Table Output")
    print("=" * 80)
    
    # Recalculate bonferroni_alpha for LaTeX section
    n_categories = len(category_results)
    bonferroni_alpha = 0.05 / n_categories
    
    latex_lines = []
    latex_lines.append("\\begin{table}[]")
    latex_lines.append("\\begin{center}")
    latex_lines.append("\\scriptsize")
    latex_lines.append("\\caption{Paired Comparison of Character-Level vs. Byte Pair Encoding Tokenization on MCC Scores "
                      "Across Different Genomic Features in a 4 layer Mamba-DNA Model (Matched by Seed)} \\label{table:paired_statistical_tests}")
    latex_lines.append("\\begin{tabular}{@{}crlr@{}}")
    latex_lines.append("\\toprule")
    latex_lines.append("\\textbf{Category} & \\textbf{t-statistic} & \\textbf{p-value} & "
                      "\\textbf{\\begin{tabular}[c]{@{}c@{}}Mean difference \\\\ (char - bpe)\\end{tabular}} \\\\ \\midrule")
    
    # Add rows for each category
    for category in sorted(category_results.keys()):
        results = category_results[category]
        
        # Format values
        t_stat = f"{results['t_statistic']:.4f}"
        p_val = f"{results['p_value']:.4f}"
        mean_diff = f"{results['mean_diff']:.4f}"
        
        # Add significance marker to p-value based on Bonferroni correction
        if results['p_value'] < 0.001:
            p_val += "***"
        elif results['p_value'] < 0.01:
            p_val += "**"
        elif results['p_value'] < 0.05:
            p_val += "*"
        
        # Check if Bonferroni significant
        if results['p_value'] < bonferroni_alpha:
            p_val += "†"
            # Use row coloring for Bonferroni significant results
            # Teal for char better, magenta for bpe better
            if results['mean_diff'] > 0:  # char > bpe
                row_start = "\\rowcolor{teal!25}"
            else:  # bpe > char
                row_start = "\\rowcolor{magenta!25}"
        else:
            row_start = ""
        
        # Replace underscores in category names for LaTeX
        category_display = category.replace('_', ' ')
        
        # Determine superior method
        if results['p_value'] < 0.05:
            superior = 'CHAR' if results['mean_diff'] > 0 else 'BPE'
        else:
            superior = '-'
        
        # Build row
        if row_start:
            latex_lines.append(f"{row_start}{category_display} & {t_stat} & {p_val} & {mean_diff} \\\\")
        else:
            latex_lines.append(f"{category_display} & {t_stat} & {p_val} & {mean_diff} \\\\")
    
    latex_lines.append("\\midrule")
    latex_lines.append("\\multicolumn{4}{l}{* p $< 0.05$, ** p $< 0.01$, *** p $< 0.001$} \\\\")
    latex_lines.append(f"\\multicolumn{{4}}{{l}}{{† Bonferroni corrected ($\\alpha = {bonferroni_alpha:.4f}$)}} \\\\")
    latex_lines.append("\\multicolumn{4}{l}{\\cellcolor{teal!25} Character tokenization superior (Bonferroni significant)} \\\\")
    latex_lines.append("\\multicolumn{4}{l}{\\cellcolor{magenta!25} BPE tokenization superior (Bonferroni significant)}")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{center}")
    latex_lines.append("\\end{table}")
    
    # Print LaTeX table
    print("\n".join(latex_lines))
    
    # Save LaTeX table to file
    with open('../results_tables/mamba_paired_significance_table.tex', 'w') as f:
        f.write("\n".join(latex_lines))
    print(f"\nLaTeX table saved to: ../results_tables/mamba_paired_significance_table.tex")

if __name__ == "__main__":
    calculate_mamba_significance_paired()