#!/usr/bin/env python3
"""
Calculate statistical significance of differences between Mamba-char and Mamba-bpe by individual task.
Uses paired t-tests matching replicates by seed for more accurate comparison.
"""

import pandas as pd
import numpy as np
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

def calculate_mamba_significance_paired_by_task():
    """
    Perform paired t-tests comparing Mamba-char vs Mamba-bpe for each individual task.
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
        # Organism Classification
        "demo_coding_vs_intergenomic": "organism",
        "demo_coding_vs_intergenomic_seqs": "organism",
        "demo_human_or_worm": "organism",
        # Virus Variant Detection
        "covid": "virus variant detection"
    }
    
    # Task display names for LaTeX
    task_display_names = {
        'tf_0': 'human tfp 0',
        'tf_1': 'human tfp 1',
        'tf_2': 'human tfp 2',
        'tf_3': 'human tfp 3',
        'tf_4': 'human tfp 4',
        'mouse_0': 'mouse tfp 0',
        'mouse_1': 'mouse tfp 1',
        'mouse_2': 'mouse tfp 2',
        'mouse_3': 'mouse tfp 3',
        'mouse_4': 'mouse tfp 4',
        'H2AFZ': 'H2AFZ',
        'H3K27ac': 'H3K27ac',
        'H3K27me3': 'H3K27me3',
        'H3K36me3': 'H3K36me3',
        'H3K4me1': 'H3K4me1',
        'H3K4me2': 'H3K4me2',
        'H3K4me3': 'H3K4me3',
        'H3K9ac': 'H3K9ac',
        'H3K9me3': 'H3K9me3',
        'H4K20me1': 'H4K20me1',
        'covid': 'virus covid',
        'demo_coding_vs_intergenomic': 'coding vs intergenomic',
        'demo_human_or_worm': 'human or worm',
        'human_enhancers_cohn': 'human enhancers cohn',
        'human_enhancers_ensembl': 'human enhancers ensembl',
        'human_ensembl_regulatory': 'human ensembl regulatory',
        'human_nontata_promoters': 'human nontata promoters',
        'human_ocr_ensembl': 'human ocr ensembl',
        'dummy_mouse_enhancers': 'dummy mouse enhancers',
        'prom_core_all': 'prom core all',
        'prom_core_notata': 'prom core notata',
        'prom_core_tata': 'prom core tata',
        'prom_300_all': 'prom 300 all',
        'prom_300_notata': 'prom 300 notata',
        'prom_300_tata': 'prom 300 tata',
        'splice_sites_all': 'splice sites all',
        'splice_sites_acceptors': 'splice sites acceptors',
        'splice_sites_donors': 'splice sites donors',
        'enhancers': 'enhancers',
        'enhancers_types': 'enhancers types',
        'reconstructed': 'reconstructed',
        'promoter_all': 'promoter all',
        'promoter_no_tata': 'promoter no tata',
        'promoter_tata': 'promoter tata'
    }
    
    # Filter dataframes to only include valid tasks and Mamba models
    mamba_df = df[(df['task'].isin(valid_tasks)) & 
                  (df['model'].isin(['Mamba-bpe', 'Mamba-char'])) &
                  (df['mcc'].notna())]
    
    print(f"\nTotal valid Mamba rows: {len(mamba_df)}")
    print(f"Tasks with data: {sorted(mamba_df['task'].unique())}")
    
    # Process each task to perform paired t-test
    task_results = []
    
    # Count total number of tests for Bonferroni correction
    n_tests = 0
    for task in sorted(valid_tasks):
        task_data = mamba_df[mamba_df['task'] == task]
        if len(task_data) > 0:
            bpe_data = task_data[task_data['model'] == 'Mamba-bpe']
            char_data = task_data[task_data['model'] == 'Mamba-char']
            if len(bpe_data) > 0 and len(char_data) > 0:
                bpe_seeds = set(bpe_data['seed'].unique())
                char_seeds = set(char_data['seed'].unique())
                common_seeds = bpe_seeds.intersection(char_seeds)
                if len(common_seeds) >= 2:
                    n_tests += 1
    
    print(f"\nTotal number of tests for Bonferroni correction: {n_tests}")
    bonferroni_alpha = 0.05 / n_tests
    print(f"Bonferroni-corrected significance level: {bonferroni_alpha:.4f}")
    
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
        
        # Print diagnostic information
        print(f"\n{'='*60}")
        print(f"Task: {task}")
        print(f"  Mamba-bpe seeds ({len(bpe_seeds)}): {sorted(bpe_seeds)}")
        print(f"  Mamba-char seeds ({len(char_seeds)}): {sorted(char_seeds)}")
        print(f"  Common seeds ({len(common_seeds)}): {sorted(common_seeds)}")
        print(f"  BPE-only seeds: {sorted(bpe_seeds - common_seeds)}")
        print(f"  Char-only seeds: {sorted(char_seeds - common_seeds)}")
        
        if len(common_seeds) < 2:  # Need at least 2 paired observations
            print(f"  SKIPPING: Only {len(common_seeds)} paired observations")
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
        
        # Perform paired t-test
        t_stat, p_value = stats.ttest_rel(paired_mcc_char, paired_mcc_bpe)
        
        # Calculate statistics
        mean_diff = np.mean(paired_mcc_char - paired_mcc_bpe)
        std_diff = np.std(paired_mcc_char - paired_mcc_bpe, ddof=1)
        
        # Get category
        category = task_to_category.get(task, 'unknown')
        
        # Determine superior method with Bonferroni correction
        if p_value < bonferroni_alpha:
            superior = 'char' if mean_diff > 0 else 'bpe'
            bonferroni_sig = True
        else:
            superior = 'ns'  # not significant
            bonferroni_sig = False
        
        # Store result
        task_result = {
            'task': task,
            'task_display': task_display_names.get(task, task),
            'category': category,
            'n_paired': len(common_seeds),
            'mean_bpe': np.mean(paired_mcc_bpe),
            'mean_char': np.mean(paired_mcc_char),
            'mean_diff': mean_diff,
            'std_diff': std_diff,
            't_statistic': t_stat,
            'p_value': p_value,
            'superior': superior,
            'bonferroni_sig': bonferroni_sig
        }
        
        task_results.append(task_result)
    
    print(f"\nTasks with paired data: {len(task_results)}")
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print("SEED MATCHING SUMMARY")
    print(f"{'='*60}")
    total_tasks_analyzed = 0
    total_with_10_plus_pairs = 0
    total_with_exact_10_pairs = 0
    total_with_less_than_10 = 0
    
    for result in task_results:
        total_tasks_analyzed += 1
        n_paired = result['n_paired']
        if n_paired >= 10:
            total_with_10_plus_pairs += 1
            if n_paired == 10:
                total_with_exact_10_pairs += 1
        else:
            total_with_less_than_10 += 1
    
    print(f"Total tasks analyzed: {total_tasks_analyzed}")
    print(f"Tasks with exactly 10 matched pairs: {total_with_exact_10_pairs}")
    print(f"Tasks with 10+ matched pairs: {total_with_10_plus_pairs}")
    print(f"Tasks with <10 matched pairs: {total_with_less_than_10}")
    print(f"{'='*60}")
    
    # Sort results by category and then by task
    task_results.sort(key=lambda x: (x['category'], x['task']))
    
    # Print results
    print("\n" + "=" * 80)
    print("Paired T-test Results by Individual Task (Matched by Seed)")
    print("=" * 80)
    
    current_category = None
    for result in task_results:
        if result['category'] != current_category:
            current_category = result['category']
            print(f"\n{current_category.upper()}")
            print("-" * 40)
        
        print(f"\n{result['task_display']}:")
        print(f"  Paired observations: {result['n_paired']}")
        print(f"  Mean MCC (Mamba-char): {result['mean_char']:.4f}")
        print(f"  Mean MCC (Mamba-bpe): {result['mean_bpe']:.4f}")
        print(f"  Mean difference (char - bpe): {result['mean_diff']:.4f} ± {result['std_diff']:.4f}")
        print(f"  t-statistic: {result['t_statistic']:.4f}")
        print(f"  p-value: {result['p_value']:.4f}", end="")
        
        # Add significance markers
        if result['p_value'] < 0.001:
            print(" ***", end="")
        elif result['p_value'] < 0.01:
            print(" **", end="")
        elif result['p_value'] < 0.05:
            print(" *", end="")
        else:
            print(" (ns)", end="")
        
        if result['bonferroni_sig']:
            print(" [Bonferroni significant]")
        else:
            print()
        
        if result['superior'] != 'ns':
            print(f"  → Mamba-{result['superior']} is superior")
    
    # Save results
    results_df = pd.DataFrame(task_results)
    os.makedirs('../results_tables', exist_ok=True)
    results_df.to_csv('../results_tables/mamba_paired_significance_by_task.csv', index=False)
    print(f"\nResults saved to: ../results_tables/mamba_paired_significance_by_task.csv")
    
    # Create LaTeX table
    print("\n" + "=" * 80)
    print("LaTeX Table Output")
    print("=" * 80)
    
    latex_lines = []
    latex_lines.append("\\begin{table}[]")
    latex_lines.append("\\begin{center}")
    latex_lines.append("\\scriptsize")
    latex_lines.append("\\caption{Task-Level Paired Comparison of Character-Level vs. Byte Pair Encoding Tokenization "
                      "on MCC Scores in a 4 layer Mamba-DNA Model (Matched by Seed)} \\label{table:paired_task_tests}")
    latex_lines.append("\\begin{tabular}{@{}lcrlr@{}}")
    latex_lines.append("\\toprule")
    latex_lines.append("\\textbf{Task} & \\textbf{n} & \\textbf{t-stat} & \\textbf{p-value} & "
                      "\\textbf{\\begin{tabular}[c]{@{}c@{}}Mean diff \\\\ (char - bpe)\\end{tabular}} \\\\ \\midrule")
    
    current_category = None
    for result in task_results:
        # Category header
        if result['category'] != current_category:
            if current_category is not None:
                latex_lines.append("\\midrule")
            current_category = result['category']
            category_display = current_category.replace('_', ' ')
            latex_lines.append(f"\\multicolumn{{5}}{{c}}{{\\textbf{{{category_display.title()}}}}} \\\\")
            latex_lines.append("\\midrule")
        
        # Format values
        t_stat = f"{result['t_statistic']:.3f}"
        p_val = f"{result['p_value']:.3f}"
        mean_diff = f"{result['mean_diff']:.3f}"
        
        # Add significance marker to p-value
        if result['p_value'] < 0.001:
            p_val += "***"
        elif result['p_value'] < 0.01:
            p_val += "**"
        elif result['p_value'] < 0.05:
            p_val += "*"
        
        # Add Bonferroni marker
        if result['bonferroni_sig']:
            p_val += "†"
        
        # Row coloring based on Bonferroni significance and direction
        row_start = ""
        if result['bonferroni_sig']:
            if result['mean_diff'] > 0:  # char > bpe
                row_start = "\\rowcolor{teal!25}"
            else:  # bpe > char
                row_start = "\\rowcolor{magenta!25}"
        
        # Task name with proper escaping
        task_display = result['task_display'].replace('_', '\\_')
        
        # Build row
        row = f"{task_display} & {result['n_paired']} & {t_stat} & {p_val} & {mean_diff} \\\\"
        if row_start:
            row = row_start + row
        latex_lines.append(row)
    
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\multicolumn{5}{l}{* p $< 0.05$, ** p $< 0.01$, *** p $< 0.001$} \\\\")
    latex_lines.append(f"\\multicolumn{{5}}{{l}}{{† Bonferroni corrected ($\\alpha = {bonferroni_alpha:.4f}$)}} \\\\")
    latex_lines.append("\\multicolumn{5}{l}{\\cellcolor{teal!25} Character tokenization superior (Bonferroni significant)} \\\\")
    latex_lines.append("\\multicolumn{5}{l}{\\cellcolor{magenta!25} BPE tokenization superior (Bonferroni significant)}")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{center}")
    latex_lines.append("\\end{table}")
    
    # Print LaTeX table
    print("\n".join(latex_lines))
    
    # Save LaTeX table to file
    with open('../results_tables/mamba_paired_significance_by_task_table.tex', 'w') as f:
        f.write("\n".join(latex_lines))
    print(f"\nLaTeX table saved to: ../results_tables/mamba_paired_significance_by_task_table.tex")

if __name__ == "__main__":
    calculate_mamba_significance_paired_by_task()