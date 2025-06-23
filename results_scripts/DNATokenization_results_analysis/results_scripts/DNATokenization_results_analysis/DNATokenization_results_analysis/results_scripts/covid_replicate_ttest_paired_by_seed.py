#!/usr/bin/env python3
"""
Extract COVID task replicates for Mamba models and perform paired t-test matching by seed.
"""

import pandas as pd
import numpy as np
from scipy import stats
import os

def analyze_covid_replicates_paired():
    """
    Extract COVID replicates and perform paired t-test on Mamba-char vs Mamba-bpe matching by seed.
    """
    
    print("=" * 80)
    print("COVID Task Paired Replicate Analysis (Matched by Seed)")
    print("=" * 80)
    
    # Read the state space results
    ss_file = '../final_combined_ss_results/best_replicates_ss.csv'
    
    if not os.path.exists(ss_file):
        print(f"Error: File {ss_file} not found!")
        return
    
    # Read the CSV file
    print(f"\nReading file: {ss_file}")
    df = pd.read_csv(ss_file)
    
    # Display column names to understand the structure
    print(f"\nColumns in the file: {df.columns.tolist()}")
    
    # Filter for COVID task and Mamba models
    covid_df = df[(df['task'] == 'covid') & (df['model'].isin(['Mamba-bpe', 'Mamba-char']))]
    
    print(f"\nTotal COVID rows found: {len(covid_df)}")
    
    if len(covid_df) == 0:
        print("No COVID data found for Mamba models!")
        return
    
    # Check what MCC column name is used
    mcc_col = 'mcc' if 'mcc' in covid_df.columns else None
    
    if not mcc_col:
        print("Error: No MCC column found!")
        return
    
    # Create separate dataframes for each model
    mamba_bpe_df = covid_df[covid_df['model'] == 'Mamba-bpe'].copy()
    mamba_char_df = covid_df[covid_df['model'] == 'Mamba-char'].copy()
    
    # Remove NaN values
    mamba_bpe_df = mamba_bpe_df[mamba_bpe_df[mcc_col].notna()]
    mamba_char_df = mamba_char_df[mamba_char_df[mcc_col].notna()]
    
    print(f"\nMamba-bpe: {len(mamba_bpe_df)} valid replicates")
    print(f"Mamba-char: {len(mamba_char_df)} valid replicates")
    
    # Check if seed column exists
    if 'seed' not in covid_df.columns:
        print("\nError: No 'seed' column found in the data!")
        return
    
    # Get unique seeds for each model
    bpe_seeds = set(mamba_bpe_df['seed'].unique())
    char_seeds = set(mamba_char_df['seed'].unique())
    common_seeds = bpe_seeds.intersection(char_seeds)
    
    print(f"\nSeed analysis:")
    print(f"  Mamba-bpe seeds: {sorted(bpe_seeds)}")
    print(f"  Mamba-char seeds: {sorted(char_seeds)}")
    print(f"  Common seeds: {sorted(common_seeds)}")
    print(f"  Number of paired observations: {len(common_seeds)}")
    
    if len(common_seeds) < 2:
        print("\nError: Not enough paired observations for t-test (need at least 2)")
        return
    
    # Create paired data
    paired_data = []
    for seed in sorted(common_seeds):
        bpe_value = mamba_bpe_df[mamba_bpe_df['seed'] == seed][mcc_col].iloc[0]
        char_value = mamba_char_df[mamba_char_df['seed'] == seed][mcc_col].iloc[0]
        
        paired_data.append({
            'seed': seed,
            'mamba_bpe_mcc': bpe_value,
            'mamba_char_mcc': char_value,
            'difference': char_value - bpe_value
        })
    
    paired_df = pd.DataFrame(paired_data)
    
    print("\nPaired data:")
    print(paired_df.to_string(index=False))
    
    # Extract values for t-test
    bpe_values = paired_df['mamba_bpe_mcc'].values
    char_values = paired_df['mamba_char_mcc'].values
    differences = paired_df['difference'].values
    
    # Perform paired t-test
    t_stat, p_value = stats.ttest_rel(char_values, bpe_values)
    
    # Calculate statistics
    mean_bpe = np.mean(bpe_values)
    mean_char = np.mean(char_values)
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)  # Sample standard deviation
    se_diff = std_diff / np.sqrt(len(differences))
    
    # Calculate 95% confidence interval for the mean difference
    t_critical = stats.t.ppf(0.975, len(differences) - 1)
    ci_lower = mean_diff - t_critical * se_diff
    ci_upper = mean_diff + t_critical * se_diff
    
    # Calculate Cohen's d for paired samples
    cohens_d = mean_diff / std_diff if std_diff > 0 else 0
    
    print("\n" + "=" * 80)
    print("Paired T-test Results (Matched by Seed)")
    print("=" * 80)
    print(f"Number of paired observations: {len(paired_df)}")
    print(f"\nMamba-bpe mean MCC: {mean_bpe:.4f}")
    print(f"Mamba-char mean MCC: {mean_char:.4f}")
    print(f"\nMean difference (Char - BPE): {mean_diff:.4f}")
    print(f"Standard deviation of differences: {std_diff:.4f}")
    print(f"Standard error of mean difference: {se_diff:.4f}")
    print(f"95% CI for mean difference: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"\nt-statistic: {t_stat:.4f}")
    print(f"p-value: {p_value:.4f}")
    print(f"Cohen's d (effect size): {cohens_d:.4f}")
    
    # Interpretation
    if p_value < 0.05:
        if mean_diff > 0:
            print(f"\nResult: Mamba-char significantly outperforms Mamba-bpe (p = {p_value:.4f})")
        else:
            print(f"\nResult: Mamba-bpe significantly outperforms Mamba-char (p = {p_value:.4f})")
    else:
        print(f"\nResult: No significant difference between models (p = {p_value:.4f})")
    
    # Effect size interpretation
    if abs(cohens_d) < 0.2:
        effect_size = "negligible"
    elif abs(cohens_d) < 0.5:
        effect_size = "small"
    elif abs(cohens_d) < 0.8:
        effect_size = "medium"
    else:
        effect_size = "large"
    print(f"Effect size is {effect_size}")
    
    # Save results
    results = {
        'task': 'covid',
        'n_paired_observations': len(paired_df),
        'mean_mcc_bpe': mean_bpe,
        'mean_mcc_char': mean_char,
        'mean_difference': mean_diff,
        'std_difference': std_diff,
        'se_difference': se_diff,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'test_type': 'Paired (matched by seed)',
        'significant': 'Yes' if p_value < 0.05 else 'No'
    }
    
    # Save summary to CSV
    results_df = pd.DataFrame([results])
    output_file = '../results_tables/covid_paired_ttest_results.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\nSummary results saved to: {output_file}")
    
    # Save detailed paired data
    paired_output = '../results_tables/covid_paired_data.csv'
    paired_df.to_csv(paired_output, index=False)
    print(f"Detailed paired data saved to: {paired_output}")
    
    # Check for unpaired data
    unpaired_bpe_seeds = bpe_seeds - common_seeds
    unpaired_char_seeds = char_seeds - common_seeds
    
    if unpaired_bpe_seeds or unpaired_char_seeds:
        print("\n" + "=" * 80)
        print("Unpaired Data (excluded from analysis):")
        print("=" * 80)
        if unpaired_bpe_seeds:
            print(f"Mamba-bpe only seeds: {sorted(unpaired_bpe_seeds)}")
            for seed in sorted(unpaired_bpe_seeds):
                value = mamba_bpe_df[mamba_bpe_df['seed'] == seed][mcc_col].iloc[0]
                print(f"  Seed {seed}: MCC = {value:.4f}")
        
        if unpaired_char_seeds:
            print(f"Mamba-char only seeds: {sorted(unpaired_char_seeds)}")
            for seed in sorted(unpaired_char_seeds):
                value = mamba_char_df[mamba_char_df['seed'] == seed][mcc_col].iloc[0]
                print(f"  Seed {seed}: MCC = {value:.4f}")

if __name__ == "__main__":
    analyze_covid_replicates_paired()