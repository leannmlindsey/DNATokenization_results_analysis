#!/usr/bin/env python3
"""
Extract COVID task replicates for Mamba models and perform t-test on individual replicates.
"""

import pandas as pd
import numpy as np
from scipy import stats
import os

def analyze_covid_replicates():
    """
    Extract COVID replicates and perform paired t-test on Mamba-char vs Mamba-bpe.
    """
    
    print("=" * 80)
    print("COVID Task Replicate-Level Analysis")
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
    
    # Display the data structure
    print("\nFirst few rows of COVID data:")
    print(covid_df.head())
    
    # Check what MCC column name is used
    mcc_columns = [col for col in covid_df.columns if 'mcc' in col.lower()]
    print(f"\nMCC-related columns found: {mcc_columns}")
    
    # Use the appropriate MCC column
    mcc_col = 'mcc' if 'mcc' in covid_df.columns else mcc_columns[0] if mcc_columns else None
    
    if not mcc_col:
        print("Error: No MCC column found!")
        return
    
    # Extract replicates for each model
    mamba_bpe_data = covid_df[covid_df['model'] == 'Mamba-bpe'][mcc_col].values
    mamba_char_data = covid_df[covid_df['model'] == 'Mamba-char'][mcc_col].values
    
    print(f"\nMamba-bpe replicates: {len(mamba_bpe_data)}")
    print(f"MCC values: {mamba_bpe_data}")
    print(f"Mean ± SD: {np.mean(mamba_bpe_data):.4f} ± {np.std(mamba_bpe_data):.4f}")
    
    print(f"\nMamba-char replicates: {len(mamba_char_data)}")
    print(f"MCC values: {mamba_char_data}")
    print(f"Mean ± SD: {np.mean(mamba_char_data):.4f} ± {np.std(mamba_char_data):.4f}")
    
    # Check if we have the same number of replicates
    if len(mamba_bpe_data) != len(mamba_char_data):
        print(f"\nWarning: Different number of replicates! BPE: {len(mamba_bpe_data)}, Char: {len(mamba_char_data)}")
        print("Performing unpaired t-test instead...")
        
        # Unpaired t-test
        t_stat, p_value = stats.ttest_ind(mamba_char_data, mamba_bpe_data)
        test_type = "Unpaired (Independent samples)"
    else:
        print(f"\nSame number of replicates ({len(mamba_bpe_data)}). Performing paired t-test...")
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(mamba_char_data, mamba_bpe_data)
        test_type = "Paired"
    
    # Calculate effect size (Cohen's d)
    mean_diff = np.mean(mamba_char_data) - np.mean(mamba_bpe_data)
    pooled_sd = np.sqrt((np.std(mamba_char_data)**2 + np.std(mamba_bpe_data)**2) / 2)
    cohens_d = mean_diff / pooled_sd if pooled_sd > 0 else 0
    
    print("\n" + "=" * 80)
    print(f"T-test Results ({test_type})")
    print("=" * 80)
    print(f"Mean difference (Char - BPE): {mean_diff:.4f}")
    print(f"t-statistic: {t_stat:.4f}")
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
    
    # Also check if there's attention model data for comparison
    print("\n" + "=" * 80)
    print("Checking attention model results for comparison...")
    print("=" * 80)
    
    attn_file = '../final_combined_attn_results/best_replicates.csv'
    if os.path.exists(attn_file):
        attn_df = pd.read_csv(attn_file)
        covid_attn = attn_df[attn_df['task'] == 'covid']
        if len(covid_attn) > 0:
            print(f"\nFound {len(covid_attn)} COVID entries in attention results")
            print("Models present:", covid_attn['model'].unique())
        else:
            print("\nNo COVID data found in attention results")
    else:
        print(f"\nAttention results file not found: {attn_file}")
    
    # Save detailed results
    results = {
        'task': 'covid',
        'n_replicates_bpe': len(mamba_bpe_data),
        'n_replicates_char': len(mamba_char_data),
        'mean_mcc_bpe': np.mean(mamba_bpe_data),
        'std_mcc_bpe': np.std(mamba_bpe_data),
        'mean_mcc_char': np.mean(mamba_char_data),
        'std_mcc_char': np.std(mamba_char_data),
        'mean_difference': mean_diff,
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'test_type': test_type,
        'significant': 'Yes' if p_value < 0.05 else 'No'
    }
    
    # Save to CSV
    results_df = pd.DataFrame([results])
    output_file = '../results_tables/covid_replicate_ttest_results.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    analyze_covid_replicates()