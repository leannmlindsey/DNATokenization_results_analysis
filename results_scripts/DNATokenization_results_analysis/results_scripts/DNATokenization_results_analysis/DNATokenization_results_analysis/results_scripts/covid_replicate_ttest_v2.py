#!/usr/bin/env python3
"""
Extract COVID task replicates for Mamba models and perform t-test on individual replicates.
Version 2: Handles NaN values properly
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
    print("COVID Task Replicate-Level Analysis (v2)")
    print("=" * 80)
    
    # Read the state space results
    ss_file = '../final_combined_ss_results/best_replicates_ss.csv'
    
    if not os.path.exists(ss_file):
        print(f"Error: File {ss_file} not found!")
        print("\nLooking for alternative files...")
        
        # Check what files exist
        ss_dir = '../final_combined_ss_results/'
        if os.path.exists(ss_dir):
            files = [f for f in os.listdir(ss_dir) if f.endswith('.csv')]
            print(f"CSV files in {ss_dir}:")
            for f in files:
                print(f"  - {f}")
        
        # Also check for the main combined file
        alt_file = '../final_combined_ss_results/combined_ss_results_cleaned.csv'
        if os.path.exists(alt_file):
            print(f"\nUsing alternative file: {alt_file}")
            ss_file = alt_file
        else:
            return
    
    # Read the CSV file
    print(f"\nReading file: {ss_file}")
    df = pd.read_csv(ss_file)
    
    # Display column names to understand the structure
    print(f"\nColumns in the file: {df.columns.tolist()}")
    print(f"Total rows in file: {len(df)}")
    
    # Check unique tasks
    print(f"\nUnique tasks in file: {sorted(df['task'].unique())}")
    
    # Filter for COVID task and Mamba models
    covid_df = df[(df['task'] == 'covid') & (df['model'].isin(['Mamba-bpe', 'Mamba-char']))]
    
    print(f"\nTotal COVID rows found: {len(covid_df)}")
    
    if len(covid_df) == 0:
        print("No COVID data found for Mamba models!")
        return
    
    # Check what MCC column name is used
    mcc_columns = [col for col in covid_df.columns if 'mcc' in col.lower()]
    print(f"\nMCC-related columns found: {mcc_columns}")
    
    # Use the appropriate MCC column
    mcc_col = 'mcc' if 'mcc' in covid_df.columns else mcc_columns[0] if mcc_columns else None
    
    if not mcc_col:
        print("Error: No MCC column found!")
        return
    
    # Separate by model and remove NaN values
    mamba_bpe_df = covid_df[covid_df['model'] == 'Mamba-bpe']
    mamba_char_df = covid_df[covid_df['model'] == 'Mamba-char']
    
    # Extract non-NaN MCC values
    mamba_bpe_data = mamba_bpe_df[mcc_col].dropna().values
    mamba_char_data = mamba_char_df[mcc_col].dropna().values
    
    print(f"\nMamba-bpe:")
    print(f"  Total rows: {len(mamba_bpe_df)}")
    print(f"  Valid MCC values: {len(mamba_bpe_data)}")
    print(f"  NaN values: {len(mamba_bpe_df) - len(mamba_bpe_data)}")
    if len(mamba_bpe_data) > 0:
        print(f"  MCC values: {mamba_bpe_data}")
        print(f"  Mean ± SD: {np.mean(mamba_bpe_data):.4f} ± {np.std(mamba_bpe_data):.4f}")
        print(f"  Min: {np.min(mamba_bpe_data):.4f}, Max: {np.max(mamba_bpe_data):.4f}")
    
    print(f"\nMamba-char:")
    print(f"  Total rows: {len(mamba_char_df)}")
    print(f"  Valid MCC values: {len(mamba_char_data)}")
    print(f"  NaN values: {len(mamba_char_df) - len(mamba_char_data)}")
    if len(mamba_char_data) > 0:
        print(f"  MCC values: {mamba_char_data}")
        print(f"  Mean ± SD: {np.mean(mamba_char_data):.4f} ± {np.std(mamba_char_data):.4f}")
        print(f"  Min: {np.min(mamba_char_data):.4f}, Max: {np.max(mamba_char_data):.4f}")
    
    # Check if we have enough data for t-test
    if len(mamba_bpe_data) < 2 or len(mamba_char_data) < 2:
        print("\nError: Not enough valid data points for t-test (need at least 2 per group)")
        return
    
    # Perform unpaired t-test (since we can't guarantee pairing with NaN values)
    t_stat, p_value = stats.ttest_ind(mamba_char_data, mamba_bpe_data)
    
    # Calculate effect size (Cohen's d)
    mean_diff = np.mean(mamba_char_data) - np.mean(mamba_bpe_data)
    pooled_sd = np.sqrt((np.std(mamba_char_data)**2 + np.std(mamba_bpe_data)**2) / 2)
    cohens_d = mean_diff / pooled_sd if pooled_sd > 0 else 0
    
    print("\n" + "=" * 80)
    print("T-test Results (Independent samples)")
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
    
    # Save detailed results
    results = {
        'task': 'covid',
        'n_valid_replicates_bpe': len(mamba_bpe_data),
        'n_total_replicates_bpe': len(mamba_bpe_df),
        'n_valid_replicates_char': len(mamba_char_data),
        'n_total_replicates_char': len(mamba_char_df),
        'mean_mcc_bpe': np.mean(mamba_bpe_data) if len(mamba_bpe_data) > 0 else np.nan,
        'std_mcc_bpe': np.std(mamba_bpe_data) if len(mamba_bpe_data) > 0 else np.nan,
        'mean_mcc_char': np.mean(mamba_char_data) if len(mamba_char_data) > 0 else np.nan,
        'std_mcc_char': np.std(mamba_char_data) if len(mamba_char_data) > 0 else np.nan,
        'mean_difference': mean_diff,
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'test_type': 'Independent samples',
        'significant': 'Yes' if p_value < 0.05 else 'No'
    }
    
    # Save to CSV
    results_df = pd.DataFrame([results])
    output_file = '../results_tables/covid_replicate_ttest_results_v2.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    # Also display the run names to understand what data we have
    print("\n" + "=" * 80)
    print("Run details for valid data:")
    print("=" * 80)
    
    print("\nMamba-bpe runs with valid MCC:")
    valid_bpe = mamba_bpe_df[mamba_bpe_df[mcc_col].notna()]
    for _, row in valid_bpe.iterrows():
        print(f"  - {row['run_name']}: MCC={row[mcc_col]:.4f}")
    
    print("\nMamba-char runs with valid MCC:")
    valid_char = mamba_char_df[mamba_char_df[mcc_col].notna()]
    for _, row in valid_char.iterrows():
        print(f"  - {row['run_name']}: MCC={row[mcc_col]:.4f}")

if __name__ == "__main__":
    analyze_covid_replicates()