import pandas as pd
import numpy as np

def combine_csv_files():
    """
    Combine five CSV files with different column structures.
    Missing columns will be filled with NaN values.
    """
    
    # Read the CSV files
    print("Reading CSV files...")
    df1 = pd.read_csv('../attn_results/combined_results_repeated_07.21.2025.csv')
    df2 = pd.read_csv('../attn_results/combined_results_orig_07.21.2025.csv')
    df3 = pd.read_csv('../attn_results/anisa_finetune_results_with_lr_06.11.2025.csv')
    df4 = pd.read_csv('../attn_results/covid_repeated.csv')
    df5 = pd.read_csv('../attn_results/dnabert1_bridges_gb_anisa_replicates.csv')
    
    # Fix incorrect benchmark labeling in df5 (all should be GB, not GUE)
    df5['benchmark'] = df5['benchmark'].replace('GUE', 'GB')
    print(f"Fixed benchmark labeling in df5: GUE -> GB")
    
    print(f"File 1 shape: {df1.shape}")
    print(f"File 2 shape: {df2.shape}")
    print(f"File 3 shape before filtering: {df3.shape}")
    print(f"File 4 shape: {df4.shape}")
    print(f"File 5 shape: {df5.shape}")
    
    # Add source identifier to track which file each row came from
    df1['source_file'] = 'combined_results_repeated_07.21.2025.csv'
    df2['source_file'] = 'combined_results_orig_07.21.2025.csv'
    df3['source_file'] = 'anisa_finetune_results_with_lr_06.11.2025.csv'
    df4['source_file'] = 'covid_repeated.csv'
    df5['source_file'] = 'dnabert1_bridges_gb_anisa_replicates.csv'
    
    # Clean benchmark column in df1 and df2
    df1['benchmark'] = df1['benchmark'].replace({
        'GB_repeated': 'GB',
        'GB_repated_anisacodebase': 'GB',
        'GB_repeated_anisacodebase': 'GB'
    })
    df2['benchmark'] = df2['benchmark'].replace({
        'GB_repeated': 'GB',
        'GB_repated_anisacodebase': 'GB',
        'GB_repeated_anisacodebase': 'GB'
    })
    
    # Clean model names in df1 and df2
    df1['model'] = df1['model'].replace({
        'DNABERT2_repeated': 'DNABERT2'
    })
    df2['model'] = df2['model'].replace({
        'DNABERT2_repeated': 'DNABERT2'
    })
    
    # Drop covid task from df1 and df2 only for DNABERT2 (use only df4 for DNABERT2 covid)
    df1 = df1[~((df1['task'] == 'covid') & (df1['model'] == 'DNABERT2'))]
    df2 = df2[~((df2['task'] == 'covid') & (df2['model'] == 'DNABERT2'))]
    
    # Process df3: rename task_benchmark to benchmark and clean values
    df3_renamed = df3.rename(columns={
        'task_benchmark': 'benchmark'  # Map task_benchmark to benchmark
    })
    
    # Clean benchmark values in df3
    df3_renamed['benchmark'] = df3_renamed['benchmark'].replace({
        'Genomic Benchmark': 'GB',
        'Nucleotide Transformer': 'NTv1'
    })
    
    # Clean model names in df3
    df3_renamed['model'] = df3_renamed['model'].replace({
        'DNABERT-1 (6-mer)': 'DNABERT',
        'DNABERT-2': 'DNABERT2'
    })
    
    # Clean model names in df4 (covid file)
    df4['model'] = df4['model'].replace({
        'dnabert2': 'DNABERT2'
    })
    
    # Drop rows with specific models from df3
    models_to_drop = ['HyenaDNA (1k)', 'NT 500M 1000G']
    df3_renamed = df3_renamed[~df3_renamed['model'].isin(models_to_drop)]
    
    # Drop rows containing 'GPT' in the model name
    df3_renamed = df3_renamed[~df3_renamed['model'].str.contains('GPT', case=False, na=False)]
    
    # Drop Anisa's results for NTv2 benchmark (she didn't use revised input files)
    df3_renamed = df3_renamed[df3_renamed['benchmark'] != 'NTv2']
    
    # Drop Anisa's results for specific tasks that have updated versions in NTv2
    anisa_tasks_to_drop = [
        'H2AFZ', 'H3', 'H3K14ac', 'H3K27ac', 'H3K27me3', 'H3K36me3',
        'H3K4me1', 'H3K4me2', 'H3K4me3', 'H3K79me3', 'H3K9ac', 'H3K9me3',
        'H4', 'H4ac', 'H4K20me1'
    ]
    df3_renamed = df3_renamed[~df3_renamed['task'].isin(anisa_tasks_to_drop)]
    
    # Report on filtering results
    print(f"File 3 shape after filtering: {df3_renamed.shape}")
    print(f"Rows dropped from File 3: {df3.shape[0] - df3_renamed.shape[0]}")
    
    # Debug: Check for NaN in task columns after all processing
    print("\n=== DEBUGGING: Task columns after all processing ===")
    print(f"df1 task NaN count after processing: {df1['task'].isna().sum()}")
    print(f"df2 task NaN count after processing: {df2['task'].isna().sum()}")
    print(f"df3_renamed task NaN count after processing: {df3_renamed['task'].isna().sum()}")
    print(f"df4 task NaN count after processing: {df4['task'].isna().sum()}")
    
    # Check if filtering removed rows and affected task column
    print(f"df3 original task unique: {df3['task'].unique()}")
    print(f"df3_renamed filtered task unique: {df3_renamed['task'].unique()}")
    print(f"df4 task unique: {df4['task'].unique()}")
    
    # Verify no NaN values were introduced
    if df1['task'].isna().any() or df2['task'].isna().any() or df3_renamed['task'].isna().any() or df4['task'].isna().any():
        print("⚠️  WARNING: NaN values found in task columns after processing!")
        print("This should not happen - investigating...")
        
        # Check for empty rows or corrupted data
        print(f"df1 empty rows: {df1.isnull().all(axis=1).sum()}")
        print(f"df2 empty rows: {df2.isnull().all(axis=1).sum()}")  
        print(f"df3_renamed empty rows: {df3_renamed.isnull().all(axis=1).sum()}")
        print(f"df4 empty rows: {df4.isnull().all(axis=1).sum()}")
    else:
        print("✓ No NaN values in task columns after processing")
    print("=" * 60)
    
    # Drop task_category column from df3
    if 'task_category' in df3_renamed.columns:
        df3_renamed = df3_renamed.drop('task_category', axis=1)
    
    # Add seed column to Anisa's data (she uses seed 42 for all experiments)
    df3_renamed['seed'] = 42
    
    # Define the complete set of columns (union of all columns)
    all_columns = [
        'directory_name', 'model', 'lr', 'benchmark', 'task', 'seed',
        'accuracy', 'mcc', 'f1', 'precision', 'recall', 'runtime',
        'replicate_number', 'sequence_length', 'epoch', 'source_file'
    ]
    
    # Add missing columns to each dataframe with NaN values
    def add_missing_columns(df, target_columns):
        """Add missing columns with NaN values"""
        for col in target_columns:
            if col not in df.columns:
                df[col] = np.nan
        return df[target_columns]  # Reorder columns to match target order
    
    # Standardize all dataframes to have the same columns
    df1_standard = add_missing_columns(df1, all_columns)
    df2_standard = add_missing_columns(df2, all_columns)
    df3_standard = add_missing_columns(df3_renamed, all_columns)
    df4_standard = add_missing_columns(df4, all_columns)
    df5_standard = add_missing_columns(df5, all_columns)
    
    # Combine all dataframes
    print("\nCombining dataframes...")
    combined_df = pd.concat([df1_standard, df2_standard, df3_standard, df4_standard, df5_standard], 
                           ignore_index=True, sort=False)
    
    # Debug: Check task column in combined dataframe
    print(f"\n=== DEBUGGING: Combined dataframe task column ===")
    print(f"Combined task NaN count: {combined_df['task'].isna().sum()}")
    print(f"Combined task unique values: {combined_df['task'].unique()}")
    
    if combined_df['task'].isna().any():
        print("⚠️  NaN values found in combined task column!")
        print("Rows with NaN tasks:")
        nan_task_rows = combined_df[combined_df['task'].isna()]
        print(nan_task_rows[['source_file', 'model', 'task', 'benchmark']].head(10))
    else:
        print("✓ No NaN values in combined task column")
    print("=" * 60)
    
    # Drop rows with NaN in task or mcc columns
    print(f"\nBefore filtering - Combined dataframe shape: {combined_df.shape}")
    rows_before = len(combined_df)
    
    combined_df = combined_df.dropna(subset=['task', 'mcc'])
    
    rows_after = len(combined_df)
    rows_dropped = rows_before - rows_after
    print(f"After filtering - Combined dataframe shape: {combined_df.shape}")
    print(f"Dropped {rows_dropped} rows with NaN in task or mcc columns")
    
    # Display summary information
    print(f"\nTotal rows: {len(combined_df)}")
    print(f"Columns: {list(combined_df.columns)}")
    
    # Show data completeness by source
    print("\nData completeness by source file:")
    completeness = combined_df.groupby('source_file').apply(
        lambda x: (x.notna().sum() / len(x) * 100).round(2)
    )
    print(completeness)
    
    # Show sample of missing values by column
    print("\nMissing values by column:")
    missing_counts = combined_df.isnull().sum()
    missing_pct = (missing_counts / len(combined_df) * 100).round(2)
    missing_summary = pd.DataFrame({
        'Missing_Count': missing_counts,
        'Missing_Percentage': missing_pct
    })
    print(missing_summary[missing_summary['Missing_Count'] > 0])
    
    # Save the combined results
    output_filename = '../final_combined_attn_results/combined_all_results_06.11.2025.csv'
    combined_df.to_csv(output_filename, index=False)
    print(f"\nCombined results saved to: {output_filename}")
    
    # Display first few rows as preview
    print("\nPreview of combined data:")
    print(combined_df.head())
    
    return combined_df

# Additional utility function to explore the data
def explore_combined_data(df):
    """
    Explore the combined dataset to understand the data better
    """
    print("=== DATA EXPLORATION ===")
    
    # Unique values in key columns
    print(f"\nUnique models: {df['model'].nunique()}")
    print(f"Models: {sorted(df['model'].unique())}")
    
    print(f"\nUnique benchmarks: {df['benchmark'].nunique()}")
    # Handle NaN values in benchmark column for sorting  
    benchmark_values = df['benchmark'].dropna().unique()
    print(f"Benchmarks: {sorted(benchmark_values)}")
    if df['benchmark'].isna().any():
        print(f"Note: {df['benchmark'].isna().sum()} rows have NaN benchmark values")
    
    print(f"\nUnique tasks: {df['task'].nunique()}")
    # Handle NaN values in task column for sorting
    task_values = df['task'].dropna().unique()
    print(f"Tasks: {sorted(task_values)}")
    if df['task'].isna().any():
        print(f"Note: {df['task'].isna().sum()} rows have NaN task values")
    
    # Performance metrics summary
    print(f"\nPerformance metrics summary:")
    metrics = ['accuracy', 'f1', 'mcc', 'precision', 'recall']
    for metric in metrics:
        if metric in df.columns and df[metric].notna().any():
            print(f"{metric}: mean={df[metric].mean():.3f}, "
                  f"std={df[metric].std():.3f}, "
                  f"count={df[metric].count()}")

if __name__ == "__main__":
    # Combine the CSV files
    combined_data = combine_csv_files()
    
    # Explore the combined data
    explore_combined_data(combined_data)
    
    print("\n=== COLUMN MAPPING SUMMARY ===")
    print("Files 1&2 → File 3 mappings:")
    print("- task_benchmark → benchmark")
    print("- Added seed=42 to File 3 (Anisa's standard)")
    print("- Dropped task_category from File 3")
    print("Benchmark standardization:")
    print("- Files 1&2: 'GB_repeated', 'GB_repated_anisacodebase', 'GB_repeated_anisacodebase' → 'GB'")
    print("- File 3: 'Genomic Benchmark' → 'GB', 'Nucleotide Transformer' → 'NTv1'")
    print("Model standardization:")
    print("- Files 1&2: 'DNABERT2_repeated' → 'DNABERT2'")
    print("- File 3: 'DNABERT-1 (6-mer)' → 'DNABERT', 'DNABERT-2' → 'DNABERT2'")
    print("- File 3: Dropped rows with models containing 'GPT', 'HyenaDNA (1k)', 'NT 500M 1000G'")
    print("- Common: model, task, accuracy, f1, mcc")
    print("- File 3 unique: replicate_number, sequence_length, epoch")
    print("- Files 1&2 unique: directory_name, lr, precision, recall, runtime")
