import pandas as pd
import numpy as np

def clean_ss_results():
    """
    Clean state space results by removing hyperparameter combinations where 
    all seeds failed (have NA for both accuracy and mcc).
    """
    
    # Read the state space results
    df = pd.read_csv('../ss_results/combined_ss_results_header.csv')
    
    print(f"Initial dataset: {len(df)} rows")
    
    # Standardize task names to match attention model naming
    #task_name_mapping = {
    #    'demo_coding_vs_intergenomic_seqs': 'demo_coding_vs_intergenomic'
    #}
    
    # Apply task name mapping
    #df['task'] = df['task'].replace(task_name_mapping)
    
    # Report any task name changes
    #for old_name, new_name in task_name_mapping.items():
    #    count = len(df[df['task'] == new_name])
    #    if count > 0:
    #        print(f"Renamed task: {old_name} → {new_name} ({count} rows)")
    
    # Rename 'model' column to 'run_name'
    df = df.rename(columns={'model': 'run_name'})
    
    # Create 'model' column based on run_name
    def extract_model_name(run_name):
        run_name_lower = run_name.lower()
        if 'caduceus' in run_name_lower:
            return 'Caduceus'
        elif 'hyena' in run_name_lower:
            return 'HyenaDNA'
        elif 'mamba' in run_name_lower and 'char' in run_name_lower:
            return 'Mamba-char'
        elif 'mamba' in run_name_lower and 'bpe' in run_name_lower:
            return 'Mamba-bpe'
        elif 'cnn' in run_name_lower:
            return 'CNN'
        else:
            print(f"WARNING: Unknown model type for run_name: {run_name}")
            return 'Unknown'
    
    df['model'] = df['run_name'].apply(extract_model_name)
    
    # Extract rc-aug hyperparameter from run_name
    def extract_rc_aug(run_name):
        if 'rc_aug-true' in run_name.lower():
            return True
        elif 'rc_aug-false' in run_name.lower():
            return False
        else:
            return None  # Not specified in run_name
    
    df['rc_aug'] = df['run_name'].apply(extract_rc_aug)
    
    # Replace 'N/A' strings and empty strings with NaN
    df['accuracy'] = df['accuracy'].replace(['N/A', ''], np.nan)
    df['mcc'] = df['mcc'].replace(['N/A', ''], np.nan)
    
    # Convert to numeric
    df['accuracy'] = pd.to_numeric(df['accuracy'], errors='coerce')
    df['mcc'] = pd.to_numeric(df['mcc'], errors='coerce')
    
    # Drop all rows where task starts with 'emp_'
    emp_tasks = df[df['task'].str.startswith('emp_', na=False)]
    emp_task_count = len(emp_tasks)
    emp_task_names = emp_tasks['task'].unique()
    df = df[~df['task'].str.startswith('emp_', na=False)]
    
    if emp_task_count > 0:
        print(f"\nDropped {emp_task_count} rows with emp_ tasks:")
        for task in sorted(emp_task_names):
            task_count = len(emp_tasks[emp_tasks['task'] == task])
            print(f"  {task}: {task_count} rows")
    
    # Group by task, run_name (which uniquely identifies the model configuration)
    # This prevents grouping different model architectures together
    grouping_cols = ['task', 'run_name']
    
    # Find groups where ALL seeds have NaN for both accuracy and mcc
    def all_seeds_failed(group):
        """Check if all seeds in this group failed (both acc and mcc are NaN for all rows)"""
        both_nan = group['accuracy'].isna() & group['mcc'].isna()
        return both_nan.all()
    
    # Identify failed groups
    failed_groups = df.groupby(grouping_cols).apply(all_seeds_failed)
    failed_combinations = failed_groups[failed_groups].index
    
    print(f"Found {len(failed_combinations)} failed hyperparameter combinations:")
    for combo in failed_combinations:
        task, run_name = combo
        # Get additional info for display
        combo_data = df[(df['task'] == task) & (df['run_name'] == run_name)].iloc[0]
        model = combo_data['model']
        lr = combo_data['learning_rate']
        batch_size = combo_data['batch_size']
        rc_aug = combo_data['rc_aug']
        print(f"  {task} | {run_name} | {model} | lr={lr} | batch_size={batch_size} | rc_aug={rc_aug}")
    
    # Create a mask to identify rows that belong to failed combinations
    failed_mask = pd.Series(False, index=df.index)
    for combo in failed_combinations:
        task, run_name = combo
        mask = ((df['task'] == task) & 
                (df['run_name'] == run_name))
        failed_mask |= mask
    
    # Remove failed combinations
    df_cleaned = df[~failed_mask].copy()
    
    print(f"\nCleaned dataset: {len(df_cleaned)} rows")
    print(f"Removed {len(df) - len(df_cleaned)} rows from failed combinations")
    
    # Save cleaned results
    output_file = '../final_combined_ss_results/combined_ss_results_cleaned.csv'
    df_cleaned.to_csv(output_file, index=False)
    
    print(f"\nCleaned results saved to: {output_file}")
    
    # Summary of remaining data
    print(f"\n=== CLEANED DATA SUMMARY ===")
    print(f"Tasks: {len(df_cleaned['task'].unique())}")
    print(f"Models: {df_cleaned['model'].unique()}")
    print(f"Learning rates: {sorted(df_cleaned['learning_rate'].unique())}")
    print(f"Batch sizes: {sorted(df_cleaned['batch_size'].unique())}")
    print(f"RC-aug values: {sorted([x for x in df_cleaned['rc_aug'].unique() if x is not None])}")
    
    # Show rc_aug distribution
    print(f"\nRC-aug distribution:")
    rc_aug_counts = df_cleaned['rc_aug'].value_counts(dropna=False)
    for rc_aug, count in rc_aug_counts.items():
        print(f"  {rc_aug}: {count} rows")
    
    # Show model mapping results
    print(f"\nModel mapping from run_name:")
    model_counts = df_cleaned['model'].value_counts()
    for model, count in model_counts.items():
        print(f"  {model}: {count} rows")
    
    # Check for unknown models
    unknown_models = df_cleaned[df_cleaned['model'] == 'Unknown']['run_name'].unique()
    if len(unknown_models) > 0:
        print(f"\nWARNING: Found {len(unknown_models)} run_names that couldn't be mapped:")
        for run_name in unknown_models[:5]:  # Show first 5
            print(f"  {run_name}")
    
    # Check for remaining NaN values
    remaining_acc_nan = df_cleaned['accuracy'].isna().sum()
    remaining_mcc_nan = df_cleaned['mcc'].isna().sum()
    print(f"\nRemaining NaN values:")
    print(f"  Accuracy: {remaining_acc_nan}")
    print(f"  MCC: {remaining_mcc_nan}")
    
    return df_cleaned

if __name__ == "__main__":
    cleaned_data = clean_ss_results()