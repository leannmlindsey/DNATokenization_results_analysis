import pandas as pd
import numpy as np
from collections import Counter
import logging
import os
from datetime import datetime

def find_best_hyperparameters_ss():
    """
    Find the best hyperparameter combination for each state space model on each task based on MCC score.
    Hyperparameters: learning_rate and batch_size
    """
    
    # Set up logging
    log_dir = '../final_combined_ss_results'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'hyperparameter_selection_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting hyperparameter selection for state space models")
    
    # Read the cleaned state space results
    df = pd.read_csv('../final_combined_ss_results/combined_ss_results_cleaned.csv')
    
    # Standardize task names
    df['task'] = df['task'].replace({
        'demo_coding_vs_intergenomic_seqs': 'demo_coding_vs_intergenomic',
        'dummy_mouse_enhancers_ensembl': 'dummy_mouse_enhancers',
        'virus_covid': 'covid',  # Standardize to match attention models
        'splice_reconstructed': 'reconstructed',  # Standardize to match attention models
        'drosophila_enhancers_stark': 'drosophilia_enhancers',  # Note: attention models use misspelling
        'human_tf_0': 'tf_0',
        'human_tf_1': 'tf_1',
        'human_tf_2': 'tf_2',
        'human_tf_3': 'tf_3',
        'human_tf_4': 'tf_4'
    })
    
    print(f"Loaded dataset with {len(df)} rows and {len(df['task'].unique())} unique tasks")
    
    # Define hyperparameter columns for state space models
    hyperparam_cols = ['learning_rate', 'batch_size', 'rc_aug']
    
    # Group by task and model combination
    results = []
    best_replicates_list = []
    all_nan_analysis = []
    all_hyperparams_analysis = []  # New: store all hyperparameter combinations
    
    for task in sorted(df['task'].unique()):
        task_data = df[df['task'] == task].copy()
        
        print(f"Task: {task}")
        
        # Process each model separately for this task
        for model in sorted(task_data['model'].unique()):
            model_data = task_data[task_data['model'] == model].copy()
            
            # Store original data for NaN analysis
            model_data_with_nans = model_data.copy()
            
            # Filter out rows with NaN MCC values for best hyperparameter selection
            nan_rows = model_data['mcc'].isna().sum()
            model_data = model_data[model_data['mcc'].notna()]
            
            if nan_rows > 0:
                print(f"  Note: Filtered out {nan_rows} rows with NaN MCC values for {model}")
            
            # Skip models with missing hyperparameter data
            missing_hyperparam = model_data[hyperparam_cols].isnull().any(axis=1).sum()
            if missing_hyperparam > 0:
                print(f"  Skipping model '{model}': {missing_hyperparam} rows with missing hyperparameters")
                continue
                
            # First, count NaN values for each hyperparameter combination (using original data with NaNs)
            nan_analysis = []
            for (lr, bs, rc_aug), group in model_data_with_nans.groupby(hyperparam_cols):
                total_seeds = len(group)
                mcc_nans = group['mcc'].isna().sum()
                acc_nans = group['accuracy'].isna().sum()
                
                nan_analysis.append({
                    'task': task,
                    'model': model,
                    'learning_rate': lr,
                    'batch_size': bs,
                    'rc_aug': rc_aug,
                    'total_seeds': total_seeds,
                    'mcc_nans': mcc_nans,
                    'acc_nans': acc_nans,
                    'mcc_success_rate': (total_seeds - mcc_nans) / total_seeds,
                    'acc_success_rate': (total_seeds - acc_nans) / total_seeds
                })
                
                if mcc_nans > 0 or acc_nans > 0:
                    print(f"    WARNING: {model} lr={lr} bs={bs} rc_aug={rc_aug} has {mcc_nans} MCC NaNs, {acc_nans} Acc NaNs out of {total_seeds} seeds")
            
            # Add to overall NaN analysis
            all_nan_analysis.extend(nan_analysis)
            
            # Group by hyperparameter combination and calculate statistics (excluding NaN values)
            grouped = model_data.groupby(hyperparam_cols).agg({
                'mcc': ['mean', 'std', 'count'],
                'accuracy': ['mean', 'std'],
                'f1': ['mean', 'std'],
                'precision': ['mean', 'std'],
                'recall': ['mean', 'std']
            }).reset_index()
            
            # Check if grouping resulted in empty dataframe
            if len(grouped) == 0:
                print(f"  Skipping model '{model}': No valid hyperparameter combinations found")
                continue
            
            # Flatten column names
            grouped.columns = [
                'learning_rate', 'batch_size', 'rc_aug',
                'mcc_mean', 'mcc_std', 'replicate_count',
                'accuracy_mean', 'accuracy_std',
                'f1_mean', 'f1_std',
                'precision_mean', 'precision_std',
                'recall_mean', 'recall_std'
            ]
            
            # Handle NaN standard deviations (when only 1 replicate)
            grouped['mcc_std'] = grouped['mcc_std'].fillna(0)
            grouped['accuracy_std'] = grouped['accuracy_std'].fillna(0)
            grouped['f1_std'] = grouped['f1_std'].fillna(0)
            grouped['precision_std'] = grouped['precision_std'].fillna(0)
            grouped['recall_std'] = grouped['recall_std'].fillna(0)
            
            # Check if this is a Mamba model that requires stricter filtering
            is_mamba_model = model in ['Mamba-char', 'Mamba-bpe']
            
            if is_mamba_model:
                # For Mamba models, require at least 10 replicates
                grouped_10_replicates = grouped[grouped['replicate_count'] >= 10]
                
                if len(grouped_10_replicates) > 0:
                    grouped_filtered = grouped_10_replicates
                    logger.info(f"  {model}: Found {len(grouped_filtered)} combinations with >= 10 replicates (required for Mamba models)")
                else:
                    # Skip this model if no combinations have at least 10 replicates
                    logger.error(f"  {model}: SKIPPING - No hyperparameter combinations with >= 10 replicates (required for Mamba models)")
                    print(f"  ERROR: {model} - No hyperparameter combinations with >= 10 replicates. Skipping this model (10 replicates required for Mamba models).")
                    continue
            else:
                # For non-Mamba models, use the original logic
                # First try to find combinations with at least 10 replicates
                grouped_10_replicates = grouped[grouped['replicate_count'] >= 10]
                
                if len(grouped_10_replicates) > 0:
                    # Use combinations with at least 10 replicates
                    grouped_filtered = grouped_10_replicates
                    logger.info(f"  {model}: Found {len(grouped_filtered)} combinations with >= 10 replicates")
                else:
                    # Fall back to combinations with at least 3 replicates
                    grouped_3_replicates = grouped[grouped['replicate_count'] >= 3]
                    
                    if len(grouped_3_replicates) > 0:
                        grouped_filtered = grouped_3_replicates
                        logger.warning(f"  {model}: No combinations with >= 10 replicates found. Using {len(grouped_filtered)} combinations with >= 3 replicates")
                        print(f"  WARNING: {model} - No hyperparameter combinations with >= 10 replicates. Using combinations with >= 3 replicates instead.")
                    else:
                        # Use all combinations if none meet minimum requirements
                        grouped_filtered = grouped
                        logger.warning(f"  {model}: No combinations with >= 3 replicates found. Using all {len(grouped_filtered)} combinations")
                        print(f"  WARNING: {model} - No hyperparameter combinations with >= 3 replicates. Using all available combinations.")
            
            # Check if any combinations are available
            if len(grouped_filtered) == 0:
                logger.error(f"  Skipping model '{model}': No hyperparameter combinations found")
                print(f"  Skipping model '{model}': No hyperparameter combinations found")
                continue
            
            # Find best combination for this model (highest MCC, tie-break with accuracy)
            # First, sort by MCC in descending order
            grouped_sorted = grouped_filtered.sort_values('mcc_mean', ascending=False)
            
            # Get the best combination
            best_idx = grouped_sorted.iloc[0]
            
            # Log warning if best combination has fewer than 10 replicates
            if best_idx['replicate_count'] < 10:
                logger.warning(f"  {model}: Best hyperparameter combination has only {best_idx['replicate_count']} replicates (< 10)")
                logger.warning(f"    Best params: lr={best_idx['learning_rate']}, batch_size={best_idx['batch_size']}, rc_aug={best_idx['rc_aug']}")
                logger.warning(f"    MCC: {best_idx['mcc_mean']:.4f} ± {best_idx['mcc_std']:.4f}")
            
            # Check if best has fewer than 10 replicates and if there's a better option
            if best_idx['replicate_count'] < 10 and len(grouped_sorted) > 1:
                # Look for alternatives with at least 10 replicates
                alternatives = grouped_sorted[grouped_sorted['replicate_count'] >= 10]
                
                if len(alternatives) > 0:
                    # Check if the best alternative has MCC within 0.015 of the best
                    best_alternative = alternatives.iloc[0]
                    mcc_diff = best_idx['mcc_mean'] - best_alternative['mcc_mean']
                    
                    if mcc_diff <= 0.015:
                        logger.info(f"    Choosing alternative with more replicates (MCC diff: {mcc_diff:.4f})")
                        logger.info(f"    Original best: MCC={best_idx['mcc_mean']:.4f} with {best_idx['replicate_count']} replicates")
                        logger.info(f"    Selected: MCC={best_alternative['mcc_mean']:.4f} with {best_alternative['replicate_count']} replicates")
                        print(f"    Note: Choosing alternative with more replicates (MCC diff: {mcc_diff:.4f})")
                        print(f"    Original best: MCC={best_idx['mcc_mean']:.4f} with {best_idx['replicate_count']} replicates")
                        print(f"    Selected: MCC={best_alternative['mcc_mean']:.4f} with {best_alternative['replicate_count']} replicates")
                        best_idx = best_alternative
            
            # Process all hyperparameter combinations from the original data
            all_hp_combos = model_data_with_nans[hyperparam_cols].drop_duplicates()
            
            for _, hp_combo in all_hp_combos.iterrows():
                lr = hp_combo['learning_rate']
                bs = hp_combo['batch_size'] 
                rc_aug = hp_combo['rc_aug']
                
                # Get data for this combination from original data (with NaNs)
                hp_combo_data_with_nans = model_data_with_nans[
                    (model_data_with_nans['learning_rate'] == lr) & 
                    (model_data_with_nans['batch_size'] == bs) &
                    (model_data_with_nans['rc_aug'] == rc_aug)
                ]
                
                # Get data for this combination from filtered data (without NaNs)
                hp_combo_data_filtered = model_data[
                    (model_data['learning_rate'] == lr) & 
                    (model_data['batch_size'] == bs) &
                    (model_data['rc_aug'] == rc_aug)
                ]
                
                # Count NaN values
                nan_count = hp_combo_data_with_nans['mcc'].isna().sum()
                total_count = len(hp_combo_data_with_nans)
                valid_count = len(hp_combo_data_filtered)
                
                # Check if this combination exists in grouped (has valid data)
                grouped_match = grouped[
                    (grouped['learning_rate'] == lr) & 
                    (grouped['batch_size'] == bs) &
                    (grouped['rc_aug'] == rc_aug)
                ]
                
                if len(grouped_match) > 0:
                    # Use statistics from grouped data
                    hp_stats = grouped_match.iloc[0]
                    accuracy_mean = hp_stats['accuracy_mean']
                    accuracy_std = hp_stats['accuracy_std']
                    mcc_mean = hp_stats['mcc_mean']
                    mcc_std = hp_stats['mcc_std']
                else:
                    # All values were NaN - set stats to NaN
                    accuracy_mean = np.nan
                    accuracy_std = np.nan
                    mcc_mean = np.nan
                    mcc_std = np.nan
                
                # Check if this is the best combination
                is_best = (lr == best_idx['learning_rate'] and 
                          bs == best_idx['batch_size'] and
                          rc_aug == best_idx['rc_aug'])
                
                all_hyperparams_analysis.append({
                    'benchmark': hp_combo_data_with_nans['benchmark'].iloc[0] if 'benchmark' in hp_combo_data_with_nans.columns and len(hp_combo_data_with_nans) > 0 else 'N/A',
                    'task': task,
                    'model': model,
                    'learning_rate': lr,
                    'batch_size': bs,
                    'rc_aug': rc_aug,
                    'num_replicates': total_count,
                    'num_valid_replicates': valid_count,
                    'num_nan_replicates': nan_count,
                    'accuracy_mean': accuracy_mean,
                    'accuracy_std': accuracy_std,
                    'mcc_mean': mcc_mean,
                    'mcc_std': mcc_std,
                    'is_best': is_best
                })
            
            # Get the actual replicate data for this best combination
            best_combo_data = model_data[
                (model_data['learning_rate'] == best_idx['learning_rate']) & 
                (model_data['batch_size'] == best_idx['batch_size']) &
                (model_data['rc_aug'] == best_idx['rc_aug'])
            ]
            
            # For state space models, no source file splitting needed - use all replicates
            final_replicates = best_combo_data.copy()
            
            print(f"  {model}: lr={best_idx['learning_rate']} | batch_size={best_idx['batch_size']} | rc_aug={best_idx['rc_aug']}")
            print(f"    MCC: {best_idx['mcc_mean']:.4f} ± {best_idx['mcc_std']:.4f}, Acc: {best_idx['accuracy_mean']:.4f} ± {best_idx['accuracy_std']:.4f} ({best_idx['replicate_count']} replicates)")
            
            # Add final replicates to the best replicates list
            best_replicates_list.append(final_replicates)
            
            # Create result entry
            result = {
                'task': task,
                'model': model,
                'best_learning_rate': best_idx['learning_rate'],
                'best_batch_size': best_idx['batch_size'],
                'best_rc_aug': best_idx['rc_aug'],
                'mcc_mean': best_idx['mcc_mean'],
                'mcc_std': best_idx['mcc_std'],
                'accuracy_mean': best_idx['accuracy_mean'],
                'accuracy_std': best_idx['accuracy_std'],
                'f1_mean': best_idx['f1_mean'],
                'f1_std': best_idx['f1_std'],
                'precision_mean': best_idx['precision_mean'],
                'precision_std': best_idx['precision_std'],
                'recall_mean': best_idx['recall_mean'],
                'recall_std': best_idx['recall_std'],
                'replicate_count': best_idx['replicate_count']
            }
            
            results.append(result)
        print()
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Combine all best replicates into a single DataFrame
    best_replicates_df = pd.concat(best_replicates_list, ignore_index=True)
    
    # Convert NaN analysis to DataFrame
    nan_analysis_df = pd.DataFrame(all_nan_analysis)
    
    # Save results
    output_file = '../final_combined_ss_results/best_hyperparameters_by_task_ss.csv'
    replicates_file = '../final_combined_ss_results/best_replicates_ss.csv'
    nan_analysis_file = '../final_combined_ss_results/ss_nan_analysis.csv'
    all_hyperparams_file = '../final_combined_ss_results/all_hyperparameters_analysis_ss.csv'
    
    results_df.to_csv(output_file, index=False)
    best_replicates_df.to_csv(replicates_file, index=False)
    nan_analysis_df.to_csv(nan_analysis_file, index=False)
    
    # Save all hyperparameter combinations analysis
    all_hyperparams_df = pd.DataFrame(all_hyperparams_analysis)
    all_hyperparams_df.to_csv(all_hyperparams_file, index=False)
    
    logger.info(f"\nResults saved to: {output_file}")
    logger.info(f"Best replicates saved to: {replicates_file}")
    logger.info(f"NaN analysis saved to: {nan_analysis_file}")
    logger.info(f"All hyperparameters analysis saved to: {all_hyperparams_file}")
    logger.info(f"Total best replicates: {len(best_replicates_df)}")
    logger.info(f"Log file saved to: {log_file}")
    
    print(f"\nResults saved to: {output_file}")
    print(f"Best replicates saved to: {replicates_file}")
    print(f"NaN analysis saved to: {nan_analysis_file}")
    print(f"All hyperparameters analysis saved to: {all_hyperparams_file}")
    print(f"Total best replicates: {len(best_replicates_df)}")
    print(f"\nLog file with warnings saved to: {log_file}")
    
    # Summary of NaN issues
    problematic_combos = nan_analysis_df[(nan_analysis_df['mcc_nans'] > 0) | (nan_analysis_df['acc_nans'] > 0)]
    if len(problematic_combos) > 0:
        print(f"\nFound {len(problematic_combos)} hyperparameter combinations with NaN values")
        worst_combos = problematic_combos.nlargest(5, 'mcc_nans')
        print("Top 5 most problematic combinations (by MCC NaNs):")
        for _, row in worst_combos.iterrows():
            print(f"  {row['model']} | {row['task']} | lr={row['learning_rate']} bs={row['batch_size']} rc_aug={row['rc_aug']} | {row['mcc_nans']}/{row['total_seeds']} MCC NaNs")
    
    # Summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    print(f"Total task-model combinations analyzed: {len(results_df)}")
    print(f"Average replicate count: {results_df['replicate_count'].mean():.1f}")
    print(f"Average MCC of best combinations: {results_df['mcc_mean'].mean():.4f}")
    
    # Additional summary for replicate counts
    below_10_replicates = results_df[results_df['replicate_count'] < 10]
    below_3_replicates = results_df[results_df['replicate_count'] < 3]
    
    if len(below_10_replicates) > 0:
        logger.warning(f"\n{len(below_10_replicates)} task-model combinations have < 10 replicates:")
        print(f"\nWARNING: {len(below_10_replicates)} task-model combinations have < 10 replicates")
        for _, row in below_10_replicates.iterrows():
            logger.warning(f"  {row['model']} on {row['task']}: {row['replicate_count']} replicates")
            
    if len(below_3_replicates) > 0:
        logger.error(f"\nCRITICAL: {len(below_3_replicates)} task-model combinations have < 3 replicates:")
        print(f"\nCRITICAL WARNING: {len(below_3_replicates)} task-model combinations have < 3 replicates")
        for _, row in below_3_replicates.iterrows():
            logger.error(f"  {row['model']} on {row['task']}: {row['replicate_count']} replicates")
    
    # Model distribution
    print(f"\nModel representation:")
    model_counts = results_df['model'].value_counts()
    for model, count in model_counts.items():
        print(f"  {model}: {count} task combinations")
    
    # Learning rate distribution by model
    print(f"\nBest learning rate distribution by model:")
    for model in sorted(results_df['model'].unique()):
        model_data = results_df[results_df['model'] == model]
        lr_counts = model_data['best_learning_rate'].value_counts()
        print(f"  {model}:")
        for lr, count in lr_counts.items():
            print(f"    {lr}: {count} tasks ({count/len(model_data)*100:.1f}%)")
    
    # Batch size distribution by model
    print(f"\nBest batch size distribution by model:")
    for model in sorted(results_df['model'].unique()):
        model_data = results_df[results_df['model'] == model]
        batch_counts = model_data['best_batch_size'].value_counts()
        print(f"  {model}:")
        for batch_size, count in batch_counts.items():
            print(f"    {batch_size}: {count} tasks ({count/len(model_data)*100:.1f}%)")
    
    # RC-aug distribution by model
    print(f"\nBest rc_aug distribution by model:")
    for model in sorted(results_df['model'].unique()):
        model_data = results_df[results_df['model'] == model]
        rc_aug_counts = model_data['best_rc_aug'].value_counts(dropna=False)
        print(f"  {model}:")
        for rc_aug, count in rc_aug_counts.items():
            print(f"    {rc_aug}: {count} tasks ({count/len(model_data)*100:.1f}%)")
    
    return results_df

if __name__ == "__main__":
    best_results = find_best_hyperparameters_ss()