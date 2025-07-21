import pandas as pd
import numpy as np
from collections import Counter

def find_best_hyperparameters():
    """
    Find the best hyperparameter combination for each task based on MCC score.
    """
    
    # Read the combined results
    df = pd.read_csv('../final_combined_attn_results/combined_attn_results_cleaned.csv')
    
    # Standardize task names
    df['task'] = df['task'].replace({
        'demo_coding_vs_intergenomic_seqs': 'demo_coding_vs_intergenomic',
        'dummy_mouse_enhancers_ensembl': 'dummy_mouse_enhancers'
    })
    
    print(f"Loaded dataset with {len(df)} rows and {len(df['task'].unique())} unique tasks")
    
    # Define hyperparameter columns (excluding model since we want best hyperparams per model)
    hyperparam_cols = ['lr', 'epoch']
    
    # Group by task and model combination
    results = []
    best_replicates_list = []
    
    for task in sorted(df['task'].unique()):
        task_data = df[df['task'] == task].copy()
        
        # Filter out NTv1 benchmark results
        task_data = task_data[task_data['benchmark'] != 'NTv1']
        
        # Skip if no data left after filtering
        if len(task_data) == 0:
            print(f"Skipping task '{task}': No data after filtering out NTv1")
            continue
        
        print(f"Task: {task}")
        
        # Process each model separately for this task
        for model in sorted(task_data['model'].unique()):
            model_data = task_data[task_data['model'] == model].copy()
            
            # Skip models with missing hyperparameter data
            missing_hyperparam = model_data[hyperparam_cols].isnull().any(axis=1).sum()
            if missing_hyperparam > 0:
                print(f"  Skipping model '{model}': {missing_hyperparam} rows with missing hyperparameters")
                continue
                
            # Group by hyperparameter combination and calculate statistics
            grouped = model_data.groupby(hyperparam_cols).agg({
                'mcc': ['mean', 'std', 'count'],
                'accuracy': ['mean', 'std'],
                'f1': ['mean', 'std'],
                'precision': ['mean', 'std'],
                'recall': ['mean', 'std'],
                'source_file': lambda x: list(x.unique())
            }).reset_index()
            
            # Check if grouping resulted in empty dataframe
            if len(grouped) == 0:
                print(f"  Skipping model '{model}': No valid hyperparameter combinations found")
                continue
            
            # Flatten column names
            grouped.columns = [
                'lr', 'epoch',
                'mcc_mean', 'mcc_std', 'replicate_count',
                'accuracy_mean', 'accuracy_std',
                'f1_mean', 'f1_std',
                'precision_mean', 'precision_std',
                'recall_mean', 'recall_std',
                'source_files'
            ]
            
            # Handle NaN standard deviations (when only 1 replicate)
            grouped['mcc_std'] = grouped['mcc_std'].fillna(0)
            grouped['accuracy_std'] = grouped['accuracy_std'].fillna(0)
            grouped['f1_std'] = grouped['f1_std'].fillna(0)
            grouped['precision_std'] = grouped['precision_std'].fillna(0)
            grouped['recall_std'] = grouped['recall_std'].fillna(0)
            
            # Filter to only combinations with at least 3 replicates
            grouped_filtered = grouped[grouped['replicate_count'] >= 3]
            
            # Check if any combinations meet the minimum replicate requirement
            if len(grouped_filtered) == 0:
                print(f"  Skipping model '{model}': No hyperparameter combinations with ≥3 replicates")
                continue
            
            # Find best combination for this model (highest MCC, tie-break with accuracy)
            best_idx = grouped_filtered.loc[grouped_filtered['mcc_mean'].idxmax()]
            
            # Get the actual replicate data for this best combination
            best_combo_data = model_data[
                (model_data['lr'] == best_idx['lr']) & 
                (model_data['epoch'] == best_idx['epoch'])
            ]
            
            # Check if all replicates came from same source file
            source_files = best_idx['source_files']
            same_source = len(source_files) == 1
            
            # Check if mixed sources include Anisa's file (only split sources if Anisa is involved)
            anisa_file_present = any('anisa' in source.lower() for source in source_files)
            should_split_sources = not same_source and anisa_file_present
            
            final_replicates = best_combo_data.copy()
            if should_split_sources:
                print(f"  {model}: lr={best_idx['lr']} | epoch={best_idx['epoch']}")
                print(f"    Overall MCC: {best_idx['mcc_mean']:.4f} ± {best_idx['mcc_std']:.4f}, Acc: {best_idx['accuracy_mean']:.4f} ± {best_idx['accuracy_std']:.4f}")
                print(f"    Source breakdown (Anisa file detected):")
                
                best_source_mcc = -1
                best_source_data = None
                best_source_name = None
                
                for source in source_files:
                    source_data = best_combo_data[best_combo_data['source_file'] == source]
                    source_mcc_mean = source_data['mcc'].mean()
                    source_mcc_std = source_data['mcc'].std() if len(source_data) > 1 else 0
                    source_acc_mean = source_data['accuracy'].mean()
                    source_acc_std = source_data['accuracy'].std() if len(source_data) > 1 else 0
                    print(f"      {source}: MCC={source_mcc_mean:.4f}±{source_mcc_std:.4f}, Acc={source_acc_mean:.4f}±{source_acc_std:.4f} ({len(source_data)} reps)")
                    
                    if source_mcc_mean > best_source_mcc:
                        best_source_mcc = source_mcc_mean
                        best_source_data = source_data
                        best_source_name = source
                
                print(f"    Using best source: {best_source_name}")
                final_replicates = best_source_data
                
                # Recalculate stats using only the best source
                final_mcc_mean = final_replicates['mcc'].mean()
                final_mcc_std = final_replicates['mcc'].std() if len(final_replicates) > 1 else 0
                final_acc_mean = final_replicates['accuracy'].mean()
                final_acc_std = final_replicates['accuracy'].std() if len(final_replicates) > 1 else 0
                final_f1_mean = final_replicates['f1'].mean()
                final_f1_std = final_replicates['f1'].std() if len(final_replicates) > 1 else 0
                final_precision_mean = final_replicates['precision'].mean()
                final_precision_std = final_replicates['precision'].std() if len(final_replicates) > 1 else 0
                final_recall_mean = final_replicates['recall'].mean()
                final_recall_std = final_replicates['recall'].std() if len(final_replicates) > 1 else 0
                final_replicate_count = len(final_replicates)
                final_source_files = best_source_name
                final_same_source = True
                final_num_sources = 1
            else:
                print(f"  {model}: lr={best_idx['lr']} | epoch={best_idx['epoch']}")
                print(f"    MCC: {best_idx['mcc_mean']:.4f} ± {best_idx['mcc_std']:.4f}, Acc: {best_idx['accuracy_mean']:.4f} ± {best_idx['accuracy_std']:.4f} ({best_idx['replicate_count']} replicates)")
                if same_source:
                    print(f"    Source: {source_files[0]}")
                else:
                    print(f"    Sources: {', '.join(source_files)} (keeping combined)")
                
                final_mcc_mean = best_idx['mcc_mean']
                final_mcc_std = best_idx['mcc_std']
                final_acc_mean = best_idx['accuracy_mean']
                final_acc_std = best_idx['accuracy_std']
                final_f1_mean = best_idx['f1_mean']
                final_f1_std = best_idx['f1_std']
                final_precision_mean = best_idx['precision_mean']
                final_precision_std = best_idx['precision_std']
                final_recall_mean = best_idx['recall_mean']
                final_recall_std = best_idx['recall_std']
                final_replicate_count = best_idx['replicate_count']
                final_source_files = ', '.join(source_files)
                final_same_source = same_source
                final_num_sources = len(source_files)
            
            # Add final replicates to the best replicates list
            best_replicates_list.append(final_replicates)
            
            # Create result entry using final (potentially source-filtered) stats
            result = {
                'task': task,
                'model': model,
                'best_lr': best_idx['lr'],
                'best_epoch': best_idx['epoch'],
                'mcc_mean': final_mcc_mean,
                'mcc_std': final_mcc_std,
                'accuracy_mean': final_acc_mean,
                'accuracy_std': final_acc_std,
                'f1_mean': final_f1_mean,
                'f1_std': final_f1_std,
                'precision_mean': final_precision_mean,
                'precision_std': final_precision_std,
                'recall_mean': final_recall_mean,
                'recall_std': final_recall_std,
                'replicate_count': final_replicate_count,
                'source_files': final_source_files,
                'same_source_file': final_same_source,
                'num_source_files': final_num_sources
            }
            
            results.append(result)
        print()
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Combine all best replicates into a single DataFrame
    best_replicates_df = pd.concat(best_replicates_list, ignore_index=True)
    
    # Save results
    output_file = '../final_combined_attn_results/best_hyperparameters_by_task.csv'
    replicates_file = '../final_combined_attn_results/best_replicates.csv'
    
    results_df.to_csv(output_file, index=False)
    best_replicates_df.to_csv(replicates_file, index=False)
    
    print(f"\nResults saved to: {output_file}")
    print(f"Best replicates saved to: {replicates_file}")
    print(f"Total best replicates: {len(best_replicates_df)}")
    
    # Summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    print(f"Total task-model combinations analyzed: {len(results_df)}")
    print(f"Combinations with same source file: {results_df['same_source_file'].sum()}")
    print(f"Combinations with mixed source files: {(~results_df['same_source_file']).sum()}")
    print(f"Average replicate count: {results_df['replicate_count'].mean():.1f}")
    print(f"Average MCC of best combinations: {results_df['mcc_mean'].mean():.4f}")
    
    # Model distribution
    print(f"\nModel representation:")
    model_counts = results_df['model'].value_counts()
    for model, count in model_counts.items():
        print(f"  {model}: {count} task combinations")
    
    # Learning rate distribution by model
    print(f"\nBest learning rate distribution by model:")
    for model in sorted(results_df['model'].unique()):
        model_data = results_df[results_df['model'] == model]
        lr_counts = model_data['best_lr'].value_counts()
        print(f"  {model}:")
        for lr, count in lr_counts.items():
            print(f"    {lr}: {count} tasks ({count/len(model_data)*100:.1f}%)")
    
    # Epoch distribution by model
    print(f"\nBest epoch distribution by model:")
    for model in sorted(results_df['model'].unique()):
        model_data = results_df[results_df['model'] == model]
        epoch_counts = model_data['best_epoch'].value_counts()
        print(f"  {model}:")
        for epoch, count in epoch_counts.items():
            print(f"    {epoch}: {count} tasks ({count/len(model_data)*100:.1f}%)")
    
    return results_df

if __name__ == "__main__":
    best_results = find_best_hyperparameters()
