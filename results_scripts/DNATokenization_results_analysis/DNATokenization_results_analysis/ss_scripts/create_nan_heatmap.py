import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def create_nan_heatmap():
    """
    Create a heatmap showing NaN counts for NTv2 benchmark tasks.
    X-axis: Tasks in NTv2 benchmark
    Y-axis: Model/learning_rate/batch_size combinations
    Values: Number of NaN values for MCC
    """
    
    # Read the cleaned state space results
    df = pd.read_csv('../final_combined_ss_results/combined_ss_results_cleaned.csv')
    
    print(f"Loaded dataset with {len(df)} rows")
    
    # Convert metrics to numeric (handle any remaining string values)
    df['mcc'] = pd.to_numeric(df['mcc'], errors='coerce')
    df['accuracy'] = pd.to_numeric(df['accuracy'], errors='coerce')
    
    # Define NTv2 benchmark tasks
    ntv2_tasks = [
        'H2AFZ', 'H3', 'H3K14ac', 'H3K27ac', 'H3K27me3', 'H3K36me3',
        'H3K4me1', 'H3K4me2', 'H3K4me3', 'H3K79me3', 'H3K9ac', 'H3K9me3',
        'H4', 'H4ac', 'H4K20me1'
    ]
    
    # Filter to only NTv2 benchmark tasks
    print(f"Available tasks in dataset: {sorted(df['task'].unique())}")
    ntv2_data = df[df['task'].isin(ntv2_tasks)].copy()
    
    if len(ntv2_data) == 0:
        print("No NTv2 tasks found in the dataset.")
        print("Available columns:")
        print(df.columns.tolist())
        print("Sample data:")
        print(df.head())
        return
    
    print(f"Found {len(ntv2_data)} rows for NTv2 benchmark")
    print(f"Tasks in NTv2: {sorted(ntv2_data['task'].unique())}")
    
    # Create model/hyperparameter combination labels (shortened format)
    ntv2_data['model_combo'] = (ntv2_data['model'].astype(str) + '/' + 
                               ntv2_data['learning_rate'].astype(str) + '/' + 
                               ntv2_data['batch_size'].astype(str))
    
    # Group by task and model_combo, count NaN values for MCC
    nan_summary = []
    
    for task in sorted(ntv2_data['task'].unique()):
        task_data = ntv2_data[ntv2_data['task'] == task]
        
        for combo in sorted(task_data['model_combo'].unique()):
            combo_data = task_data[task_data['model_combo'] == combo]
            
            total_seeds = len(combo_data)
            mcc_nans = combo_data['mcc'].isna().sum()
            acc_nans = combo_data['accuracy'].isna().sum()
            
            # Calculate percentage of NaN values
            mcc_nan_pct = (mcc_nans / total_seeds * 100) if total_seeds > 0 else 0
            acc_nan_pct = (acc_nans / total_seeds * 100) if total_seeds > 0 else 0
            
            nan_summary.append({
                'task': task,
                'model_combo': combo,
                'total_seeds': total_seeds,
                'mcc_nans': mcc_nans,
                'mcc_nan_pct': mcc_nan_pct,
                'acc_nans': acc_nans,
                'acc_nan_pct': acc_nan_pct
            })
    
    # Convert to DataFrame
    nan_df = pd.DataFrame(nan_summary)
    
    if len(nan_df) == 0:
        print("No data to plot")
        return
    
    # Create pivot table for heatmap (using percentage of NaN values)
    heatmap_data = nan_df.pivot(index='model_combo', columns='task', values='mcc_nan_pct')
    
    # Fill NaN values with 0 (means no data for that combination)
    heatmap_data = heatmap_data.fillna(0)
    
    print(f"Heatmap dimensions: {heatmap_data.shape}")
    print(f"Tasks (columns): {len(heatmap_data.columns)}")
    print(f"Model combinations (rows): {len(heatmap_data.index)}")
    
    # Create the heatmap
    plt.figure(figsize=(max(16, len(heatmap_data.columns) * 1.0), 
                       max(10, len(heatmap_data.index) * 0.5)))
    
    # Create heatmap with custom colormap (no annotations, Blues colormap)
    sns.heatmap(heatmap_data, 
                annot=False,
                cmap='Blues',
                cbar_kws={'label': 'Percentage of MCC NaN values (%)'},
                linewidths=0.5)
    
    plt.title('NaN Values Heatmap for NTv2 Benchmark Tasks\n(State Space Models)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Tasks', fontsize=14)
    plt.ylabel('Model/Learning Rate/Batch Size', fontsize=14)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    
    # Adjust layout with more padding to prevent label cutoff
    plt.tight_layout(pad=2.0)
    
    # Save the plot as PDF
    output_file = '../ss_figures/ntv2_nan_heatmap.pdf'
    plt.savefig(output_file, format='pdf', bbox_inches='tight')
    print(f"\nHeatmap saved to: {output_file}")
    
    # Save the underlying data
    data_file = '../ss_figures/ntv2_nan_summary.csv'
    nan_df.to_csv(data_file, index=False)
    print(f"Data saved to: {data_file}")
    
    # Show summary statistics
    print(f"\n=== SUMMARY ===")
    total_combinations = len(nan_df)
    combinations_with_nans = len(nan_df[nan_df['mcc_nans'] > 0])
    print(f"Total task-model combinations: {total_combinations}")
    print(f"Combinations with NaN values: {combinations_with_nans} ({combinations_with_nans/total_combinations*100:.1f}%)")
    
    if combinations_with_nans > 0:
        print(f"\nWorst combinations (highest NaN percentage):")
        worst = nan_df.nlargest(5, 'mcc_nan_pct')
        for _, row in worst.iterrows():
            print(f"  {row['task']} | {row['model_combo']}: {row['mcc_nan_pct']:.1f}% ({row['mcc_nans']}/{row['total_seeds']} NaNs)")
    
    # Show the plot
    plt.show()
    
    return heatmap_data

if __name__ == "__main__":
    heatmap_data = create_nan_heatmap()