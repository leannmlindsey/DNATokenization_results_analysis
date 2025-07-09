import pandas as pd
import numpy as np
from collections import defaultdict

def create_latex_tables():
    """
    Create LaTeX tables for each benchmark showing hyperparameters tested and best ones
    for state space models.
    """
    
    # Read the all hyperparameters file which has all combinations including NaN results
    all_hp_df = pd.read_csv('../final_combined_ss_results/all_hyperparameters_analysis_ss.csv')
    
    # Read the raw data to get ALL hyperparameter combinations including those that resulted in all NaN
    raw_df = pd.read_csv('../ss_results/combined_ss_results_header.csv')
    
    # Function to extract model name from the model column in raw data
    def extract_model_from_raw(model_str):
        if 'caduceus' in model_str.lower():
            return 'Caduceus'
        elif 'hyenadna' in model_str.lower():
            return 'HyenaDNA'
        elif 'mamba' in model_str.lower():
            if 'char' in model_str.lower():
                return 'Mamba-char'
            else:
                return 'Mamba-bpe'
        elif 'cnn' in model_str.lower():
            return 'CNN'
        else:
            return 'Unknown'
    
    # Extract clean model names from raw data
    raw_df['clean_model'] = raw_df['model'].apply(extract_model_from_raw)
    
    # Drop CNN model rows from raw data since no hyperparameter tuning was done
    raw_df = raw_df[raw_df['clean_model'] != 'CNN']
    
    # Task to benchmark mapping
    task_benchmark_map = {
        # GB tasks
        'demo_coding_vs_intergenomic_seqs': 'GB',
        'demo_human_or_worm': 'GB',
        'human_enhancers_cohn': 'GB',
        'human_enhancers_ensembl': 'GB',
        'human_ensembl_regulatory': 'GB',
        'human_nontata_promoters': 'GB',
        'human_ocr_ensembl': 'GB',
        'dummy_mouse_enhancers': 'GB',
        # GUE tasks
        'prom_300_all': 'GUE',
        'prom_300_notata': 'GUE',
        'prom_300_tata': 'GUE',
        'prom_core_all': 'GUE',
        'prom_core_notata': 'GUE',
        'prom_core_tata': 'GUE',
        'tf_0': 'GUE',
        'tf_1': 'GUE',
        'tf_2': 'GUE',
        'tf_3': 'GUE',
        'tf_4': 'GUE',
        'mouse_0': 'GUE',
        'mouse_1': 'GUE',
        'mouse_2': 'GUE',
        'mouse_3': 'GUE',
        'mouse_4': 'GUE',
        'reconstructed': 'GUE',
         'covid': 'GUE',
        # NTv2 tasks
        'H2AFZ': 'NTv2',
        'H3': 'NTv2',
        'H3K14ac': 'NTv2',
        'H3K27ac': 'NTv2',
        'H3K27me3': 'NTv2',
        'H3K36me3': 'NTv2',
        'H3K4me1': 'NTv2',
        'H3K4me2': 'NTv2',
        'H3K4me3': 'NTv2',
        'H3K79me3': 'NTv2',
        'H3K9ac': 'NTv2',
        'H3K9me3': 'NTv2',
        'H4': 'NTv2',
        'H4ac': 'NTv2',
        'H4K20me1': 'NTv2',
        'promoter_all': 'NTv2',
        'promoter_no_tata': 'NTv2',
        'promoter_tata': 'NTv2',
        'splice_sites_all': 'NTv2',
        'splice_sites_acceptors': 'NTv2',
        'splice_sites_donors': 'NTv2',
        'enhancers': 'NTv2',
        'enhancers_types': 'NTv2',
       
    }
    
    # Map tasks to benchmarks
    all_hp_df['task_benchmark'] = all_hp_df['task'].map(task_benchmark_map)
    
    # Drop CNN model rows since no hyperparameter tuning was done
    all_hp_df = all_hp_df[all_hp_df['model'] != 'CNN']
    
    # Map benchmarks for consistency (excluding NTv1)
    benchmark_map = {
        'GB': 'GB',
        'GUE': 'GUE', 
        'NTv2': 'NTv2'
    }
    
    # Map benchmark names for captions
    benchmark_names = {
        'GB': 'Genomic Benchmark',
        'GUE': 'GUE',
        'NTv2': 'Nucleotide Transformer (revised)'
    }
    
    # Clean up task names for LaTeX (escape underscores)
    def escape_latex(text):
        if pd.isna(text):
            return text
        return str(text).replace('_', '\\_')
    
    # Get unique benchmarks
    benchmarks = all_hp_df['task_benchmark'].dropna().unique()
    
    # Process each benchmark
    for benchmark in sorted(benchmarks):
        if benchmark not in benchmark_map:
            continue
            
        print(f"\n{'='*80}")
        print(f"Processing benchmark: {benchmark}")
        print(f"{'='*80}")
        
        # Filter data for this benchmark
        benchmark_data = all_hp_df[all_hp_df['task_benchmark'] == benchmark].copy()
        
        # Skip if no data
        if len(benchmark_data) == 0:
            continue
        
        # Build results list - one row per task-model combination
        results = []
        
        # Get unique task-model combinations
        task_models = benchmark_data.groupby(['task', 'model']).first().reset_index()
        
        for _, tm_row in task_models.iterrows():
            task = tm_row['task']
            model = tm_row['model']
            
            # Get all hyperparameter combinations for this task-model
            task_model_data = benchmark_data[
                (benchmark_data['task'] == task) & 
                (benchmark_data['model'] == model)
            ]
            
            # Get the ranges of hyperparameters tested from RAW data (includes all-NaN combinations)
            raw_task_model_data = raw_df[
                (raw_df['task'] == task) & 
                (raw_df['clean_model'] == model)
            ]
            
            # Get unique learning rates and batch sizes from raw data
            lr_values = sorted(raw_task_model_data['learning_rate'].dropna().unique())
            bs_values = sorted(raw_task_model_data['batch_size'].dropna().unique())
            
            # If no values found in raw data, fall back to cleaned data
            if len(lr_values) == 0:
                lr_values = sorted(task_model_data['learning_rate'].dropna().unique())
            if len(bs_values) == 0:
                bs_values = sorted(task_model_data['batch_size'].dropna().unique())
                
            lr_range = '[' + ', '.join([f'{lr:.1e}' for lr in lr_values]) + ']'
            bs_range = '[' + ', '.join([str(int(bs)) for bs in bs_values]) + ']'
            
            # Find the best hyperparameter combination (marked with is_best=True)
            best_row = task_model_data[task_model_data['is_best'] == True]
            
            if len(best_row) == 0:
                # No valid results for any hyperparameter combination
                # Pick the first one to show that we tried
                best_row = task_model_data.iloc[[0]]
            
            best_row = best_row.iloc[0]
            
            # Handle cases where all replicates resulted in NaN
            if best_row['num_valid_replicates'] == 0:
                mcc_str = 'NaN'
                acc_str = 'NaN'
                mcc_sd_str = 'NaN'
                acc_sd_str = 'NaN'
            else:
                mcc_str = f"{best_row['mcc_mean']:.3f}" if not pd.isna(best_row['mcc_mean']) else 'NaN'
                acc_str = f"{best_row['accuracy_mean']:.3f}" if not pd.isna(best_row['accuracy_mean']) else 'NaN'
                mcc_sd_str = f"{best_row['mcc_std']:.4f}" if not pd.isna(best_row['mcc_std']) else 'NaN'
                acc_sd_str = f"{best_row['accuracy_std']:.4f}" if not pd.isna(best_row['accuracy_std']) else 'NaN'
            
            # Format the row data
            results.append({
                'Task': escape_latex(task),
                'Model': model,
                'LR': f"{best_row['learning_rate']:.1e}",
                'BS': int(best_row['batch_size']),
                'MCC': mcc_str,
                'Acc': acc_str,
                'N': int(best_row['num_valid_replicates']),
                'MCC SD': mcc_sd_str,
                'Acc SD': acc_sd_str,
                'LR Range': lr_range,
                'BS Range': bs_range
            })
        
        # Convert to DataFrame and sort
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(['Task', 'Model'])
        
        # Determine max rows per page based on benchmark
        if benchmark in ['GUE', 'NTv2']:
            max_rows_per_page = 37  # Increased to fit GUE on 2 pages instead of 3
        else:
            max_rows_per_page = 52  # GB can fit on one page
        
        # Split into pages if needed
        n_rows = len(results_df)
        n_pages = (n_rows + max_rows_per_page - 1) // max_rows_per_page
        
        all_latex_content = []
        
        for page in range(n_pages):
            start_idx = page * max_rows_per_page
            end_idx = min((page + 1) * max_rows_per_page, n_rows)
            page_df = results_df.iloc[start_idx:end_idx]
            
            # Generate LaTeX table for this page
            latex_lines = []
            latex_lines.append(f"\\begin{{landscape}}")
            latex_lines.append(f"\\begin{{table}}[]")
            
            if n_pages > 1:
                latex_lines.append(f"\\caption{{Best hyperparameters for {benchmark_names[benchmark]} benchmark (Part {page + 1} of {n_pages})}}")
                latex_lines.append(f"\\label{{table:hp_{benchmark.lower()}{page + 1}}}")
            else:
                latex_lines.append(f"\\caption{{Best hyperparameters for {benchmark_names[benchmark]} benchmark}}")
                latex_lines.append(f"\\label{{table:hp_{benchmark.lower()}}}")
            
            latex_lines.append("\\begin{tabular}{@{}lcccccccccc@{}}")
            latex_lines.append("\\toprule")
            latex_lines.append("Task & Model & LR & BS & MCC & Acc & N & MCC SD & Acc SD & LR Range & BS Range \\\\ \\midrule")
            
            for _, row in page_df.iterrows():
                latex_lines.append(f"{row['Task']} & {row['Model']} & {row['LR']} & {row['BS']} & "
                      f"{row['MCC']} & {row['Acc']} & {row['N']} & {row['MCC SD']} & {row['Acc SD']} & "
                      f"{row['LR Range']} & {row['BS Range']} \\\\")
            
            latex_lines.append("\\bottomrule")
            latex_lines.append("\\end{tabular}")
            latex_lines.append("\\end{table}")
            latex_lines.append("\\end{landscape}")
            
            # Print to console
            print("\n".join(latex_lines))
            
            # Add to all content
            all_latex_content.extend(latex_lines)
            if page < n_pages - 1:
                all_latex_content.append("")  # Add blank line between pages
        
        # Save all pages to file
        output_file = f'../ss_tables/latex_table_{benchmark.lower()}_ss.tex'
        with open(output_file, 'w') as f:
            f.write("\n".join(all_latex_content))
        
        print(f"\nTable saved to: {output_file} ({n_pages} page(s))")
        
    # Create a summary table of all benchmarks
    print("\n\n=== SUMMARY TABLE ===")
    summary_results = []
    
    for benchmark in sorted(benchmarks):
        if benchmark not in benchmark_map:
            continue
            
        benchmark_data = all_hp_df[all_hp_df['task_benchmark'] == benchmark]
        n_tasks = benchmark_data['task'].nunique()
        n_models = benchmark_data['model'].nunique()
        n_experiments = benchmark_data['num_replicates'].sum()
        n_valid_experiments = benchmark_data['num_valid_replicates'].sum()
        
        # Calculate averages only for valid experiments
        valid_data = benchmark_data[benchmark_data['num_valid_replicates'] > 0]
        if len(valid_data) > 0:
            avg_mcc = valid_data['mcc_mean'].mean()
            avg_acc = valid_data['accuracy_mean'].mean()
        else:
            avg_mcc = 0.0
            avg_acc = 0.0
        
        summary_results.append({
            'Benchmark': benchmark,
            'Tasks': n_tasks,
            'Models': n_models,
            'Total Experiments': int(n_experiments),
            'Valid Experiments': int(n_valid_experiments),
            'Avg MCC': f"{avg_mcc:.3f}",
            'Avg Acc': f"{avg_acc:.3f}"
        })
    
    summary_df = pd.DataFrame(summary_results)
    
    print("\n\\begin{table}[]")
    print("\\caption{Summary of state space model benchmarks}")
    print("\\label{table:ss_benchmark_summary}")
    print("\\begin{tabular}{@{}lcccccc@{}}")
    print("\\toprule")
    print("Benchmark & Tasks & Models & Total Exp. & Valid Exp. & Avg MCC & Avg Acc \\\\ \\midrule")
    
    for _, row in summary_df.iterrows():
        print(f"{row['Benchmark']} & {row['Tasks']} & {row['Models']} & "
              f"{row['Total Experiments']} & {row['Valid Experiments']} & {row['Avg MCC']} & {row['Avg Acc']} \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")

if __name__ == "__main__":
    create_latex_tables()
