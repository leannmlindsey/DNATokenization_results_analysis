import pandas as pd
import numpy as np
from collections import defaultdict

def create_latex_tables():
    """
    Create LaTeX tables for each benchmark showing hyperparameters tested and best ones
    for state space models.
    """
    
    # Read the best replicates file which has all the data
    best_replicates_df = pd.read_csv('../final_combined_ss_results/best_replicates_ss.csv')
    
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
        'covid': 'NTv2'
    }
    
    # Add benchmark column
    best_replicates_df['benchmark'] = best_replicates_df['task'].map(task_benchmark_map)
    
    # Drop CNN model rows since no hyperparameter tuning was done
    best_replicates_df = best_replicates_df[best_replicates_df['model'] != 'CNN']
    
    # Read the full results to get hyperparameter ranges
    full_results_df = pd.read_csv('../final_combined_ss_results/combined_ss_results_cleaned.csv')
    
    # Also drop CNN from full results
    full_results_df = full_results_df[full_results_df['model'] != 'CNN']
    
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
    benchmarks = best_replicates_df['benchmark'].dropna().unique()
    
    # Process each benchmark
    for benchmark in sorted(benchmarks):
        if benchmark not in benchmark_map:
            continue
            
        print(f"\n{'='*80}")
        print(f"Processing benchmark: {benchmark}")
        print(f"{'='*80}")
        
        # Filter data for this benchmark
        benchmark_data = best_replicates_df[best_replicates_df['benchmark'] == benchmark].copy()
        
        # Skip if no data
        if len(benchmark_data) == 0:
            continue
            
        # Group by task and model to compute statistics
        grouped = benchmark_data.groupby(['task', 'model', 'learning_rate', 'batch_size']).agg({
            'mcc': ['mean', 'std', 'count'],
            'accuracy': ['mean', 'std']
        }).reset_index()
        
        # Flatten column names
        grouped.columns = ['task', 'model', 'learning_rate', 'batch_size', 
                          'mcc_mean', 'mcc_std', 'count', 'accuracy_mean', 'accuracy_std']
        
        # Build results list
        results = []
        
        for _, row in grouped.iterrows():
            task = row['task']
            model = row['model']
            
            # Get hyperparameter ranges for this task-model combination from full results
            task_model_data = full_results_df[
                (full_results_df['task'] == task) & 
                (full_results_df['model'] == model)
            ]
            
            # Get unique learning rates and batch sizes
            lr_values = sorted(task_model_data['learning_rate'].dropna().unique())
            bs_values = sorted(task_model_data['batch_size'].dropna().unique())
            
            # Format hyperparameter ranges
            lr_range = '[' + ', '.join([f'{lr:.1e}' for lr in lr_values]) + ']'
            bs_range = '[' + ', '.join([str(int(bs)) for bs in bs_values]) + ']'
            
            # Format the row data
            results.append({
                'Task': escape_latex(task),
                'Model': model,
                'LR': f"{row['learning_rate']:.1e}",
                'BS': int(row['batch_size']),
                'MCC': f"{row['mcc_mean']:.3f}",
                'Acc': f"{row['accuracy_mean']:.3f}",
                'N': int(row['count']),
                'MCC SD': f"{row['mcc_std']:.4f}",
                'Acc SD': f"{row['accuracy_std']:.4f}",
                'LR Range': lr_range,
                'BS Range': bs_range
            })
        
        # Convert to DataFrame and sort
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(['Task', 'Model'])
        
        # Determine max rows per page based on benchmark
        if benchmark in ['GUE', 'NTv2']:
            max_rows_per_page = 35  # Increased to fit GUE on 2 pages instead of 3
        else:
            max_rows_per_page = 50  # GB can fit on one page
        
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
            
        benchmark_data = best_replicates_df[best_replicates_df['benchmark'] == benchmark]
        n_tasks = benchmark_data['task'].nunique()
        n_models = benchmark_data['model'].nunique()
        n_experiments = len(benchmark_data)
        avg_mcc = benchmark_data['mcc'].mean()
        avg_acc = benchmark_data['accuracy'].mean()
        
        summary_results.append({
            'Benchmark': benchmark,
            'Tasks': n_tasks,
            'Models': n_models,
            'Experiments': int(n_experiments),
            'Avg MCC': f"{avg_mcc:.3f}",
            'Avg Acc': f"{avg_acc:.3f}"
        })
    
    summary_df = pd.DataFrame(summary_results)
    
    print("\n\\begin{table}[]")
    print("\\caption{Summary of state space model benchmarks}")
    print("\\label{table:ss_benchmark_summary}")
    print("\\begin{tabular}{@{}lccccc@{}}")
    print("\\toprule")
    print("Benchmark & Tasks & Models & Experiments & Avg MCC & Avg Acc \\\\ \\midrule")
    
    for _, row in summary_df.iterrows():
        print(f"{row['Benchmark']} & {row['Tasks']} & {row['Models']} & "
              f"{row['Experiments']} & {row['Avg MCC']} & {row['Avg Acc']} \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")

if __name__ == "__main__":
    create_latex_tables()
