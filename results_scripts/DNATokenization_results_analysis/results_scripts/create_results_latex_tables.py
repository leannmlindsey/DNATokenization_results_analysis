import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_name_mapping import get_display_name

def create_results_latex_tables():
    """
    Create LaTeX tables for results combining both attention and state space models.
    Creates separate tables for accuracy and MCC for each benchmark.
    """
    
    # Read the best hyperparameters files
    ss_df = pd.read_csv('../final_combined_ss_results/best_hyperparameters_by_task_ss.csv')
    attn_df = pd.read_csv('../final_combined_attn_results/best_hyperparameters_by_task.csv')
    
    # Task to benchmark mapping
    task_benchmark_map = {
        # GB tasks
        'demo_coding_vs_intergenomic': 'GB',
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
	    'enhancers': 'NTv2',
        'enhancers_types': 'NTv2',
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
        'splice_sites_donors': 'NTv2'
    }
    
    # Add benchmark column
    ss_df['benchmark'] = ss_df['task'].map(task_benchmark_map)
    attn_df['benchmark'] = attn_df['task'].map(task_benchmark_map)
    
    # Combine the dataframes
    # Select only the columns we need and rename to match
    ss_columns = ['task', 'model', 'benchmark', 'mcc_mean', 'accuracy_mean', 'replicate_count']
    attn_columns = ['task', 'model', 'benchmark', 'mcc_mean', 'accuracy_mean', 'replicate_count']
    
    ss_subset = ss_df[ss_columns].rename(columns={
        'mcc_mean': 'mean_mcc',
        'accuracy_mean': 'mean_accuracy',
        'replicate_count': 'count'
    })
    
    attn_subset = attn_df[attn_columns].rename(columns={
        'mcc_mean': 'mean_mcc',
        'accuracy_mean': 'mean_accuracy',
        'replicate_count': 'count'
    })
    
    # Combine
    combined_df = pd.concat([ss_subset, attn_subset], ignore_index=True)
    
    # Model display names and order
    model_order = [
        'CNN',
        'GPT',
        'DNABERT2',
        'Mamba-bpe',
        'NTv2',
        'DNABERT',
        'HyenaDNA',
        'Mamba-char',
        'Caduceus'
    ]
    
    model_display_names = {
        'CNN': 'CNN\\\\ (char)',
        'GPT': 'GPT-2\\\\ (bpe)',
        'DNABERT2': 'DNABERT2\\\\ (bpe)',
        'Mamba-bpe': 'Mamba\\\\ (bpe)',
        'NTv2': 'NT\\\\ (blocked k-mer)',
        'DNABERT': 'DNABERT\\\\ (k-mer)',
        'HyenaDNA': 'HyenaDNA\\\\ (char)',
        'Mamba-char': 'Mamba\\\\ (char)',
        'Caduceus': 'Caduceus\\\\ (char)'
    }
    
    # Task display names for GB
    task_display_names = {
        'dummy_mouse_enhancers': 'Mouse Enhancers',
        'demo_coding_vs_intergenomic': 'Coding vs. Intergenic',
        'demo_human_or_worm': 'Human vs. Worm',
        'human_enhancers_cohn': 'Enhancers Cohn',
        'human_enhancers_ensembl': 'Enhancers Ensembl',
        'human_ensembl_regulatory': 'Ensembl Regulatory',
        'human_nontata_promoters': 'Non-Tata Promoters',
        'human_ocr_ensembl': 'OCR Ensembl'
    }
    
    # Benchmark names
    benchmark_names = {
        'GB': 'Genomic Benchmark',
        'GUE': 'GUE',
        'NTv2': 'Nucleotide Transformer (revised)'
    }
    
    # Process each benchmark
    for benchmark in ['GB', 'GUE', 'NTv2']:
        benchmark_data = combined_df[combined_df['benchmark'] == benchmark]
        
        if len(benchmark_data) == 0:
            continue
            
        # Get unique tasks for this benchmark
        tasks = sorted(benchmark_data['task'].unique())
        
        # Use specific task order for each benchmark
        if benchmark == 'GB':
            task_order = [
                'dummy_mouse_enhancers',
                'demo_coding_vs_intergenomic',
                'demo_human_or_worm',
                'human_enhancers_cohn',
                'human_enhancers_ensembl',
                'human_ensembl_regulatory',
                'human_nontata_promoters',
                'human_ocr_ensembl'
            ]
            tasks = [t for t in task_order if t in tasks]
        elif benchmark == 'GUE':
            task_order = [
                # Core promoter tasks (matching paper order)
                'prom_core_all',
                'prom_core_notata',
                'prom_core_tata',
                # 300bp promoter tasks  
                'prom_300_all',
                'prom_300_notata',
                'prom_300_tata',
                # Human TF tasks
                'tf_0',
                'tf_1',
                'tf_2',
                'tf_3',
                'tf_4',
                # Splice sites
                'splice_sites_all',
                # Mouse TF tasks
                'mouse_0',
                'mouse_1',
                'mouse_2',
                'mouse_3',
                'mouse_4',
                # Virus
                'covid',
                # Any remaining GUE tasks
                'enhancers',
                'enhancers_types',
                'splice_sites_acceptors',
                'splice_sites_donors',
                'reconstructed'
            ]
            # Keep only tasks that exist in the data
            tasks = [t for t in task_order if t in tasks]
        elif benchmark == 'NTv2':
            task_order = [
                # Histone modifications
                'H2AFZ',
                'H3K27ac',
                'H3K27me3',
                'H3K36me3',
                'H3K4me1',
                'H3K4me2',
                'H3K4me3',
                'H3K9ac',
                'H3K9me3',
                'H4K20me1',
                # Additional histone tasks if present
                'H3',
                'H3K14ac',
                'H3K79me3',
                'H4',
                'H4ac',
                # Promoter tasks
                'promoter_all',
                'promoter_no_tata',
                'promoter_tata',
                # Enhancer tasks
                'enhancers',
                'enhancers_types',
                # Splice sites
                'splice_sites_all',
                'splice_sites_acceptors',
                'splice_sites_donors'
            ]
            # Keep only tasks that exist in the data
            tasks = [t for t in task_order if t in tasks]
        
        # Create accuracy table
        print(f"\n{'='*80}")
        print(f"Creating Accuracy table for {benchmark}")
        print(f"{'='*80}")
        
        create_metric_table(benchmark_data, tasks, 'mean_accuracy', 'Accuracy', 
                          benchmark, benchmark_names[benchmark], model_order, 
                          model_display_names, task_display_names)
        
        # Create MCC table
        print(f"\n{'='*80}")
        print(f"Creating MCC table for {benchmark}")
        print(f"{'='*80}")
        
        create_metric_table(benchmark_data, tasks, 'mean_mcc', 'MCC', 
                          benchmark, benchmark_names[benchmark], model_order, 
                          model_display_names, task_display_names)

def create_metric_table(data, tasks, metric_col, metric_name, benchmark, 
                       benchmark_full_name, model_order, model_display_names, 
                       task_display_names):
    """
    Create a LaTeX table for a specific metric (accuracy or MCC).
    """
    
    # Create pivot table
    pivot = data.pivot(index='task', columns='model', values=metric_col)
    
    # Check for missing model-task combinations
    print(f"\n=== Checking for missing values in {benchmark} {metric_name} table ===")
    for task in tasks:
        if task in pivot.index:
            for model in model_order:
                if model not in pivot.columns or pd.isna(pivot.loc[task, model]):
                    print(f"MISSING: Benchmark={benchmark}, Task={task}, Model={model}, Metric={metric_name}")
        else:
            print(f"MISSING: Benchmark={benchmark}, Task={task} - entire task missing from data")
    
    # Ensure all models from model_order are included
    # Add missing model columns with NaN values
    for model in model_order:
        if model not in pivot.columns:
            pivot[model] = np.nan
    
    # Reorder columns based on model_order
    pivot = pivot[model_order]
    available_models = model_order
    
    # Start building LaTeX table
    latex_lines = []
    latex_lines.append("\\begin{table*}[h!]")
    latex_lines.append("\\small")
    latex_lines.append("\\begin{center}")
    latex_lines.append(f"\\caption{{{metric_name} Scores on the {benchmark_full_name}. "
                      f"The highest score for each dataset is highlighted in bold and underlined.}}")
    latex_lines.append(f"\\label{{table: {benchmark} Results {metric_name.upper()}}}")
    
    # Create column specification
    n_cols = len(available_models)
    if n_cols <= 2:
        col_spec = "l|" + "c" * n_cols
    elif n_cols <= 5:
        col_spec = "l|cc|" + "c" * (n_cols - 2)
    else:
        # Split as 2|3|4 or similar
        col_spec = "l|cc|ccc|" + "c" * (n_cols - 5)
    
    latex_lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex_lines.append("\\hline")
    
    # Create header row
    header_parts = [""]
    for model in available_models:
        display_name = model_display_names.get(model, model)
        # Format for multiline header
        header_parts.append(f"\\textbf{{\\begin{{tabular}}[c]{{@{{}}c@{{}}}}{display_name}\\end{{tabular}}}}")
    
    latex_lines.append(" & ".join(header_parts) + " \\\\ \\hline")
    
    # Add data rows
    for task in tasks:
        if task not in pivot.index:
            continue
            
        # Get display name for task using mapping
        display_task = get_display_name(task)
        # For tasks not in mapping, clean up the name
        if display_task == task:
            display_task = task.replace('_', '\\_')
        
        row_values = [display_task]
        
        # Get values for this row
        task_values = pivot.loc[task]
        
        # Find maximum value (handling NaN)
        max_val = task_values.max()
        
        # Format each value
        for model in available_models:
            if model in task_values.index and not pd.isna(task_values[model]):
                val = task_values[model]
                # Format to 3 decimal places
                formatted_val = f"{val:.3f}"
                
                # Check if this is the maximum
                if val == max_val:
                    formatted_val = f"{{\\ul \\textbf{{{formatted_val}}}}}"
                
                row_values.append(formatted_val)
            else:
                row_values.append("-")
        
        latex_lines.append(" & ".join(row_values) + " \\\\")
    
    # Add footer
    latex_lines.append("\\hline")
    
    # Add category row
    if len(available_models) > 5:
        latex_lines.append("\\multicolumn{1}{l|}{} & \\multicolumn{2}{c|}{\\textit{baseline}} & "
                          "\\multicolumn{3}{c|}{\\textit{Sub-word Tokenization}} & "
                          "\\multicolumn{4}{c}{\\textit{Nucleotide Level Tokenization}} \\\\ \\hline")
    
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{center}")
    latex_lines.append("\\end{table*}")
    
    # Print the table
    print("\n".join(latex_lines))
    
    # Save to file
    filename = f"../results_tables/latex_table_{benchmark.lower()}_{metric_name.lower()}.tex"
    with open(filename, 'w') as f:
        f.write("\n".join(latex_lines))
    
    print(f"\nTable saved to: {filename}")

if __name__ == "__main__":
    print("Starting to create results LaTeX tables...")
    print("=" * 80)
    create_results_latex_tables()
    print("\n" + "=" * 80)
    print("Finished creating LaTeX tables. Check above for any MISSING values.")
