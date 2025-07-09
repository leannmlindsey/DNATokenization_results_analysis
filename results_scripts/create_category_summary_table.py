#!/usr/bin/env python3
"""
Create a summary LaTeX table grouped by task category.
"""

import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_category_summary_table():
    """
    Create a LaTeX table summarizing MCC scores by category across all models.
    """
    
    # Task to category mapping (without benchmark prefixes)
    task_to_category = {
        # Regulatory Elements
        "human_ensembl_regulatory": "Regulatory Elements",
        "human_ocr_ensembl": "Regulatory Elements",
        "human_nontata_promoters": "Regulatory Elements",

        # Promoter Detection
        "prom_core_all": "Promoter Detection",
        "prom_core_tata": "Promoter Detection",
        "prom_core_notata": "Promoter Detection",
        "prom_300_all": "Promoter Detection",
        "prom_300_tata": "Promoter Detection",
        "prom_300_notata": "Promoter Detection",
        "promoter_all": "Promoter Detection",
        "promoter_no_tata": "Promoter Detection",
        "promoter_tata": "Promoter Detection",

        # Enhancer Detection
        "dummy_mouse_enhancers": "Enhancer Detection",
        "human_enhancers_cohn": "Enhancer Detection",
        "human_enhancers_ensembl": "Enhancer Detection",
        "enhancers": "Enhancer Detection",
        "enhancers_types": "Enhancer Detection",

        # Transcription Factor Binding Site Prediction
        "tf_0": "Transcription Factor Binding Site Prediction",
        "tf_1": "Transcription Factor Binding Site Prediction",
        "tf_2": "Transcription Factor Binding Site Prediction",
        "tf_3": "Transcription Factor Binding Site Prediction",
        "tf_4": "Transcription Factor Binding Site Prediction",
        "mouse_0": "Transcription Factor Binding Site Prediction",
        "mouse_1": "Transcription Factor Binding Site Prediction",
        "mouse_2": "Transcription Factor Binding Site Prediction",
        "mouse_3": "Transcription Factor Binding Site Prediction",
        "mouse_4": "Transcription Factor Binding Site Prediction",

        # Epigenetic Marks Prediction
        "H2AFZ": "Epigenetic Marks Prediction",
        "H3K27ac": "Epigenetic Marks Prediction",
        "H3K27me3": "Epigenetic Marks Prediction",
        "H3K36me3": "Epigenetic Marks Prediction",
        "H3K4me1": "Epigenetic Marks Prediction",
        "H3K4me2": "Epigenetic Marks Prediction",
        "H3K4me3": "Epigenetic Marks Prediction",
        "H3K9ac": "Epigenetic Marks Prediction",
        "H3K9me3": "Epigenetic Marks Prediction",
        "H4K20me1": "Epigenetic Marks Prediction",

        # Splice Site Detection
        "splice_sites_all": "Splice Site Detection",
        "splice_sites_acceptors": "Splice Site Detection",
        "splice_sites_donors": "Splice Site Detection",
        "reconstructed": "Splice Site Detection",

        # Coding vs Non-coding
        "demo_coding_vs_intergenomic": "Coding vs Non-coding",
        "demo_coding_vs_intergenomic_seqs": "Coding vs Non-coding",
        
        # Taxonomic Classification
        "demo_human_or_worm": "Taxonomic Classification",
        
        # Virus Variant Detection
        "covid": "Virus Variant Detection"
    }
    
    # Task to benchmark mapping
    task_benchmark_map = {
        # GB tasks
        'demo_coding_vs_intergenomic': 'GB',
        'demo_coding_vs_intergenomic_seqs': 'GB',
        'demo_human_or_worm': 'GB',
        'human_enhancers_cohn': 'GB',
        'human_enhancers_ensembl': 'GB',
        'human_ensembl_regulatory': 'GB',
        'human_nontata_promoters': 'GB',
        'human_ocr_ensembl': 'GB',
        'dummy_mouse_enhancers': 'GB',
        # GUE tasks
        'prom_core_all': 'GUE',
        'prom_core_notata': 'GUE',
        'prom_core_tata': 'GUE',
        'prom_300_all': 'GUE',
        'prom_300_notata': 'GUE',
        'prom_300_tata': 'GUE',
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
        'H3K27ac': 'NTv2',
        'H3K27me3': 'NTv2',
        'H3K36me3': 'NTv2',
        'H3K4me1': 'NTv2',
        'H3K4me2': 'NTv2',
        'H3K4me3': 'NTv2',
        'H3K9ac': 'NTv2',
        'H3K9me3': 'NTv2',
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
    
    # Read the best hyperparameters files
    ss_df = pd.read_csv('../final_combined_ss_results/best_hyperparameters_by_task_ss.csv')
    attn_df = pd.read_csv('../final_combined_attn_results/best_hyperparameters_by_task.csv')
    
    # Add benchmark column
    ss_df['benchmark'] = ss_df['task'].map(task_benchmark_map)
    attn_df['benchmark'] = attn_df['task'].map(task_benchmark_map)
    
    # Combine the dataframes
    ss_subset = ss_df[['task', 'model', 'benchmark', 'mcc_mean']].rename(columns={'mcc_mean': 'mean_mcc'})
    attn_subset = attn_df[['task', 'model', 'benchmark', 'mcc_mean']].rename(columns={'mcc_mean': 'mean_mcc'})
    combined_df = pd.concat([ss_subset, attn_subset], ignore_index=True)
    
    # Add category column
    combined_df['category'] = combined_df['task'].map(task_to_category)
    
    # Model order
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
    
    # Category display names (shortened for table)
    category_display_names = {
        'Regulatory Elements': 'regulatory',
        'Promoter Detection': 'promoters',
        'Enhancer Detection': 'enhancers',
        'Transcription Factor Binding Site Prediction': 'transcription factors',
        'Epigenetic Marks Prediction': 'epigenetic marks',
        'Splice Site Detection': 'splice sites',
        'Organism Classification': 'organism',
        'Virus Variant Detection': 'virus variant detection'
    }
    
    # Calculate mean MCC by category and model
    category_means = combined_df.groupby(['category', 'model'])['mean_mcc'].mean().reset_index()
    category_pivot = category_means.pivot(index='category', columns='model', values='mean_mcc')
    
    # Calculate mean MCC by benchmark and model
    benchmark_means = combined_df.groupby(['benchmark', 'model'])['mean_mcc'].mean().reset_index()
    benchmark_pivot = benchmark_means.pivot(index='benchmark', columns='model', values='mean_mcc')
    
    # Calculate overall mean MCC by model
    overall_means = combined_df.groupby('model')['mean_mcc'].mean()
    
    # Ensure all models are in the pivots
    for model in model_order:
        if model not in category_pivot.columns:
            category_pivot[model] = np.nan
        if model not in benchmark_pivot.columns:
            benchmark_pivot[model] = np.nan
        if model not in overall_means.index:
            overall_means[model] = np.nan
    
    # Reorder columns
    category_pivot = category_pivot[model_order]
    benchmark_pivot = benchmark_pivot[model_order]
    overall_means = overall_means[model_order]
    
    # Start building LaTeX table
    latex_lines = []
    latex_lines.append("\\begin{table*}[]")
    latex_lines.append("\\begin{center}")
    latex_lines.append("\\scriptsize")
    latex_lines.append("\\caption{Overview of MCC scores across models summarized by task category and benchmark. "
                      "The highest performing model in each row is highlighted in bold and underlined. "
                      "The benchmarks included are: Genomic Benchmark (GB) \\cite{gresova_genomic_2023}, "
                      "Nucleotide Transformer Tasks (NTT) \\cite{dalla-torre_nucleotide_2024}, "
                      "GUE (Genome Understanding Evaluation) \\cite{zhou_DNABERT-2_2023}} \\label{table:results_summary}")
    latex_lines.append("\\begin{tabular}{@{}c|cc|ccc|cccc@{}}")
    latex_lines.append("\\toprule")
    
    # Header row
    header = "\\textbf{Category}"
    model_headers = {
        'CNN': '\\textbf{CNN}',
        'GPT': '\\textbf{GPT-2}',
        'DNABERT2': '\\textbf{\\begin{tabular}[c]{@{}c@{}}DNABERT-2\\\\ (bpe)\\end{tabular}}',
        'Mamba-bpe': '\\textbf{\\begin{tabular}[c]{@{}c@{}}Mamba\\\\ (bpe)\\end{tabular}}',
        'NTv2': '\\textbf{\\begin{tabular}[c]{@{}c@{}}NT\\\\ (blocked \\\\ k-mer)\\end{tabular}}',
        'DNABERT': '\\textbf{\\begin{tabular}[c]{@{}c@{}}DNABERT\\\\ (k-mer)\\end{tabular}}',
        'HyenaDNA': '\\textbf{\\begin{tabular}[c]{@{}c@{}}HyenaDNA\\\\ (char)\\end{tabular}}',
        'Mamba-char': '\\textbf{\\begin{tabular}[c]{@{}c@{}}Mamba\\\\ (char)\\end{tabular}}',
        'Caduceus': '\\textbf{\\begin{tabular}[c]{@{}c@{}}Caduceus\\\\ (char)\\end{tabular}}'
    }
    
    for model in model_order:
        header += " & " + model_headers[model]
    header += " \\\\ \\midrule"
    latex_lines.append(header)
    
    # Model size row (placeholder - you can update with actual values)
    latex_lines.append("model size (parameters) & 464K & 125M & 117M & - & 500M & 117M & 13.1M & 1.8M & 3.9M \\\\ \\midrule")
    
    # Category rows
    category_order = [
        'Regulatory Elements',
        'Promoter Detection', 
        'Enhancer Detection',
        'Transcription Factor Binding Site Prediction',
        'Epigenetic Marks Prediction',
        'Splice Site Detection',
        'Organism Classification',
        'Virus Variant Detection'
    ]
    
    for category in category_order:
        if category in category_pivot.index:
            row = category_display_names[category]
            values = category_pivot.loc[category]
            
            # Find max value (overall best in the row)
            max_val = values.max()
            
            for model in model_order:
                val = values[model]
                if pd.isna(val):
                    row += " & -"
                else:
                    formatted = f"{val:.3f}"
                    # Bold and underline if overall max
                    if val == max_val:
                        formatted = f"\\textbf{{\\underline{{{formatted}}}}}"
                    row += f" & {formatted}"
            row += " \\\\"
            latex_lines.append(row)
    
    latex_lines.append("\\midrule")
    
    # Benchmark rows
    benchmark_display_names = {
        'GUE': 'GUE',
        'GB': 'Genomic Benchmark',
        'NTv2': 'Nucleotide Transformer Tasks'
    }
    
    for benchmark in ['GUE', 'GB', 'NTv2']:
        if benchmark in benchmark_pivot.index:
            row = benchmark_display_names[benchmark]
            values = benchmark_pivot.loc[benchmark]
            
            # Find max value
            max_val = values.max()
            
            for model in model_order:
                val = values[model]
                if pd.isna(val):
                    row += " & -"
                else:
                    formatted = f"{val:.3f}"
                    # Bold and underline if overall max
                    if val == max_val:
                        formatted = f"\\textbf{{\\underline{{{formatted}}}}}"
                    row += f" & {formatted}"
            row += " \\\\"
            latex_lines.append(row)
    
    latex_lines.append("\\midrule")
    
    # Overall row
    row = "Overall"
    max_val = overall_means.max()
    for model in model_order:
        val = overall_means[model]
        if pd.isna(val):
            row += " & -"
        else:
            formatted = f"{val:.3f}"
            # Bold and underline if overall max
            if val == max_val:
                formatted = f"\\textbf{{\\underline{{{formatted}}}}}"
            row += f" & {formatted}"
    row += " \\\\"
    latex_lines.append(row)
    
    latex_lines.append("\\midrule")
    
    # Category row
    latex_lines.append("\\multicolumn{1}{l|}{} & \\multicolumn{2}{c|}{\\textit{Baseline}} & "
                      "\\multicolumn{3}{c|}{\\textit{Sub-word Tokenization}} & "
                      "\\multicolumn{4}{c}{\\textit{Nucleotide Level Tokenization}}")
    
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{center}")
    latex_lines.append("\\end{table*}")
    
    # Print the table
    print("\n".join(latex_lines))
    
    # Save to file
    os.makedirs('../results_tables', exist_ok=True)
    with open('../results_tables/category_summary_table.tex', 'w') as f:
        f.write("\n".join(latex_lines))
    
    print("\n\nTable saved to: ../results_tables/category_summary_table.tex")

if __name__ == "__main__":
    print("Creating category summary table...")
    print("=" * 80)
    create_category_summary_table()