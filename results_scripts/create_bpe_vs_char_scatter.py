#!/usr/bin/env python3
"""
Create a 3-panel scatter plot comparing Mamba-bpe vs Mamba-char across benchmarks.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_bpe_vs_char(ax, df, df_subset, bpe, char, xlabel, ylabel, title, range_min, range_max):
    """
    Create a scatter plot comparing BPE vs Char tokenization for a specific benchmark.
    """
    print(f"Plotting {title}: range {range_min} to {range_max}")
    
    # Text markers for each category (using first letter as marker, except some special cases)
    marker_map = {
        'enhancers': '$E$',
        'coding': '$C$',
        'taxonomic': '$X$',
        'regulatory': '$R$',
        'promoters': '$P$',
        'transcription factors': '$T$',
        'splice sites': '$S$',
        'epigenetic marks': '$M$',
        'virus variant': '$V$'
    }
    
    # Colors for each category
    color_map = {
        'enhancers': "#76b7b2",
        'coding': "#76b7b2",
        'taxonomic': "#59a14f",
        'regulatory': "#bab0ab",
        'promoters': "#edc949",
        'transcription factors': "#f28e2c",
        'splice sites': "#e15759",
        'epigenetic marks': "#4e79a7",
        'virus variant': "#af7aa1"
    }
    
    unique_category = df_subset['Category'].unique()
    print(f"Categories in {title}: {unique_category}")
    
    # Plot each category
    for category in unique_category:
        subset = df_subset[df_subset['Category'] == category]
        if len(subset) > 0:
            ax.scatter(subset[bpe], subset[char], 
                      c=color_map.get(category, 'gray'), 
                      marker=marker_map.get(category, 'o'), 
                      label=None, s=65)
    
    # Set axis limits and labels
    ax.set_xlim([range_min, range_max])
    ax.set_ylim([range_min, range_max])
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    
    # Add diagonal line
    ax.plot([range_min, range_max], [range_min, range_max], color='#d3d3d3', linestyle='--')
    
    # Add grid
    ax.grid(True, which="both", ls="--", alpha=0.3)
    
    # Create legend for categories present in this subplot
    for category in unique_category:
        if category in color_map and category in marker_map:
            ax.scatter([], [], color=color_map[category], marker=marker_map[category], 
                      label=f'{category}', s=65)
    
    legend = ax.legend(loc='lower right', fontsize=10)
    legend.get_frame().set_linewidth(0)
    
    return ax

def create_bpe_vs_char_scatter():
    """
    Create a 3-panel figure comparing Mamba-bpe vs Mamba-char across benchmarks.
    """
    
    # Read the best hyperparameters files
    ss_df = pd.read_csv('../final_combined_ss_results/best_hyperparameters_by_task_ss.csv')
    attn_df = pd.read_csv('../final_combined_attn_results/best_hyperparameters_by_task.csv')
    
    # Define valid tasks (same as heatmap)
    valid_tasks = {
        # GB tasks
        'demo_coding_vs_intergenomic', 'demo_human_or_worm',
        'human_enhancers_cohn', 'human_enhancers_ensembl', 'human_ensembl_regulatory',
        'human_nontata_promoters', 'human_ocr_ensembl', 'dummy_mouse_enhancers',
        # GUE tasks
        'prom_core_all', 'prom_core_notata', 'prom_core_tata', 'prom_300_all', 'prom_300_notata',
        'prom_300_tata', 'tf_0', 'tf_1', 'tf_2', 'tf_3', 'tf_4', 'splice_sites_all',
        'splice_sites_acceptors', 'splice_sites_donors', 'enhancers', 'enhancers_types',
        'mouse_0', 'mouse_1', 'mouse_2', 'mouse_3', 'mouse_4', 'reconstructed', 'covid',
        # NTv2 tasks
        'H2AFZ', 'H3K27ac', 'H3K27me3', 'H3K36me3', 'H3K4me1', 'H3K4me2', 'H3K4me3',
        'H3K9ac', 'H3K9me3', 'H4K20me1', 'promoter_all', 'promoter_no_tata', 'promoter_tata'
    }
    
    # Task to category mapping
    task_to_category = {
        # Regulatory Elements
        "human_ensembl_regulatory": "regulatory",
        "human_ocr_ensembl": "regulatory",
        "human_nontata_promoters": "regulatory",
        # Promoter Detection
        "prom_core_all": "promoters",
        "prom_core_tata": "promoters",
        "prom_core_notata": "promoters",
        "prom_300_all": "promoters",
        "prom_300_tata": "promoters",
        "prom_300_notata": "promoters",
        "promoter_all": "promoters",
        "promoter_no_tata": "promoters",
        "promoter_tata": "promoters",
        # Enhancer Detection
        "dummy_mouse_enhancers": "enhancers",
        "human_enhancers_cohn": "enhancers",
        "human_enhancers_ensembl": "enhancers",
        "enhancers": "enhancers",
        "enhancers_types": "enhancers",
        # Transcription Factor Binding
        "tf_0": "transcription factors",
        "tf_1": "transcription factors",
        "tf_2": "transcription factors",
        "tf_3": "transcription factors",
        "tf_4": "transcription factors",
        "mouse_0": "transcription factors",
        "mouse_1": "transcription factors",
        "mouse_2": "transcription factors",
        "mouse_3": "transcription factors",
        "mouse_4": "transcription factors",
        # Epigenetic Marks
        "H2AFZ": "epigenetic marks",
        "H3K27ac": "epigenetic marks",
        "H3K27me3": "epigenetic marks",
        "H3K36me3": "epigenetic marks",
        "H3K4me1": "epigenetic marks",
        "H3K4me2": "epigenetic marks",
        "H3K4me3": "epigenetic marks",
        "H3K9ac": "epigenetic marks",
        "H3K9me3": "epigenetic marks",
        "H4K20me1": "epigenetic marks",
        # Splice Sites
        "splice_sites_all": "splice sites",
        "splice_sites_acceptors": "splice sites",
        "splice_sites_donors": "splice sites",
        "reconstructed": "splice sites",
        # Coding vs Non-coding
        "demo_coding_vs_intergenomic": "coding",
        "demo_coding_vs_intergenomic_seqs": "coding",
        # Taxonomic Classification
        "demo_human_or_worm": "taxonomic",
        # Virus Variant Detection
        "covid": "virus variant"
    }
    
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
    
    # Filter dataframes to only include valid tasks
    ss_df_filtered = ss_df[ss_df['task'].isin(valid_tasks)]
    
    # Get only Mamba models from state space results
    mamba_bpe = ss_df_filtered[ss_df_filtered['model'] == 'Mamba-bpe'][['task', 'mcc_mean']].rename(
        columns={'mcc_mean': 'Mamba-bpe'})
    mamba_char = ss_df_filtered[ss_df_filtered['model'] == 'Mamba-char'][['task', 'mcc_mean']].rename(
        columns={'mcc_mean': 'Mamba-char'})
    
    # Merge Mamba results
    bpevschar = mamba_bpe.merge(mamba_char, on='task', how='outer')
    
    # Add benchmark and category columns
    bpevschar['Benchmark'] = bpevschar['task'].map(task_benchmark_map)
    bpevschar['Category'] = bpevschar['task'].map(task_to_category)
    bpevschar['Dataset'] = bpevschar['task']  # Keep task name as dataset
    
    # Fill NaN with 0
    bpevschar['Mamba-char'] = bpevschar['Mamba-char'].fillna(0)
    bpevschar['Mamba-bpe'] = bpevschar['Mamba-bpe'].fillna(0)
    
    # Add a dummy Layers column (used in original code)
    bpevschar['Layers'] = 4
    
    # Split by benchmark
    bpevschar_gue = bpevschar[bpevschar['Benchmark'] == 'GUE'].copy()
    bpevschar_gb = bpevschar[bpevschar['Benchmark'] == 'GB'].copy()
    bpevschar_nt = bpevschar[bpevschar['Benchmark'] == 'NTv2'].copy()
    
    # Update benchmark display names
    benchmark_display = {
        'GB': 'Genomic Benchmark',
        'GUE': 'GUE',
        'NTv2': 'Nucleotide Transformer Tasks'
    }
    
    # Create the 3-panel figure
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # Plot each benchmark
    scatter1 = plot_bpe_vs_char(axes[0], bpevschar, bpevschar_nt, 
                               'Mamba-bpe', 'Mamba-char', 'BPE MCC', 'Char MCC', 
                               'Nucleotide Transformer Tasks', 0.25, 1)
    
    scatter2 = plot_bpe_vs_char(axes[1], bpevschar, bpevschar_gb, 
                               'Mamba-bpe', 'Mamba-char', 'BPE MCC', 'Char MCC', 
                               'Genomic Benchmark', 0.25, 1)
    
    scatter3 = plot_bpe_vs_char(axes[2], bpevschar, bpevschar_gue, 
                               'Mamba-bpe', 'Mamba-char', 'BPE MCC', 'Char MCC', 
                               'GUE', 0.25, 1)
    
    # Adjust layout
    fig.subplots_adjust(hspace=0.35)
    
    # Save figure
    os.makedirs('../results_figures', exist_ok=True)
    filename = '../results_figures/bpevschar.pdf'
    fig.savefig(filename, format='pdf', bbox_inches='tight')
    print(f"\nScatter plot saved to: {filename}")
    
    # Also save as PNG
    fig.savefig('../results_figures/bpevschar.png', format='png', bbox_inches='tight', dpi=150)
    
    # Also save as SVG
    fig.savefig('../results_figures/bpevschar.svg', format='svg', bbox_inches='tight')
    print(f"SVG version saved to: ../results_figures/bpevschar.svg")
    
    plt.close()
    
    # Print summary
    print(f"\nTotal tasks plotted: {len(bpevschar)}")
    print(f"Tasks per benchmark:")
    print(f"  NTv2: {len(bpevschar_nt)}")
    print(f"  GB: {len(bpevschar_gb)}")
    print(f"  GUE: {len(bpevschar_gue)}")
    
    # Print tasks with zero values
    zero_tasks = bpevschar[(bpevschar['Mamba-bpe'] == 0) | (bpevschar['Mamba-char'] == 0)]
    if len(zero_tasks) > 0:
        print(f"\nTasks with zero MCC values:")
        for _, row in zero_tasks.iterrows():
            print(f"  {row['Dataset']}: Mamba-bpe={row['Mamba-bpe']:.3f}, Mamba-char={row['Mamba-char']:.3f}")
    
    # Calculate maximum mean difference between Mamba-char and Mamba-bpe
    print(f"\n{'='*80}")
    print("MAXIMUM MEAN DIFFERENCES BETWEEN MAMBA-CHAR AND MAMBA-BPE:")
    print(f"{'='*80}")
    
    # For MCC values
    bpevschar['mcc_diff'] = bpevschar['Mamba-char'] - bpevschar['Mamba-bpe']
    max_mcc_diff_idx = bpevschar['mcc_diff'].abs().idxmax()
    max_mcc_diff_row = bpevschar.loc[max_mcc_diff_idx]
    
    print(f"\nMaximum MCC difference:")
    print(f"  Task: {max_mcc_diff_row['Dataset']}")
    print(f"  Benchmark: {max_mcc_diff_row['Benchmark']}")
    print(f"  Mamba-char MCC: {max_mcc_diff_row['Mamba-char']:.4f}")
    print(f"  Mamba-bpe MCC: {max_mcc_diff_row['Mamba-bpe']:.4f}")
    print(f"  Difference (char - bpe): {max_mcc_diff_row['mcc_diff']:.4f}")
    if max_mcc_diff_row['mcc_diff'] > 0:
        print(f"  --> Mamba-char performs better by {max_mcc_diff_row['mcc_diff']:.4f}")
    else:
        print(f"  --> Mamba-bpe performs better by {abs(max_mcc_diff_row['mcc_diff']):.4f}")
    
    # Now need to get accuracy data
    # Merge with the original dataframes to get accuracy
    mamba_bpe_acc = ss_df_filtered[ss_df_filtered['model'] == 'Mamba-bpe'][['task', 'accuracy_mean']].rename(
        columns={'accuracy_mean': 'Mamba-bpe-acc'})
    mamba_char_acc = ss_df_filtered[ss_df_filtered['model'] == 'Mamba-char'][['task', 'accuracy_mean']].rename(
        columns={'accuracy_mean': 'Mamba-char-acc'})
    
    # Merge accuracy data
    acc_comparison = mamba_bpe_acc.merge(mamba_char_acc, on='task', how='outer')
    acc_comparison['acc_diff'] = acc_comparison['Mamba-char-acc'] - acc_comparison['Mamba-bpe-acc']
    
    # Find maximum accuracy difference
    max_acc_diff_idx = acc_comparison['acc_diff'].abs().idxmax()
    max_acc_diff_row = acc_comparison.loc[max_acc_diff_idx]
    
    print(f"\nMaximum Accuracy difference:")
    print(f"  Task: {max_acc_diff_row['task']}")
    print(f"  Mamba-char Accuracy: {max_acc_diff_row['Mamba-char-acc']:.4f}")
    print(f"  Mamba-bpe Accuracy: {max_acc_diff_row['Mamba-bpe-acc']:.4f}")
    print(f"  Difference (char - bpe): {max_acc_diff_row['acc_diff']:.4f}")
    if max_acc_diff_row['acc_diff'] > 0:
        print(f"  --> Mamba-char performs better by {max_acc_diff_row['acc_diff']:.4f}")
    else:
        print(f"  --> Mamba-bpe performs better by {abs(max_acc_diff_row['acc_diff']):.4f}")
    
    print(f"{'='*80}")

if __name__ == "__main__":
    print("Creating BPE vs Char scatter plot...")
    print("=" * 80)
    create_bpe_vs_char_scatter()