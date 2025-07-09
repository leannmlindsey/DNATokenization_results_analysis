#!/usr/bin/env python3
"""
Create a heatmap showing MCC scores for all models and tasks.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def create_mcc_heatmap():
    """
    Create a heatmap visualization of MCC scores across all models and tasks.
    """
    
    # Read the best hyperparameters files
    ss_df = pd.read_csv('../final_combined_ss_results/best_hyperparameters_by_task_ss.csv')
    attn_df = pd.read_csv('../final_combined_attn_results/best_hyperparameters_by_task.csv')
    
    # Define valid tasks for GB, GUE, and NTv2 (excluding emp_ and phage_fragments)
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
    
    # Filter dataframes to only include valid tasks
    ss_df_filtered = ss_df[ss_df['task'].isin(valid_tasks)]
    attn_df_filtered = attn_df[attn_df['task'].isin(valid_tasks)]
    
    # Combine the dataframes
    ss_subset = ss_df_filtered[['task', 'model', 'mcc_mean']].rename(columns={'mcc_mean': 'MCC'})
    attn_subset = attn_df_filtered[['task', 'model', 'mcc_mean']].rename(columns={'mcc_mean': 'MCC'})
    combined_df = pd.concat([ss_subset, attn_subset], ignore_index=True)
    
    # Task ordering - group by category
    task_order = {
        # Regulatory Elements (100s)
        'human_ensembl_regulatory': 100,
        'human_ocr_ensembl': 101,
        
        # Promoter Detection (200s)
        'human_nontata_promoters': 200,
        'prom_core_all': 201,
        'prom_core_notata': 202,
        'prom_core_tata': 203,
        'prom_300_all': 204,
        'prom_300_notata': 205,
        'prom_300_tata': 206,
        'promoter_all': 210,
        'promoter_no_tata': 211,
        'promoter_tata': 212,
        
        # Enhancer Detection (300s)
        'dummy_mouse_enhancers': 300,
        'human_enhancers_cohn': 301,
        'human_enhancers_ensembl': 302,
        'enhancers': 310,
        'enhancers_types': 311,
        
        # Transcription Factor Binding (400s)
        'tf_0': 400,
        'tf_1': 401,
        'tf_2': 402,
        'tf_3': 403,
        'tf_4': 404,
        'mouse_0': 405,
        'mouse_1': 406,
        'mouse_2': 407,
        'mouse_3': 408,
        'mouse_4': 409,
        
        # Epigenetic Marks (500s)
        'H2AFZ': 500,
        'H3K27ac': 501,
        'H3K27me3': 502,
        'H3K36me3': 503,
        'H3K4me1': 504,
        'H3K4me2': 505,
        'H3K4me3': 506,
        'H3K9ac': 507,
        'H3K9me3': 508,
        'H4K20me1': 509,
        
        # Splice Sites (600s)
        'splice_sites_all': 600,
        'reconstructed': 601,
        'splice_sites_acceptors': 611,
        'splice_sites_donors': 612,
        
        # Organism Classification (700s)
        'demo_coding_vs_intergenomic': 700,
        'demo_human_or_worm': 701,
        'covid': 705,
    }
    
    # Model order and display names
    model_order = ['GPT', 'DNABERT2', 'Mamba-bpe', 'NTv2', 'CNN', 
                   'DNABERT', 'HyenaDNA', 'Mamba-char', 'Caduceus']
    
    model_display_names = {
        'CNN': 'CNN',
        'GPT': 'GPT-2',
        'DNABERT2': 'DNABERT2-bpe',
        'Mamba-bpe': 'Mamba-bpe',
        'NTv2': 'NT-kmer',
        'DNABERT': 'DNABERT1-kmer',
        'HyenaDNA': 'HyenaDNA-char',
        'Mamba-char': 'Mamba-char',
        'Caduceus': 'Caduceus-char'
    }
    
    # Task display names
    task_display_names = {
        # Use the pretty names from task_name_mapping
        'prom_core_all': 'prom_core_all',
        'prom_core_notata': 'prom_core_notata',
        'prom_core_tata': 'prom_core_tata',
        'prom_300_all': 'prom_300_all',
        'prom_300_notata': 'prom_300_notata',
        'prom_300_tata': 'prom_300_tata',
        'tf_0': 'human_tfp_0',
        'tf_1': 'human_tfp_1',
        'tf_2': 'human_tfp_2',
        'tf_3': 'human_tfp_3',
        'tf_4': 'human_tfp_4',
        'splice_sites_all': 'splice_site_prediction',
        'mouse_0': 'mouse_tfp_0',
        'mouse_1': 'mouse_tfp_1',
        'mouse_2': 'mouse_tfp_2',
        'mouse_3': 'mouse_tfp_3',
        'mouse_4': 'mouse_tfp_4',
        'covid': 'covid_variant',
        'reconstructed': 'splice_reconstructed',
        'dummy_mouse_enhancers': 'mouse_enhancers_ensembl',
        'demo_coding_vs_intergenomic': 'coding_vs_intergenomic',
        'demo_human_or_worm': 'human_or_worm',
        'human_enhancers_cohn': 'human_enhancers_cohn',
        'human_enhancers_ensembl': 'human_enhancers_ensembl',
        'human_ensembl_regulatory': 'human_ensembl_regulatory',
        'human_nontata_promoters': 'human_nontata_promoters',
        'human_ocr_ensembl': 'human_ocr_ensembl',
        'promoter_all': 'promoter_all',
        'promoter_no_tata': 'promoter_no_tata',
        'promoter_tata': 'promoter_tata',
        'enhancers': 'enhancers',
        'enhancers_types': 'enhancers_types',
        'splice_sites_acceptors': 'splice_sites_acceptors',
        'splice_sites_donors': 'splice_sites_donors',
        'H2AFZ': 'H2AFZ',
        'H3K27ac': 'H3K27ac',
        'H3K27me3': 'H3K27me3',
        'H3K36me3': 'H3K36me3',
        'H3K4me1': 'H3K4me1',
        'H3K4me2': 'H3K4me2',
        'H3K4me3': 'H3K4me3',
        'H3K9ac': 'H3K9ac',
        'H3K9me3': 'H3K9me3',
        'H4K20me1': 'H4K20me1'
    }
    
    # Create pivot table
    pivot_df = combined_df.pivot(index='task', columns='model', values='MCC')
    
    # Add ordering column
    pivot_df['order'] = pivot_df.index.map(task_order)
    
    # Sort by order
    pivot_df = pivot_df.sort_values('order')
    
    # Drop the order column
    pivot_df = pivot_df.drop(columns=['order'])
    
    # Reorder columns to match model order
    pivot_df = pivot_df[model_order]
    
    # Fill NaN with 0
    pivot_df = pivot_df.fillna(0)
    
    # Apply task display names
    pivot_df.index = pivot_df.index.map(lambda x: task_display_names.get(x, x))
    
    # Apply model display names
    pivot_df.columns = [model_display_names.get(col, col) for col in pivot_df.columns]
    
    # Transpose the dataframe so models are rows and tasks are columns
    pivot_df_transposed = pivot_df.T
    
    # Create heatmap with horizontal orientation
    width = 16  # Wider for horizontal orientation
    height = 6  # Shorter height
    
    fig, ax = plt.subplots(figsize=(width, height))
    
    # Create the heatmap with transposed data
    im = ax.imshow(pivot_df_transposed.values, cmap='GnBu', vmin=0, aspect='auto')
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax, orientation='vertical', fraction=0.01, pad=0.02)
    cbar.set_label('MCC', fontsize=12)
    
    # Set ticks and labels
    # X-axis: tasks (columns of transposed df)
    ax.set_xticks(np.arange(len(pivot_df_transposed.columns)))
    ax.set_xticklabels(pivot_df_transposed.columns)
    # Y-axis: models (rows of transposed df)
    ax.set_yticks(np.arange(len(pivot_df_transposed.index)))
    ax.set_yticklabels(pivot_df_transposed.index, fontsize=12)
    
    # Move x-axis to top
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    
    # Rotate x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor", fontsize=12)
    
    # Adjust layout as in original code
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    fig.tight_layout()
    
    # Save figure
    os.makedirs('../results_figures', exist_ok=True)
    filename = '../results_figures/Horizontal_Heatmap.pdf'
    plt.savefig(filename, format='pdf', bbox_inches='tight')
    print(f"Heatmap saved to: {filename}")
    
    # Also save as PNG for quick viewing
    plt.savefig('../results_figures/Horizontal_Heatmap.png', format='png', bbox_inches='tight', dpi=150)
    
    # Also save as SVG
    plt.savefig('../results_figures/Horizontal_Heatmap.svg', format='svg', bbox_inches='tight')
    print(f"SVG version saved to: ../results_figures/Horizontal_Heatmap.svg")
    
    plt.close()
    
    # Print summary statistics
    print(f"\nHeatmap created with {len(pivot_df_transposed.columns)} tasks and {len(pivot_df_transposed.index)} models")
    print(f"MCC range: {pivot_df.values.min():.3f} to {pivot_df.values.max():.3f}")
    print(f"\nIncluded benchmarks: GB, GUE, NTv2 (excluding NTv1 emp_ tasks and phage_fragments)")
    
    # Print tasks with missing data (zeros)
    zero_counts = (pivot_df == 0).sum(axis=1)
    if zero_counts.sum() > 0:
        print("\nTasks with missing data (zeros):")
        for task in zero_counts[zero_counts > 0].index:
            missing_models = pivot_df.columns[pivot_df.loc[task] == 0].tolist()
            print(f"  {task}: {missing_models}")

if __name__ == "__main__":
    print("Creating MCC heatmap...")
    print("=" * 80)
    create_mcc_heatmap()