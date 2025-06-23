#!/usr/bin/env python3
"""
Task name mapping for prettier display in LaTeX tables
"""

# Mapping from internal task names to display names
TASK_NAME_MAPPING = {
    # Promoter tasks
    'prom_core_all': 'prom\_core\_all',
    'prom_core_notata': 'prom\_core\_notata', 
    'prom_core_tata': 'prom\_core\_tata',
    'prom_300_all': 'prom\_300\_all',
    'prom_300_notata': 'prom\_300\_notata',
    'prom_300_tata': 'prom\_300\_tata',
    
    # TF binding tasks
    'tf_0': 'human\_tfp\_0',
    'tf_1': 'human\_tfp\_1',
    'tf_2': 'human\_tfp\_2',
    'tf_3': 'human\_tfp\_3',
    'tf_4': 'human\_tfp\_4',
    
    # Splice site tasks
    'splice_sites_all': 'splice\_site\_prediction',
    'splice_sites_acceptors': 'splice\_sites\_acceptors',
    'splice_sites_donors': 'splice\_sites\_donors',
    
    # Mouse TF tasks
    'mouse_0': 'mouse\_tfp\_0',
    'mouse_1': 'mouse\_tfp\_1',
    'mouse_2': 'mouse\_tfp\_2',
    'mouse_3': 'mouse\_tfp\_3',
    'mouse_4': 'mouse\_tfp\_4',
    
    # Other tasks
    'covid': 'virus\_covid',
    'reconstructed': 'splice\_reconstructed',
    
    # GB tasks (using shorter display names)
    'dummy_mouse_enhancers': 'Mouse Enhancers',
    'demo_coding_vs_intergenomic': 'Coding vs. Intergenic',
    'demo_human_or_worm': 'Human vs. Worm',
    'human_enhancers_cohn': 'Enhancers Cohn',
    'human_enhancers_ensembl': 'Enhancers Ensembl',
    'human_ensembl_regulatory': 'Ensembl Regulatory',
    'human_nontata_promoters': 'Non-Tata Promoters',
    'human_ocr_ensembl': 'OCR Ensembl',
    
    # NTv2 tasks
    'promoter_all': 'promoter\_all',
    'promoter_no_tata': 'promoter\_no\_tata',
    'promoter_tata': 'promoter\_tata',
    
    # Keep others as is
    'enhancers': 'enhancers',
    'enhancers_types': 'enhancers\_types',
}

def get_display_name(task_name):
    """Get the display name for a task, defaulting to the original if not mapped"""
    return TASK_NAME_MAPPING.get(task_name, task_name)