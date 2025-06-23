#!/usr/bin/env python3

import os
import json
import csv
import re
from pathlib import Path

def extract_path_info(file_path):
    """Extract information from the file path."""
    path_parts = Path(file_path).parts
    
    # Find the model directory (e.g., 'dnabert2')
    model = None
    seed = None
    
    for i, part in enumerate(path_parts):
        if part in ['dnabert2', 'dnabert1_6', 'nt']:
            model = part
            # The next part should be the seed number
            if i + 1 < len(path_parts):
                try:
                    seed = int(path_parts[i + 1])
                except ValueError:
                    pass
            break
    
    # Extract learning rate and task from the final directory name
    final_dir = Path(file_path).parent.name
    lr = None
    task = None
    
    # Pattern: DNABERT2__2e-5_covid_seed13
    match = re.match(r'.*?__([^_]+)_([^_]+)_seed\d+', final_dir)
    if match:
        lr = match.group(1)
        task = match.group(2)
    
    return {
        'model': model,
        'lr': lr,
        'seed': seed,
        'task': task
    }

def extract_results_from_json(file_path):
    """Extract results from a single eval_results.json file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        path_info = extract_path_info(file_path)
        
        result = {
            'directory_name': str(file_path),
            'model': path_info['model'],
            'lr': path_info['lr'],
            'benchmark': 'GUE',  # As specified in requirements
            'task': path_info['task'],
            'seed': path_info['seed'],
            'accuracy': data.get('eval_accuracy', ''),
            'mcc': data.get('eval_matthews_correlation', ''),
            'f1': data.get('eval_f1', ''),
            'precision': data.get('eval_precision', ''),
            'recall': data.get('eval_recall', ''),
            'runtime': data.get('eval_runtime', ''),
            'replicate_number': '',  # Not available in the data
            'sequence_length': '',   # Not available in the data
            'epoch': data.get('epoch', '')
        }
        
        return result
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def find_eval_results_files(root_dir):
    """Find all eval_results.json files in the directory tree."""
    eval_files = []
    
    for root, dirs, files in os.walk(root_dir):
        if 'eval_results.json' in files:
            eval_files.append(os.path.join(root, 'eval_results.json'))
    
    return eval_files

def main():
    # Set the root directory to search
    root_dir = "RAW_RESULTS"
    
    if not os.path.exists(root_dir):
        print(f"Directory {root_dir} not found!")
        return
    
    # Find all eval_results.json files
    eval_files = find_eval_results_files(root_dir)
    print(f"Found {len(eval_files)} eval_results.json files")
    
    # Extract results from each file
    results = []
    for file_path in eval_files:
        result = extract_results_from_json(file_path)
        if result:
            results.append(result)
    
    # Write to CSV
    output_file = 'extracted_results.csv'
    
    if results:
        fieldnames = [
            'directory_name', 'model', 'lr', 'benchmark', 'task', 'seed',
            'accuracy', 'mcc', 'f1', 'precision', 'recall', 'runtime',
            'replicate_number', 'sequence_length', 'epoch'
        ]
        
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        print(f"Results extracted to {output_file}")
        print(f"Total records: {len(results)}")
    else:
        print("No results extracted!")

if __name__ == "__main__":
    main()