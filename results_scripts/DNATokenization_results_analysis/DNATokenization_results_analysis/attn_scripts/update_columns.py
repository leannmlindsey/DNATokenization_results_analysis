import pandas as pd
import re
import os

def process_csv_with_directory_info(input_file, output_file=None, learning_rate="3e-5"):
    """
    Process CSV file to fill in missing columns based on directory name information.
    
    Parameters:
    input_file (str): Path to input CSV file
    output_file (str): Path to output CSV file (optional, defaults to input_file with '_processed' suffix)
    learning_rate (str): Learning rate to use for all rows (default: "3e-5")
    """
    
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Process each row
    for idx, row in df.iterrows():
        directory_path = row['directory_name']
        
        # Extract the directory name from the full path
        # Get the part that contains the experiment info (before '/results/')
        if '/results/' in directory_path:
            experiment_dir = directory_path.split('/results/')[0]
            experiment_name = os.path.basename(experiment_dir)
        else:
            # Fallback: use the last directory in the path before the filename
            path_parts = directory_path.split('/')
            experiment_name = path_parts[-3] if len(path_parts) >= 3 else path_parts[-2]
        
        # Extract model (DNABERT1_6 -> DNABERT)
        model_match = re.search(r'DNABERT\d*_\d*', experiment_name)
        if model_match:
            df.at[idx, 'model'] = 'DNABERT'
        
        # Set learning rate
        df.at[idx, 'lr'] = learning_rate
        
        # Extract task (everything between model and seed)
        task_match = re.search(r'DNABERT\d*_\d*_(.+)_seed\d+', experiment_name)
        if task_match:
            df.at[idx, 'task'] = task_match.group(1)
        
        # Extract seed number
        seed_match = re.search(r'seed(\d+)', experiment_name)
        if seed_match:
            df.at[idx, 'seed'] = int(seed_match.group(1))
    
    # Set default output filename if not provided
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_processed.csv"
    
    # Save the processed CSV
    df.to_csv(output_file, index=False)
    
    print(f"Processed CSV saved to: {output_file}")
    print(f"Filled in {len(df)} rows with missing information")
    
    # Display summary of extracted information
    print("\nSummary of extracted information:")
    print(f"Models: {df['model'].value_counts().to_dict()}")
    print(f"Tasks: {df['task'].value_counts().to_dict()}")
    print(f"Seeds: {sorted(df['seed'].dropna().unique())}")
    print(f"Learning rate: {df['lr'].iloc[0]}")
    
    return df

def preview_extraction(csv_file, num_rows=5):
    """
    Preview what information would be extracted from the first few rows.
    
    Parameters:
    csv_file (str): Path to CSV file
    num_rows (int): Number of rows to preview (default: 5)
    """
    df = pd.read_csv(csv_file)
    
    print("Preview of information extraction:")
    print("=" * 80)
    
    for idx in range(min(num_rows, len(df))):
        directory_path = df.iloc[idx]['directory_name']
        
        # Extract experiment name
        if '/results/' in directory_path:
            experiment_dir = directory_path.split('/results/')[0]
            experiment_name = os.path.basename(experiment_dir)
        else:
            path_parts = directory_path.split('/')
            experiment_name = path_parts[-3] if len(path_parts) >= 3 else path_parts[-2]
        
        # Extract information
        model_match = re.search(r'DNABERT\d*_\d*', experiment_name)
        model = 'DNABERT' if model_match else 'Not found'
        
        task_match = re.search(r'DNABERT\d*_\d*_(.+)_seed\d+', experiment_name)
        task = task_match.group(1) if task_match else 'Not found'
        
        seed_match = re.search(r'seed(\d+)', experiment_name)
        seed = seed_match.group(1) if seed_match else 'Not found'
        
        print(f"Row {idx + 1}:")
        print(f"  Directory: {experiment_name}")
        print(f"  Model: {model}")
        print(f"  Task: {task}")
        print(f"  Seed: {seed}")
        print(f"  Learning Rate: 3e-5")
        print()

# Example usage
if __name__ == "__main__":
    input_csv = "extracted_results.csv"  # Replace with your actual file path
    
    # Preview extraction before processing
    print("Previewing extraction...")
    # preview_extraction(input_csv)
    
    processed_df = process_csv_with_directory_info(
         input_csv, 
         output_file="processed_results.csv",
         learning_rate="3e-5"
     )
    
