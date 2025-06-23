import pandas as pd

# Read the CSV file
df = pd.read_csv('/Users/leannmlindsey/Dropbox/NIH_WORK/TOKENIZATION/submission_06.11.2025/RESULTS/attn_results/anisa_finetune_results_06.11.2025.csv')

# Add learning rate column based on model
def get_learning_rate(model):
    if 'DNABERT-1' in model:
        return 3e-5
    elif 'DNABERT-2' in model:
        return 3e-5
    elif 'NT 500M 1000G' in model:
        return 1e-4
    else:
        return None

df['lr'] = df['model'].apply(get_learning_rate)

# Save as new CSV file
output_file = '/Users/leannmlindsey/Dropbox/NIH_WORK/TOKENIZATION/submission_06.11.2025/RESULTS/attn_results/anisa_finetune_results_with_lr_06.11.2025.csv'
df.to_csv(output_file, index=False)

print(f'Learning rates added successfully')
print(f'Output saved to: {output_file}')
print(f'Total rows: {len(df)}')
print(f'Unique learning rates: {sorted(df["lr"].unique())}')
