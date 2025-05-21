import pandas as pd
import os

parquet_file_path = os.path.join('data', 'output', 'processed_triage.parquet')

try:
    df = pd.read_parquet(parquet_file_path)
    unique_urgency_levels = df['doğru triyaj'].unique().tolist()
    print("Unique medical urgency levels:")
    print(unique_urgency_levels)
except FileNotFoundError:
    print(f"Error: The file {parquet_file_path} was not found.")
except KeyError:
    print("Error: 'doğru triyaj' column not found in the DataFrame.")
except Exception as e:
    print(f"An error occurred: {e}")