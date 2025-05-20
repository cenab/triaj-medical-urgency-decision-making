import pandas as pd
import os

# Define the path to the parquet file
# Assuming the script is in data/ and the parquet file is in output/
script_dir = os.path.dirname(__file__)
# Construct the absolute path to the parquet file
workspace_dir = "/Users/batu/Documents/DEVELOPMENT/triaj-medical-urgency-decision-making/data"
parquet_file_path = os.path.join(workspace_dir, 'output', 'processed_triage.parquet')

try:
    # Read the parquet file into a pandas DataFrame
    df = pd.read_parquet(parquet_file_path)

    # Display the first few rows of the DataFrame
    print(f"Successfully loaded data from {parquet_file_path}")
    print("DataFrame Head:")
    print(df.head())

    # Optionally, print some info about the DataFrame
    print("\nDataFrame Info:")
    df.info()

except FileNotFoundError:
    print(f"Error: The file {parquet_file_path} was not found.")
except Exception as e:
    print(f"An error occurred while reading the parquet file: {e}")