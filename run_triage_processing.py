import logging
from data.triage_data_processor import TriageDataProcessor
import numpy as np

if __name__ == "__main__":
    # Optional: configure logging to see info in the console
    logging.basicConfig(level=logging.INFO)

    # Initialize the processor (customize args if needed)
    processor = TriageDataProcessor(
        data_file="data/triaj_data.xls",
        output_dir="output",
        label_col="urgency",  # Change if your label column is different
        scaler_type="standard"  # or 'minmax'
    )

    # Run the full pipeline
    result = processor.prepare_data()

    # Print summary
    print("Processed triage data summary:")
    print(f"Features shape: {result['X'].shape}")
    if result['y'] is not None:
        print(f"Labels shape: {result['y'].shape}")
        print(f"Train set size: {None if result['X_train'] is None else result['X_train'].shape[0]}")
        print(f"Test set size: {None if result['X_test'] is None else result['X_test'].shape[0]}")
        print(f"Label distribution: {dict(zip(*np.unique(result['y'], return_counts=True)))}")
    print(f"Feature names: {result['feature_names']}")
    print("Processed data and statistics saved in the 'output' directory.") 