import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path

# Setup logging
logger = logging.getLogger(__name__)

class TriageDataProcessor:
    """Main class for processing medical triage data from Excel."""
    
    def __init__(
        self,
        data_file: str = "triaj_data.csv",
        output_dir: str = "output",
        label_col: str = "urgency",
        random_state: int = 42,
        test_size: float = 0.2,
        n_splits: int = 5,
        scaler_type: str = 'standard'
    ):
        """
        Initialize the triage data processor.
        Args:
            data_file: Path to the triage Excel file
            output_dir: Path to save processed data
            label_col: Name of the label column (default: 'urgency')
            random_state: Random seed for reproducibility
            test_size: Proportion of data to use for testing
            n_splits: Number of cross-validation folds
            scaler_type: Type of feature scaling ('standard' or 'minmax')
        """
        self.data_file = Path(data_file)
        self.output_dir = Path(output_dir)
        self.label_col = label_col
        self.random_state = random_state
        self.test_size = test_size
        self.n_splits = n_splits
        
        # Initialize scaler
        if scaler_type.lower() == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type.lower() == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unsupported scaler type: {scaler_type}")
        
        self.data: Optional[pd.DataFrame] = None
        self.X: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None
        self.feature_names: Optional[List[str]] = None
        self.cv_folds: List[Tuple[np.ndarray, np.ndarray]] = []
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self) -> pd.DataFrame:
        """
        Load triage data from the Excel file.
        Returns:
            DataFrame containing all triage data.
        """
        logger.info(f"Loading data from {self.data_file}")
        try:
            df = pd.read_csv(self.data_file, encoding='utf-8-sig')
        except Exception as e:
            logger.error(f"Error loading {self.data_file}: {e}")
            raise
        self.data = df
        logger.info(f"Loaded {len(self.data)} records from {self.data_file}")
        return self.data

    def clean_data(self) -> pd.DataFrame:
        """
        Clean the loaded data by handling missing values and outliers.
        Returns:
            Cleaned DataFrame
        """
        if self.data is None:
            self.load_data()
        logger.info("Cleaning data...")
        # Handle missing values
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        self.data[numeric_columns] = self.data[numeric_columns].fillna(0)
        # Handle categorical missing values
        cat_columns = self.data.select_dtypes(include=["object", "category"]).columns
        self.data[cat_columns] = self.data[cat_columns].fillna("Unknown")
        # Remove duplicate rows
        self.data = self.data.drop_duplicates()
        logger.info(f"Cleaned data shape: {self.data.shape}")
        return self.data

    def extract_features(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Extract features and label from the cleaned data.
        Returns:
            Tuple of (feature_matrix, labels)
        """
        if self.data is None:
            self.clean_data()
        logger.info("Extracting features from triage data")
        # Exclude label column from features
        feature_columns = [col for col in self.data.columns if col != self.label_col]
        # Convert categorical features to dummies
        X = pd.get_dummies(self.data[feature_columns], drop_first=True)
        self.feature_names = X.columns.tolist()
        self.X = X.values
        # Extract label if present
        if self.label_col in self.data.columns:
            y = self.data[self.label_col].values
            self.y = y
        else:
            y = None
            self.y = None
        logger.info(f"Extracted {self.X.shape[1]} features for {self.X.shape[0]} records")
        if y is not None:
            logger.info(f"Label distribution: {pd.Series(y).value_counts().to_dict()}")
        return self.X, self.y

    def normalize_features(self) -> np.ndarray:
        """
        Normalize the feature matrix using the configured scaler.
        Returns:
            Normalized feature matrix
        """
        if self.X is None:
            self.extract_features()
        logger.info(f"Normalizing features using {type(self.scaler).__name__}")
        self.X_normalized = self.scaler.fit_transform(self.X)
        return self.X_normalized

    def create_cv_folds(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create stratified cross-validation folds if label is present.
        Returns:
            List of (train_indices, test_indices) for each fold
        """
        if self.y is None:
            logger.warning("No label column found; cannot create stratified folds.")
            return []
        logger.info(f"Creating {self.n_splits} stratified cross-validation folds")
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        self.cv_folds = list(skf.split(self.X, self.y))
        return self.cv_folds

    def save_processed_data(self):
        """Save the processed data in multiple formats."""
        if self.data is None:
            self.load_data()
        logger.info("Saving processed data...")
        # Save full dataset as parquet
        parquet_path = self.output_dir / 'processed_triage.parquet'
        self.data.to_parquet(parquet_path)
        logger.info(f"Saved full dataset to: {parquet_path}")
        # Save sample as CSV
        csv_path = self.output_dir / 'processed_triage_sample.csv'
        self.data.head(1000).to_csv(csv_path, index=False)
        logger.info(f"Saved sample (1000 rows) to: {csv_path}")
        # Save statistics
        stats_path = self.output_dir / 'triage_statistics.txt'
        with open(stats_path, 'w') as f:
            f.write("Triage Dataset Statistics\n")
            f.write("========================\n\n")
            f.write(f"Total number of records: {len(self.data)}\n")
            f.write(f"Number of features: {len(self.data.columns)}\n\n")
            # Label distribution
            if self.label_col in self.data.columns:
                f.write("Label Distribution\n")
                f.write("-----------------\n")
                label_counts = self.data[self.label_col].value_counts()
                for label, count in label_counts.items():
                    f.write(f"{label}: {count}\n")
            # Feature information
            f.write("\nFeature Information\n")
            f.write("------------------\n")
            for col in self.data.columns:
                f.write(f"\n{col}:\n")
                f.write(f"  Type: {self.data[col].dtype}\n")
                f.write(f"  Non-null count: {self.data[col].count()}\n")
                if self.data[col].dtype in ['int64', 'float64']:
                    f.write(f"  Min: {self.data[col].min()}\n")
                    f.write(f"  Max: {self.data[col].max()}\n")
                    f.write(f"  Mean: {self.data[col].mean():.2f}\n")
                    f.write(f"  Std: {self.data[col].std():.2f}\n")
        logger.info(f"Saved statistics to: {stats_path}")

    def prepare_data(self) -> Dict[str, Any]:
        """
        Perform all data preparation steps and return the prepared dataset.
        Returns:
            Dictionary containing the prepared data including train/test splits
        """
        self.load_data()
        self.clean_data()
        self.extract_features()
        self.normalize_features()
        self.create_cv_folds()
        self.save_processed_data()
        # Create train/test split
        if self.y is not None:
            X_train, X_test, y_train, y_test = train_test_split(
                self.X_normalized, self.y, test_size=self.test_size, random_state=self.random_state, stratify=self.y
            )
        else:
            X_train = X_test = y_train = y_test = None
        return {
            'X': self.X_normalized,
            'y': self.y,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': self.feature_names,
            'cv_splits': self.cv_folds,
            'scaler': self.scaler
        } 