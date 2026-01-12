"""
Data Preprocessing Utilities

Functions for loading, cleaning, and preprocessing healthcare datasets.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, List
import warnings

# Suppress specific warnings that we expect and handle
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)


class HealthDataPreprocessor:
    """Preprocessor for healthcare datasets."""
    
    def __init__(self):
        self.data = None
        self.processed_data = None
        
    def load_data(self, filepath: str, **kwargs) -> pd.DataFrame:
        """
        Load data from various file formats.
        
        Args:
            filepath: Path to the data file
            **kwargs: Additional arguments for pandas read functions
            
        Returns:
            DataFrame containing the loaded data
        """
        if filepath.endswith('.csv'):
            self.data = pd.read_csv(filepath, **kwargs)
        elif filepath.endswith('.xlsx') or filepath.endswith('.xls'):
            self.data = pd.read_excel(filepath, **kwargs)
        elif filepath.endswith('.json'):
            self.data = pd.read_json(filepath, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {filepath}")
        
        print(f"Loaded data with shape: {self.data.shape}")
        return self.data
    
    def handle_missing_values(self, strategy: str = 'mean', columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            strategy: Strategy for handling missing values ('mean', 'median', 'mode', 'drop')
            columns: List of columns to process (None for all)
            
        Returns:
            DataFrame with handled missing values
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        df = self.data.copy()
        cols = columns if columns else df.columns
        
        for col in cols:
            if col not in df.columns:
                continue
                
            if strategy == 'mean' and df[col].dtype in ['int64', 'float64']:
                df[col].fillna(df[col].mean(), inplace=True)
            elif strategy == 'median' and df[col].dtype in ['int64', 'float64']:
                df[col].fillna(df[col].median(), inplace=True)
            elif strategy == 'mode':
                mode_values = df[col].mode()
                fill_value = mode_values[0] if len(mode_values) > 0 else (0 if df[col].dtype in ['int64', 'float64'] else 'Unknown')
                df[col].fillna(fill_value, inplace=True)
            elif strategy == 'drop':
                df.dropna(subset=[col], inplace=True)
        
        print(f"After handling missing values: {df.shape}")
        self.processed_data = df
        return df
    
    def normalize_features(self, columns: List[str], method: str = 'standard') -> pd.DataFrame:
        """
        Normalize numerical features.
        
        Args:
            columns: List of columns to normalize
            method: Normalization method ('standard', 'minmax')
            
        Returns:
            DataFrame with normalized features
        """
        if self.processed_data is None:
            df = self.data.copy() if self.data is not None else None
            if df is None:
                raise ValueError("No data loaded.")
        else:
            df = self.processed_data.copy()
        
        for col in columns:
            if col not in df.columns or df[col].dtype not in ['int64', 'float64']:
                continue
            
            if method == 'standard':
                mean = df[col].mean()
                std = df[col].std()
                df[col] = (df[col] - mean) / std if std != 0 else df[col]
            elif method == 'minmax':
                min_val = df[col].min()
                max_val = df[col].max()
                df[col] = (df[col] - min_val) / (max_val - min_val) if max_val != min_val else df[col]
        
        self.processed_data = df
        return df
    
    def encode_categorical(self, columns: List[str], method: str = 'onehot') -> pd.DataFrame:
        """
        Encode categorical variables.
        
        Args:
            columns: List of columns to encode
            method: Encoding method ('onehot', 'label')
            
        Returns:
            DataFrame with encoded categorical variables
        """
        if self.processed_data is None:
            df = self.data.copy() if self.data is not None else None
            if df is None:
                raise ValueError("No data loaded.")
        else:
            df = self.processed_data.copy()
        
        for col in columns:
            if col not in df.columns:
                continue
            
            if method == 'onehot':
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df, dummies], axis=1)
                df.drop(col, axis=1, inplace=True)
            elif method == 'label':
                df[col] = pd.Categorical(df[col]).codes
        
        self.processed_data = df
        return df
    
    def create_time_features(self, date_column: str) -> pd.DataFrame:
        """
        Create time-based features from a date column.
        
        Args:
            date_column: Name of the date column
            
        Returns:
            DataFrame with additional time features
        """
        if self.processed_data is None:
            df = self.data.copy() if self.data is not None else None
            if df is None:
                raise ValueError("No data loaded.")
        else:
            df = self.processed_data.copy()
        
        if date_column not in df.columns:
            raise ValueError(f"Column {date_column} not found in data.")
        
        df[date_column] = pd.to_datetime(df[date_column])
        df['year'] = df[date_column].dt.year
        df['month'] = df[date_column].dt.month
        df['day'] = df[date_column].dt.day
        df['dayofweek'] = df[date_column].dt.dayofweek
        df['quarter'] = df[date_column].dt.quarter
        
        self.processed_data = df
        return df
    
    def remove_outliers(self, columns: List[str], method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
        """
        Remove outliers from numerical columns.
        
        Args:
            columns: List of columns to process
            method: Method for outlier detection ('iqr', 'zscore')
            threshold: Threshold for outlier detection
            
        Returns:
            DataFrame with outliers removed
        """
        if self.processed_data is None:
            df = self.data.copy() if self.data is not None else None
            if df is None:
                raise ValueError("No data loaded.")
        else:
            df = self.processed_data.copy()
        
        for col in columns:
            if col not in df.columns or df[col].dtype not in ['int64', 'float64']:
                continue
            
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            elif method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                df = df[z_scores < threshold]
        
        print(f"After removing outliers: {df.shape}")
        self.processed_data = df
        return df
    
    def save_processed_data(self, filepath: str):
        """
        Save processed data to file.
        
        Args:
            filepath: Path to save the processed data
        """
        if self.processed_data is None:
            raise ValueError("No processed data available.")
        
        if filepath.endswith('.csv'):
            self.processed_data.to_csv(filepath, index=False)
        elif filepath.endswith('.xlsx'):
            self.processed_data.to_excel(filepath, index=False)
        else:
            raise ValueError(f"Unsupported file format: {filepath}")
        
        print(f"Processed data saved to: {filepath}")


def load_sample_health_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    Generate sample healthcare data for demonstration purposes.
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        DataFrame with sample healthcare data
    """
    np.random.seed(42)
    
    data = {
        'patient_id': range(1, n_samples + 1),
        'age': np.random.randint(18, 90, n_samples),
        'gender': np.random.choice(['M', 'F'], n_samples),
        'blood_pressure': np.random.randint(90, 180, n_samples),
        'heart_rate': np.random.randint(60, 120, n_samples),
        'temperature': np.random.uniform(36.0, 39.5, n_samples),
        'respiratory_rate': np.random.randint(12, 25, n_samples),
        'admission_type': np.random.choice(['Emergency', 'Urgent', 'Elective'], n_samples),
        'diagnosis_category': np.random.choice(['Cardiovascular', 'Respiratory', 'Infectious', 'Other'], n_samples),
        'length_of_stay': np.random.randint(1, 30, n_samples),
        'readmission': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        'mortality': np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
    }
    
    return pd.DataFrame(data)
