"""
Utility Functions for Healthcare Data Analysis

Helper functions for various common tasks.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import json
from pathlib import Path


def calculate_basic_statistics(data: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
    """
    Calculate basic statistics for numerical columns.
    
    Args:
        data: Input DataFrame
        columns: List of columns to analyze (None for all numerical)
        
    Returns:
        DataFrame with statistics
    """
    if columns is None:
        columns = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    stats = pd.DataFrame({
        'Mean': data[columns].mean(),
        'Median': data[columns].median(),
        'Std': data[columns].std(),
        'Min': data[columns].min(),
        'Max': data[columns].max(),
        'Q1': data[columns].quantile(0.25),
        'Q3': data[columns].quantile(0.75),
        'Missing': data[columns].isnull().sum(),
        'Missing %': (data[columns].isnull().sum() / len(data) * 100).round(2)
    })
    
    return stats


def generate_model_report(metrics: Dict[str, Any], model_name: str = "Model") -> str:
    """
    Generate a formatted report from model metrics.
    
    Args:
        metrics: Dictionary of metrics
        model_name: Name of the model
        
    Returns:
        Formatted report string
    """
    report = f"\n{'='*60}\n"
    report += f"{model_name} Performance Report\n"
    report += f"{'='*60}\n\n"
    
    for key, value in metrics.items():
        if key not in ['confusion_matrix', 'classification_report']:
            if isinstance(value, (int, float)):
                report += f"{key.replace('_', ' ').title()}: {value:.4f}\n"
            else:
                report += f"{key.replace('_', ' ').title()}: {value}\n"
    
    report += f"\n{'='*60}\n"
    return report


def save_model_results(results: Dict[str, Any], filepath: str):
    """
    Save model results to JSON file.
    
    Args:
        results: Dictionary of results to save
        filepath: Path to save the file
    """
    # Convert numpy arrays to lists for JSON serialization
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            serializable_results[key] = value.tolist()
        elif isinstance(value, (np.integer, np.floating)):
            serializable_results[key] = float(value)
        else:
            serializable_results[key] = value
    
    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"Results saved to {filepath}")


def load_model_results(filepath: str) -> Dict[str, Any]:
    """
    Load model results from JSON file.
    
    Args:
        filepath: Path to the results file
        
    Returns:
        Dictionary of results
    """
    with open(filepath, 'r') as f:
        results = json.load(f)
    
    return results


def compare_models(results_dict: Dict[str, Dict[str, float]], metric: str = 'accuracy') -> pd.DataFrame:
    """
    Compare multiple models based on a specific metric.
    
    Args:
        results_dict: Dictionary with model names as keys and metrics as values
        metric: Metric to compare
        
    Returns:
        DataFrame with comparison results
    """
    comparison = pd.DataFrame({
        'Model': list(results_dict.keys()),
        metric.title(): [results[metric] for results in results_dict.values()]
    })
    
    comparison = comparison.sort_values(metric.title(), ascending=False)
    return comparison


def create_summary_statistics(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Create comprehensive summary statistics for a dataset.
    
    Args:
        data: Input DataFrame
        
    Returns:
        Dictionary with summary statistics
    """
    summary = {
        'total_records': len(data),
        'total_features': len(data.columns),
        'numerical_features': len(data.select_dtypes(include=['int64', 'float64']).columns),
        'categorical_features': len(data.select_dtypes(include=['object', 'category']).columns),
        'missing_values': data.isnull().sum().sum(),
        'missing_percentage': (data.isnull().sum().sum() / (len(data) * len(data.columns)) * 100),
        'duplicate_rows': data.duplicated().sum(),
        'memory_usage_mb': data.memory_usage(deep=True).sum() / 1024**2
    }
    
    return summary


def export_predictions(y_true: np.ndarray, y_pred: np.ndarray, 
                      ids: np.ndarray = None, filepath: str = None) -> pd.DataFrame:
    """
    Export predictions to a DataFrame or CSV file.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        ids: Optional IDs for each prediction
        filepath: Optional path to save CSV
        
    Returns:
        DataFrame with predictions
    """
    if ids is None:
        ids = np.arange(len(y_true))
    
    predictions_df = pd.DataFrame({
        'ID': ids,
        'True_Value': y_true,
        'Predicted_Value': y_pred,
        'Correct': y_true == y_pred
    })
    
    if filepath:
        predictions_df.to_csv(filepath, index=False)
        print(f"Predictions exported to {filepath}")
    
    return predictions_df


def calculate_confidence_interval(data: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Calculate confidence interval for a dataset.
    
    Args:
        data: Input data
        confidence: Confidence level (default 0.95)
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    from scipy import stats
    
    mean = np.mean(data)
    std_err = stats.sem(data)
    margin = std_err * stats.t.ppf((1 + confidence) / 2, len(data) - 1)
    
    return (mean - margin, mean + margin)


def print_dataset_info(data: pd.DataFrame, name: str = "Dataset"):
    """
    Print comprehensive information about a dataset.
    
    Args:
        data: Input DataFrame
        name: Name of the dataset
    """
    print(f"\n{'='*60}")
    print(f"{name} Information")
    print(f"{'='*60}\n")
    
    print(f"Shape: {data.shape[0]} rows × {data.shape[1]} columns")
    print(f"\nColumn Types:")
    print(data.dtypes.value_counts())
    
    print(f"\nMissing Values:")
    missing = data.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("No missing values")
    
    print(f"\nMemory Usage: {data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print(f"\n{'='*60}\n")


def validate_data_quality(data: pd.DataFrame, 
                         max_missing_pct: float = 0.5,
                         check_duplicates: bool = True) -> Dict[str, Any]:
    """
    Validate data quality and return quality report.
    
    Args:
        data: Input DataFrame
        max_missing_pct: Maximum allowed missing percentage per column
        check_duplicates: Whether to check for duplicate rows
        
    Returns:
        Dictionary with quality metrics and issues
    """
    issues = []
    
    # Check missing values
    missing_pct = data.isnull().sum() / len(data)
    high_missing = missing_pct[missing_pct > max_missing_pct]
    
    if len(high_missing) > 0:
        issues.append({
            'type': 'high_missing_values',
            'columns': high_missing.index.tolist(),
            'percentages': high_missing.values.tolist()
        })
    
    # Check duplicates
    if check_duplicates:
        n_duplicates = data.duplicated().sum()
        if n_duplicates > 0:
            issues.append({
                'type': 'duplicate_rows',
                'count': n_duplicates,
                'percentage': n_duplicates / len(data) * 100
            })
    
    # Check for constant columns
    constant_cols = [col for col in data.columns if data[col].nunique() <= 1]
    if constant_cols:
        issues.append({
            'type': 'constant_columns',
            'columns': constant_cols
        })
    
    quality_score = 100 - (len(issues) * 10)  # Simple quality score
    
    return {
        'quality_score': max(0, quality_score),
        'issues': issues,
        'passed': len(issues) == 0
    }
