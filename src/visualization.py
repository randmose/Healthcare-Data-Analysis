"""
Visualization Utilities for Healthcare Data Analysis

Functions for creating various plots and visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Tuple
import warnings

# Suppress specific matplotlib warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def plot_feature_distributions(data: pd.DataFrame, columns: List[str], ncols: int = 3, figsize: Tuple[int, int] = (15, 10)):
    """
    Plot distributions of multiple features.
    
    Args:
        data: Input DataFrame
        columns: List of columns to plot
        ncols: Number of columns in the subplot grid
        figsize: Figure size
    """
    n_features = len(columns)
    nrows = (n_features + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = axes.flatten() if n_features > 1 else [axes]
    
    for idx, col in enumerate(columns):
        if col in data.columns:
            if data[col].dtype in ['int64', 'float64']:
                axes[idx].hist(data[col].dropna(), bins=30, edgecolor='black', alpha=0.7)
                axes[idx].set_title(f'Distribution of {col}')
                axes[idx].set_xlabel(col)
                axes[idx].set_ylabel('Frequency')
            else:
                data[col].value_counts().plot(kind='bar', ax=axes[idx], edgecolor='black')
                axes[idx].set_title(f'Distribution of {col}')
                axes[idx].set_xlabel(col)
                axes[idx].set_ylabel('Count')
                axes[idx].tick_params(axis='x', rotation=45)
    
    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    return fig


def plot_correlation_matrix(data: pd.DataFrame, figsize: Tuple[int, int] = (12, 10), method: str = 'pearson'):
    """
    Plot correlation matrix heatmap.
    
    Args:
        data: Input DataFrame
        figsize: Figure size
        method: Correlation method ('pearson', 'spearman', 'kendall')
    """
    # Select only numerical columns
    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns
    
    if len(numerical_cols) == 0:
        print("No numerical columns found for correlation matrix.")
        return None
    
    corr_matrix = data[numerical_cols].corr(method=method)
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, ax=ax)
    ax.set_title(f'{method.capitalize()} Correlation Matrix')
    plt.tight_layout()
    return fig


def plot_feature_importance(feature_importance: pd.DataFrame, top_n: int = 15, figsize: Tuple[int, int] = (10, 8)):
    """
    Plot feature importance from a trained model.
    
    Args:
        feature_importance: DataFrame with 'feature' and 'importance' columns
        top_n: Number of top features to display
        figsize: Figure size
    """
    if feature_importance is None or len(feature_importance) == 0:
        print("No feature importance data available.")
        return None
    
    top_features = feature_importance.head(top_n)
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(data=top_features, x='importance', y='feature', ax=ax, palette='viridis')
    ax.set_title(f'Top {top_n} Most Important Features')
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    plt.tight_layout()
    return fig


def plot_confusion_matrix(cm: np.ndarray, labels: Optional[List[str]] = None, figsize: Tuple[int, int] = (8, 6)):
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix array
        labels: Class labels
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=labels if labels else 'auto',
                yticklabels=labels if labels else 'auto')
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.tight_layout()
    return fig


def plot_time_series(data: pd.DataFrame, date_column: str, value_column: str, 
                     title: Optional[str] = None, figsize: Tuple[int, int] = (14, 6)):
    """
    Plot time series data.
    
    Args:
        data: Input DataFrame
        date_column: Name of the date column
        value_column: Name of the value column to plot
        title: Plot title
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    data_sorted = data.sort_values(date_column)
    ax.plot(data_sorted[date_column], data_sorted[value_column], linewidth=2)
    ax.set_xlabel('Date')
    ax.set_ylabel(value_column)
    ax.set_title(title if title else f'{value_column} Over Time')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


def plot_prediction_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, 
                               title: str = 'Predictions vs Actual', 
                               figsize: Tuple[int, int] = (10, 6)):
    """
    Plot predicted values vs actual values.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        title: Plot title
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.scatter(y_true, y_pred, alpha=0.5)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
            'r--', lw=2, label='Perfect Prediction')
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_model_comparison(results: dict, metric: str = 'accuracy', 
                          title: Optional[str] = None, figsize: Tuple[int, int] = (10, 6)):
    """
    Plot comparison of multiple models.
    
    Args:
        results: Dictionary with model names as keys and metrics as values
        metric: Metric to compare
        title: Plot title
        figsize: Figure size
    """
    models = list(results.keys())
    values = [results[model].get(metric, 0) for model in models]
    
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(models, values, edgecolor='black', alpha=0.7)
    
    # Color bars
    colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(title if title else f'Model Comparison - {metric.replace("_", " ").title()}')
    ax.set_ylim([0, max(values) * 1.2])
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on bars
    for i, v in enumerate(values):
        ax.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    return fig


def plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray, 
                   title: str = 'ROC Curve', figsize: Tuple[int, int] = (8, 6)):
    """
    Plot ROC curve for binary classification.
    
    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities for positive class
        title: Plot title
        figsize: Figure size
    """
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def create_health_dashboard(data: pd.DataFrame, save_path: Optional[str] = None):
    """
    Create a comprehensive health data dashboard.
    
    Args:
        data: Input DataFrame
        save_path: Path to save the dashboard image
    """
    fig = plt.figure(figsize=(20, 12))
    
    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Select numerical columns
    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns[:6]
    
    # Plot distributions
    for idx, col in enumerate(numerical_cols[:6]):
        row = idx // 3
        col_pos = idx % 3
        ax = fig.add_subplot(gs[row, col_pos])
        
        if data[col].dtype in ['int64', 'float64']:
            ax.hist(data[col].dropna(), bins=30, edgecolor='black', alpha=0.7, color='steelblue')
            ax.set_title(f'Distribution of {col}', fontsize=10)
            ax.set_xlabel(col, fontsize=8)
            ax.set_ylabel('Frequency', fontsize=8)
    
    plt.suptitle('Healthcare Data Dashboard', fontsize=16, y=0.995)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Dashboard saved to {save_path}")
    
    return fig


def save_plot(fig, filepath: str, dpi: int = 300):
    """
    Save a matplotlib figure to file.
    
    Args:
        fig: Matplotlib figure object
        filepath: Path to save the figure
        dpi: Resolution in dots per inch
    """
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    print(f"Plot saved to {filepath}")
