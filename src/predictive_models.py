"""
Predictive Models for Healthcare Analysis

Models for predicting disease outbreaks, hospital admissions, and patient outcomes.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.preprocessing import StandardScaler
import warnings

# Suppress specific warnings that we expect and handle
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')


class DiseaseOutbreakPredictor:
    """Predict disease outbreaks using classification models."""
    
    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize the disease outbreak predictor.
        
        Args:
            model_type: Type of model ('random_forest', 'gradient_boosting', 'logistic')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
        
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        elif model_type == 'logistic':
            self.model = LogisticRegression(random_state=42, max_iter=1000)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, float]:
        """
        Train the disease outbreak prediction model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Dictionary with training metrics
        """
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Get training predictions
        y_pred = self.model.predict(X_train_scaled)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_train, y_pred),
            'precision': precision_score(y_train, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_train, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_train, y_pred, average='weighted', zero_division=0)
        }
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        print(f"Training completed. Accuracy: {metrics['accuracy']:.4f}")
        return metrics
    
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X_test: Test features
            
        Returns:
            Predicted labels
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        X_test_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_test_scaled)
    
    def predict_proba(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X_test: Test features
            
        Returns:
            Prediction probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        X_test_scaled = self.scaler.transform(X_test)
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_test_scaled)
        else:
            raise ValueError("Model does not support probability predictions.")
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Evaluate model performance on test data.
        
        Args:
            X_test: Test features
            y_test: True labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        y_pred = self.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred)
        }
        
        # Add ROC AUC for binary classification
        if len(np.unique(y_test)) == 2 and hasattr(self.model, 'predict_proba'):
            y_proba = self.predict_proba(X_test)[:, 1]
            metrics['roc_auc'] = roc_auc_score(y_test, y_proba)
        
        return metrics


class HospitalAdmissionPredictor:
    """Predict hospital admissions using regression models."""
    
    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize the hospital admission predictor.
        
        Args:
            model_type: Type of model ('random_forest', 'linear')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
        
        if model_type == 'random_forest':
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_type == 'linear':
            self.model = LinearRegression()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, float]:
        """
        Train the hospital admission prediction model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Dictionary with training metrics
        """
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Get training predictions
        y_pred = self.model.predict(X_train_scaled)
        
        # Calculate metrics
        metrics = {
            'mse': mean_squared_error(y_train, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_train, y_pred)),
            'mae': mean_absolute_error(y_train, y_pred),
            'r2': r2_score(y_train, y_pred)
        }
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        print(f"Training completed. R² Score: {metrics['r2']:.4f}, RMSE: {metrics['rmse']:.4f}")
        return metrics
    
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X_test: Test features
            
        Returns:
            Predicted values
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        X_test_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_test_scaled)
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate model performance on test data.
        
        Args:
            X_test: Test features
            y_test: True values
            
        Returns:
            Dictionary with evaluation metrics
        """
        y_pred = self.predict(X_test)
        
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
        
        return metrics


class PatientOutcomePredictor:
    """Predict patient outcomes (readmission, mortality) using classification models."""
    
    def __init__(self, outcome_type: str = 'readmission', model_type: str = 'random_forest'):
        """
        Initialize the patient outcome predictor.
        
        Args:
            outcome_type: Type of outcome to predict ('readmission', 'mortality')
            model_type: Type of model ('random_forest', 'gradient_boosting', 'logistic')
        """
        self.outcome_type = outcome_type
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
        
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight='balanced'
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        elif model_type == 'logistic':
            self.model = LogisticRegression(
                random_state=42,
                max_iter=1000,
                class_weight='balanced'
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, float]:
        """
        Train the patient outcome prediction model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Dictionary with training metrics
        """
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Get training predictions
        y_pred = self.model.predict(X_train_scaled)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_train, y_pred),
            'precision': precision_score(y_train, y_pred, average='binary', zero_division=0),
            'recall': recall_score(y_train, y_pred, average='binary', zero_division=0),
            'f1_score': f1_score(y_train, y_pred, average='binary', zero_division=0)
        }
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        print(f"Training completed for {self.outcome_type} prediction.")
        print(f"Accuracy: {metrics['accuracy']:.4f}, F1-Score: {metrics['f1_score']:.4f}")
        return metrics
    
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X_test: Test features
            
        Returns:
            Predicted labels
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        X_test_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_test_scaled)
    
    def predict_proba(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X_test: Test features
            
        Returns:
            Prediction probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        X_test_scaled = self.scaler.transform(X_test)
        return self.model.predict_proba(X_test_scaled)
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Evaluate model performance on test data.
        
        Args:
            X_test: Test features
            y_test: True labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='binary', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='binary', zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average='binary', zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred)
        }
        
        return metrics


def prepare_features_and_target(
    data: pd.DataFrame,
    target_column: str,
    feature_columns: Optional[list] = None,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepare features and target for model training.
    
    Args:
        data: Input DataFrame
        target_column: Name of the target column
        feature_columns: List of feature column names (None for all except target)
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    if feature_columns is None:
        feature_columns = [col for col in data.columns if col != target_column]
    
    X = data[feature_columns]
    y = data[target_column]
    
    # Check if we should stratify (for classification with reasonable class counts)
    should_stratify = False
    if y.dtype == 'object' or len(y.unique()) < 10:
        # Ensure no NaN values in target for stratification
        if not y.isnull().any():
            should_stratify = True
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, 
        stratify=y if should_stratify else None
    )
    
    print(f"Train set size: {len(X_train)}, Test set size: {len(X_test)}")
    return X_train, X_test, y_train, y_test
