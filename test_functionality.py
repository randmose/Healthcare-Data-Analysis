"""
Basic Tests for Healthcare Data Analysis

Simple tests to verify core functionality works as expected.
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

import pandas as pd
import numpy as np
from src.data_preprocessing import HealthDataPreprocessor, load_sample_health_data
from src.predictive_models import (
    DiseaseOutbreakPredictor,
    HospitalAdmissionPredictor,
    PatientOutcomePredictor,
    prepare_features_and_target
)


def test_data_loading():
    """Test data loading functionality."""
    print("Testing data loading...")
    data = load_sample_health_data(n_samples=100)
    assert data is not None, "Data should not be None"
    assert len(data) == 100, "Should have 100 samples"
    assert 'patient_id' in data.columns, "Should have patient_id column"
    print("✓ Data loading test passed")


def test_data_preprocessing():
    """Test data preprocessing functionality."""
    print("Testing data preprocessing...")
    
    preprocessor = HealthDataPreprocessor()
    data = load_sample_health_data(n_samples=100)
    preprocessor.data = data
    
    # Test missing value handling
    processed = preprocessor.handle_missing_values(strategy='mean')
    assert processed is not None, "Processed data should not be None"
    
    # Test categorical encoding
    processed = preprocessor.encode_categorical(['gender'], method='onehot')
    assert processed is not None, "Encoded data should not be None"
    
    print("✓ Data preprocessing test passed")


def test_patient_outcome_predictor():
    """Test patient outcome prediction model."""
    print("Testing patient outcome predictor...")
    
    # Load and prepare data
    data = load_sample_health_data(n_samples=200)
    preprocessor = HealthDataPreprocessor()
    preprocessor.data = data
    data = preprocessor.handle_missing_values(strategy='mean')
    data = preprocessor.encode_categorical(['gender', 'admission_type', 'diagnosis_category'])
    
    # Prepare features
    feature_cols = [col for col in data.columns if col not in ['patient_id', 'readmission', 'mortality']]
    X_train, X_test, y_train, y_test = prepare_features_and_target(
        data, 'readmission', feature_cols, test_size=0.3
    )
    
    # Train model
    predictor = PatientOutcomePredictor(outcome_type='readmission')
    metrics = predictor.train(X_train, y_train)
    
    assert metrics is not None, "Training metrics should not be None"
    assert 'accuracy' in metrics, "Should have accuracy metric"
    
    # Test predictions
    predictions = predictor.predict(X_test)
    assert predictions is not None, "Predictions should not be None"
    assert len(predictions) == len(X_test), "Should have prediction for each test sample"
    
    # Test evaluation
    eval_metrics = predictor.evaluate(X_test, y_test)
    assert eval_metrics is not None, "Evaluation metrics should not be None"
    assert 'accuracy' in eval_metrics, "Should have accuracy in evaluation"
    
    print("✓ Patient outcome predictor test passed")


def test_hospital_admission_predictor():
    """Test hospital admission prediction model."""
    print("Testing hospital admission predictor...")
    
    # Load and prepare data
    data = load_sample_health_data(n_samples=200)
    preprocessor = HealthDataPreprocessor()
    preprocessor.data = data
    data = preprocessor.handle_missing_values(strategy='mean')
    data = preprocessor.encode_categorical(['gender', 'admission_type', 'diagnosis_category'])
    
    # Prepare features
    feature_cols = [col for col in data.columns 
                    if col not in ['patient_id', 'readmission', 'mortality', 'length_of_stay']]
    X_train, X_test, y_train, y_test = prepare_features_and_target(
        data, 'length_of_stay', feature_cols, test_size=0.3
    )
    
    # Train model
    predictor = HospitalAdmissionPredictor(model_type='random_forest')
    metrics = predictor.train(X_train, y_train)
    
    assert metrics is not None, "Training metrics should not be None"
    assert 'r2' in metrics, "Should have R² metric"
    
    # Test predictions
    predictions = predictor.predict(X_test)
    assert predictions is not None, "Predictions should not be None"
    assert len(predictions) == len(X_test), "Should have prediction for each test sample"
    
    print("✓ Hospital admission predictor test passed")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running Healthcare Data Analysis Tests")
    print("=" * 60)
    print()
    
    try:
        test_data_loading()
        test_data_preprocessing()
        test_patient_outcome_predictor()
        test_hospital_admission_predictor()
        
        print()
        print("=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        return True
    except Exception as e:
        print()
        print("=" * 60)
        print(f"Tests failed: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
