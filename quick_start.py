"""
Quick Start Example for Healthcare Data Analysis

This script demonstrates how to quickly get started with the healthcare
data analysis tools.
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.data_preprocessing import load_sample_health_data, HealthDataPreprocessor
from src.predictive_models import PatientOutcomePredictor, prepare_features_and_target
from src.visualization import plot_confusion_matrix
import matplotlib.pyplot as plt


def quick_start_example():
    """Run a quick example analysis."""
    
    print("Healthcare Data Analysis - Quick Start Example")
    print("=" * 60)
    
    # Step 1: Load sample data
    print("\n1. Loading sample data...")
    data = load_sample_health_data(n_samples=500)
    print(f"   Loaded {len(data)} patient records")
    
    # Step 2: Preprocess data
    print("\n2. Preprocessing data...")
    preprocessor = HealthDataPreprocessor()
    preprocessor.data = data
    data = preprocessor.handle_missing_values(strategy='mean')
    data = preprocessor.encode_categorical(['gender', 'admission_type', 'diagnosis_category'])
    print(f"   Processed data shape: {data.shape}")
    
    # Step 3: Train a prediction model
    print("\n3. Training readmission prediction model...")
    feature_cols = [col for col in data.columns if col not in ['patient_id', 'readmission', 'mortality']]
    X_train, X_test, y_train, y_test = prepare_features_and_target(
        data, 'readmission', feature_cols
    )
    
    predictor = PatientOutcomePredictor(outcome_type='readmission')
    predictor.train(X_train, y_train)
    
    # Step 4: Evaluate the model
    print("\n4. Evaluating model performance...")
    metrics = predictor.evaluate(X_test, y_test)
    
    print("\n   Results:")
    print(f"   - Accuracy:  {metrics['accuracy']:.4f}")
    print(f"   - Precision: {metrics['precision']:.4f}")
    print(f"   - Recall:    {metrics['recall']:.4f}")
    print(f"   - F1-Score:  {metrics['f1_score']:.4f}")
    print(f"   - ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    # Step 5: Visualize results
    print("\n5. Generating visualizations...")
    fig = plot_confusion_matrix(
        metrics['confusion_matrix'],
        labels=['No Readmission', 'Readmission']
    )
    
    # Create output directory if it doesn't exist
    output_dir = Path(__file__).parent / 'output'
    output_dir.mkdir(exist_ok=True)
    
    plt.savefig(output_dir / 'quick_start_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved confusion matrix to: {output_dir / 'quick_start_confusion_matrix.png'}")
    
    print("\n" + "=" * 60)
    print("Quick start example completed successfully!")
    print("\nNext steps:")
    print("  - Run the full analysis: python src/main_analysis.py")
    print("  - Explore the Jupyter notebook: jupyter notebook notebooks/exploratory_analysis.ipynb")
    print("  - Load your own data and customize the analysis")


if __name__ == "__main__":
    quick_start_example()
