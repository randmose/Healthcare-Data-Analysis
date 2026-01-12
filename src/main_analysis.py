"""
Main Analysis Script for Healthcare Data

This script demonstrates the complete healthcare data analysis pipeline,
including data preprocessing, model training, evaluation, and visualization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add src directory to path
sys.path.append(str(Path(__file__).parent))

from data_preprocessing import HealthDataPreprocessor, load_sample_health_data
from predictive_models import (
    DiseaseOutbreakPredictor,
    HospitalAdmissionPredictor,
    PatientOutcomePredictor,
    prepare_features_and_target
)
from visualization import (
    plot_feature_distributions,
    plot_correlation_matrix,
    plot_feature_importance,
    plot_confusion_matrix,
    plot_prediction_vs_actual,
    create_health_dashboard
)


def main():
    """Run the complete healthcare data analysis pipeline."""
    
    print("=" * 80)
    print("HEALTHCARE DATA ANALYSIS")
    print("=" * 80)
    print()
    
    # Create directories
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Step 1: Load and preprocess data
    print("[1/6] Loading and preprocessing data...")
    print("-" * 80)
    
    # Load sample data (in practice, replace with actual dataset)
    data = load_sample_health_data(n_samples=1000)
    print(f"Loaded {len(data)} patient records")
    print(f"Columns: {', '.join(data.columns)}")
    print()
    
    # Initialize preprocessor
    preprocessor = HealthDataPreprocessor()
    preprocessor.data = data
    
    # Handle missing values
    data = preprocessor.handle_missing_values(strategy='mean')
    
    # Encode categorical variables
    categorical_cols = ['gender', 'admission_type', 'diagnosis_category']
    data = preprocessor.encode_categorical(categorical_cols, method='onehot')
    
    print(f"After preprocessing: {data.shape}")
    print()
    
    # Step 2: Exploratory Data Analysis
    print("[2/6] Performing exploratory data analysis...")
    print("-" * 80)
    
    # Select numerical columns for visualization
    numerical_cols = ['age', 'blood_pressure', 'heart_rate', 'temperature', 
                      'respiratory_rate', 'length_of_stay']
    
    # Create visualizations
    fig1 = plot_feature_distributions(data, numerical_cols)
    plt.savefig(output_dir / "feature_distributions.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Feature distributions plot saved")
    
    fig2 = plot_correlation_matrix(data)
    plt.savefig(output_dir / "correlation_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Correlation matrix saved")
    
    # Create dashboard
    fig3 = create_health_dashboard(data, save_path=output_dir / "dashboard.png")
    plt.close()
    print("✓ Health dashboard saved")
    print()
    
    # Step 3: Predict Patient Readmissions
    print("[3/6] Training readmission prediction model...")
    print("-" * 80)
    
    # Prepare features for readmission prediction
    feature_cols = [col for col in data.columns if col not in ['patient_id', 'readmission', 'mortality']]
    X_train, X_test, y_train, y_test = prepare_features_and_target(
        data, 'readmission', feature_cols
    )
    
    # Train readmission predictor
    readmission_predictor = PatientOutcomePredictor(
        outcome_type='readmission',
        model_type='random_forest'
    )
    train_metrics = readmission_predictor.train(X_train, y_train)
    
    # Evaluate model
    test_metrics = readmission_predictor.evaluate(X_test, y_test)
    print(f"\nReadmission Prediction Results:")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1-Score:  {test_metrics['f1_score']:.4f}")
    print(f"  ROC-AUC:   {test_metrics['roc_auc']:.4f}")
    
    # Plot confusion matrix
    fig4 = plot_confusion_matrix(
        test_metrics['confusion_matrix'],
        labels=['No Readmission', 'Readmission']
    )
    plt.savefig(output_dir / "readmission_confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Confusion matrix saved")
    
    # Plot feature importance
    if readmission_predictor.feature_importance is not None:
        fig5 = plot_feature_importance(readmission_predictor.feature_importance)
        plt.savefig(output_dir / "readmission_feature_importance.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Feature importance plot saved")
    print()
    
    # Step 4: Predict Mortality Risk
    print("[4/6] Training mortality prediction model...")
    print("-" * 80)
    
    # Prepare features for mortality prediction
    X_train_m, X_test_m, y_train_m, y_test_m = prepare_features_and_target(
        data, 'mortality', feature_cols
    )
    
    # Train mortality predictor
    mortality_predictor = PatientOutcomePredictor(
        outcome_type='mortality',
        model_type='gradient_boosting'
    )
    train_metrics_m = mortality_predictor.train(X_train_m, y_train_m)
    
    # Evaluate model
    test_metrics_m = mortality_predictor.evaluate(X_test_m, y_test_m)
    print(f"\nMortality Prediction Results:")
    print(f"  Accuracy:  {test_metrics_m['accuracy']:.4f}")
    print(f"  Precision: {test_metrics_m['precision']:.4f}")
    print(f"  Recall:    {test_metrics_m['recall']:.4f}")
    print(f"  F1-Score:  {test_metrics_m['f1_score']:.4f}")
    print(f"  ROC-AUC:   {test_metrics_m['roc_auc']:.4f}")
    
    # Plot confusion matrix
    fig6 = plot_confusion_matrix(
        test_metrics_m['confusion_matrix'],
        labels=['Survived', 'Deceased']
    )
    plt.savefig(output_dir / "mortality_confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Confusion matrix saved")
    print()
    
    # Step 5: Predict Length of Stay
    print("[5/6] Training length of stay prediction model...")
    print("-" * 80)
    
    # Prepare features for length of stay prediction
    los_feature_cols = [col for col in data.columns 
                        if col not in ['patient_id', 'readmission', 'mortality', 'length_of_stay']]
    X_train_los, X_test_los, y_train_los, y_test_los = prepare_features_and_target(
        data, 'length_of_stay', los_feature_cols
    )
    
    # Train hospital admission predictor (using length of stay as proxy)
    los_predictor = HospitalAdmissionPredictor(model_type='random_forest')
    train_metrics_los = los_predictor.train(X_train_los, y_train_los)
    
    # Evaluate model
    test_metrics_los = los_predictor.evaluate(X_test_los, y_test_los)
    print(f"\nLength of Stay Prediction Results:")
    print(f"  R² Score: {test_metrics_los['r2']:.4f}")
    print(f"  RMSE:     {test_metrics_los['rmse']:.4f}")
    print(f"  MAE:      {test_metrics_los['mae']:.4f}")
    
    # Plot predictions vs actual
    y_pred_los = los_predictor.predict(X_test_los)
    fig7 = plot_prediction_vs_actual(
        y_test_los.values,
        y_pred_los,
        title='Length of Stay: Predictions vs Actual'
    )
    plt.savefig(output_dir / "los_predictions.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Predictions plot saved")
    
    # Plot feature importance
    if los_predictor.feature_importance is not None:
        fig8 = plot_feature_importance(los_predictor.feature_importance)
        plt.savefig(output_dir / "los_feature_importance.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Feature importance plot saved")
    print()
    
    # Step 6: Summary Report
    print("[6/6] Generating summary report...")
    print("-" * 80)
    
    # Create summary report
    summary = {
        'Readmission Model': {
            'Accuracy': test_metrics['accuracy'],
            'F1-Score': test_metrics['f1_score'],
            'ROC-AUC': test_metrics['roc_auc']
        },
        'Mortality Model': {
            'Accuracy': test_metrics_m['accuracy'],
            'F1-Score': test_metrics_m['f1_score'],
            'ROC-AUC': test_metrics_m['roc_auc']
        },
        'Length of Stay Model': {
            'R² Score': test_metrics_los['r2'],
            'RMSE': test_metrics_los['rmse'],
            'MAE': test_metrics_los['mae']
        }
    }
    
    # Save summary to file
    summary_df = pd.DataFrame(summary).T
    summary_df.to_csv(output_dir / "model_summary.csv")
    print("✓ Model summary saved to CSV")
    
    print()
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nAll results have been saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - feature_distributions.png")
    print("  - correlation_matrix.png")
    print("  - dashboard.png")
    print("  - readmission_confusion_matrix.png")
    print("  - readmission_feature_importance.png")
    print("  - mortality_confusion_matrix.png")
    print("  - los_predictions.png")
    print("  - los_feature_importance.png")
    print("  - model_summary.csv")
    print()
    print("Summary of Results:")
    print(summary_df.to_string())
    print()


if __name__ == "__main__":
    main()
