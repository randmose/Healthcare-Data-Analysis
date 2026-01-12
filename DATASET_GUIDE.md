# Dataset Guide

This guide provides information about the suggested public health datasets and how to use them with this project.

## Recommended Datasets

### 1. World Bank - Health Nutrition and Population Statistics

**About**: Comprehensive global health indicators from the World Bank's open data portal.

**URL**: https://data.worldbank.org/topic/health

**Key Indicators**:
- Disease prevalence rates
- Mortality rates by cause
- Healthcare infrastructure metrics
- Vaccination coverage
- Life expectancy statistics

**Usage**:
```python
from src.data_preprocessing import HealthDataPreprocessor

preprocessor = HealthDataPreprocessor()
data = preprocessor.load_data('data/raw/world_bank_health.csv')
```

**Example Use Cases**:
- Global disease trend analysis
- Healthcare system comparisons
- Mortality rate predictions

---

### 2. Data.gov - Death Rates for Suicide

**About**: Detailed suicide death rates broken down by demographics (sex, race, Hispanic origin, and age) in the United States.

**URL**: https://catalog.data.gov/dataset/death-rates-for-suicide-by-sex-race-hispanic-origin-and-age-united-states

**Data Fields**:
- Year
- Sex
- Race
- Hispanic Origin
- Age Group
- Death Rate per 100,000

**Usage**:
```python
from src.data_preprocessing import HealthDataPreprocessor

preprocessor = HealthDataPreprocessor()
data = preprocessor.load_data('data/raw/suicide_rates.csv')

# Create time features if analyzing trends
data = preprocessor.create_time_features('year')
```

**Example Use Cases**:
- Demographic risk factor analysis
- Trend prediction over time
- Public health intervention planning

---

### 3. Kaggle - Heart Disease Dataset

**About**: Classic heart disease dataset from UCI Machine Learning Repository, available on Kaggle.

**URL**: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset

**Features**:
- Age, sex, chest pain type
- Resting blood pressure
- Cholesterol level
- Fasting blood sugar
- ECG results
- Maximum heart rate achieved
- Exercise-induced angina
- Target: Presence of heart disease

**Usage**:
```python
from src.predictive_models import DiseaseOutbreakPredictor, prepare_features_and_target

# Load data
preprocessor = HealthDataPreprocessor()
data = preprocessor.load_data('data/raw/heart_disease.csv')

# Prepare features
X_train, X_test, y_train, y_test = prepare_features_and_target(
    data, 'target', test_size=0.2
)

# Train model
predictor = DiseaseOutbreakPredictor(model_type='random_forest')
predictor.train(X_train, y_train)
```

**Example Use Cases**:
- Heart disease risk prediction
- Risk factor identification
- Patient outcome classification

---

### 4. Kaggle - Diabetes Dataset

**About**: Pima Indians Diabetes Database for predicting diabetes onset.

**URL**: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

**Features**:
- Pregnancies
- Glucose level
- Blood pressure
- Skin thickness
- Insulin level
- BMI
- Diabetes pedigree function
- Age
- Outcome (diabetes diagnosis)

**Example Use Cases**:
- Diabetes risk prediction
- Feature importance analysis
- Early detection systems

---

### 5. Kaggle - COVID-19 Datasets

**About**: Various COVID-19 datasets including case counts, deaths, and vaccinations.

**URL**: https://www.kaggle.com/datasets?search=covid-19

**Example Use Cases**:
- Disease outbreak prediction
- Vaccination impact analysis
- Time series forecasting

---

## Data Preprocessing Tips

### Handling Missing Values
```python
# For numerical features, use mean/median
data = preprocessor.handle_missing_values(strategy='mean', columns=['age', 'blood_pressure'])

# For categorical features, use mode
data = preprocessor.handle_missing_values(strategy='mode', columns=['gender', 'race'])
```

### Encoding Categorical Variables
```python
# One-hot encoding (recommended for most ML models)
data = preprocessor.encode_categorical(['gender', 'race'], method='onehot')

# Label encoding (for ordinal data)
data = preprocessor.encode_categorical(['severity_level'], method='label')
```

### Handling Time Series Data
```python
# Create time-based features
data = preprocessor.create_time_features('date_column')

# This creates: year, month, day, dayofweek, quarter columns
```

### Dealing with Outliers
```python
# Using IQR method (recommended)
data = preprocessor.remove_outliers(['blood_pressure', 'cholesterol'], method='iqr', threshold=1.5)

# Using Z-score method
data = preprocessor.remove_outliers(['age', 'bmi'], method='zscore', threshold=3)
```

## Model Selection Guide

### Classification Problems (Binary/Multi-class)
- **Disease presence/absence**: Use `DiseaseOutbreakPredictor` or `PatientOutcomePredictor`
- **Readmission risk**: Use `PatientOutcomePredictor(outcome_type='readmission')`
- **Mortality risk**: Use `PatientOutcomePredictor(outcome_type='mortality')`

### Regression Problems
- **Length of stay**: Use `HospitalAdmissionPredictor`
- **Cost prediction**: Use `HospitalAdmissionPredictor`
- **Risk scores**: Use `HospitalAdmissionPredictor`

## Example Workflows

### Complete Analysis Pipeline
```python
# 1. Load data
preprocessor = HealthDataPreprocessor()
data = preprocessor.load_data('data/raw/dataset.csv')

# 2. Preprocess
data = preprocessor.handle_missing_values(strategy='mean')
data = preprocessor.encode_categorical(['categorical_col'], method='onehot')
data = preprocessor.normalize_features(['numerical_col'], method='standard')

# 3. Save processed data
preprocessor.save_processed_data('data/processed/processed_dataset.csv')

# 4. Train model
from src.predictive_models import prepare_features_and_target
X_train, X_test, y_train, y_test = prepare_features_and_target(
    data, 'target_column'
)

predictor = PatientOutcomePredictor()
predictor.train(X_train, y_train)

# 5. Evaluate
metrics = predictor.evaluate(X_test, y_test)
print(f"Accuracy: {metrics['accuracy']:.4f}")

# 6. Visualize
from src.visualization import plot_confusion_matrix, plot_feature_importance
plot_confusion_matrix(metrics['confusion_matrix'])
plot_feature_importance(predictor.feature_importance)
```

## Getting Help

- Check the main README.md for installation and setup
- See example notebooks in `notebooks/` directory
- Run `python quick_start.py` for a simple example
- Run `python src/main_analysis.py` for a complete analysis

## Contributing New Datasets

If you'd like to contribute support for new datasets:
1. Add a data loader function to `src/data_preprocessing.py`
2. Document the dataset in this guide
3. Create an example notebook
4. Submit a pull request

See CONTRIBUTING.md for more details.
