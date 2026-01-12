# Data Directory

This directory contains the healthcare datasets used for analysis.

## Directory Structure

- `raw/`: Original, immutable data dump
- `processed/`: Cleaned and processed data ready for analysis

## Suggested Datasets

### 1. World Bank - Health Nutrition and Population Statistics
- **Source**: https://data.worldbank.org/topic/health
- **Description**: Global health indicators including disease prevalence, mortality rates, and healthcare access
- **Format**: CSV, Excel
- **Recommended files**: Place in `raw/` directory

### 2. Data.gov - Death Rates for Suicide
- **Source**: https://catalog.data.gov/dataset/death-rates-for-suicide-by-sex-race-hispanic-origin-and-age-united-states
- **Description**: Suicide death rates by demographics
- **Format**: CSV
- **Recommended files**: Place in `raw/` directory

### 3. Kaggle Health Datasets
- **Source**: https://www.kaggle.com/datasets
- **Popular datasets**:
  - Heart Disease UCI
  - Diabetes Dataset
  - COVID-19 Data
  - Hospital Readmissions
- **Format**: CSV
- **Recommended files**: Place in `raw/` directory

## Usage

1. Download datasets from the sources above
2. Place raw data files in the `raw/` directory
3. Run the preprocessing scripts to generate processed data in the `processed/` directory

## Data Loading

```python
from src.data_preprocessing import HealthDataPreprocessor

# Load your data
preprocessor = HealthDataPreprocessor()
data = preprocessor.load_data('data/raw/your_dataset.csv')

# Process the data
data = preprocessor.handle_missing_values(strategy='mean')
data = preprocessor.encode_categorical(['gender', 'category'], method='onehot')

# Save processed data
preprocessor.save_processed_data('data/processed/processed_dataset.csv')
```

## Sample Data

If you don't have real datasets yet, the project includes a sample data generator:

```python
from src.data_preprocessing import load_sample_health_data

# Generate sample healthcare data
sample_data = load_sample_health_data(n_samples=1000)
```

## Notes

- Keep raw data in the `raw/` directory unchanged
- All processing should be done on copies of the data
- Processed data should be saved in the `processed/` directory
- Large data files (>50MB) are excluded from version control via .gitignore
