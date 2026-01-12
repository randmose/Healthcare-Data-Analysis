# Healthcare Data Analysis

A comprehensive data analysis project for predicting disease outbreaks, hospital admissions, and patient outcomes using public health datasets.

## Overview

This project utilizes machine learning and statistical analysis to:
- Predict disease outbreak patterns
- Forecast hospital admission rates
- Analyze patient outcomes
- Provide insights from public health data

## Datasets

This project is designed to work with various public health datasets including:
- **World Bank Open Data**: Health Nutrition and Population Statistics
- **Data.gov**: Death rates for suicide by demographics
- **Kaggle**: Various health-related datasets

## Project Structure

```
Healthcare-Data-Analysis/
├── data/
│   ├── raw/              # Original, immutable data
│   └── processed/        # Cleaned, processed data
├── models/               # Trained models
├── notebooks/            # Jupyter notebooks for analysis
├── src/                  # Source code for the project
├── utils/                # Utility functions
├── requirements.txt      # Project dependencies
└── README.md            # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/randmose/Healthcare-Data-Analysis.git
cd Healthcare-Data-Analysis
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Main Analysis

```bash
python src/main_analysis.py
```

### Exploratory Data Analysis

Open the Jupyter notebooks in the `notebooks/` directory:
```bash
jupyter notebook notebooks/exploratory_analysis.ipynb
```

### Making Predictions

```python
from src.predictive_models import DiseaseOutbreakPredictor

# Load your data
predictor = DiseaseOutbreakPredictor()
predictor.load_data('data/processed/health_data.csv')
predictor.train_model()
predictions = predictor.predict(test_data)
```

## Features

- **Data Preprocessing**: Automated cleaning and transformation of health datasets
- **Exploratory Data Analysis**: Visualization and statistical analysis tools
- **Predictive Modeling**: Multiple ML models for forecasting health outcomes
- **Visualization**: Interactive charts and graphs for insights
- **Model Evaluation**: Comprehensive performance metrics

## Models Implemented

1. **Disease Outbreak Prediction**: Time-series analysis and classification models
2. **Hospital Admission Forecasting**: Regression models with seasonal components
3. **Patient Outcome Analysis**: Risk stratification and outcome prediction

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Contact

For questions or suggestions, please open an issue on GitHub.