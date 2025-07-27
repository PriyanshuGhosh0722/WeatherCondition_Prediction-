# README: Weather Prediction using XGBoost Regressor

This repository contains a Jupyter notebook for weather data analysis and temperature prediction using machine learning techniques, primarily featuring the XGBoost Regressor, with data processing and feature engineering workflows. The dataset covers hourly weather data over several years.

## Overview

- **Goal:** Predict weather parameters (primarily temperature) using historic weather data and machine learning.
- **Main Model:** XGBoost Regressor, tuned via GridSearchCV.
- **Features Used:** Temperature, Humidity, Visibility, Time-of-day & Day-of-year cyclic encodings, among others.
- **Data Source:** Hourly weather dataset spanning multiple years.

## Workflow Summary

### 1. Data Loading and Initial Exploration

- The notebook loads a weather dataset with columns:
  - Date and Time
  - Summary & Precip Type
  - Temperature (C), Apparent Temperature
  - Humidity, Wind Speed/Bearing
  - Visibility, Pressure
- Exploration includes:
  - Data shape reporting.
  - Display of sample rows and feature statistics (mean, std, min, max, quartiles).

### 2. Data Preprocessing

- **Dealing with Missing Values:** Checks and handles missing values.
- **Categorical Encoding:**
  - Converts text features (`Summary`, `Precip Type`) into numerical representations.
- **Feature Engineering:**
  - Extraction of hour and day from timestamps.
  - Creation of cyclic features for hour and day (using sine and cosine transforms).
- **Feature Selection:**  
  - Utilizes `SelectKBest` to find the most important predictors for the temperature.

### 3. Data Normalization

- Normalizes features (mostly using z-score standardization), ensuring all model input features are on compatible scales.

### 4. Model Building: XGBoost Regressor

- The model of choice is **XGBRegressor** (from XGBoost).
- **Parameter Tuning:**
  - Uses `GridSearchCV` for hyperparameter search over options such as:
    - `learning_rate`: [0.01, 0.1, 0.3]
    - `max_depth`: [3,  - `min_child_weight`: [1]
    - `num_boost_round`: [50, 
    - `subsample`: [0.8, 1.0]
- **Cross-validation:** 3-fold CV to assess mean squared error.

### 5. Model Evaluation

- Fits the final XGBoost model on the training set after parameter selection.
- Evaluates predictive performance (MSE or RMSE) on test/validation split.
- Analyzes feature importance, if applicable.

### 6. Results & Observations

- Reports on the best hyperparameters found via GridSearchCV.
- Discusses model accuracy and prediction reliability.
- (Not all outputs are shown here; see the notebook for details and plots.)

## Usage

1. Clone this repository
2. Install requirements (see below)
3. Run the notebook:  
   - Open `21bcs8733_priyanshu_weather.ipynb` in JupyterLab or similar environment
   - Run the cells sequentially

## Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn
- xgboost
- matplotlib (for any plots/visualizations)

Install packages using pip if needed:
```bash
pip install pandas numpy scikit-learn xgboost matplotlib
```

## Folder Structure

- `21bcs8733_priyanshu_weather.ipynb`  
  *The main notebook containing all code, analysis pipeline, and results.*
- (Data file(s) are referenced in the notebook; ensure your data path matches.)

## File Details / Noteworthy Code Steps

- **Data Cleaning:**  
  Handles anomalies, inconsistent encodings, and normalization.
- **Feature Encoding:**  
  Numerical transformation of cyclic time features improves prediction.
- **Model Training:**  
  Uses extensive grid search and cross-validation for parameter robustness.
- **Testing & Results:**  
  Shows statistical summaries and test predictions.

## Notes

- The notebook is self-contained and can be used as a template for similar time-series regression tasks.
- Feature selection and normalization steps are critical for optimal model accuracy.

## Results and Next Steps

- Model achieves effective prediction on the dataset.
- Future improvements could include:
  - Further data cleaning.
  - Integration of additional features (e.g., external weather indices).
  - Experimentation with other regression algorithms for comparison.

*For any issues or questions, please create an issue or pull request.*
