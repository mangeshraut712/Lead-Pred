<div align="center">

# 🎯 Lead Prediction System

A logistic-regression pipeline for predicting lead conversion from historical sales data.

![Python](https://img.shields.io/badge/Python-3.x-3776ab?style=flat&logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-Data%20Prep-150458?style=flat&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML%20Pipeline-f7931e?style=flat&logo=scikitlearn&logoColor=white)
![Logistic Regression](https://img.shields.io/badge/Model-Logistic%20Regression-4c78a8?style=flat)

</div>

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Stack](#stack)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Scripts](#scripts)
- [Outputs](#outputs)

## Overview

Lead-Pred trains a reusable scikit-learn pipeline that cleans lead data, engineers a few signal-rich features, and predicts whether a prospective lead is likely to convert. The repository also includes the saved artifacts and a prediction workflow for new CSV inputs.

## Features

- Data cleaning and feature engineering built into the training flow.
- ColumnTransformer-based preprocessing with imputation, encoding, and scaling.
- Logistic Regression tuned with cross-validation and GridSearchCV.
- Saved model artifacts for reproducible predictions on new leads.

## Stack

- Python, pandas, NumPy, scikit-learn, and joblib.
- Jupyter Notebook for exploratory analysis.
- CSV-based input/output for training and scoring.

## Quick Start

### Train

```bash
python scripts/tune_model.py
```

This creates the fitted pipeline and evaluation outputs in `artifacts/`.

### Predict

```bash
python scripts/predict_conversion.py Leads.csv --output predictions.csv
```

The prediction script reuses the saved pipeline and writes scored leads to a new CSV file.

## Project Structure

```text
.
├── Leads.csv
├── scripts/
│   ├── tune_model.py
│   ├── predict_conversion.py
│   └── model.py
├── artifacts/
│   ├── full_pipeline.pkl
│   ├── feature_list.csv
│   ├── feature_importance.csv
│   ├── model_performance.txt
│   └── logistic_model.pkl
├── Analysis.ipynb
└── predictions.csv
```

## Scripts

- `tune_model.py` trains and evaluates the conversion model.
- `predict_conversion.py` scores new lead data with the saved pipeline.
- `model.py` contains the reusable modeling helpers used by the workflow.

## Outputs

- `artifacts/full_pipeline.pkl` stores the full preprocessing + model pipeline.
- `artifacts/model_performance.txt` summarizes the evaluation metrics.
- `artifacts/feature_importance.csv` and `artifacts/feature_list.csv` document the learned features.
