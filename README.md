### Product Performance Optimization System: Lead Conversion Prediction

This project focuses on building a machine learning pipeline to predict lead conversion for institution 'X'. The analysis utilizes Logistic Regression to model the patterns in lead data and predict the likelihood of conversion. The system incorporates a robust preprocessing pipeline and evaluates the model's performance using standard metrics.

Project Overview
The primary goal of this project is to develop a reliable system for predicting whether a prospective lead will convert. By leveraging historical lead data, the system trains a Logistic Regression model to identify key factors influencing conversion and provide a prediction for new leads.

Methodology
The project employs a standard machine learning workflow:

Data Loading and Initial Cleaning: Loading the raw lead data and performing initial cleaning steps, including handling identifier columns and rows with missing target values.

Feature Engineering: Creating new, potentially more informative features from the existing data (e.g., average time per visit, activity flags).

Robust Preprocessing Pipeline: Utilizing a scikit-learn Pipeline with a ColumnTransformer to handle:

Missing Value Imputation: Filling missing numerical values with the median and categorical values with a placeholder like 'Unknown' (learned from training data).

Categorical Encoding: Converting categorical features into a numerical format using One-Hot Encoding.

Feature Scaling: Scaling numerical features to a standard range.
This pipeline ensures that all preprocessing steps are learned solely from the training data and applied consistently to both training and new data.

Model Training and Hyperparameter Tuning: Training a Logistic Regression model and optimizing its hyperparameters using GridSearchCV with cross-validation to find the best-performing configuration.

Model Evaluation: Assessing the performance of the trained model on a held-out test set using various metrics.

Evaluation Metrics
The model's performance is evaluated using several key metrics relevant to classification tasks:

ROC AUC (Receiver Operating Characteristic - Area Under Curve): Measures the model's ability to distinguish between the positive and negative classes. A higher AUC indicates better performance.

Confusion Matrix: Provides a breakdown of correct and incorrect predictions for each class (True Positives, True Negatives, False Positives, False Negatives).

Classification Report: Includes Precision, Recall (Sensitivity), F1-Score, and Support for each class, providing a comprehensive view of the model's performance across different thresholds.

Project Structure
The main components of the project are:

Leads.csv: The input dataset containing lead information.

scripts/tune_model.py: Python script for data loading, preprocessing (including building and fitting the pipeline), feature engineering, model training, hyperparameter tuning, evaluation, and saving the fitted pipeline and other artifacts.

scripts/predict_conversion.py: Python script for loading the saved pipeline and using it to make predictions on new lead data.

artifacts/: A directory created by tune_model.py to store the outputs of the training process.

Getting Started
Follow these steps to run the project and make predictions:

Prerequisites
Python 3.x

Required Python libraries (pandas, numpy, scikit-learn, joblib). You might need a requirements.txt file to manage dependencies, but you can install them manually if needed (e.g., pip install pandas numpy scikit-learn joblib).

Data
Ensure you have the Leads.csv file in the main project directory.

Step 1: Train and Tune the Model
This step runs the script that loads the data, performs all preprocessing steps within a scikit-learn pipeline, trains and tunes the Logistic Regression model, evaluates it, and saves the necessary files to the artifacts directory.

Navigate to your project's root directory in the terminal and run the following command:

python scripts/tune_model.py

What this script does:

Loads Leads.csv.

Cleans column names and performs initial feature engineering.

Splits the data into training and testing sets.

Builds and fits a scikit-learn Pipeline on the training data. This pipeline includes steps for imputing missing values (learning medians/modes from training data), one-hot encoding categorical features (learning categories from training data), and scaling numerical features (learning mean/std from training data).

Performs hyperparameter tuning for the Logistic Regression model using GridSearchCV and cross-validation on the data processed by the pipeline.

Evaluates the best model (which is part of the best pipeline) on the test set.

Saves the following files into the artifacts/ directory:

full_pipeline.pkl: The entire fitted scikit-learn pipeline, containing all preprocessing steps and the trained model. This is the main artifact needed for prediction.

feature_list.csv: A list of the final feature names used by the model after all preprocessing (useful for inspection).

model_performance.txt: A summary of the model's evaluation metrics on the test set.

feature_importance.csv: The calculated importance (coefficients) of each feature in the trained Logistic Regression model.

Wait for this script to complete successfully. You will see output indicating the progress, best parameters found, evaluation results, and confirmation of saved artifacts.

Step 2: Predict on New Lead Data
This step uses the saved full_pipeline.pkl to preprocess new lead data (in this case, the same Leads.csv file for demonstration) and generate conversion predictions.

Navigate to your project's root directory in the terminal and run the following command:

python scripts/predict_conversion.py Leads.csv --output predictions.csv

Leads.csv: Specifies the input file containing the lead data you want to predict on.

--output predictions.csv: Specifies the name of the output file where the predictions will be saved. You can change predictions.csv to your desired filename.

What this script does:

Loads the specified input CSV file (Leads.csv).

Performs the initial cleaning and feature engineering steps that were applied before the scikit-learn Pipeline in tune_model.py.

Loads the saved artifacts/full_pipeline.pkl.

Passes the initially processed data to the loaded pipeline's .predict() and .predict_proba() methods. The pipeline automatically applies the learned imputation, encoding, and scaling transformations consistently.

Generates the predicted conversion class and probability for each lead.

Saves the original input data along with the new Predicted_Converted and Conversion_Probability columns to the specified output CSV file (predictions.csv).

After this script completes, you will find the predictions.csv file in your main project directory containing the results.

Key Features and Improvements
End-to-End Pipeline: The use of sklearn.pipeline.Pipeline encapsulates the entire workflow from raw data (after initial FE) to prediction, ensuring consistency.

Robust Preprocessing: sklearn.compose.ColumnTransformer and sklearn.impute.SimpleImputer handle missing values and different data types systematically.

Data Leakage Prevention: Imputation and scaling parameters are learned only from the training data within the pipeline.

Simplified Prediction: The predict_conversion.py script is simplified, only requiring loading the full pipeline and applying it to new data after initial feature engineering.

This updated system provides a more reliable and maintainable approach to lead conversion prediction.

---

<!-- codex:project-diagram:start -->

## Project Diagram

```mermaid
flowchart LR
    A["Raw Data"] --> B["Preprocessing"]
    B --> C["Model Training"]
    C --> D["Predictions / Reports"]
```

_Core machine-learning workflow from source data to final artifacts._

<!-- codex:project-diagram:end -->
