import pandas as pd
import joblib
import argparse
from pathlib import Path
import numpy as np
import os

# Suppress warnings during prediction if needed
import warnings
warnings.filterwarnings("ignore")

def load_pipeline(pipeline_path):
    """Loads the trained scikit-learn pipeline."""
    try:
        full_pipeline = joblib.load(pipeline_path)
        print(f"✅ Loaded full pipeline from: {pipeline_path}")
        return full_pipeline
    except FileNotFoundError:
        print(f"❌ Error: Pipeline file not found at {pipeline_path}")
        print("Please ensure the full fitted pipeline was saved by tune_model.py")
        raise # Re-raise the exception
    except Exception as e:
        print(f"❌ Error loading pipeline: {e}")
        raise # Re-raise other exceptions


def preprocess_new_data_for_pipeline(new_data):
    """
    Performs initial cleaning and feature engineering steps that are
    applied *before* the scikit-learn pipeline in tune_model.py.
    The pipeline will handle imputation, encoding, and scaling.
    """
    print("🧹 Performing initial cleaning and feature engineering (steps before pipeline)...")

    # Ensure column names are cleaned first, exactly as in tune_model.py
    new_data.columns = new_data.columns.str.strip().str.replace(" ", "_").str.replace("-", "_")
    # print("Columns after cleaning:", new_data.columns.tolist()) # Debug print

    # --- Apply Feature Engineering (Mirroring tune_model.py BEFORE ColumnTransformer) ---
    # These steps create new columns or modify existing ones BEFORE the main pipeline processing.

    # Create interaction features if relevant columns exist
    if 'TotalVisits' in new_data.columns and 'Total_Time_Spent_on_Website' in new_data.columns:
        # Coerce to numeric first, errors become NaN (pipeline imputer handles these)
        new_data['TotalVisits'] = pd.to_numeric(new_data['TotalVisits'], errors='coerce')
        new_data['Total_Time_Spent_on_Website'] = pd.to_numeric(new_data['Total_Time_Spent_on_Website'], errors='coerce')
        # Add a small epsilon to avoid division by zero *after* imputation (handled by pipeline)
        # Calculation here will produce NaNs if inputs are NaN, which is fine as pipeline imputer handles them
        new_data['Avg_Time_Per_Visit'] = new_data['Total_Time_Spent_on_Website'] / (new_data['TotalVisits'] + 1e-6)
        # print(" Created Avg_Time_Per_Visit.") # Debug print
    else:
         print("⚠️ Warning: Required columns for Avg_Time_Per_Visit not found.")


    # Add email and SMS activity flags if relevant column exists
    if 'Last_Activity' in new_data.columns:
         # Ensure 'Last_Activity' is treated as string, NaNs handled by pipeline imputer
         new_data['Last_Activity'] = new_data['Last_Activity'].astype(str) # Convert to string, NaNs become 'nan' string or stay NaN depending on original type
         new_data['Is_Email_Activity'] = new_data['Last_Activity'].str.contains('Email', case=False, na=False).astype(int)
         new_data['Is_SMS_Activity'] = new_data['Last_Activity'].str.contains('SMS', case=False, na=False).astype(int)
         # print(" Created email/SMS activity flags.") # Debug print
    else:
         print("⚠️ Warning: 'Last_Activity' column not found for email/SMS flags.")

    # --- IMPORTANT: Do NOT manually handle imputation, encoding, or scaling here ---
    # The loaded pipeline will handle these steps internally using the parameters learned during training.

    # Return the DataFrame after initial cleaning and feature engineering
    return new_data

def predict_from_csv(input_csv, output_csv, pipeline_path):
    """
    Loads data, performs initial preprocessing (steps before pipeline),
    loads the pipeline, makes predictions using the pipeline, and saves the results.
    """
    print(f"📊 Loading data from: {input_csv}")
    new_data = pd.read_csv(input_csv)

    # Load the trained pipeline
    full_pipeline = load_pipeline(pipeline_path)

    # Perform initial cleaning and feature engineering (steps before the pipeline)
    # Pass a copy to avoid modifying the original DataFrame loaded from CSV
    data_for_pipeline = preprocess_new_data_for_pipeline(new_data.copy())

    # --- IMPORTANT: Drop columns that were dropped in tune_model.py BEFORE the pipeline ---
    # Ensure the columns passed to the pipeline match the columns passed to pipeline.fit(X_train, y_train)
    # In tune_model.py, 'Prospect_ID' was dropped before splitting and fitting the pipeline.
    # If 'Converted' was also in the input CSV and you want predictions for all rows,
    # you should drop it here too before passing to predict, as the pipeline expects features only.
    columns_to_drop_before_pipeline = ['Prospect_ID'] # Add 'Converted' if it's in your input CSV
    data_for_pipeline = data_for_pipeline.drop(columns=columns_to_drop_before_pipeline, errors='ignore')
    print(f"Dropped columns before passing to pipeline: {columns_to_drop_before_pipeline}")
    # print("Columns passed to pipeline:", data_for_pipeline.columns.tolist()) # Debug print


    # Make predictions using the full pipeline
    # The pipeline handles all remaining preprocessing (imputation, encoding, scaling) internally
    print("\n🔮 Making predictions using the pipeline...")
    try:
        # Pass the DataFrame *after* initial feature engineering and dropping columns
        predictions = full_pipeline.predict(data_for_pipeline)
        probabilities = full_pipeline.predict_proba(data_for_pipeline)[:, 1]
        print("✅ Predictions made.")
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        print("This might be due to data format issues or columns not matching expectations.")
        print("Ensure the columns in the input CSV (after initial cleaning/FE) match")
        print("the columns in X_train BEFORE it was passed to pipeline.fit() in tune_model.py.")
        import traceback
        traceback.print_exc()
        return # Exit if prediction fails


    # Save results
    # Create a new DataFrame for results, including original data and predictions
    result_df = new_data.copy() # Start with the original data
    result_df["Predicted_Converted"] = predictions
    result_df["Conversion_Probability"] = probabilities.round(4)

    print(f"\n💾 Saving predictions to: {output_csv}")
    result_df.to_csv(output_csv, index=False)
    print(f"✅ Predictions saved to: {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lead Conversion Prediction Tool")
    parser.add_argument("input_csv", help="Path to the input CSV with new lead data")
    parser.add_argument("--output", default="predictions.csv", help="Output CSV file name")
    args = parser.parse_args()

    # Define the path to the saved pipeline file.
    PIPELINE_PATH = "artifacts/full_pipeline.pkl"

    predict_from_csv(
        input_csv=args.input_csv,
        output_csv=args.output,
        pipeline_path=PIPELINE_PATH # Pass the pipeline path
    )
