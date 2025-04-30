
# This code is for reference and shows how prediction might be done.
# The actual prediction script is predict_conversion.py and should be run separately.

import pandas as pd
import joblib
import os
import warnings
import numpy as np

# Suppress warnings during prediction if needed
warnings.filterwarnings("ignore")

def predict_lead_conversion(lead_data_input,
                           model_path="artifacts/logistic_model.pkl",
                           scaler_path="artifacts/scaler.pkl",
                           feature_list_path="artifacts/feature_list.csv",
                           imputation_values_path="artifacts/imputation_values.pkl"): # Added imputation values path
    """
    Predicts lead conversion probability for new lead data using saved artifacts.

    Parameters:
    lead_data_input (pd.DataFrame or dict): DataFrame or dictionary containing new lead data.
                                            Must contain columns used during training *before* encoding.
    model_path (str): Path to the saved trained model (.pkl).
    scaler_path (str): Path to the saved fitted scaler (.pkl).
    feature_list_path (str): Path to the CSV file containing the list of features used during training (after encoding).
    imputation_values_path (str): Path to the .pkl file containing imputation values calculated from training data.

    Returns:
    tuple: (predicted_class (int), predicted_probability (float)) or (None, None) if an error occurs.
           predicted_class: 0 for not converted, 1 for converted.
           predicted_probability: Probability of belonging to class 1 (converted).
    """
    try:
        # --- Load Artifacts ---
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
            return None, None
        if not os.path.exists(scaler_path):
            print(f"Error: Scaler file not found at {scaler_path}")
            return None, None
        if not os.path.exists(feature_list_path):
            print(f"Error: Feature list file not found at {feature_list_path}")
            return None, None
        if not os.path.exists(imputation_values_path): # Check for imputation values file
             print(f"Error: Imputation values file not found at {imputation_values_path}")
             return None, None


        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        training_features_df = pd.read_csv(feature_list_path)
        training_columns = training_features_df['Feature'].tolist() # Get the exact column names/order
        imputation_values = joblib.load(imputation_values_path) # Load imputation values


        # --- Prepare Input Data ---
        # Convert dict to DataFrame if necessary
        if isinstance(lead_data_input, dict):
            lead_df = pd.DataFrame([lead_data_input])
        elif isinstance(lead_data_input, pd.DataFrame):
            lead_df = lead_data_input.copy()
        else:
            print("Error: Input data must be a pandas DataFrame or a dictionary.")
            return None, None

        # --- Preprocessing and Feature Engineering (Mirror Training Steps) ---
        # Apply imputation using the values calculated from training data
        for col, value in imputation_values.items():
            if col in lead_df.columns:
                lead_df[col] = lead_df[col].fillna(value)
            else:
                 print(f"⚠️ Warning: Imputation value provided for column '{col}' not found in input data.")


        # Apply same feature engineering steps as in tune_model.py
        if 'TotalVisits' in lead_df.columns and 'Total Time Spent on Website' in lead_df.columns:
            lead_df['TotalVisits'] = pd.to_numeric(lead_df['TotalVisits'], errors='coerce').fillna(0)
            lead_df['Total Time Spent on Website'] = pd.to_numeric(lead_df['Total Time Spent on Website'], errors='coerce').fillna(0)
            lead_df['Avg_Time_Per_Visit'] = lead_df['Total Time Spent on Website'] / (lead_df['TotalVisits'] + 1e-6)

        if 'Last Activity' in lead_df.columns:
             lead_df['Last Activity'] = lead_df['Last Activity'].astype(str).fillna('Unknown') # Ensure string type
             lead_df['Is_Email_Activity'] = lead_df['Last Activity'].str.contains('Email', case=False, na=False).astype(int)
             lead_df['Is_SMS_Activity'] = lead_df['Last Activity'].str.contains('SMS', case=False, na=False).astype(int)


        # --- One-Hot Encode ---
        # Identify categorical columns in the input (must match original categorical columns used in training)
        # This requires knowing the original categorical columns. Ideally, save this list too.
        # For now, assuming columns that are 'object' type after imputation are categorical.
        categorical_cols_input = lead_df.select_dtypes(include='object').columns.tolist()
        lead_encoded = pd.get_dummies(lead_df, columns=categorical_cols_input, drop_first=True, dummy_na=False)


        # --- Align Columns with Training Data ---
        # Add missing columns (that were in training) with value 0
        missing_cols = set(training_columns) - set(lead_encoded.columns)
        for col in missing_cols:
            lead_encoded[col] = 0

        # Ensure the order of columns matches the training data
        # Keep only columns that were present during training
        lead_encoded = lead_encoded[training_columns]

        # Final check for NaNs before scaling
        if lead_encoded.isnull().sum().sum() > 0:
            print("❌ Error: NaNs found in processed data before scaling!")
            print(lead_encoded.isnull().sum()[lead_encoded.isnull().sum() > 0])
            return None, None

        # --- Scale Features ---
        lead_scaled = scaler.transform(lead_encoded)

        # --- Predict ---
        predicted_class = model.predict(lead_scaled)[0]
        predicted_probability = model.predict_proba(lead_scaled)[0][1] # Probability of class 1

        return int(predicted_class), float(predicted_probability)

    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback for debugging
        return None, None

# Example Usage (within the prediction script file):
if __name__ == '__main__':
    # Create sample lead data (dictionary) - Column names MUST match the original CSV
    # Ensure data types match (e.g., numeric columns should have numbers, categorical as strings)
    sample_lead = {
        'TotalVisits': 5.0,
        'Total Time Spent on Website': 674.0,
        'Page Views Per Visit': 2.5,
        'Lead Origin': 'API',
        'Lead Source': 'Google',
        'Do Not Email': 'No',
        'Do Not Call': 'No',
        'Search': 'No',
        'Magazine': 'No',
        'Newspaper Article': 'No',
        'X Education Forums': 'No',
        'Newspaper': 'No',
        'Digital Advertisement': 'No',
        'Through Recommendations': 'No',
        'Receive More Updates About Our Courses': 'No',
        'Update me on Supply Chain Content': 'No',
        'Get updates on DM Content': 'No',
        'I agree to pay the amount through cheque': 'No',
        'A free copy of Mastering The Interview': 'No',
        'Last Activity': 'Email Opened',
        'Specialization': 'Select',
        'What is your current occupation': 'Unemployed',
        'What matters most to you in choosing a course': 'Better Career Prospects',
        'City': 'Select',
        'Lead Quality': 'Low in Relevance',
        'Asymmetrique Activity Index': '02.Medium',
        'Asymmetrique Profile Index': '02.Medium',
        'Asymmetrique Activity Score': 14.0,
        'Asymmetrique Profile Score': 15.0
        # ... add all other relevant columns from the original Leads.csv with sample values
    }

    # Convert sample lead dict to DataFrame
    sample_df = pd.DataFrame([sample_lead])

    # Make prediction
    # Ensure paths are correct relative to where you run the prediction script
    pred_class, pred_prob = predict_lead_conversion(sample_df,
                                                    model_path="artifacts/logistic_model.pkl",
                                                    scaler_path="artifacts/scaler.pkl",
                                                    feature_list_path="artifacts/feature_list.csv",
                                                    imputation_values_path="artifacts/imputation_values.pkl") # Pass the path

    if pred_class is not None:
        print(f"\n--- Sample Prediction ---")
        print(f"Predicted Class: {'Converted' if pred_class == 1 else 'Not Converted'} ({pred_class})")
        print(f"Predicted Conversion Probability: {pred_prob:.4f}")

