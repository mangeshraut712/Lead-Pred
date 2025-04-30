"""
Enhanced Lead Conversion Prediction Pipeline using scikit-learn Pipeline

This script implements a refined machine learning pipeline for predicting lead conversion:
1. Data loading and initial cleaning
2. Feature engineering
3. Defines a robust preprocessing pipeline using ColumnTransformer and SimpleImputer
4. Trains the model within the pipeline with hyperparameter tuning
5. Comprehensive model evaluation
6. Saves the entire fitted pipeline (preprocessing + model) and other artifacts.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer # Import SimpleImputer
from sklearn.compose import ColumnTransformer # Import ColumnTransformer
from sklearn.pipeline import Pipeline # Import Pipeline
import joblib
import os
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)


# Define paths
DATA_PATH = "Leads.csv" # Assuming Leads.csv is in the main project folder
ARTIFACTS_PATH = "artifacts"
PIPELINE_PATH = os.path.join(ARTIFACTS_PATH, "full_pipeline.pkl") # Save the entire pipeline
FEATURE_LIST_PATH = os.path.join(ARTIFACTS_PATH, "feature_list.csv") # Still useful for reference
# IMPUTATION_VALUES_PATH and SCALER_PATH are no longer needed separately when saving the pipeline

# Ensure artifacts directory exists
os.makedirs(ARTIFACTS_PATH, exist_ok=True)

def load_data(path):
    """Loads data from a CSV file."""
    print("📊 Loading data...")
    df = pd.read_csv(path)
    print("✅ Data loaded.")
    return df

def clean_column_names(df):
    """Cleans column names by stripping whitespace and replacing special characters."""
    print("🧹 Cleaning column names...")
    df.columns = df.columns.str.strip().str.replace(" ", "_").str.replace("-", "_")
    print("✅ Column names cleaned.")
    return df

# --- 1. Load and initial cleaning ---
df = load_data(DATA_PATH)
df = clean_column_names(df)

# Drop identifier column if it exists
df.drop(columns=["Prospect_ID"], inplace=True, errors='ignore') # Use cleaned column name

# Drop rows where the target variable is missing (essential)
TARGET_COLUMN = 'Converted' # Define target column name
if TARGET_COLUMN not in df.columns:
    print(f"❌ Error: Target column '{TARGET_COLUMN}' not found in the data.")
    exit()
df.dropna(subset=[TARGET_COLUMN], inplace=True)
df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int) # Ensure target is integer


# --- 2. Feature Engineering (Applied before splitting and pipeline) ---
# Features derived from existing columns are created here.
print("⚙️ Creating additional features...")

# Create interaction features if relevant columns exist
# Ensure columns are numeric before division and handle potential NaNs/zeros *before* this step
# We will handle NaNs systematically in the pipeline, but ensure types are consistent for calculation
if 'TotalVisits' in df.columns and 'Total_Time_Spent_on_Website' in df.columns:
    # Coerce to numeric first, errors become NaN, these will be handled by imputer later
    df['TotalVisits'] = pd.to_numeric(df['TotalVisits'], errors='coerce')
    df['Total_Time_Spent_on_Website'] = pd.to_numeric(df['Total_Time_Spent_on_Website'], errors='coerce')
    # Add a small epsilon to avoid division by zero *after* imputation
    # Note: Calculation here will produce NaNs if inputs are NaN, which is fine as imputer handles them
    df['Avg_Time_Per_Visit'] = df['Total_Time_Spent_on_Website'] / (df['TotalVisits'] + 1e-6)


# Add email and SMS activity flags if relevant column exists
if 'Last_Activity' in df.columns:
    # Ensure 'Last_Activity' is treated as string, NaNs will be handled by imputer later
    df['Last_Activity'] = df['Last_Activity'].astype(str) # Convert to string, NaNs become 'nan' string or stay NaN depending on original type
    df['Is_Email_Activity'] = df['Last_Activity'].str.contains('Email', case=False, na=False).astype(int)
    df['Is_SMS_Activity'] = df['Last_Activity'].str.contains('SMS', case=False, na=False).astype(int)


# --- 3. Prepare Features and Target & Split Data ---
print("🔧 Preparing features and target variable and splitting data...")
X = df.drop(TARGET_COLUMN, axis=1)
y = df[TARGET_COLUMN]

# Split data into training and testing sets *before* defining the full preprocessing pipeline
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42, # Ensures reproducibility
    stratify=y      # Important for classification, especially if classes are imbalanced
)
print("✅ Data split into training and testing sets.")

# Identify column types *after* feature engineering but *before* imputation/encoding
# This is crucial for the ColumnTransformer
numerical_features = X_train.select_dtypes(include=np.number).columns.tolist()
categorical_features = X_train.select_dtypes(include='object').columns.tolist()

print(f"\nIdentified {len(numerical_features)} numerical features and {len(categorical_features)} categorical features for pipeline.")
# print("Numerical features:", numerical_features) # Optional debug
# print("Categorical features:", categorical_features) # Optional debug


# --- 4. Create Preprocessing Pipeline using ColumnTransformer ---
# Define transformers for different column types
# Numerical Pipeline: Impute with median, then scale
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')), # Impute numerical NaNs with median
    ('scaler', StandardScaler()) # Scale numerical features
])

# Categorical Pipeline: Impute with a constant ('Unknown'), then one-hot encode
# Use handle_unknown='ignore' to handle categories not seen during training
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')), # Impute categorical NaNs with 'Unknown'
    ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first')) # One-hot encode, drop first category to avoid multicollinearity
])

# Create a ColumnTransformer to apply different transformers to different columns
# Ensure all columns are accounted for
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough' # Keep any columns not specified (shouldn't be any if lists are correct)
)

print("\n✅ Preprocessing pipeline created using ColumnTransformer.")


# --- 5. Create Full Machine Learning Pipeline (Preprocessing + Model) ---
# Combine the preprocessor and the model into a single pipeline
full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor), # The preprocessing steps
    ('classifier', LogisticRegression(max_iter=10000, random_state=42)) # The model
])

print("✅ Full ML pipeline created (Preprocessing + Logistic Regression).")


# --- 6. Hyperparameter Tuning with Grid Search on the Full Pipeline ---
print("\n🔍 Performing expanded hyperparameter tuning on the full pipeline...")

# Define the parameter grid for the *pipeline*.
# Parameter names need to be prefixed with the step name (__parametername)
param_grid = {
    'classifier__C': [0.01, 0.1, 1, 10, 100],            # Regularization strength for the classifier
    'classifier__penalty': ['l1', 'l2'],                 # Regularization type for the classifier
    'classifier__solver': ['liblinear', 'saga'],         # Solvers compatible with L1/L2
    'classifier__class_weight': [None, 'balanced']       # To handle potential class imbalance
    # You could also tune preprocessor steps, e.g., 'preprocessor__num__imputer__strategy': ['median', 'mean']
}

# Use StratifiedKFold for cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=full_pipeline, # Use the full pipeline as the estimator
    param_grid=param_grid,
    cv=cv,                    # Use StratifiedKFold
    scoring='roc_auc',        # Metric to optimize
    verbose=1,                # Show progress
    n_jobs=-1                 # Use all available CPU cores
)

# Fit the grid search to the raw training data (X_train, y_train).
# The pipeline handles all preprocessing internally during fitting.
try:
    print("\nFitting GridSearchCV on training data...")
    grid_search.fit(X_train, y_train) # Fit on raw training data
    print("✅ GridSearchCV fitting complete.")
except Exception as e:
    print(f"Error during GridSearchCV fitting: {e}")
    # Consider adding more specific error handling if needed
    exit()


# --- 7. Get the Best Pipeline ---
# The best estimator from GridSearchCV is the best fitted pipeline
best_pipeline = grid_search.best_estimator_
print(f"\n✅ Best parameters found for the pipeline: {grid_search.best_params_}")
print(f"✅ Best cross-validation ROC AUC score: {grid_search.best_score_:.4f}")


# --- 8. Model Evaluation ---
print("\n📈 Evaluating the best pipeline on the test set...")
try:
    # Predict directly using the best pipeline on the raw test data
    y_pred = best_pipeline.predict(X_test)
    y_proba = best_pipeline.predict_proba(X_test)[:, 1] # Probabilities for the positive class

    # Calculate metrics
    auc_score = roc_auc_score(y_test, y_proba)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    print(f"✅ Test Set AUC Score: {auc_score:.4f}")
    print("✅ Test Set Confusion Matrix:\n", conf_matrix)
    print("✅ Test Set Classification Report:\n", class_report)

    # Save model performance metrics
    try:
        with open(os.path.join(ARTIFACTS_PATH, "model_performance.txt"), "w") as f:
            f.write(f"Best Parameters: {grid_search.best_params_}\n")
            f.write(f"Best Cross-Validation ROC AUC Score: {grid_search.best_score_:.4f}\n\n")
            f.write("--- Test Set Performance ---\n")
            f.write(f"AUC Score: {auc_score:.4f}\n")
            f.write(f"Confusion Matrix:\n{conf_matrix}\n")
            f.write(f"Classification Report:\n{class_report}\n")
        print(f"✅ Model performance saved to: {os.path.join(ARTIFACTS_PATH, 'model_performance.txt')}")
    except Exception as e:
        print(f"Error saving model performance: {e}")

except Exception as e:
    print(f"Error during model evaluation: {e}")


# --- 9. Save the Full Pipeline ---
print("\n💾 Saving the full fitted pipeline...")
try:
    joblib.dump(best_pipeline, PIPELINE_PATH)
    print(f"✅ Full pipeline saved to {PIPELINE_PATH}")
except Exception as e:
    print(f"Error saving the pipeline: {e}")

# --- 10. Save Feature List (Features *after* preprocessing) ---
# Getting feature names after ColumnTransformer and OneHotEncoder is a bit tricky.
# The easiest way is often to transform a sample of the training data and get columns.
try:
    # Get the preprocessor step from the best pipeline
    fitted_preprocessor = best_pipeline.named_steps['preprocessor']

    # Transform a small sample of the training data to get the final feature names
    # Use the original X_train here
    X_train_transformed_sample = fitted_preprocessor.transform(X_train.head())

    # Get the feature names after transformation
    # This requires scikit-learn version >= 0.24 for get_feature_names_out
    # If using an older version, this step might be more complex
    try:
        feature_names_out = fitted_preprocessor.get_feature_names_out()
        feature_names = feature_names_out.tolist()
        print(f"Identified {len(feature_names)} features after preprocessing.")
        pd.DataFrame({"Feature": feature_names}).to_csv(FEATURE_LIST_PATH, index=False)
        print(f"✅ Feature list saved to: {FEATURE_LIST_PATH}")
    except AttributeError:
        print("⚠️ Warning: scikit-learn version might not support get_feature_names_out.")
        print("Feature list saved might not be accurate for OHE columns.")
        # Fallback: Save columns from X_train before OHE for basic reference
        pd.DataFrame({"Feature": X_train.columns.tolist()}).to_csv(FEATURE_LIST_PATH, index=False)
        print(f"✅ Basic feature list (before OHE) saved to: {FEATURE_LIST_PATH}")

except Exception as e:
    print(f"Error saving feature list: {e}")


# --- 11. Feature Importance Analysis (for Logistic Regression within Pipeline) ---
print("\n🔍 Analyzing feature importance...")
# Get the fitted classifier from the best pipeline
final_model = best_pipeline.named_steps['classifier']

# Check if the final model has coefficients (Logistic Regression does)
if hasattr(final_model, 'coef_'):
    # Get importance (coefficients)
    importance = final_model.coef_[0]

    # Ensure we have feature names corresponding to the coefficients
    if 'feature_names' in locals() and len(feature_names) == len(importance):
        # Create DataFrame
        feature_importance = pd.DataFrame({
            'Feature': feature_names, # Use the saved list of feature names after OHE
            'Importance': importance
        })

        # Sort by absolute importance for better ranking view, but display actual coefficient
        feature_importance['Abs_Importance'] = feature_importance['Importance'].abs()
        feature_importance = feature_importance.sort_values('Abs_Importance', ascending=False)

        # Save complete feature importance
        try:
            feature_importance[['Feature', 'Importance']].to_csv(os.path.join(ARTIFACTS_PATH, "feature_importance.csv"), index=False)
            print(f"✅ Feature importance saved to: {os.path.join(ARTIFACTS_PATH, 'feature_importance.csv')}")
        except Exception as e:
            print(f"Error saving feature importance: {e}")

        # Display top features
        print("\n🔝 Top 15 Important Features (by absolute coefficient value):")
        for i in range(min(15, len(feature_importance))):
            feature = feature_importance.iloc[i]['Feature']
            imp_value = feature_importance.iloc[i]['Importance']
            print(f"  {i+1}. {feature}: {imp_value:.4f}")
    else:
        print("Could not retrieve feature names correctly for importance analysis or mismatch in length.")
        print(f"Number of coefficients: {len(importance) if hasattr(final_model, 'coef_') else 'N/A'}")
        print(f"Number of feature names: {len(feature_names) if 'feature_names' in locals() else 'N/A'}")

else:
    print("Selected model does not have 'coef_' attribute for feature importance analysis.")


# --- 12. Prediction Script Code (for reference) ---
def save_prediction_code_reference():
    """Saves the prediction function code using the saved pipeline for reference."""
    print("\n📝 Saving prediction function code (for reference)...")
    prediction_script_content = """
# This code is for reference and shows how prediction might be done
# using the saved scikit-learn pipeline.
# The actual prediction script is predict_conversion.py and should be run separately.

import pandas as pd
import joblib
import os
import warnings
import numpy as np

# Suppress warnings during prediction if needed
warnings.filterwarnings("ignore")

def predict_lead_conversion_with_pipeline(lead_data_input,
                                         pipeline_path="artifacts/full_pipeline.pkl"):
    \"\"\"
    Predicts lead conversion probability for new lead data using the saved pipeline.

    Parameters:
    lead_data_input (pd.DataFrame or dict): DataFrame or dictionary containing new lead data.
                                            Must contain columns used during training *before* preprocessing pipeline.
    pipeline_path (str): Path to the saved fitted scikit-learn pipeline (.pkl).

    Returns:
    tuple: (predicted_class (int), predicted_probability (float)) or (None, None) if an error occurs.
           predicted_class: 0 for not converted, 1 for converted.
           predicted_probability: Probability of belonging to class 1 (converted).
    \"\"\"
    try:
        # --- Load Pipeline ---
        if not os.path.exists(pipeline_path):
            print(f"Error: Pipeline file not found at {pipeline_path}")
            return None, None

        full_pipeline = joblib.load(pipeline_path)
        print("✅ Full pipeline loaded.")

        # --- Prepare Input Data ---
        # Convert dict to DataFrame if necessary
        if isinstance(lead_data_input, dict):
            lead_df = pd.DataFrame([lead_data_input])
        elif isinstance(lead_data_input, pd.DataFrame):
            lead_df = lead_data_input.copy()
        else:
            print("Error: Input data must be a pandas DataFrame or a dictionary.")
            return None, None

        # --- Apply Feature Engineering (steps *before* the pipeline) ---
        # Mirror the feature engineering steps applied in tune_model.py BEFORE the pipeline
        # Ensure column names are cleaned first
        lead_df.columns = lead_df.columns.str.strip().str.replace(" ", "_").str.replace("-", "_")

        # Create interaction features if relevant columns exist
        if 'TotalVisits' in lead_df.columns and 'Total_Time_Spent_on_Website' in lead_df.columns:
            # Coerce to numeric first, errors become NaN (pipeline imputer handles these)
            lead_df['TotalVisits'] = pd.to_numeric(lead_df['TotalVisits'], errors='coerce')
            lead_df['Total_Time_Spent_on_Website'] = pd.to_numeric(lead_df['Total_Time_Spent_on_Website'], errors='coerce')
            # Add a small epsilon to avoid division by zero *after* imputation (handled by pipeline)
            lead_df['Avg_Time_Per_Visit'] = lead_df['Total_Time_Spent_on_Website'] / (lead_df['TotalVisits'] + 1e-6)


        # Add email and SMS activity flags if relevant column exists
        if 'Last_Activity' in lead_df.columns:
             # Ensure 'Last_Activity' is treated as string, NaNs handled by pipeline imputer
             lead_df['Last_Activity'] = lead_df['Last_Activity'].astype(str)
             lead_df['Is_Email_Activity'] = lead_df['Last_Activity'].str.contains('Email', case=False, na=False).astype(int)
             lead_df['Is_SMS_Activity'] = lead_df['Last_Activity'].str.contains('SMS', case=False, na=False).astype(int)

        # --- Predict using the full pipeline ---
        # The pipeline handles imputation, encoding, and scaling internally
        # Pass the DataFrame *after* initial feature engineering
        predicted_class = full_pipeline.predict(lead_df)[0]
        predicted_probability = full_pipeline.predict_proba(lead_df)[0][1] # Probability of class 1

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

    # Make prediction using the pipeline
    pred_class, pred_prob = predict_lead_conversion_with_pipeline(sample_df,
                                                                 pipeline_path="artifacts/full_pipeline.pkl")

    if pred_class is not None:
        print(f"\\n--- Sample Prediction ---")
        print(f"Predicted Class: {'Converted' if pred_class == 1 else 'Not Converted'} ({pred_class})")
        print(f"Predicted Conversion Probability: {pred_prob:.4f}")

"""
    try:
        with open(os.path.join(ARTIFACTS_PATH, "prediction_code_pipeline.py"), "w") as f:
            f.write(prediction_script_content)
        print(f"✅ Prediction function code (pipeline version) saved to {os.path.join(ARTIFACTS_PATH, 'prediction_code_pipeline.py')}")
    except Exception as e:
        print(f"Error saving prediction code: {e}")


print("\n✨ Pipeline completed successfully!")
print("🔧 Increased max_iter for Logistic Regression to potentially avoid convergence warnings.")
print("💾 Full fitted pipeline, feature list, performance metrics, and prediction code saved to artifacts directory.")
