import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")


def load_data(path="Leads.csv"):
    """Load the dataset from CSV file."""
    return pd.read_csv(path)


def preprocess_data(df):
    """Preprocess the data for modeling."""
    # Drop identifier columns if they exist
    if "Prospect ID" in df.columns:
        df = df.drop(columns=["Prospect ID"])
    
    # Drop rows with null in target
    df = df.dropna(subset=["Converted"])
    
    # Fill missing values
    df = df.fillna("Unknown")
    
    # Separate features and target
    X = df.drop("Converted", axis=1)
    y = df["Converted"]
    
    # Encode categorical variables
    X_encoded = pd.get_dummies(X, drop_first=True)
    
    # Split the data
    return train_test_split(X_encoded, y, test_size=0.3, random_state=42)


def train_model(X_train, y_train):
    """Train a logistic regression model."""
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance using AUC and confusion matrix."""
    preds = model.predict(X_test)
    auc = roc_auc_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    print("AUC Score:", auc)
    print("Confusion Matrix:\n", cm)


if __name__ == "__main__":
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)