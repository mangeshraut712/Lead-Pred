"""
Lead Conversion Prediction Pipeline

This script implements a complete machine learning pipeline for predicting lead conversion:
1. Data loading and preprocessing
2. Feature engineering with one-hot encoding
3. Model training with hyperparameter tuning
4. Model evaluation and feature importance analysis
5. Model and preprocessing artifacts persistence
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Create artifacts directory
os.makedirs("artifacts", exist_ok=True)

# 1. Load and preprocess data
print("📊 Loading and preprocessing data...")
df = pd.read_csv("Leads.csv")
df.drop(columns=["Prospect ID"], inplace=True, errors='ignore')
df.dropna(subset=["Converted"], inplace=True)
df.fillna("Unknown", inplace=True)

# 2. Feature engineering
X = pd.get_dummies(df.drop("Converted", axis=1), drop_first=True)
y = df["Converted"]

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. Feature scaling
print("⚖️ Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Hyperparameter tuning with grid search
print("🔍 Performing hyperparameter tuning...")
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l2'],
    'solver': ['lbfgs']
}

grid = GridSearchCV(
    LogisticRegression(max_iter=1000), 
    param_grid, 
    cv=5, 
    scoring='roc_auc',
    verbose=1
)
grid.fit(X_train_scaled, y_train)

# 6. Get the best model
best_model = grid.best_estimator_
print(f"✅ Best parameters: {grid.best_params_}")

# 7. Model evaluation
print("\n📈 Evaluating model...")
y_pred = best_model.predict(X_test_scaled)
y_proba = best_model.predict_proba(X_test_scaled)[:, 1]

print(f"✅ AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
print("✅ Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 8. Save model and scaler
print("\n💾 Saving model artifacts...")
joblib.dump(best_model, "artifacts/logistic_model.pkl")
joblib.dump(scaler, "artifacts/scaler.pkl")
print("✅ Model and scaler saved to artifacts directory.")

# 9. Analyze feature importance
print("\n🔍 Analyzing feature importance...")
features = X.columns
importance = best_model.coef_[0]
top_features = pd.Series(importance, index=features).sort_values(ascending=False)

# Save feature importance
top_features.to_csv("artifacts/feature_importance.csv")
print("\n🔍 Top 10 Important Features:")
for feature, importance in top_features.head(10).items():
    print(f"  • {feature}: {importance:.4f}")

print("\n✨ Pipeline completed successfully!")