"""
Machine Learning Model Training Pipeline for Loan Approval Prediction
Trains Random Forest and Logistic Regression classifiers, evaluates metrics,
and exports the serialized model artifacts.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import joblib

def train_and_save_model():
    print("Loading and cleaning loan approval dataset...")
    df = pd.read_csv("loan_approval_dataset.csv")
    
    clean_df = df.dropna(subset=["Application_ID", "Annual_Income", "Loan_Amount", "Credit_Score"]).drop_duplicates(subset=["Application_ID"])
    clean_df = clean_df[(clean_df["Annual_Income"] > 0) & (clean_df["Credit_Score"].between(300, 850))].copy()
    
    # Feature columns
    feature_cols = [
        "Applicant_Age",
        "Annual_Income",
        "Loan_Amount",
        "Credit_Score",
        "DTI_Ratio",
        "Collateral_Value",
        "Past_Defaults",
        "Loan_Term_Months"
    ]
    
    X = clean_df[feature_cols]
    y = (clean_df["Loan_Status"].str.strip().str.lower() == "approved").astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    print(f"Dataset split: {len(X_train)} train rows, {len(X_test)} test rows.")
    
    # 1. Random Forest Classifier
    rf_model = RandomForestClassifier(
        n_estimators=120,
        max_depth=12,
        min_samples_split=8,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    
    rf_preds = rf_model.predict(X_test)
    rf_probs = rf_model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, rf_preds)
    roc_auc = roc_auc_score(y_test, rf_probs)
    f1 = f1_score(y_test, rf_preds)
    prec = precision_score(y_test, rf_preds)
    rec = recall_score(y_test, rf_preds)
    cm = confusion_matrix(y_test, rf_preds)
    
    print("\n=== Random Forest Model Performance ===")
    print(f"Accuracy:  {acc*100:.2f}%")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"Precision: {prec*100:.2f}%")
    print(f"Recall:    {rec*100:.2f}%")
    print(f"F1-Score:  {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, rf_preds))
    
    # Feature Importances
    importances = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": rf_model.feature_importances_
    }).sort_values("Importance", ascending=False)
    print("\nFeature Importances:")
    print(importances.to_string(index=False))
    
    # Package model metadata
    model_artifact = {
        "model": rf_model,
        "feature_cols": feature_cols,
        "metrics": {
            "accuracy": round(acc, 4),
            "roc_auc": round(roc_auc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "confusion_matrix": cm.tolist()
        },
        "feature_importances": importances.to_dict(orient="records")
    }
    
    joblib.dump(model_artifact, "loan_model.joblib")
    print("\nModel artifact saved successfully to 'loan_model.joblib'!")
    return model_artifact

if __name__ == "__main__":
    train_and_save_model()

