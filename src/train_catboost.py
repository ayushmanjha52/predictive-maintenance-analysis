"""
Improved Model Training with Proper Evaluation + Feature Names Saving
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from config import MODELS_DIR, REPORTS_DIR, RANDOM_STATE
from data_loader import load_master_events
from feature_engineering import prepare_modeling_data


def train_and_evaluate(min_samples_per_class=5):
    print("Loading data...")
    df = load_master_events()
    
    print("Creating features...")
    df_model, vectorizer = prepare_modeling_data(df)
    
    y = df_model["field_device"].astype(str)
    
    # Remove rare classes
    class_counts = y.value_counts()
    rare_classes = class_counts[class_counts < min_samples_per_class].index.tolist()
    if rare_classes:
        mask = ~y.isin(rare_classes)
        df_model = df_model[mask]
        y = y[mask]
    
    X = df_model.drop(columns=["field_device"], errors="ignore")
    X = X.select_dtypes(include=[np.number])
    
    print(f"Training on {X.shape[1]} features with {len(X)} samples.")
    
    # Model
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1
    )
    
    # === Cross Validation ===
    print("\nRunning 5-Fold Cross Validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    y_pred = cross_val_predict(model, X, y, cv=cv)
    
    # Reports
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    report = classification_report(y, y_pred, zero_division=0, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(REPORTS_DIR / "classification_report.csv")
    
    print("\n=== Cross-Validated Classification Report ===")
    print(classification_report(y, y_pred, zero_division=0))
    
    # Confusion Matrix
    labels = sorted(y.unique())
    cm = confusion_matrix(y, y_pred, labels=labels)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix (5-Fold CV)")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=300)
    plt.close()
    
    print(f"\nReports saved to: {REPORTS_DIR}")
    
    # === Final model on full data ===
    print("\nTraining final model on full data...")
    model.fit(X, y)
    
    # Save model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model = False  # Not needed for RandomForest
    joblib.dump(model, MODELS_DIR / "cause_classifier_model.pkl")
    joblib.dump(vectorizer, MODELS_DIR / "tfidf_vectorizer.pkl")
    
    # ====================== NEW: Save feature names ======================
    feature_names = X.columns.tolist()
    joblib.dump(feature_names, MODELS_DIR / "feature_names.pkl")
    print(f"Feature names saved successfully ({len(feature_names)} features)")
    # ===================================================================
    
    print("Final model and artifacts saved successfully.")
    
    return model


if __name__ == "__main__":
    train_and_evaluate()