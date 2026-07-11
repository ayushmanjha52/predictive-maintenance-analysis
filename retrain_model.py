"""
Retraining Pipeline for Combi Mill PdM Model
Run this script whenever you have new delay data to update the model.
"""

import numpy as np
import joblib
from datetime import datetime
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import sys

sys.path.append(str(Path(__file__).parent))

from src.data_loader import load_master_events
from src.feature_engineering import prepare_modeling_data
from config import MODELS_DIR, RANDOM_STATE


def retrain_model(min_samples_per_class=5):
    print("=" * 60)
    print("STARTING MODEL RETRAINING")
    print("=" * 60)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load latest data
    print("\n[1/5] Loading data...")
    df = load_master_events()
    print(f"Total records loaded: {len(df)}")

    # Prepare features
    print("\n[2/5] Creating features...")
    df_model, vectorizer = prepare_modeling_data(df)

    y = df_model["field_device"].astype(str)

    # Remove rare classes
    class_counts = y.value_counts()
    rare_classes = class_counts[class_counts < min_samples_per_class].index.tolist()
    if rare_classes:
        mask = ~y.isin(rare_classes)
        df_model = df_model[mask]
        y = y[mask]
        print(f"Removed rare classes: {rare_classes}")

    X = df_model.drop(columns=["field_device"], errors="ignore")
    X = X.select_dtypes(include=[np.number])

    print(f"Final training samples: {len(X)} | Features: {X.shape[1]}")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    # Train new model
    print("\n[3/5] Training new Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=400,           # Slightly higher for better performance
        max_depth=15,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # Evaluate
    print("\n[4/5] Evaluating new model...")
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)

    print("\nNew Model Performance:")
    print(f"Accuracy: {report['accuracy']:.3f}")
    print(f"Weighted F1: {report['weighted avg']['f1-score']:.3f}")

    # Save new model with timestamp
    print("\n[5/5] Saving new model version...")

    versioned_model_name = f"cause_classifier_model_{timestamp}.pkl"
    versioned_vectorizer_name = f"tfidf_vectorizer_{timestamp}.pkl"
    versioned_features_name = f"feature_names_{timestamp}.pkl"

    joblib.dump(model, MODELS_DIR / versioned_model_name)
    joblib.dump(vectorizer, MODELS_DIR / versioned_vectorizer_name)
    joblib.dump(X.columns.tolist(), MODELS_DIR / versioned_features_name)

    # Also update the "latest" files (used by app.py and predict.py)
    joblib.dump(model, MODELS_DIR / "cause_classifier_model.pkl")
    joblib.dump(vectorizer, MODELS_DIR / "tfidf_vectorizer.pkl")
    joblib.dump(X.columns.tolist(), MODELS_DIR / "feature_names.pkl")

    print("\n Retraining completed successfully!")
    print(f"New model version saved as: {versioned_model_name}")
    print("Latest model files updated.")

    return model


if __name__ == "__main__":
    retrain_model()