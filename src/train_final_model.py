"""
Final Production Model Training - Random Forest (Best Parameters)
"""

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, f1_score

from config import MODELS_DIR, RANDOM_STATE
from data_loader import load_master_events
from feature_engineering import prepare_modeling_data


def train_final_model():
    print("=" * 60)
    print("TRAINING FINAL RANDOM FOREST MODEL (Best Parameters)")
    print("=" * 60)

    # Load data
    print("\n[1/4] Loading data...")
    df = load_master_events()

    # Prepare features
    print("[2/4] Creating features...")
    df_model, vectorizer = prepare_modeling_data(df)

    y = df_model["field_device"].astype(str)
    X = df_model.drop(columns=["field_device"], errors="ignore")
    X = X.select_dtypes(include=[np.number])

    print(f"Training samples: {len(X)} | Features: {X.shape[1]}")

    # Best hyperparameters from tuning
    best_params = {
        'n_estimators': 430,
        'max_depth': None,
        'max_features': 'log2',
        'min_samples_split': 8,
        'min_samples_leaf': 1,
        'class_weight': 'balanced_subsample',
        'random_state': RANDOM_STATE,
        'n_jobs': -1
    }

    model = RandomForestClassifier(**best_params)

    # Cross Validation
    print("\n[3/4] Running 5-Fold Cross Validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    y_pred = cross_val_predict(model, X, y, cv=cv)

    weighted_f1 = f1_score(y, y_pred, average='weighted')
    print(f"\nCross-Validated Weighted F1 Score: {weighted_f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(y, y_pred, zero_division=0))

    # Train final model on full data
    print("[4/4] Training final model on full data...")
    model.fit(X, y)

    # Save all artifacts
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / "final_random_forest_model.pkl")
    joblib.dump(vectorizer, MODELS_DIR / "tfidf_vectorizer.pkl")
    joblib.dump(X.columns.tolist(), MODELS_DIR / "feature_names.pkl")

    print("\n" + "=" * 60)
    print(" FINAL MODEL SAVED SUCCESSFULLY")
    print("=" * 60)
    print(f"Model saved at: {MODELS_DIR / 'final_random_forest_model.pkl'}")

    return model


if __name__ == "__main__":
    train_final_model()