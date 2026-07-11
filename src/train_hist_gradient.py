"""
Training using HistGradientBoostingClassifier (Stronger than Random Forest)
"""

import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, f1_score

from config import MODELS_DIR, RANDOM_STATE
from data_loader import load_master_events
from feature_engineering import prepare_modeling_data


def train_hist_gradient():
    print("Loading data...")
    df = load_master_events()

    print("Creating features...")
    df_model, vectorizer = prepare_modeling_data(df)

    y = df_model["field_device"].astype(str)
    X = df_model.drop(columns=["field_device"], errors="ignore")
    X = X.select_dtypes(include=[np.number])

    print(f"Training on {X.shape[1]} features with {len(X)} samples.")

    # HistGradientBoostingClassifier
    model = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_depth=10,
        min_samples_leaf=20,
        random_state=RANDOM_STATE,
        class_weight="balanced"
    )

    print("\nRunning 5-Fold Cross Validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    y_pred = cross_val_predict(model, X, y, cv=cv)

    # Evaluation
    print("\n" + "="*60)
    print("CROSS-VALIDATED PERFORMANCE (HistGradientBoosting)")
    print("="*60)
    print(classification_report(y, y_pred, zero_division=0))

    weighted_f1 = f1_score(y, y_pred, average='weighted')
    print(f"Weighted F1 Score: {weighted_f1:.4f}")

    # Train final model on full data
    print("\nTraining final model on full data...")
    model.fit(X, y)

    # Save model and vectorizer
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / "hist_gradient_model.pkl")
    joblib.dump(vectorizer, MODELS_DIR / "tfidf_vectorizer.pkl")
    joblib.dump(X.columns.tolist(), MODELS_DIR / "feature_names.pkl")

    print("\n HistGradientBoosting model saved successfully!")
    print(f"Location: {MODELS_DIR / 'hist_gradient_model.pkl'}")

    return model


if __name__ == "__main__":
    train_hist_gradient()