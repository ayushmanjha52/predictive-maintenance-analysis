"""
Hyperparameter Tuning for Combi Mill PdM Model
"""

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import classification_report
from scipy.stats import randint

from src.config import MODELS_DIR, RANDOM_STATE          # ← Fixed import
from src.data_loader import load_master_events
from src.feature_engineering import prepare_modeling_data


def tune_hyperparameters():
    print("Loading data and preparing features...")
    df = load_master_events()
    df_model, vectorizer = prepare_modeling_data(df)

    y = df_model["field_device"].astype(str)
    X = df_model.drop(columns=["field_device"], errors="ignore")
    X = X.select_dtypes(include=[np.number])

    print(f"Data shape: {X.shape}")

    # Parameter distribution
    param_dist = {
        'n_estimators': randint(200, 800),
        'max_depth': [10, 15, 20, 25, None],
        'min_samples_split': randint(2, 10),
        'min_samples_leaf': randint(1, 5),
        'max_features': ['sqrt', 'log2', None],
        'class_weight': ['balanced', 'balanced_subsample']
    }

    base_model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)

    print("\nStarting RandomizedSearchCV (this may take some time)...")

    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=30,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        scoring='f1_weighted',
        verbose=2,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    random_search.fit(X, y)

    print("\n" + "="*60)
    print("BEST HYPERPARAMETERS:")
    print(random_search.best_params_)
    print(f"\nBest Weighted F1 Score: {random_search.best_score_:.4f}")

    best_model = random_search.best_estimator_

    # Evaluate
    y_pred = best_model.predict(X)
    print("\nClassification Report:")
    print(classification_report(y, y_pred, zero_division=0))

    # Save best model
    print("\nSaving best model...")
    joblib.dump(best_model, MODELS_DIR / "best_random_forest_model.pkl")
    joblib.dump(random_search.best_params_, MODELS_DIR / "best_hyperparameters.pkl")

    print("Best model saved successfully!")


if __name__ == "__main__":
    tune_hyperparameters()