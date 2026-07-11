"""
True Forward-in-Time Validation
Train on Nov 2025 – Apr 2026, Test on May 2026
"""

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, f1_score
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.data_loader import load_master_events
from src.feature_engineering import prepare_modeling_data


def run_forward_time_validation():
    print("=" * 60)
    print("FORWARD-IN-TIME VALIDATION")
    print("Train: Nov 2025 – Apr 2026 | Test: May 2026")
    print("=" * 60)

    df = load_master_events()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Split by time
    train_df = df[df['date'] < '2026-05-01']
    test_df = df[(df['date'] >= '2026-05-01') & (df['date'] < '2026-06-01')]

    print(f"\nTraining samples: {len(train_df)}")
    print(f"Testing samples:  {len(test_df)}")

    if len(test_df) == 0:
        print("No data found in May 2026. Please check your date range.")
        return

    # Prepare features on training data only
    train_model_df, vectorizer = prepare_modeling_data(train_df)

    y_train = train_model_df["field_device"].astype(str)
    X_train = train_model_df.drop(columns=["field_device"], errors="ignore")
    X_train = X_train.select_dtypes(include=[np.number])

    # Train model (using best parameters)
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(
        n_estimators=430,
        max_depth=None,
        max_features='log2',
        min_samples_split=8,
        min_samples_leaf=1,
        class_weight='balanced_subsample',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Prepare test data using the same vectorizer
    test_model_df, _ = prepare_modeling_data(test_df)

    y_test = test_model_df["field_device"].astype(str)
    X_test = test_model_df.drop(columns=["field_device"], errors="ignore")
    X_test = X_test.select_dtypes(include=[np.number])

    # Align features
    feature_names = X_train.columns.tolist()
    X_test = X_test.reindex(columns=feature_names, fill_value=0)

    # Predict
    y_pred = model.predict(X_test)

    print("\n" + "=" * 60)
    print("FORWARD-IN-TIME TEST RESULTS (May 2026)")
    print("=" * 60)
    print(classification_report(y_test, y_pred, zero_division=0))

    weighted_f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"Weighted F1 Score on Unseen May Data: {weighted_f1:.4f}")

    return weighted_f1


if __name__ == "__main__":
    run_forward_time_validation()