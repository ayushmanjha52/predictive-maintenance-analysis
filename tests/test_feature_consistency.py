"""
Critical Test: Ensures features created during training match features at prediction time.
This prevents silent bugs in production.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent.parent))

from src.data_loader import load_master_events
from src.feature_engineering import prepare_modeling_data


def test_feature_alignment():
    print("Testing feature alignment between training and prediction...")

    # Load training data and features
    df = load_master_events()
    df_model, vectorizer = prepare_modeling_data(df)

    # Get training feature names
    train_features = df_model.select_dtypes(include=[np.number]).columns.tolist()

    # Simulate a new prediction (single row)
    sample_text = "encoder feedback missing at stand 3"
    df_new = pd.DataFrame({"reason_text": [sample_text]})
    df_new["clean_text"] = df_new["reason_text"].str.lower().str.replace(r'[^a-z0-9\s]', ' ', regex=True)

    tfidf_matrix = vectorizer.transform(df_new["clean_text"])
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=[f"tfidf_{feat}" for feat in vectorizer.get_feature_names_out()]
    )

    df_new = pd.concat([df_new, tfidf_df], axis=1)
    df_new = create_domain_features(df_new)  # from feature_engineering

    pred_features = df_new.select_dtypes(include=[np.number]).columns.tolist()

    # Check alignment
    missing_in_pred = set(train_features) - set(pred_features)
    extra_in_pred = set(pred_features) - set(train_features)

    print(f"Training features: {len(train_features)}")
    print(f"Prediction features: {len(pred_features)}")
    print(f"Missing in prediction: {len(missing_in_pred)}")
    print(f"Extra in prediction: {len(extra_in_pred)}")

    assert len(missing_in_pred) == 0, f"Features missing in prediction: {missing_in_pred}"
    print("✅ Feature alignment test passed!")


if __name__ == "__main__":
    from src.feature_engineering import create_domain_features
    test_feature_alignment()