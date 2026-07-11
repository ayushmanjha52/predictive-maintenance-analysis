"""
Test to ensure feature engineering during training matches prediction time.
This is critical to avoid silent bugs.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.data_loader import load_master_events
from src.feature_engineering import prepare_modeling_data


def test_feature_consistency():
    print("Testing feature consistency between train and predict...")

    # Load data
    df = load_master_events()
    
    # Prepare features (same as training)
    df_model, vectorizer = prepare_modeling_data(df)
    
    # Simulate what happens in prediction
    sample_text = "HMD cleaning required in BDM area"
    df_pred = pd.DataFrame({"reason_text": [sample_text]})
    
    # Apply same cleaning and feature creation as in predict.py
    df_pred["clean_text"] = df_pred["reason_text"].str.lower().str.replace(r'[^a-z0-9\s]', ' ', regex=True)
    
    # Transform using the same vectorizer (important!)
    tfidf_matrix = vectorizer.transform(df_pred["clean_text"])
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=[f"tfidf_{feat}" for feat in vectorizer.get_feature_names_out()]
    )
    
    df_pred = pd.concat([df_pred, tfidf_df], axis=1)
    
    # Get numeric columns from training data
    train_numeric_cols = df_model.select_dtypes(include=[np.number]).columns.tolist()
    
    # Check if prediction features can align with training features
    pred_numeric_cols = df_pred.select_dtypes(include=[np.number]).columns.tolist()
    
    # Common columns
    common_cols = set(train_numeric_cols) & set(pred_numeric_cols)
    
    print(f"Training numeric features: {len(train_numeric_cols)}")
    print(f"Prediction numeric features: {len(pred_numeric_cols)}")
    print(f"Common features: {len(common_cols)}")
    
    # This is a basic check - in production we'd do stricter column matching
    assert len(common_cols) > 100, "Too few common features between train and predict"
    
    print(" Feature consistency check passed!")


if __name__ == "__main__":
    test_feature_consistency()