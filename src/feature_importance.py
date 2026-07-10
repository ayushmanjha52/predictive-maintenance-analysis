"""
Improved Feature Importance Analysis
Shows actual feature names (TF-IDF words + domain features)
"""

import joblib
import pandas as pd
import numpy as np
from config import MODELS_DIR
from data_loader import load_master_events
from feature_engineering import prepare_modeling_data


def show_feature_importance(top_n=25):
    print("Loading model and data...")
    
    model_path = MODELS_DIR / "cause_classifier_model.pkl"
    if not model_path.exists():
        print("Model not found. Please train the model first.")
        return
    
    model = joblib.load(model_path)
    
    # Load data and recreate features to get correct column names
    df = load_master_events()
    df_model, _ = prepare_modeling_data(df)
    
    # Remove rare classes (same as training)
    y = df_model["field_device"].astype(str)
    class_counts = y.value_counts()
    rare_classes = class_counts[class_counts < 5].index.tolist()
    if rare_classes:
        mask = ~y.isin(rare_classes)
        df_model = df_model[mask]
    
    X = df_model.drop(columns=["field_device"], errors="ignore")
    X = X.select_dtypes(include=[np.number])
    
    feature_names = X.columns.tolist()
    importances = model.feature_importances_
    
    # Create DataFrame for better display
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    print("\n" + "="*70)
    print(f"           TOP {top_n} MOST IMPORTANT FEATURES")
    print("="*70)
    print(importance_df.head(top_n).to_string(index=False))
    
    # Optional: Save to CSV for report
    importance_df.to_csv(MODELS_DIR / "feature_importance.csv", index=False)
    print(f"\nFeature importance saved to: {MODELS_DIR / 'feature_importance.csv'}")
    
    return importance_df


if __name__ == "__main__":
    show_feature_importance(top_n=25)