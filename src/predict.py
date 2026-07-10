"""
Simple Prediction Script for Combi Mill PdM Model
(Corrected version - uses saved vectorizer properly)
"""

import joblib
import pandas as pd
import numpy as np


from config import MODELS_DIR
from feature_engineering import create_domain_features


def load_model_and_vectorizer():
    """Load the trained model and TF-IDF vectorizer."""
    model_path = MODELS_DIR / "cause_classifier_model.pkl"
    vectorizer_path = MODELS_DIR / "tfidf_vectorizer.pkl"
    
    if not model_path.exists() or not vectorizer_path.exists():
        raise FileNotFoundError("Model or vectorizer not found. Please train the model first.")
    
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    
    return model, vectorizer


def clean_text(text):
    import re
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def predict_device(delay_text: str, model, vectorizer, top_n=3):
    """
    Predict the most likely field device(s) for a given delay description.
    """
    # Create input dataframe
    df_input = pd.DataFrame({"reason_text": [delay_text]})
    
    # Clean text
    df_input["clean_text"] = df_input["reason_text"].apply(clean_text)
    
    # === IMPORTANT: Use transform() instead of fit_transform() ===
    tfidf_matrix = vectorizer.transform(df_input["clean_text"])
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=[f"tfidf_{feat}" for feat in vectorizer.get_feature_names_out()],
        index=df_input.index
    )
    
    # Combine with domain features
    df_input = pd.concat([df_input, tfidf_df], axis=1)
    df_input = create_domain_features(df_input)
    
    # Keep only numeric columns
    X = df_input.select_dtypes(include=[np.number])
    
    # Predict
    proba = model.predict_proba(X)[0]
    classes = model.classes_
    
    # Get top predictions
    top_indices = proba.argsort()[-top_n:][::-1]
    
    results = []
    for idx in top_indices:
        results.append({
            "Device": classes[idx],
            "Probability": round(proba[idx] * 100, 2)
        })
    
    return results


if __name__ == "__main__":
    print("Loading model...")
    model, vectorizer = load_model_and_vectorizer()
    print("Model loaded successfully!\n")
    
    print("=== Combi Mill Delay Cause Predictor ===")
    print("Type a delay description and press Enter (or type 'exit' to quit)\n")
    
    while True:
        text = input("Enter delay description: ").strip()
        
        if text.lower() in ["exit", "quit", "q"]:
            print("Exiting...")
            break
        
        if not text:
            continue
        
        predictions = predict_device(text, model, vectorizer, top_n=3)
        
        print("\nTop Predictions:")
        for i, pred in enumerate(predictions, 1):
            print(f"  {i}. {pred['Device']:<25} → {pred['Probability']}%")
        print("-" * 50 + "\n")