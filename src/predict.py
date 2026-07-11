"""
Prediction Script using Final Random Forest Model
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import re

sys.path.append(str(Path(__file__).parent))

from feature_engineering import create_domain_features


def load_model_and_vectorizer():
    model_path = Path("models/final_random_forest_model.pkl")
    vectorizer_path = Path("models/tfidf_vectorizer.pkl")
    feature_names_path = Path("models/feature_names.pkl")

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    feature_names = joblib.load(feature_names_path)

    return model, vectorizer, feature_names


def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def predict_device(delay_text: str, model, vectorizer, feature_names, top_n=3):
    df = pd.DataFrame({"reason_text": [delay_text]})
    df["clean_text"] = df["reason_text"].apply(clean_text)

    tfidf_matrix = vectorizer.transform(df["clean_text"])
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=[f"tfidf_{feat}" for feat in vectorizer.get_feature_names_out()]
    )

    df = pd.concat([df, tfidf_df], axis=1)
    df = create_domain_features(df)

    X = df.select_dtypes(include=[np.number])
    X = X.reindex(columns=feature_names, fill_value=0)

    proba = model.predict_proba(X)[0]
    classes = model.classes_

    top_indices = np.argsort(proba)[-top_n:][::-1]

    results = []
    for idx in top_indices:
        results.append({
            "Device": classes[idx],
            "Probability": round(proba[idx] * 100, 2)
        })

    return results


if __name__ == "__main__":
    print("Loading final model...")
    model, vectorizer, feature_names = load_model_and_vectorizer()
    print("Model loaded successfully!\n")

    print("=== Combi Mill Delay Cause Predictor (Final Model) ===")
    print("Type a delay description (or type 'exit' to quit)\n")

    while True:
        text = input("Enter delay description: ").strip()

        if text.lower() in ["exit", "quit", "q"]:
            print("Exiting...")
            break

        if not text:
            continue

        predictions = predict_device(text, model, vectorizer, feature_names, top_n=3)

        print("\nTop Predictions:")
        for i, pred in enumerate(predictions, 1):
            print(f"  {i}. {pred['Device']:<25} → {pred['Probability']}%")
        print("-" * 50 + "\n")