"""
Test the full prediction pipeline (similar to what app.py does)
"""

import sys
from pathlib import Path
import joblib
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from src.feature_engineering import create_domain_features


def test_prediction_works():
    print("Testing full prediction pipeline...")

    model = joblib.load("models/cause_classifier_model.pkl")
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    feature_names = joblib.load("models/feature_names.pkl")

    # Sample input
    text = "photocell flickering in cooling bed"
    df = pd.DataFrame({"reason_text": [text]})
    df["clean_text"] = df["reason_text"].str.lower().str.replace(r'[^a-z0-9\s]', ' ', regex=True)

    tfidf = vectorizer.transform(df["clean_text"])
    tfidf_df = pd.DataFrame(tfidf.toarray(), columns=[f"tfidf_{f}" for f in vectorizer.get_feature_names_out()])

    df = pd.concat([df, tfidf_df], axis=1)
    df = create_domain_features(df)

    X = df.select_dtypes(include=["number"])
    X = X.reindex(columns=feature_names, fill_value=0)

    proba = model.predict_proba(X)[0]
    pred_class = model.classes_[proba.argmax()]

    print(f"Input: {text}")
    print(f"Predicted Device: {pred_class}")
    print(f"Confidence: {round(proba.max()*100, 2)}%")

    assert pred_class in model.classes_
    print(" Prediction pipeline test passed!")


if __name__ == "__main__":
    test_prediction_works()