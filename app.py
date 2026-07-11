from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.feature_engineering import create_domain_features

app = FastAPI(title="Combi Mill PdM Predictor")

# Load model and vectorizer once when app starts
MODEL_PATH = Path("models/cause_classifier_model.pkl")
VECTORIZER_PATH = Path("models/tfidf_vectorizer.pkl")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


class DelayInput(BaseModel):
    delay_text: str


@app.post("/predict")
def predict_delay(input_data: DelayInput):
    text = input_data.delay_text
    
    # Create input DataFrame
    df = pd.DataFrame({"reason_text": [text]})
    df["clean_text"] = df["reason_text"].str.lower().str.replace(r'[^a-z0-9\s]', ' ', regex=True)
    
    # Transform using saved vectorizer (do NOT fit again)
    tfidf_matrix = vectorizer.transform(df["clean_text"])
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=[f"tfidf_{feat}" for feat in vectorizer.get_feature_names_out()]
    )
    
    df = pd.concat([df, tfidf_df], axis=1)
    df = create_domain_features(df)
    
    # Keep only numeric columns
    X = df.select_dtypes(include=[np.number])
    
    # Predict probabilities
    proba = model.predict_proba(X)[0]
    classes = model.classes_
    
    # Get top 3 predictions
    top_indices = np.argsort(proba)[-3:][::-1]
    
    results = []
    for idx in top_indices:
        results.append({
            "device": classes[idx],
            "probability": round(float(proba[idx]) * 100, 2)
        })
    
    return {
        "input_text": text,
        "predictions": results
    }


@app.get("/")
def root():
    return {"message": "Combi Mill PdM Predictor API is running"}