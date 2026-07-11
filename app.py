from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import re

sys.path.append(str(Path(__file__).parent))

from src.feature_engineering import create_domain_features

app = FastAPI(title="Combi Mill Field Device Predictor API")

# ====================== Load Final Model ======================
MODEL_PATH = Path("models/final_random_forest_model.pkl")
VECTORIZER_PATH = Path("models/tfidf_vectorizer.pkl")
FEATURE_NAMES_PATH = Path("models/feature_names.pkl")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)
feature_names = joblib.load(FEATURE_NAMES_PATH)
# ==============================================================


class DelayInput(BaseModel):
    delay_text: str


def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


@app.post("/predict")
def predict(input_data: DelayInput):
    try:
        text = input_data.delay_text.strip()

        if not text or len(text) < 5:
            raise HTTPException(status_code=400, detail="Delay text is too short or empty")

        df = pd.DataFrame({"reason_text": [text]})
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

        top_indices = np.argsort(proba)[-3:][::-1]

        predictions = []
        for idx in top_indices:
            predictions.append({
                "device": classes[idx],
                "probability": round(float(proba[idx]) * 100, 2)
            })

        top_confidence = predictions[0]["probability"]
        low_confidence = top_confidence < 40

        return {
            "input_text": text,
            "predictions": predictions,
            "low_confidence": low_confidence,
            "top_confidence": top_confidence
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": True,
        "message": "Combi Mill PdM Predictor API is running"
    }


@app.get("/model_info")
def model_info():
    return {
        "model_type": "RandomForestClassifier (Final)",
        "cross_validated_f1": "0.7179",
        "weak_classes": ["ENCODER"],
        "total_features": len(feature_names)
    }