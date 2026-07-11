import logging
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import re
import time
import csv
from datetime import datetime

sys.path.append(str(Path(__file__).parent))
from src.feature_engineering import create_domain_features
from src.fmea_risk import get_fmea_risk, calculate_combined_risk

# ====================== Logging Setup ======================
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
# ===========================================================

PREDICTION_LOG_FILE = LOGS_DIR / "predictions.csv"
if not PREDICTION_LOG_FILE.exists():
    with open(PREDICTION_LOG_FILE, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "input_text", "model_version", "top_device", 
            "top_probability", "fmea_rpn", "combined_risk_score", 
            "final_risk_level", "low_confidence"
        ])

# ====================== EXPLICIT MODEL LOADING ======================
MODEL_VERSION = "v1.0-final"
MODEL_PATH = Path("models/final_random_forest_model.pkl")
VECTORIZER_PATH = Path("models/tfidf_vectorizer.pkl")
FEATURE_NAMES_PATH = Path("models/feature_names.pkl")

try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    feature_names = joblib.load(FEATURE_NAMES_PATH)
    logger.info(f"Model loaded successfully: {MODEL_PATH.name}")
except Exception as e:
    logger.critical(f"CRITICAL: Failed to load model artifacts: {e}")
    raise RuntimeError("Model loading failed. API cannot start.") from e
# ===================================================================


app = FastAPI(title="Combi Mill Field Device Predictor API")


class DelayInput(BaseModel):
    delay_text: str


def clean_text(text: str) -> str:
    if pd.isna(text) or not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"{request.method} {request.url.path} - {response.status_code} - {process_time:.3f}s")
    return response


@app.post("/predict")
async def predict(input_data: DelayInput):
    try:
        text = input_data.delay_text.strip() if input_data.delay_text else ""

        if not text or len(text) < 5:
            raise HTTPException(status_code=400, detail="Delay text must be at least 5 characters")

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
        
        # ====================== CRITICAL: Feature Alignment ======================
        X = X.reindex(columns=feature_names, fill_value=0)
        # =======================================================================

        proba = model.predict_proba(X)[0]
        classes = model.classes_

        top_indices = np.argsort(proba)[-3:][::-1]
        predictions = [
            {"device": classes[idx], "probability": round(float(proba[idx]) * 100, 2)}
            for idx in top_indices
        ]

        top_device = predictions[0]['device']
        top_probability = predictions[0]['probability']
        low_confidence = top_probability < 40

        fmea_data = get_fmea_risk(top_device)
        combined_risk = calculate_combined_risk(top_probability, fmea_data["rpn"])

        # Log prediction
        try:
            with open(PREDICTION_LOG_FILE, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(), text, MODEL_VERSION,
                    top_device, top_probability, fmea_data["rpn"],
                    combined_risk["combined_risk_score"],
                    combined_risk["final_risk_level"], low_confidence
                ])
        except Exception as log_error:
            logger.error(f"Logging failed: {log_error}")

        return {
            "input_text": text,
            "model_version": MODEL_VERSION,
            "predictions": predictions,
            "fmea_risk": fmea_data,
            "combined_risk": combined_risk,
            "low_confidence": low_confidence
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in /predict: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal prediction error")


@app.get("/health")
async def health_check():
    """Production health check — verifies model is actually usable"""
    try:
        # Run a dummy prediction to verify everything works
        test_input = "test delay description for health check"
        df = pd.DataFrame({"reason_text": [test_input]})
        df["clean_text"] = df["reason_text"].apply(clean_text)
        tfidf = vectorizer.transform(df["clean_text"])
        tfidf_df = pd.DataFrame(tfidf.toarray(), columns=[f"tfidf_{f}" for f in vectorizer.get_feature_names_out()])
        df = pd.concat([df, tfidf_df], axis=1)
        df = create_domain_features(df)
        X = df.select_dtypes(include=[np.number]).reindex(columns=feature_names, fill_value=0)
        
        _ = model.predict_proba(X)  # Actual model inference test

        return {
            "status": "healthy",
            "model_loaded": True,
            "model_version": MODEL_VERSION,
            "message": "Model, vectorizer and feature alignment working correctly"
        }
    except Exception as e:
        logger.critical(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Model health check failed")


@app.get("/model_info")
async def model_info():
    return {
        "model_type": "RandomForestClassifier (Final)",
        "model_version": MODEL_VERSION,
        "cross_validated_f1": "0.7179",
        "weak_classes": ["ENCODER"],
        "total_features": len(feature_names),
        "last_updated": "2026-07-11"
    }


@app.get("/stats")
async def get_stats():
    try:
        if not PREDICTION_LOG_FILE.exists():
            return {"message": "No predictions logged yet"}

        df = pd.read_csv(PREDICTION_LOG_FILE)
        if df.empty:
            return {"message": "No predictions logged yet"}

        return {
            "total_predictions": len(df),
            "average_top_confidence": round(df["top_probability"].mean(), 2),
            "low_confidence_count": int((df["low_confidence"] ).sum()),
            "risk_level_distribution": df["final_risk_level"].value_counts().to_dict(),
            "top_predicted_devices": df["top_device"].value_counts().head(5).to_dict(),
        }
    except Exception as e:
        logger.error(f"Stats generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Error generating statistics")


@app.get("/forecast")
async def get_delay_forecast(months: int = 3):
    from src.forecasting import  get_monthly_delay_forecast
    try:
        return get_monthly_delay_forecast(forecast_months=months)
    except Exception as e:
        logger.error(f"Forecast failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Error generating forecast")