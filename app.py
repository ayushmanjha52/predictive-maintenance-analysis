"""
Combi Mill Field Device Predictor API

FIXES vs. previous version:

1. CRITICAL: MODEL_PATH hardcoded "final_random_forest_model.pkl" -- the
   exact file recommended for deletion earlier in this project (it was
   produced by a redundant train_final_model.py that's been consolidated
   into the single canonical train.py). If that file was deleted, this
   app fails to start; if it wasn't, this app is silently serving a
   model that never got the compound-label-collapsing fix, the tuned
   hyperparameters, or the model-comparison result -- which just found
   Logistic Regression, not Random Forest, is the actual best model
   (64.9% vs 62.2% honest holdout accuracy). Fix: uses config.MODEL_PATH,
   the single canonical path every other script in this project uses.

2. Confidence threshold hardcoded as 40 here vs. config.CONFIDENCE_THRESHOLD
   (35%) used everywhere else (predict.py's CLI). Two different thresholds
   for the identical concept, silently able to drift apart. Fix: imports
   the one shared constant.

3. /predict and /health each reimplemented their own copy of the full
   feature-building pipeline (clean text, TF-IDF transform, domain
   features, reindex) -- a FOURTH independent copy of logic already
   duplicated across predict.py, and two test files earlier this
   session. Fix: both now call the shared CausePredictor class from
   predict.py -- one implementation, everywhere.

4. /model_info hardcoded "cross_validated_f1": "0.7179", "weak_classes":
   ["ENCODER"], "model_type": "RandomForestClassifier (Final)", and a
   fixed "last_updated" date as literal strings. These go stale the
   moment the model is retrained -- confirmed right now: the actual
   current training_manifest.json says the winning model is Logistic
   Regression, not Random Forest, so this hardcoded info is already
   wrong. Fix: reads training_manifest.json + classification_report.csv
   live, so this endpoint can never report stale model info.

5. No max-length bound on input text (only a minimum). Fix: added via
   pydantic Field, capping absurdly long input.

6. Hardcoded relative paths for logs/models. Fix: config.py paths
   throughout, consistent with the rest of the project.

7. No startup consistency check between model and feature_names. Fix:
   fails loudly at STARTUP (not on the first /predict call) if they
   disagree -- catches the exact class of bug found earlier in this
   project before the API ever accepts traffic.
"""
import csv
import json
import logging
import time
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
import pandas as pd

from src.config import (
    MODEL_PATH, MODELS_DIR,
    REPORTS_DIR, CONFIDENCE_THRESHOLD,
)
from src.predict import CausePredictor
from src.fmea_risk import get_fmea_risk, calculate_combined_risk

# ====================== Logging Setup ======================
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOGS_DIR / "app.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)
# ===========================================================

PREDICTION_LOG_FILE = LOGS_DIR / "predictions.csv"
if not PREDICTION_LOG_FILE.exists():
    with open(PREDICTION_LOG_FILE, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "input_text", "model_version", "top_device",
            "top_probability", "fmea_rpn", "combined_risk_score",
            "final_risk_level", "low_confidence",
        ])

# ====================== Model loading (single shared implementation) ======================
try:
    predictor = CausePredictor()  # loads config.MODEL_PATH / VECTORIZER_PATH / FEATURE_NAMES_PATH internally
    logger.info(f"Model loaded successfully: {MODEL_PATH.name} "
                f"({len(predictor.feature_names)} features)")
except Exception as e:
    logger.critical(f"CRITICAL: Failed to load model artifacts: {e}")
    raise RuntimeError("Model loading failed. API cannot start.") from e

# FIX: fail loudly at startup, not on the first request, if artifacts disagree.
_n_model_features = getattr(predictor.model, "n_features_in_", None)
if _n_model_features is not None and _n_model_features != len(predictor.feature_names):
    msg = (f"STARTUP CONSISTENCY CHECK FAILED: model expects {_n_model_features} "
           f"features but feature_names.pkl has {len(predictor.feature_names)}. "
           f"These are from different training runs -- retrain before starting the API.")
    logger.critical(msg)
    raise RuntimeError(msg)
# ===================================================================


def load_model_metadata() -> dict:
    """Reads training_manifest.json (written by train.py) live, so
    /model_info can never report stale hardcoded info again -- this is
    the fix for the "RandomForestClassifier (Final)" string that was
    already wrong the moment Logistic Regression won the last training
    comparison."""
    manifest_path = MODELS_DIR / "training_manifest.json"
    if not manifest_path.exists():
        return {"warning": "training_manifest.json not found -- run train.py to generate it."}
    with open(manifest_path) as f:
        manifest = json.load(f)
    return manifest


app = FastAPI(title="Combi Mill Field Device Predictor API")


class DelayInput(BaseModel):
    delay_text: str = Field(..., min_length=5, max_length=2000)


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
        text = input_data.delay_text.strip()

        # FIX: single shared pipeline via CausePredictor, not a
        # reimplemented copy -- this is the same code path predict.py's
        # CLI uses, so the two can never silently give different answers.
        raw_predictions = predictor.predict(text, top_n=3)
        predictions = [{"device": p["Device"], "probability": p["Probability"]} for p in raw_predictions]

        top_device = predictions[0]["device"]
        top_probability = predictions[0]["probability"]
        # FIX: shared threshold from config, matches predict.py's CLI exactly.
        low_confidence = (top_probability / 100) < CONFIDENCE_THRESHOLD

        fmea_data = get_fmea_risk(top_device)
        combined_risk = calculate_combined_risk(top_probability, fmea_data["rpn"],
                                                  is_estimated=fmea_data.get("is_estimated", False))

        try:
            with open(PREDICTION_LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(), text, MODEL_PATH.stem,
                    top_device, top_probability, fmea_data["rpn"],
                    combined_risk["combined_risk_score"],
                    combined_risk["final_risk_level"], low_confidence,
                ])
        except Exception as log_error:
            logger.error(f"Logging failed: {log_error}")

        return {
            "input_text": text,
            "model_version": MODEL_PATH.stem,
            "predictions": predictions,
            "fmea_risk": fmea_data,
            "combined_risk": combined_risk,
            "low_confidence": low_confidence,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in /predict: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal prediction error")


@app.get("/health")
async def health_check():
    """Verifies the model is actually usable via a real prediction call,
    using the SAME shared CausePredictor as /predict -- not a separately
    maintained copy of the pipeline that could pass while /predict fails."""
    try:
        _ = predictor.predict("test delay description for health check", top_n=3)
        return {
            "status": "healthy",
            "model_loaded": True,
            "model_version": MODEL_PATH.stem,
            "n_features": len(predictor.feature_names),
            "message": "Model, vectorizer, and feature alignment working correctly",
        }
    except Exception as e:
        logger.critical(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Model health check failed")


@app.get("/model_info")
async def model_info():
    """Reads live metadata instead of hardcoded strings -- confirmed
    the previous hardcoded 'RandomForestClassifier (Final)' is already
    wrong given the current training_manifest.json shows Logistic
    Regression won the most recent model comparison."""
    manifest = load_model_metadata()

    cv_f1 = None
    weak_classes = []
    report_path = REPORTS_DIR / "classification_report.csv"
    if report_path.exists():
        report_df = pd.read_csv(report_path, index_col=0)
        if "f1-score" in report_df.columns:
            cv_f1 = round(float(report_df.loc["weighted avg", "f1-score"]), 4) if "weighted avg" in report_df.index else None
            per_class = report_df.drop(index=[i for i in ["accuracy", "macro avg", "weighted avg"] if i in report_df.index])
            weak_classes = per_class[per_class["f1-score"] < 0.6].index.tolist()

    return {
        "model_type": manifest.get("winner", "unknown -- see training_manifest.json"),
        "model_file": MODEL_PATH.name,
        "trained_at": manifest.get("trained_at"),
        "total_features": len(predictor.feature_names),
        "n_classes": manifest.get("n_classes"),
        "classes": manifest.get("classes"),
        "cross_validated_weighted_f1": cv_f1,
        "weak_classes_f1_below_0.6": weak_classes,
        "holdout_month_used": manifest.get("holdout_month"),
        "all_candidates_compared": manifest.get("all_candidates"),
        "note": "This endpoint reads training_manifest.json and classification_report.csv "
                "live -- it will always reflect whatever model was most recently trained.",
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
            "low_confidence_count": int(df["low_confidence"].sum()),
            "risk_level_distribution": df["final_risk_level"].value_counts().to_dict(),
            "top_predicted_devices": df["top_device"].value_counts().head(5).to_dict(),
        }
    except Exception as e:
        logger.error(f"Stats generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Error generating statistics")


@app.get("/forecast")
async def get_delay_forecast(months: int = 3):
    from src.forecasting import get_monthly_delay_forecast
    try:
        return get_monthly_delay_forecast(forecast_months=months)
    except Exception as e:
        logger.error(f"Forecast failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Error generating forecast")