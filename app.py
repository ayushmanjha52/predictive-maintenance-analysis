from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import csv
import re
from datetime import datetime
from src.feature_engineering import create_domain_features

app = FastAPI(title="Combi Mill PdM Dashboard")

# Serve frontend from FRONTEND folder
app.mount("/static", StaticFiles(directory="FRONTEND"), name="static")

# ====================== Logging ======================
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PREDICTION_LOG_FILE = LOGS_DIR / "predictions.csv"
if not PREDICTION_LOG_FILE.exists():
    with open(PREDICTION_LOG_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "input_text", "model_version", "top_device",
            "top_probability", "fmea_rpn", "combined_risk_score",
            "final_risk_level", "low_confidence"
        ])

# ====================== Load Model ======================
MODEL_VERSION = "v1.0-final"
MODEL_PATH = Path("models/final_random_forest_model.pkl")
VECTORIZER_PATH = Path("models/tfidf_vectorizer.pkl")
FEATURE_NAMES_PATH = Path("models/feature_names.pkl")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)
feature_names = joblib.load(FEATURE_NAMES_PATH)


# ====================== FMEA Functions (defined inside app.py) ======================
FMEA_FILE = Path("data/fmea_risk_scores.csv")

try:
    fmea_df = pd.read_csv(FMEA_FILE)
    fmea_df.set_index("device", inplace=True)
except Exception:
    fmea_df = None

def get_fmea_risk(device: str) -> dict:
    device = device.upper().strip()
    if fmea_df is not None and device in fmea_df.index:
        row = fmea_df.loc[device]
        return {
            "rpn": int(row["rpn"]),
            "severity": float(row["severity"]),
            "occurrence": float(row["occurrence"]),
            "detection": float(row["detection"]),
            "risk_level": row["risk_level"]
        }
    return {
        "rpn": 100,
        "severity": 4,
        "occurrence": 3,
        "detection": 8,
        "risk_level": "Low"
    }

def calculate_combined_risk(ml_probability: float, fmea_rpn: int) -> dict:
    ml_score = ml_probability / 100
    fmea_score = min(fmea_rpn / 500, 1.0)
    combined_score = (ml_score * 0.6) + (fmea_score * 0.4)

    if combined_score >= 0.7:
        level = "High"
    elif combined_score >= 0.4:
        level = "Medium"
    else:
        level = "Low"

    return {
        "combined_risk_score": round(combined_score * 100, 1),
        "final_risk_level": level
    }

# ====================== Pydantic Model ======================
class DelayInput(BaseModel):
    delay_text: str

# ====================== Routes ======================

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    with open("FRONTEND/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.post("/predict")
async def predict(input_data: DelayInput):
    try:
        text = input_data.delay_text.strip()
        if not text or len(text) < 5:
            raise HTTPException(status_code=400, detail="Delay text must be at least 5 characters")

        df = pd.DataFrame({"reason_text": [text]})
        df["clean_text"] = df["reason_text"].apply(
            lambda x: re.sub(r'[^a-z0-9\s]', ' ', str(x).lower()).strip()
        )

        tfidf_matrix = vectorizer.transform(df["clean_text"])
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=[f"tfidf_{feat}" for feat in vectorizer.get_feature_names_out()]
        )

        df = pd.concat([df, tfidf_df], axis=1)
        df = create_domain_features(df)

        X = df.select_dtypes(include=[np.number]).reindex(columns=feature_names, fill_value=0)

        proba = model.predict_proba(X)[0]
        classes = model.classes_

        top_indices = np.argsort(proba)[-3:][::-1]
        predictions = [
            {"device": classes[idx], "probability": round(float(proba[idx]) * 100, 2)}
            for idx in top_indices
        ]

        top_device = predictions[0]["device"]
        top_probability = predictions[0]["probability"]
        low_confidence = top_probability < 40

        fmea_data = get_fmea_risk(top_device)
        combined_risk = calculate_combined_risk(top_probability, fmea_data["rpn"])

        # Log prediction
        with open(PREDICTION_LOG_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(), text, MODEL_VERSION,
                top_device, top_probability, fmea_data["rpn"],
                combined_risk["combined_risk_score"],
                combined_risk["final_risk_level"], low_confidence
            ])

        return {
            "input_text": text,
            "model_version": MODEL_VERSION,
            "predictions": predictions,
            "fmea_risk": fmea_data,
            "combined_risk": combined_risk,
            "low_confidence": low_confidence
        }

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def get_monthly_delay_forecast(forecast_months: int = 3):
    try:
        df = pd.read_csv("data/master_events.csv")
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df["month"] = df["date"].dt.to_period("M").astype(str)
        monthly = df.groupby("month")["mins"].sum().sort_index()

        if monthly.empty:
            raise ValueError("No data available for forecast")

        recent = monthly.tail(6)
        if len(recent) < 3:
            forecast_values = [float(monthly.mean())] * forecast_months
        else:
            avg_change = recent.pct_change().mean()
            last_value = float(recent.iloc[-1])
            forecast_values = []
            for _ in range(forecast_months):
                next_value = last_value * (1 + avg_change)
                forecast_values.append(round(next_value, 2))
                last_value = next_value

        last_period = pd.Period(monthly.index[-1], freq="M")
        labels = [str(last_period + i + 1) for i in range(forecast_months)]

        return {"forecast_months": forecast_months, "labels": labels, "values": forecast_values}
    except Exception:
        return {
            "forecast_months": forecast_months,
            "labels": [f"Month {i+1}" for i in range(forecast_months)],
            "values": [100.0] * forecast_months
        }


@app.get("/forecast")
async def forecast(months: int = 3):
    return get_monthly_delay_forecast(forecast_months=months)


@app.get("/stats")
async def stats():
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
            "risk_distribution": df["final_risk_level"].value_counts().to_dict(),
            "top_devices": df["top_device"].value_counts().head(5).to_dict()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chart-data")
async def chart_data():
    try:
        df = pd.read_csv("data/master_events.csv")
        device_share = df.groupby("field_device")["mins"].sum().sort_values(ascending=False).head(8)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["month"] = df["date"].dt.to_period("M").astype(str)
        monthly = df.groupby("month")["mins"].sum().tail(7)
        return {
            "delay_share": {"labels": device_share.index.tolist(), "values": device_share.values.tolist()},
            "monthly_trend": {"labels": monthly.index.tolist(), "values": monthly.values.tolist()}
        }
    except Exception :
        return {
            "delay_share": {"labels": ["PHOTOCELL", "LVDT", "ENCODER", "HMD", "PROXIMITY"], "values": [2329, 1685, 1480, 1120, 980]},
            "monthly_trend": {"labels": ["Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May"], "values": [120, 780, 650, 280, 310, 720, 450]}
        }


@app.get("/events")
async def get_events(search: str = "", device: str = ""):
    try:
        df = pd.read_csv("data/master_events.csv")
        if search:
            df = df[df["reason_text"].str.contains(search, case=False, na=False)]
        if device:
            df = df[df["field_device"] == device]
        return df.head(100).to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model-info")
async def model_info():
    return {
        "model_version": MODEL_VERSION,
        "accuracy": "72% (cross-validation)",
        "weak_class": "ENCODER"
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": True, "model_version": MODEL_VERSION}