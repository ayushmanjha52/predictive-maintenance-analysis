"""
app.py — FastAPI backend for the Combi Mill Field-Device Predictive
Maintenance dashboard.

Endpoints:
  GET  /health        - liveness/readiness + model-loaded check
  GET  /model_info     - manifest: winning model, honest holdout accuracy, classes
  POST /predict         - classify a new delay's free-text description
  GET  /stats           - Pareto (volume + severity) + FMEA fusion, dashboard summary numbers
  GET  /forecast         - monthly volume forecast, optionally per device
  GET  /events           - paginated/filterable event log for the dashboard table

Run with:
  uvicorn app:app --reload --port 8000

CORS is wide open (config.CORS_ALLOW_ORIGINS) for local development --
restrict to your real frontend origin before deploying to production.
"""
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src import config
from src.data_loader import load_master_events
from src.fmea_risk import fuse_pareto_with_fmea, load_fmea_scores
from src.forecasting import forecast_next_month
from src.predict import get_predictor

app = FastAPI(
    title="Combi Mill Field-Device Predictive Maintenance API",
    version="1.0.0",
    description="Classifies delay causes, fuses empirical Pareto data with FMEA risk, and forecasts delay volume.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_events_cache = None
_predictor = None


def get_events():
    global _events_cache
    if _events_cache is None:
        _events_cache = load_master_events()
    return _events_cache


def get_predictor_safe():
    global _predictor
    if _predictor is None:
        try:
            _predictor = get_predictor()
        except FileNotFoundError:
            raise HTTPException(
                status_code=503,
                detail="Model artifacts not found. Run `python -m src.train` before starting the API.",
            )
    return _predictor


# ---------------------------------------------------------------- Schemas --
class PredictRequest(BaseModel):
    reason_text: str = Field(..., min_length=3, description="Free-text delay description")
    mins: Optional[float] = Field(None, description="Delay duration in minutes, if known")


class PredictResponse(BaseModel):
    reason_text: str
    predicted_device: str
    confidence: float
    low_confidence: bool
    confidence_threshold: float
    top_candidates: list
    fmea_risk_level: str
    fmea_rpn: Optional[float]
    estimated_delay_minutes: Optional[float]


# --------------------------------------------------------------- Endpoints --
@app.get("/health")
def health():
    model_ready = config.MODEL_PATH.exists() and config.VECTORIZER_PATH.exists()
    return {"status": "ok", "model_ready": model_ready}


@app.get("/model_info")
def model_info():
    import json
    if not config.MANIFEST_PATH.exists():
        raise HTTPException(status_code=503, detail="No training manifest found. Run training first.")
    with open(config.MANIFEST_PATH) as f:
        manifest = json.load(f)
    return manifest


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    predictor = get_predictor_safe()
    result = predictor.predict_one(req.reason_text, mins=req.mins)
    return result


@app.get("/stats")
def stats():
    df = get_events()
    # Headline KPIs are scoped to the 9 TARGET_DEVICES this project tracks
    # (matches the reference dashboard) -- other tag categories present in
    # the data (e.g. ROLLER_TABLE, OVERTRAVEL, SEQUENCE_BREAK) are real
    # delay causes too, but aren't field-device sensors, so they're excluded
    # from the field-device coverage/volume headline numbers.
    tagged = df[df["primary_device"].isin(config.TARGET_DEVICES)]

    pareto = tagged.groupby("primary_device").agg(
        total_minutes=("mins", "sum"),
        avg_minutes=("mins", "mean"),
        max_minutes=("mins", "max"),
        events=("mins", "count"),
    ).reset_index().rename(columns={"primary_device": "device"})
    pareto["avg_minutes"] = pareto["avg_minutes"].round(1)
    pareto = pareto.sort_values("total_minutes", ascending=False)

    fmea = load_fmea_scores()
    fused = fuse_pareto_with_fmea(pareto, fmea)

    total_events = len(df)
    tagged_events = len(tagged)
    coverage_pct = round(100 * tagged_events / total_events, 1) if total_events else 0.0

    top_volume = pareto.iloc[0] if len(pareto) else None
    top_severity_row = pareto.sort_values("avg_minutes", ascending=False).iloc[0] if len(pareto) else None

    return {
        "total_field_device_delay_minutes": float(tagged["mins"].sum()),
        "total_tagged_events": int(tagged_events),
        "total_events": int(total_events),
        "unresolved_events": int(total_events - tagged_events),
        "coverage_pct": coverage_pct,
        "top_contributor_volume": {
            "device": top_volume["device"], "total_minutes": float(top_volume["total_minutes"]),
            "events": int(top_volume["events"]),
        } if top_volume is not None else None,
        "top_contributor_severity": {
            "device": top_severity_row["device"], "avg_minutes": float(top_severity_row["avg_minutes"]),
        } if top_severity_row is not None else None,
        "months_covered": sorted(df["month"].unique().tolist(), key=config.month_sort_key),
        "pareto_by_device": fused.to_dict(orient="records"),
    }


@app.get("/forecast")
def forecast(device: Optional[str] = Query(None, description="Filter to a single device, e.g. PHOTOCELL")):
    df = get_events()
    if device:
        device = device.upper()
        if device not in config.TARGET_DEVICES:
            raise HTTPException(status_code=400, detail=f"Unknown device '{device}'. Must be one of {config.TARGET_DEVICES}")
    return forecast_next_month(df, device=device)


@app.get("/forecast/all_devices")
def forecast_all_devices():
    df = get_events()
    return {d: forecast_next_month(df, device=d) for d in config.TARGET_DEVICES}


@app.get("/events")
def events(
    device: Optional[str] = Query(None),
    month: Optional[str] = Query(None),
    search: Optional[str] = Query(None, description="Substring search over reason_text"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    df = get_events().copy()
    if device:
        df = df[df["primary_device"] == device.upper()]
    if month:
        df = df[df["month"] == month]
    if search:
        df = df[df["reason_text"].str.contains(search, case=False, na=False)]

    total = len(df)
    df = df.sort_values("date", ascending=False, na_position="last")
    page = df.iloc[offset: offset + limit]

    records = page[["date", "month", "mins", "reason_text", "primary_device"]].rename(
        columns={"primary_device": "device"}
    ).fillna({"device": "UNRESOLVED"}).to_dict(orient="records")

    return {"total": total, "limit": limit, "offset": offset, "events": records}


@app.get("/devices")
def devices():
    return {"target_devices": config.TARGET_DEVICES}