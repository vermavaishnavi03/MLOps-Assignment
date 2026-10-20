"""FastAPI model-serving endpoint with structured request logs and Prometheus metrics."""
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from heart_ml.data import FEATURE_COLUMNS

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(message)s"
)
#logging.basicConfig(level=os$env:PYTHONPATH = "src".getenv("LOG_LEVEL", "INFO"),
 #                   format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("heart_ml.api")
REQUESTS = Counter("heart_api_requests_total", "Total API requests", ["method", "endpoint", "status"])
LATENCY = Histogram("heart_api_request_duration_seconds", "API request latency",
                    ["endpoint"])

MODEL_PATH = Path(os.getenv("MODEL_PATH", "artifacts/model.joblib"))
app = FastAPI(
    title="Heart Disease Risk Classification API",
    version="1.0.0",
    description=("Educational MLOps assignment API. Not a medical device; not for diagnosis."),
)
_model = None


class PredictionInput(BaseModel):
    age: float = Field(..., ge=1, le=120)
    sex: int = Field(..., ge=0, le=1, description="Dataset-coded value: 0 or 1")
    cp: int = Field(..., ge=0, le=4, description="Chest-pain category from dataset")
    trestbps: float = Field(..., ge=40, le=300)
    chol: float = Field(..., ge=50, le=800)
    fbs: int = Field(..., ge=0, le=1)
    restecg: int = Field(..., ge=0, le=2)
    thalach: float = Field(..., ge=40, le=250)
    exang: int = Field(..., ge=0, le=1)
    oldpeak: float = Field(..., ge=0, le=10)
    slope: int = Field(..., ge=0, le=3)
    ca: float = Field(..., ge=0, le=4)
    thal: float = Field(..., ge=0, le=7)


@app.middleware("http")
async def observe_requests(request: Request, call_next):
    start = time.perf_counter()
    status = "500"
    try:
        response = await call_next(request)
        status = str(response.status_code)
        return response
    finally:
        elapsed = time.perf_counter() - start
        endpoint = request.url.path
        REQUESTS.labels(request.method, endpoint, status).inc()
        LATENCY.labels(endpoint).observe(elapsed)
        logger.info(json.dumps({
            "event": "http_request", "method": request.method, "path": endpoint,
            "status": status, "duration_seconds": round(elapsed, 6),
        }))


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Run `python -m heart_ml.train` first."
            )
        _model = joblib.load(MODEL_PATH)
    return _model


@app.get("/")
def root():
    return {"service": "heart-disease-classifier", "docs": "/docs",
            "health": "/health", "metrics": "/metrics"}


@app.get("/health")
def health():
    return {"status": "ok", "model_file_exists": MODEL_PATH.exists()}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict")
def predict(payload: PredictionInput):
    try:
        model = get_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    row: dict[str, Any] = payload.model_dump()
    frame = pd.DataFrame([[row[name] for name in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
    prediction = int(model.predict(frame)[0])
    probabilities = model.predict_proba(frame)[0]
    confidence = float(max(probabilities))
    return {
        "prediction": prediction,
        "label": "disease_present" if prediction == 1 else "disease_not_present",
        "confidence": round(confidence, 4),
        "class_probabilities": {
            "disease_not_present": round(float(probabilities[0]), 4),
            "disease_present": round(float(probabilities[1]), 4),
        },
        "disclaimer": "Educational output only. This is not a diagnosis or medical advice.",
    }
