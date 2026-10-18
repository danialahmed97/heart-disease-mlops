"""FastAPI service exposing the heart-disease model."""
import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator

from api.schemas import (BatchRequest, BatchResponse, PatientFeatures,
                         PredictionResponse)
from src.config import MODEL_PATH, RAW_FEATURES, METADATA_PATH

MODEL_FILE = Path(os.getenv("MODEL_PATH", MODEL_PATH))
META_FILE = Path(os.getenv("METADATA_PATH", METADATA_PATH))
LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))

PREDICTIONS = Counter("heart_predictions_total",
                      "Predictions served by class", ["label"])
CONFIDENCE = Histogram("heart_prediction_confidence",
                       "Model confidence of served predictions",
                       buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0])

request_logger = logging.getLogger("api.requests")
logger = logging.getLogger("api")


def setup_request_logger() -> None:
    """Emit raw JSON lines to stdout and logs/api.log."""
    request_logger.setLevel(logging.INFO)
    request_logger.propagate = False
    if request_logger.handlers:
        return
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(LOG_DIR / "api.log"))
    except OSError as exc:  # read-only FS etc.
        logger.warning("File logging disabled: %s", exc)
    for h in handlers:
        h.setFormatter(logging.Formatter("%(message)s"))
        request_logger.addHandler(h)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model and metadata once at startup."""
    setup_request_logger()
    app.state.model = joblib.load(MODEL_FILE)
    app.state.meta = json.loads(META_FILE.read_text())
    logger.info("Loaded model %s from %s", app.state.meta["model_name"],
                MODEL_FILE)
    yield


app = FastAPI(title="Heart Disease Prediction API", version="1.0.0",
              lifespan=lifespan)
Instrumentator(excluded_handlers=["/metrics"]).instrument(app).expose(
    app, include_in_schema=False)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Structured JSON request log with latency and prediction."""
    start = time.perf_counter()
    response = await call_next(request)
    request_logger.info(json.dumps({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "method": request.method,
        "path": request.url.path,
        "status": response.status_code,
        "latency_ms": round((time.perf_counter() - start) * 1000, 2),
        "prediction": getattr(request.state, "prediction", None),
    }))
    return response


def _score(model, meta: dict, patients: list[PatientFeatures]
           ) -> list[PredictionResponse]:
    """Run the pipeline on patients and build responses."""
    df = pd.DataFrame([p.model_dump() for p in patients])[RAW_FEATURES]
    proba = model.predict_proba(df.astype("float64"))[:, 1]
    out = []
    for p in proba:
        pred = int(p >= 0.5)
        label = "Disease" if pred else "Healthy"
        conf = float(max(p, 1 - p))
        PREDICTIONS.labels(label=label).inc()
        CONFIDENCE.observe(conf)
        out.append(PredictionResponse(
            prediction=pred, label=label, confidence=round(conf, 4),
            probability_disease=round(float(p), 4),
            model_version=str(meta["version"])))
    return out


@app.get("/health")
def health(request: Request) -> dict:
    """Liveness/readiness probe."""
    return {"status": "ok",
            "model_loaded": getattr(request.app.state, "model", None)
            is not None}


@app.get("/model-info")
def model_info(request: Request) -> dict:
    """Metadata of the loaded model."""
    return request.app.state.meta


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientFeatures, request: Request) -> PredictionResponse:
    """Predict heart-disease presence for one patient."""
    result = _score(request.app.state.model, request.app.state.meta,
                    [patient])[0]
    request.state.prediction = result.label
    return result


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(batch: BatchRequest, request: Request) -> BatchResponse:
    """Predict for up to 500 patients in one call."""
    try:
        preds = _score(request.app.state.model, request.app.state.meta,
                       batch.records)
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Batch scoring failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    request.state.prediction = f"batch:{len(preds)}"
    return BatchResponse(count=len(preds), predictions=preds)
