import logging
import math
import os
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

MODEL_PATH = "models/model.joblib"
model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        logger.info("Model loaded successfully.")
    else:
        logger.warning("Model not found at %s. Run src/train.py first.", MODEL_PATH)
    yield
    # Cleanup could go here


app = FastAPI(
    title="California Housing Pricing API",
    description="API for predicting median house values in California.",
    version="1.0.0",
    lifespan=lifespan,
)

# Restrict CORS origins via environment variable (no wildcard with credentials)
allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "ALLOWED_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501"
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Accept"],
)


def _sanitize_json_value(value: object) -> object:
    """Convert non-finite floats in validation errors to strings for JSON output."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {k: _sanitize_json_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_json_value(v) for v in value]
    return value


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Return 422 even when invalid payloads contain NaN/Infinity floats."""
    sanitized = _sanitize_json_value(jsonable_encoder(exc.errors()))
    return JSONResponse(status_code=422, content={"detail": sanitized})


# Define input schema with domain guardrails
class HouseFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    MedInc: float = Field(..., ge=0.0, le=100.0, allow_inf_nan=False)
    HouseAge: float = Field(..., gt=0.0, le=200.0, allow_inf_nan=False)
    AveRooms: float = Field(..., gt=0.0, le=200.0, allow_inf_nan=False)
    AveBedrms: float = Field(..., gt=0.0, le=100.0, allow_inf_nan=False)
    Population: float = Field(..., gt=0.0, le=1_000_000.0, allow_inf_nan=False)
    AveOccup: float = Field(..., gt=0.0, le=2_000.0, allow_inf_nan=False)
    Latitude: float = Field(..., ge=-90.0, le=90.0, allow_inf_nan=False)
    Longitude: float = Field(..., ge=-180.0, le=180.0, allow_inf_nan=False)


# Define output schema
class PredictionResponse(BaseModel):
    predicted_value: float


@app.post("/predict", response_model=PredictionResponse)
def predict(features: HouseFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    # Convert input to DataFrame (feature names match model training data)
    input_data = pd.DataFrame([features.model_dump()])

    # Make prediction
    try:
        prediction = model.predict(input_data)
        return {"predicted_value": float(prediction[0])}
    except (ValueError, TypeError):
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=500, detail="Prediction failed due to an internal error."
        ) from None

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}
