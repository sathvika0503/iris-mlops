import os
from pathlib import Path
from threading import Lock

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.train import FEATURE_COLUMNS, MODEL_PATH, train_model

app = FastAPI(title="Basic MLOps Iris API", version="1.0.0")
model_lock = Lock()


class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., gt=0, le=20)
    sepal_width: float = Field(..., gt=0, le=20)
    petal_length: float = Field(..., gt=0, le=20)
    petal_width: float = Field(..., gt=0, le=20)


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model has not been trained yet.")
    return joblib.load(MODEL_PATH)


@app.on_event("startup")
def startup():
    try:
        if not MODEL_PATH.exists():
            train_model()
    except Exception as exc:
        # Keep the API available so the error can be inspected through /health.
        app.state.startup_error = str(exc)


@app.get("/")
def root():
    return {
        "project": "Basic MLOps Iris Starter",
        "docs": "/docs",
        "health": "/health",
        "retrain": "POST /retrain",
    }


@app.get("/health")
def health():
    return {
        "status": "ok" if MODEL_PATH.exists() else "model_not_ready",
        "model_exists": MODEL_PATH.exists(),
        "startup_error": getattr(app.state, "startup_error", None),
    }


@app.post("/predict")
def predict(features: IrisFeatures):
    try:
        with model_lock:
            bundle = load_model()
            row = pd.DataFrame([[getattr(features, col) for col in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
            prediction = str(bundle["model"].predict(row)[0])
            probabilities = bundle["model"].predict_proba(row)[0]
            classes = bundle["model"].classes_
        return {
            "prediction": prediction,
            "confidence": round(float(max(probabilities)), 4),
            "probabilities": {str(label): round(float(prob), 4) for label, prob in zip(classes, probabilities)},
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Prediction unavailable: {exc}")


@app.post("/retrain")
def retrain():
    try:
        result = train_model()
        return result
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Retraining failed; check CSV schema/data: {exc}")
