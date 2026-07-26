"""FastAPI app: predicts a track's Spotify stream count and supports retraining on new data.

Orange Economy Ghana project — helps artists/producers understand what drives a track's
streaming success (see summative/linear_regression/multivariate.ipynb for the full analysis).
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import pipeline
import prediction

DATA_PATH = Path(__file__).resolve().parent.parent / "linear_regression" / "data" / "Spotify_Youtube.csv"
MODEL_DIR = Path(__file__).resolve().parent.parent / "linear_regression" / "models"

app = FastAPI(
    title="Orange Economy Ghana — Streaming Success Predictor",
    description="Predicts a track's Spotify stream count from its audio and engagement features.",
    version="1.0.0",
)

# --- CORS -------------------------------------------------------------------------------
# This API is called from two kinds of clients: the Flutter mobile app (native HTTP requests,
# which are NOT subject to browser CORS at all) and, during development/grading, a browser
# hitting Swagger UI or a local web build of the app. We deliberately avoid allow_origins=["*"]:
# a wildcard would let ANY website embed a call to this API using a visitor's browser session.
# Instead we allow only origins we actually expect traffic from:
#   - localhost/127.0.0.1 on any port (Flutter web dev server uses a random port each run)
#   - the deployed Render domain itself (Swagger UI is served same-origin, so this is mostly
#     belt-and-braces for any browser-based tooling that checks the Origin header)
# Only GET/POST are exposed (the only HTTP methods this API actually implements), only the one
# request header we need (Content-Type) is allowed, and allow_credentials is False because this
# API uses no cookies or session auth — there is nothing that requires credentialed requests.
ALLOWED_ORIGIN_REGEX = r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$|^https://.*\.onrender\.com$"

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=ALLOWED_ORIGIN_REGEX,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
    allow_credentials=False,
)


class TrackFeatures(BaseModel):
    Danceability: float = Field(..., ge=0.0, le=1.0, description="Spotify danceability score")
    Energy: float = Field(..., ge=0.0, le=1.0, description="Spotify energy score")
    Loudness: float = Field(..., ge=-60.0, le=5.0, description="Overall loudness in dB")
    Speechiness: float = Field(..., ge=0.0, le=1.0)
    Acousticness: float = Field(..., ge=0.0, le=1.0)
    Instrumentalness: float = Field(..., ge=0.0, le=1.0)
    Liveness: float = Field(..., ge=0.0, le=1.0)
    Valence: float = Field(..., ge=0.0, le=1.0, description="Musical positiveness")
    Tempo: float = Field(..., ge=0.0, le=250.0, description="Tempo in BPM")
    Duration_ms: int = Field(..., ge=1000, le=6_000_000, description="Track duration in milliseconds")
    Views: int = Field(..., ge=0, le=10_000_000_000, description="YouTube views on the official video")
    Likes: int = Field(..., ge=0, le=100_000_000, description="YouTube likes on the official video")
    Comments: int = Field(..., ge=0, le=50_000_000, description="YouTube comments on the official video")
    Licensed: bool = Field(..., description="Whether the track is licensed content")
    official_video: bool = Field(..., description="Whether an official video exists")
    Key: int = Field(..., ge=0, le=11, description="Musical key (pitch class, 0-11)")
    Album_type: Literal["album", "single", "compilation"] = Field(..., description="Release type")

    model_config = {
        "json_schema_extra": {
            "example": {
                "Danceability": 0.68,
                "Energy": 0.75,
                "Loudness": -5.5,
                "Speechiness": 0.05,
                "Acousticness": 0.1,
                "Instrumentalness": 0.0,
                "Liveness": 0.12,
                "Valence": 0.55,
                "Tempo": 120.0,
                "Duration_ms": 210000,
                "Views": 5000000,
                "Likes": 250000,
                "Comments": 8000,
                "Licensed": True,
                "official_video": True,
                "Key": 5,
                "Album_type": "single",
            }
        }
    }


class PredictionResponse(BaseModel):
    predicted_stream: float
    predicted_stream_log: float


class RetrainResponse(BaseModel):
    best_model: str
    rows_used: int
    metrics: dict


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Orange Economy Ghana streaming predictor is running"}


@app.post("/predict", response_model=PredictionResponse)
def predict(track: TrackFeatures):
    try:
        result = prediction.predict_stream(track.model_dump())
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result


@app.post("/retrain", response_model=RetrainResponse)
async def retrain(file: UploadFile = File(...)):
    """Retrain the model on the existing dataset plus a newly uploaded CSV (same column schema
    as summative/linear_regression/data/Spotify_Youtube.csv), then hot-swap the served model."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file")

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        new_data_path = Path(tmp.name)

    try:
        new_df = pd.read_csv(new_data_path)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}") from exc
    finally:
        new_data_path.unlink(missing_ok=True)

    existing_df = pd.read_csv(DATA_PATH)
    missing_cols = set(existing_df.columns) - set(new_df.columns)
    if missing_cols:
        raise HTTPException(
            status_code=400,
            detail=f"Uploaded CSV is missing required columns: {sorted(missing_cols)}",
        )

    combined_df = pd.concat([existing_df, new_df[existing_df.columns]], ignore_index=True)

    df_clean = pipeline.load_and_clean_df(combined_df)
    X, y, feature_cols = pipeline.engineer_features(df_clean)
    best_model, best_name, scaler, metrics = pipeline.train_and_select_best(X, y, feature_cols)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_DIR / "best_model.joblib")
    joblib.dump(scaler, MODEL_DIR / "scaler.joblib")
    joblib.dump(feature_cols, MODEL_DIR / "feature_columns.joblib")
    (MODEL_DIR / "best_model_name.txt").write_text(best_name)

    combined_df.to_csv(DATA_PATH, index=False)

    prediction.reload_artifacts()

    return {"best_model": best_name, "rows_used": len(combined_df), "metrics": metrics}
