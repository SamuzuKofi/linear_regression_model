"""Loads the saved best model and makes a Stream-count prediction from raw track features.

This is the Task 1 -> Task 2 handoff script: it is imported by main.py's /predict endpoint,
and can also be run directly (`python prediction.py`) to demo a prediction on a sample track.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parent.parent / "linear_regression" / "models"


def _load_artifacts():
    global _model, _scaler, _feature_columns
    _model = joblib.load(MODEL_DIR / "best_model.joblib")
    _scaler = joblib.load(MODEL_DIR / "scaler.joblib")
    _feature_columns = joblib.load(MODEL_DIR / "feature_columns.joblib")


_model = None
_scaler = None
_feature_columns: list[str] = []
_load_artifacts()


def reload_artifacts():
    """Re-read model/scaler/feature_columns from disk (called after /retrain saves new ones)."""
    _load_artifacts()

NUMERIC_PASSTHROUGH = [
    "Danceability", "Energy", "Loudness", "Speechiness", "Acousticness",
    "Instrumentalness", "Liveness", "Valence", "Tempo", "Duration_ms",
]


def build_feature_row(payload: dict[str, Any], feature_columns: list[str]) -> pd.DataFrame:
    """Map a raw track payload onto the exact one-hot/scaled feature layout used at training time."""
    row = {col: 0.0 for col in feature_columns}

    for col in NUMERIC_PASSTHROUGH:
        if col in row:
            row[col] = float(payload[col])

    row["Views"] = math.log1p(payload["Views"])
    row["Likes"] = math.log1p(payload["Likes"])
    row["Comments"] = math.log1p(payload["Comments"])
    row["Licensed"] = int(bool(payload["Licensed"]))
    row["official_video"] = int(bool(payload["official_video"]))

    key_col = f"Key_{float(payload['Key'])}"
    if key_col in row:
        row[key_col] = 1

    album_col = f"Album_{payload['Album_type']}"
    if album_col in row:
        row[album_col] = 1

    return pd.DataFrame([row], columns=feature_columns)


def predict_stream(payload: dict[str, Any]) -> dict[str, float]:
    """Predict a track's Spotify stream count from its audio/engagement features."""
    row = build_feature_row(payload, _feature_columns)
    row_scaled = pd.DataFrame(_scaler.transform(row), columns=_feature_columns)

    pred_log = float(_model.predict(row_scaled)[0])
    pred_stream = math.expm1(pred_log)

    return {"predicted_stream": pred_stream, "predicted_stream_log": pred_log}


if __name__ == "__main__":
    sample_track = {
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
        "Views": 5_000_000,
        "Likes": 250_000,
        "Comments": 8_000,
        "Licensed": True,
        "official_video": True,
        "Key": 5,
        "Album_type": "single",
    }
    result = predict_stream(sample_track)
    print("Sample track:", sample_track)
    print("Predicted Spotify streams: {:,.0f}".format(result["predicted_stream"]))
