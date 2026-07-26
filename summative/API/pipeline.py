"""Shared feature-engineering + training pipeline.

This mirrors the steps in summative/linear_regression/multivariate.ipynb exactly, so the
/retrain endpoint in main.py produces a model consistent with the one explored in the notebook.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

RANDOM_STATE = 42

IDENTIFIER_COLUMNS = [
    "Unnamed: 0", "Artist", "Url_spotify", "Track", "Album", "Uri",
    "Url_youtube", "Title", "Channel", "Description",
]
REQUIRED_COLUMNS = ["Stream", "Views", "Likes", "Comments", "Danceability"]
LOG_COLUMNS = ["Stream", "Views", "Likes", "Comments"]


def load_and_clean(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.drop(columns=[c for c in IDENTIFIER_COLUMNS if c in df.columns])
    df = df.dropna(subset=[c for c in REQUIRED_COLUMNS if c in df.columns]).reset_index(drop=True)
    return df


def engineer_features(df_clean: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    fe = df_clean.copy()

    for c in LOG_COLUMNS:
        fe[c] = np.log1p(fe[c])

    fe["Licensed"] = fe["Licensed"].astype(bool).astype(int)
    fe["official_video"] = fe["official_video"].astype(bool).astype(int)

    fe = pd.get_dummies(fe, columns=["Key"], prefix="Key", drop_first=True)
    fe = pd.get_dummies(fe, columns=["Album_type"], prefix="Album", drop_first=True)

    dummy_cols = [c for c in fe.columns if c.startswith("Key_") or c.startswith("Album_")]
    fe[dummy_cols] = fe[dummy_cols].astype(int)

    feature_cols = [c for c in fe.columns if c != "Stream"]
    return fe[feature_cols], fe["Stream"], feature_cols


def _evaluate(name, model, X_train, y_train, X_test, y_test, results):
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    results[name] = {
        "model": model,
        "train_mse": mean_squared_error(y_train, pred_train),
        "test_mse": mean_squared_error(y_test, pred_test),
        "test_rmse": mean_squared_error(y_test, pred_test) ** 0.5,
        "test_mae": mean_absolute_error(y_test, pred_test),
        "test_r2": r2_score(y_test, pred_test),
    }


def train_and_select_best(X: pd.DataFrame, y: pd.Series, feature_cols: list[str]):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_cols, index=X_test.index)

    results: dict[str, dict] = {}

    lr = LinearRegression().fit(X_train_scaled, y_train)
    _evaluate("OLS Linear Regression", lr, X_train_scaled, y_train, X_test_scaled, y_test, results)

    sgd = SGDRegressor(
        loss="squared_error", penalty="l2", alpha=1e-4, learning_rate="adaptive",
        eta0=0.01, max_iter=200, random_state=RANDOM_STATE,
    ).fit(X_train_scaled, y_train)
    _evaluate("SGD Linear Regression", sgd, X_train_scaled, y_train, X_test_scaled, y_test, results)

    dt = DecisionTreeRegressor(max_depth=8, min_samples_leaf=10, random_state=RANDOM_STATE)
    dt.fit(X_train_scaled, y_train)
    _evaluate("Decision Tree", dt, X_train_scaled, y_train, X_test_scaled, y_test, results)

    rf = RandomForestRegressor(
        n_estimators=300, max_depth=12, min_samples_leaf=5, random_state=RANDOM_STATE, n_jobs=-1
    )
    rf.fit(X_train_scaled, y_train)
    _evaluate("Random Forest", rf, X_train_scaled, y_train, X_test_scaled, y_test, results)

    best_name = min(results, key=lambda k: results[k]["test_mse"])
    best_model = results[best_name]["model"]

    metrics = {
        name: {k: v for k, v in r.items() if k != "model"}
        for name, r in results.items()
    }
    return best_model, best_name, scaler, metrics
