# Linear Regression Model — Orange Economy Ghana

## Mission
Ghana's creative sector (especially music) has real export potential, but individual artists struggle to turn their craft into a sustainable income. This project predicts a track's likely Spotify stream count from its audio characteristics and its YouTube video's engagement — using real numbers once a song is live, or target numbers to plan promotion effort beforehand — giving artists and producers a practical, data-backed read on what drives streaming success. **Dataset:** [Spotify and YouTube](https://www.kaggle.com/datasets/salvatorerastelli/spotify-and-youtube) (Kaggle) — 20,718 tracks from 2,074 artists, combining Spotify audio features with YouTube engagement metrics.

**Scope note:** the model's strongest predictors (YouTube Views/Likes/Comments) are engagement metrics, so this is a *post-release trajectory / promotion-planning* tool, not a pre-release hit predictor — it's most useful once a track and its video are live (plug in real numbers to gauge potential), or beforehand with target numbers to plan how much promotion effort a given streaming goal requires.

## Live API (Swagger UI)
**https://linear-regression-model-4z71.onrender.com/docs**

The service is on Render's free tier, so the first request after a period of inactivity may take 30-60 seconds to wake up.

## Video demo


## Key visualizations (see the notebook for full interpretation)

**Correlation heatmap** — YouTube `Likes` (0.65) and `Views` (0.60) are by far the strongest predictors of `Stream`; among audio features, `Loudness` (+0.12) is strongest while `Acousticness`/`Instrumentalness` are mildly negative.

![Correlation heatmap](summative/linear_regression/correlation_heatmap.png)

**Stream distribution, before/after log transform** — raw stream counts are extremely right-skewed (skew ≈ 4.1); a `log1p` transform (applied to `Stream`, `Views`, `Likes`, `Comments`) brings that down to ≈ -0.6, which is what makes linear regression viable here.

![Stream distribution](summative/linear_regression/stream_distribution.png)

Two more plots (a views-vs-streams scatter/album-type boxplot, and the SGD gradient-descent loss curve) are generated in the notebook and saved alongside it.

## Repo structure
```
linear_regression_model/
├── summative/
│   ├── linear_regression/
│   │   ├── multivariate.ipynb   # EDA, feature engineering, model training/comparison
│   │   ├── data/                # dataset (Spotify_Youtube.csv)
│   │   └── models/              # saved best model + scaler + feature columns
│   ├── API/
│   │   ├── main.py               # FastAPI app (CORS, Pydantic validation, /predict, /retrain)
│   │   ├── prediction.py         # loads the saved model, makes a prediction (Task 1 -> Task 2 handoff)
│   │   ├── pipeline.py           # shared feature-engineering + training logic
│   │   └── requirements.txt
│   ├── FlutterApp/                # mobile app (streaming_predictor)
│   └── pyproject.toml             # uv-managed environment for the notebook/API
```

## Running the notebook / API locally

This project uses [uv](https://docs.astral.sh/uv/) for Python package management.

```bash
cd summative
uv sync                                   # installs all dependencies from pyproject.toml/uv.lock
uv run jupyter lab linear_regression/multivariate.ipynb   # open the notebook

# Run the API locally:
cd API
uv run --project .. uvicorn main:app --reload
# Swagger UI at http://127.0.0.1:8000/docs
```

## Running the mobile app

The app is a single Flutter page (`summative/FlutterApp`) that calls the live Render API — no local backend needed.

```bash
cd summative/FlutterApp
flutter pub get
flutter run -d chrome        # or: flutter run  (with a connected Android/iOS device or emulator)
```

Fill in all 17 fields (grouped into Audio characteristics, YouTube engagement, and Release details, each with inline guidance on what the value means and where to find it) and tap **Predict** to get the track's predicted Spotify stream count, along with a plain-language read of where that falls relative to the training data (e.g. "breakout hit — top 25%").

## Retraining with new data
`POST /retrain` on the API accepts a CSV upload (same columns as `Spotify_Youtube.csv`), appends it to the existing dataset, retrains and re-compares all four models (OLS linear regression, SGD/gradient-descent linear regression, Decision Tree, Random Forest), and hot-swaps the served model to whichever wins — no server restart required.
