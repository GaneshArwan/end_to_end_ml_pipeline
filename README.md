# California Housing Price Prediction Pipeline

![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.141%2B-00a393.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.61%2B-FF4B4B.svg)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.9.0-orange.svg)
[![Live Demo](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ca-housing-predictor.streamlit.app/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GaneshArwan/end_to_end_ml_pipeline/blob/main/notebooks/california_housing_analysis.ipynb)

This repository trains, evaluates, and serves regression models for California block-group median house values using 1990 U.S. Census data. It includes an MLflow training script, a ReportLab PDF evaluation generator, a validated FastAPI service, a Streamlit interface that runs either against FastAPI or standalone on Streamlit Community Cloud, and a Google Colab notebook.

## Dataset and Target Censoring

The pipeline loads the 20,640-row, 8-feature [California Housing dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html) with 0 missing values and splits it 80/20 (`16,512` train, `4,128` test, `random_state=42`). Targets are recorded in `$100,000` units (mean `2.0686`, median `1.7970`, IQR `1.1960` to `2.6472`). Exactly `965` records (`4.68%`) hit the census cap of `5.00001` (`$500,001`), which introduces right-censoring bias in upper-tail predictions.

## Modeling and Empirical Results

Three model families are trained in `src/train.py` and tracked in MLflow: Ridge Regression (`StandardScaler` + L2 linear baseline), Random Forest (`100` trees, `max_depth=10`), and a tuned `MLPRegressor` (`StandardScaler` + `RandomizedSearchCV` over `12` candidates with `3`-fold CV and `early_stopping=True`). The winning MLP uses `hidden_layer_sizes=(64, 32)`, `activation="tanh"`, `alpha=0.01`, and `learning_rate_init=0.01`.

| Model | R² | RMSE | MAE | MSE |
|---|---:|---:|---:|---:|
| MLP Neural Network (selected) | 0.7857 | 0.5299 | 0.3624 | 0.2808 |
| Random Forest | 0.7738 | 0.5445 | 0.3663 | 0.2965 |
| Ridge Regression | 0.5758 | 0.7456 | 0.5332 | 0.5559 |

Hold-out residual diagnostics on the selected MLP show:

- Mean bias (`y - y_hat`) is `-0.0154` (SD `0.5296`), median absolute error is `0.2470` (`$24,695`), and 95th-percentile absolute error is `1.0997` (`$109,971`).
- The RMSE/MAE ratio is `1.46` (`$52,987` vs `$36,238`), showing right-skewed error tails.
- Splitting the test set at the census cap yields `0.4829` RMSE on uncensored tracts (`< $500,001`) versus `1.1536` RMSE on capped tracts (`>= $500,001`).

Hold-out permutation importance (`5` repeats, `random_state=42`, `scoring="r2"`) measures sensitivity in R² drop:

| Rank | Feature | Mean ΔR² | Std |
|---:|---|---:|---:|
| 1 | Latitude | 0.9399 | ± 0.0281 |
| 2 | Longitude | 0.8301 | ± 0.0113 |
| 3 | MedInc | 0.4570 | ± 0.0106 |
| 4 | AveOccup | 0.2176 | ± 0.0070 |
| 5 | AveRooms | 0.0931 | ± 0.0037 |
| 6 | HouseAge | 0.0600 | ± 0.0031 |
| 7 | AveBedrms | 0.0332 | ± 0.0011 |
| 8 | Population | 0.0083 | ± 0.0004 |

Because `Latitude` and `Longitude` co-vary along California's coast and interior valleys, permuting each coordinate independently captures combined spatial sensitivity rather than an isolated effect.

## Repository Layout

| Path | Purpose |
|---|---|
| `src/train.py` | Trains Ridge, Random Forest, and tuned MLP, logs to MLflow, saves `models/model.joblib`, and writes the PDF report |
| `src/report.py` | Generates the 3-page two-column evaluation report at `reports/model_report.pdf` |
| `src/api.py` | FastAPI `/predict` and `/health` service with Pydantic range checks, non-finite float rejection, and `ALLOWED_ORIGINS` CORS filtering |
| `src/app.py` | Streamlit UI that calls `API_URL` when set or falls back to `models/model.joblib` for standalone deployment |
| `notebooks/california_housing_analysis.ipynb` | Exploratory analysis, residual diagnostics, and permutation importance notebook for Google Colab |
| `tests/` | Pytest suite covering API guardrails, FastAPI lifespan loading, training/report integration, and Streamlit inference |
| `.github/workflows/ci.yml` | Manual (`workflow_dispatch`) Ruff and Pytest workflow to avoid spending free-tier CI minutes on every push |

## Running Locally

```bash
# Install runtime and test dependencies
pip install -r requirements.txt

# Retrain models, log MLflow runs, and rebuild reports/model_report.pdf
python src/train.py

# Option A: launch Streamlit directly using models/model.joblib
streamlit run src/app.py

# Option B: run the FastAPI backend and point Streamlit to it
uvicorn src.api:app --host 127.0.0.1 --port 8000
# In another terminal (Linux/macOS: export API_URL=...; PowerShell: $env:API_URL=...)
API_URL=http://127.0.0.1:8000/predict streamlit run src/app.py

# Option C: run both containers via Docker Compose
docker compose up --build

# Run checks
ruff check src/ tests/
pytest tests/ --cov=src -v
```

## Cloud Deployment

### Streamlit Community Cloud

Live app: https://ca-housing-predictor.streamlit.app/

To deploy your own copy, point Streamlit Community Cloud at `GaneshArwan/end_to_end_ml_pipeline` (`main` branch, entry file `src/app.py`). When `API_URL` is unset, `src/app.py` loads `models/model.joblib` directly without requiring an external API server.

### Google Colab

Click the **Open in Colab** badge at the top of this file to open `notebooks/california_housing_analysis.ipynb` directly from GitHub.

## Limitations

Predictions come from 1990 census block-group aggregates and do not reflect post-1990 inflation, interest rates, or zoning changes. Because `4.68%` of targets are capped at `$500,001`, upper-tier properties are systematically underpredicted. Treat the model as an educational and structural baseline rather than a live appraisal tool.
