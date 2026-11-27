"""
Training pipeline for the End-to-End ML Pipeline.

Trains three models (Ridge, Random Forest, MLP Neural Network),
evaluates each, selects the best by R², logs all runs to MLflow,
and generates a comparative PDF report.
"""

import os

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from src.report import generate_report
except ModuleNotFoundError:
    from report import generate_report  # type: ignore[no-redef]


def load_data():
    """Load the California housing dataset."""
    print("Loading data...")
    housing = fetch_california_housing()
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df["target"] = housing.target
    return df


def _evaluate(model, X_test, y_test):
    """Compute standard regression metrics."""
    preds = model.predict(X_test)
    return {
        "mse": mean_squared_error(y_test, preds),
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        "mae": mean_absolute_error(y_test, preds),
        "r2": r2_score(y_test, preds),
    }


def _compute_dataset_stats(df: pd.DataFrame, feature_names: list[str]) -> dict:
    """Compute empirical dataset quality, target censoring, and feature summaries."""
    target = df["target"]
    capped_mask = target >= 5.00001
    feature_summaries = [
        {
            "name": col,
            "mean": float(df[col].mean()),
            "std": float(df[col].std()),
            "min": float(df[col].min()),
            "max": float(df[col].max()),
        }
        for col in feature_names
    ]
    return {
        "total_rows": len(df),
        "null_count": int(df.isnull().sum().sum()),
        "target": {
            "mean": float(target.mean()),
            "std": float(target.std()),
            "min": float(target.min()),
            "q25": float(target.quantile(0.25)),
            "median": float(target.median()),
            "q75": float(target.quantile(0.75)),
            "max": float(target.max()),
        },
        "censored_count": int(capped_mask.sum()),
        "censored_pct": float(capped_mask.mean() * 100.0),
        "features": feature_summaries,
    }


def _compute_residual_stats(
    model, X_test: pd.DataFrame, y_test: pd.Series
) -> dict:
    """Compute hold-out residual diagnostics and censored vs uncensored RMSE."""
    preds = np.asarray(model.predict(X_test), dtype=float)
    y_arr = np.asarray(y_test, dtype=float)
    residuals = y_arr - preds
    abs_err = np.abs(residuals)
    rmse = float(np.sqrt(mean_squared_error(y_arr, preds)))
    mae = float(mean_absolute_error(y_arr, preds))
    capped_mask = y_arr >= 5.00001
    uncensored_rmse = (
        float(np.sqrt(mean_squared_error(y_arr[~capped_mask], preds[~capped_mask])))
        if (~capped_mask).any()
        else None
    )
    censored_rmse = (
        float(np.sqrt(mean_squared_error(y_arr[capped_mask], preds[capped_mask])))
        if capped_mask.any()
        else None
    )
    return {
        "bias": float(np.mean(residuals)),
        "resid_std": float(np.std(residuals)),
        "medae": float(np.median(abs_err)),
        "p95_abs_err": float(np.percentile(abs_err, 95)),
        "rmse": rmse,
        "mae": mae,
        "rmse_mae_ratio": float(rmse / mae) if mae > 0 else 0.0,
        "uncensored_rmse": uncensored_rmse,
        "censored_rmse": censored_rmse,
    }


def train_model():
    """Train three models, compare, and persist the best."""
    df = load_data()

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    mlflow.set_experiment("california_housing_comparison")

    all_results = []
    best_model = None
    best_r2 = -np.inf
    best_name = ""

    # ------------------------------------------------------------------
    # Model 1: Ridge Regression (linear baseline)
    # ------------------------------------------------------------------
    print("\n[1/3] Training Ridge Regression...")
    ridge_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0)),
    ])
    with mlflow.start_run(run_name="Ridge Regression"):
        ridge_pipe.fit(X_train, y_train)
        ridge_metrics = _evaluate(ridge_pipe, X_test, y_test)

        mlflow.log_params({"algorithm": "Ridge", "alpha": 1.0})
        mlflow.log_metrics(ridge_metrics)
        mlflow.sklearn.log_model(ridge_pipe, name="model")

    ridge_result = {
        "name": "Ridge Regression",
        "params": {"algorithm": "Ridge", "alpha": 1.0, "scaling": "StandardScaler"},
        "metrics": ridge_metrics,
        "model": ridge_pipe,
        "feature_importances": None,
    }
    all_results.append(ridge_result)
    print(f"   R²: {ridge_metrics['r2']:.4f}  |  RMSE: {ridge_metrics['rmse']:.4f}")

    if ridge_metrics["r2"] > best_r2:
        best_r2 = ridge_metrics["r2"]
        best_model = ridge_pipe
        best_name = "Ridge Regression"

    # ------------------------------------------------------------------
    # Model 2: Random Forest Regressor (ensemble)
    # ------------------------------------------------------------------
    print("\n[2/3] Training Random Forest Regressor...")
    rf_params = {"n_estimators": 100, "max_depth": 10, "random_state": 42}
    rf_model = RandomForestRegressor(**rf_params)

    with mlflow.start_run(run_name="Random Forest"):
        rf_model.fit(X_train, y_train)
        rf_metrics = _evaluate(rf_model, X_test, y_test)

        mlflow.log_params({"algorithm": "RandomForest", **rf_params})
        mlflow.log_metrics(rf_metrics)
        mlflow.sklearn.log_model(rf_model, name="model")

    rf_result = {
        "name": "Random Forest",
        "params": {"algorithm": "RandomForest", **rf_params},
        "metrics": rf_metrics,
        "model": rf_model,
        "feature_importances": list(rf_model.feature_importances_),
    }
    all_results.append(rf_result)
    print(f"   R²: {rf_metrics['r2']:.4f}  |  RMSE: {rf_metrics['rmse']:.4f}")

    if rf_metrics["r2"] > best_r2:
        best_r2 = rf_metrics["r2"]
        best_model = rf_model
        best_name = "Random Forest"

    # ------------------------------------------------------------------
    # Model 3: MLP Neural Network (deep learning, tuned)
    # ------------------------------------------------------------------
    print("\n[3/3] Training MLP Neural Network (with hyperparameter search)...")
    mlp_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(max_iter=500, early_stopping=True, random_state=42)),
    ])

    mlp_search_space = {
        "mlp__hidden_layer_sizes": [(64, 32), (128, 64, 32), (256, 128, 64), (32, 16)],
        "mlp__activation": ["relu", "tanh"],
        "mlp__alpha": [0.0001, 0.001, 0.01],
        "mlp__learning_rate_init": [0.001, 0.005, 0.01],
    }

    search = RandomizedSearchCV(
        mlp_pipe,
        mlp_search_space,
        n_iter=12,
        cv=3,
        scoring="r2",
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )

    with mlflow.start_run(run_name="MLP Neural Network (Tuned)"):
        search.fit(X_train, y_train)
        mlp_best = search.best_estimator_
        mlp_metrics = _evaluate(mlp_best, X_test, y_test)

        best_hp = search.best_params_
        mlflow.log_params({
            "algorithm": "MLPRegressor",
            "hidden_layer_sizes": str(best_hp.get("mlp__hidden_layer_sizes")),
            "activation": best_hp.get("mlp__activation"),
            "alpha": best_hp.get("mlp__alpha"),
            "learning_rate_init": best_hp.get("mlp__learning_rate_init"),
            "cv_best_score": search.best_score_,
        })
        mlflow.log_metrics(mlp_metrics)
        mlflow.sklearn.log_model(
            mlp_best,
            name="model",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )

    mlp_result = {
        "name": "MLP Neural Network",
        "params": {
            "algorithm": "MLPRegressor",
            "hidden_layer_sizes": str(best_hp.get("mlp__hidden_layer_sizes")),
            "activation": best_hp.get("mlp__activation"),
            "alpha": best_hp.get("mlp__alpha"),
            "learning_rate_init": best_hp.get("mlp__learning_rate_init"),
            "max_iter": 500,
            "early_stopping": True,
            "tuning_method": "RandomizedSearchCV (12 iter, 3-fold CV)",
        },
        "metrics": mlp_metrics,
        "model": mlp_best,
        "feature_importances": None,
    }
    all_results.append(mlp_result)
    print(f"   R²: {mlp_metrics['r2']:.4f}  |  RMSE: {mlp_metrics['rmse']:.4f}")
    print(f"   Best hyperparameters: {best_hp}")

    if mlp_metrics["r2"] > best_r2:
        best_r2 = mlp_metrics["r2"]
        best_model = mlp_best
        best_name = "MLP Neural Network"

    # ------------------------------------------------------------------
    # Persist the best model
    # ------------------------------------------------------------------
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/model.joblib")
    print(f"\nBest model: {best_name} (R² = {best_r2:.4f})")
    print("Model saved to models/model.joblib")

    # ------------------------------------------------------------------
    # Generate comparative PDF report with empirical diagnostics
    # ------------------------------------------------------------------
    feature_names = list(X.columns)
    dataset_stats = _compute_dataset_stats(df, feature_names)
    residual_stats = _compute_residual_stats(best_model, X_test, y_test)
    perm = permutation_importance(
        best_model, X_test, y_test, n_repeats=5, random_state=42, scoring="r2"
    )
    best_result = next(res for res in all_results if res["name"] == best_name)
    best_result["feature_importances"] = [float(v) for v in perm.importances_mean]
    best_result["feature_importances_std"] = [float(v) for v in perm.importances_std]

    report_path = generate_report(
        model_results=all_results,
        best_model_name=best_name,
        feature_names=feature_names,
        train_size=len(X_train),
        test_size=len(X_test),
        dataset_stats=dataset_stats,
        residual_stats=residual_stats,
        importance_method="Permutation Importance (Test Set ΔR², 5 repeats)",
    )
    print(f"PDF report saved to {report_path}")


if __name__ == "__main__":
    train_model()
