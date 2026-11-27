import os

from src.report import generate_report


def test_generate_report(tmp_path):
    model_results = [
        {
            "name": "Dummy Model",
            "metrics": {"r2": 0.9, "mse": 0.1, "rmse": 0.316},
            "params": {"param1": 1, "param2": 2},
            "feature_importances": [0.5, 0.3, 0.2],
        }
    ]
    feature_names = ["feat1", "feat2", "feat3"]
    output_dir = tmp_path / "reports"

    report_path = generate_report(
        model_results=model_results,
        best_model_name="Dummy Model",
        feature_names=feature_names,
        train_size=100,
        test_size=20,
        output_dir=str(output_dir),
    )

    assert os.path.exists(report_path)
    assert os.path.getsize(report_path) > 0
    assert report_path.endswith("model_report.pdf")


def test_generate_report_no_feature_importances(tmp_path):
    model_results = [
        {
            "name": "Dummy Model",
            "metrics": {"r2": 0.9, "mse": 0.1, "rmse": 0.316},
            "params": {"param1": 1, "param2": 2},
            "feature_importances": None,
        }
    ]
    feature_names = ["feat1", "feat2", "feat3"]
    output_dir = tmp_path / "reports2"

    report_path = generate_report(
        model_results=model_results,
        best_model_name="Dummy Model",
        feature_names=feature_names,
        train_size=100,
        test_size=20,
        output_dir=str(output_dir),
    )

    assert os.path.exists(report_path)
    assert os.path.getsize(report_path) > 0


def test_generate_report_escapes_xml_and_zero_importances(tmp_path):
    model_results = [
        {
            "name": '<script>&"Model"',
            "metrics": {"r2": 0.5, "mse": 0.25, "rmse": 0.5},
            "params": {"<k>": "v&1"},
            "feature_importances": [0.0, 0.0, 0.0],
        }
    ]
    feature_names = ["A&B", "<C>", "D"]
    output_dir = tmp_path / "reports_escaped"

    report_path = generate_report(
        model_results=model_results,
        best_model_name='<script>&"Model"',
        feature_names=feature_names,
        train_size=80,
        test_size=20,
        output_dir=str(output_dir),
    )

    assert os.path.exists(report_path)
    assert os.path.getsize(report_path) > 0


def test_generate_report_with_full_data_science_stats(tmp_path):
    model_results = [
        {
            "name": "Ridge Regression",
            "metrics": {"r2": 0.58, "mse": 0.55, "rmse": 0.74, "mae": 0.53},
            "params": {"alpha": 1.0},
            "feature_importances": None,
        },
        {
            "name": "MLP Neural Network",
            "metrics": {"r2": 0.79, "mse": 0.28, "rmse": 0.53, "mae": 0.36},
            "params": {"hidden_layer_sizes": "(64, 32)", "activation": "tanh"},
            "feature_importances": [0.94, 0.83, 0.46],
            "feature_importances_std": [0.02, 0.01, 0.01],
        },
    ]
    feature_names = ["Latitude", "Longitude", "MedInc"]
    dataset_stats = {
        "total_rows": 20640,
        "null_count": 0,
        "target": {
            "mean": 2.0686,
            "std": 1.1540,
            "min": 0.1500,
            "q25": 1.1960,
            "median": 1.7970,
            "q75": 2.6472,
            "max": 5.0000,
        },
        "censored_count": 965,
        "censored_pct": 4.68,
        "features": [
            {"name": "Latitude", "mean": 35.63, "std": 2.14, "min": 32.54, "max": 41.95},
            {"name": "Longitude", "mean": -119.57, "std": 2.00, "min": -124.35, "max": -114.31},
            {"name": "MedInc", "mean": 3.87, "std": 1.90, "min": 0.50, "max": 15.00},
        ],
    }
    residual_stats = {
        "bias": -0.0154,
        "resid_std": 0.5297,
        "medae": 0.2510,
        "p95_abs_err": 1.0997,
        "rmse": 0.5299,
        "mae": 0.3624,
        "rmse_mae_ratio": 1.46,
        "uncensored_rmse": 0.4800,
        "censored_rmse": 1.1200,
    }

    report_path = generate_report(
        model_results=model_results,
        best_model_name="MLP Neural Network",
        feature_names=feature_names,
        train_size=16512,
        test_size=4128,
        output_dir=str(tmp_path / "reports_ds"),
        dataset_stats=dataset_stats,
        residual_stats=residual_stats,
        importance_method="Permutation Importance (Test Set ΔR²)",
    )

    assert os.path.exists(report_path)
    assert os.path.getsize(report_path) > 0
    with open(report_path, "rb") as f:
        assert f.read(4) == b"%PDF"
