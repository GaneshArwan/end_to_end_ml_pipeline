import os

import pandas as pd

from src.train import load_data


def test_load_data_returns_dataframe():
    df = load_data()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_load_data_has_expected_columns():
    df = load_data()
    expected = [
        "MedInc", "HouseAge", "AveRooms", "AveBedrms",
        "Population", "AveOccup", "Latitude", "Longitude", "target",
    ]
    for col in expected:
        assert col in df.columns, f"Missing column: {col}"


def test_load_data_no_nulls():
    df = load_data()
    assert df.isnull().sum().sum() == 0, "Dataset contains null values"


def test_model_artifact_exists_after_train():
    """Check that a model file exists (trained in a prior step or CI job)."""
    model_path = "models/model.joblib"
    if os.path.exists(model_path):
        size = os.path.getsize(model_path)
        assert size > 0, "Model file is empty"
    else:
        # In CI, model is trained before tests run, so this should pass.
        # Locally, skip gracefully if not yet trained.
        import pytest
        pytest.skip("Model not yet trained — run src/train.py first")

def test_evaluate():
    import numpy as np

    from src.train import _evaluate
    
    class DummyModel:
        def predict(self, X):
            return np.array([2.0, 3.0, 4.0])
            
    X_test = np.array([[1], [2], [3]])
    y_test = np.array([2.0, 3.0, 4.0])
    
    metrics = _evaluate(DummyModel(), X_test, y_test)
    assert metrics["mse"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["mae"] == 0.0
    assert metrics["r2"] == 1.0

def test_train_model(monkeypatch, tmp_path):
    import numpy as np

    import src.train
    from src.train import train_model
    
    # Mock data with enough samples for 3-fold CV + MLP early_stopping (10% val split)
    class MockHousing:
        def __init__(self):
            rng = np.random.default_rng(42)
            self.data = rng.random((100, 8))
            self.target = rng.random(100)
            self.feature_names = ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"]

    def mock_fetch(*args, **kwargs):
        return MockHousing()

    monkeypatch.setattr(src.train, "fetch_california_housing", mock_fetch)
    monkeypatch.chdir(tmp_path)

    # Run full train -> report pipeline in isolated tmp_path
    train_model()

    model_path = tmp_path / "models" / "model.joblib"
    report_path = tmp_path / "reports" / "model_report.pdf"
    assert os.path.exists(model_path)
    assert os.path.exists(report_path)
    assert os.path.getsize(report_path) > 0

    import math

    import joblib

    loaded_model = joblib.load(model_path)
    sample = pd.DataFrame([dict.fromkeys(MockHousing().feature_names, 0.5)])
    pred = float(loaded_model.predict(sample)[0])
    assert math.isfinite(pred)
