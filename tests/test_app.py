import joblib
import pandas as pd
import requests
from sklearn.linear_model import Ridge

from src.app import predict_local, request_prediction

SAMPLE_PAYLOAD = {
    "MedInc": 3.5,
    "HouseAge": 28.0,
    "AveRooms": 5.0,
    "AveBedrms": 1.0,
    "Population": 1400.0,
    "AveOccup": 3.0,
    "Latitude": 35.0,
    "Longitude": -119.0,
}


def test_request_prediction_success(monkeypatch):
    class DummyResponse:
        status_code = 200

        def json(self):
            return {"predicted_value": 2.5}

    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: DummyResponse())
    value, err = request_prediction("http://localhost:8000/predict", SAMPLE_PAYLOAD)
    assert err is None
    assert value == 250000.0


def test_request_prediction_non_200(monkeypatch):
    class DummyResponse:
        status_code = 503
        text = "internal detail"

    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: DummyResponse())
    value, err = request_prediction("http://localhost:8000/predict", SAMPLE_PAYLOAD)
    assert value is None
    assert err == "API returned status 503."


def test_request_prediction_timeout(monkeypatch):
    def raise_timeout(*args, **kwargs):
        raise requests.exceptions.Timeout("timed out")

    monkeypatch.setattr(requests, "post", raise_timeout)
    value, err = request_prediction("http://localhost:8000/predict", SAMPLE_PAYLOAD)
    assert value is None
    assert err is not None and "timed out" in err.lower()


def test_request_prediction_connection_error_scrubs_url(monkeypatch):
    def raise_conn(*args, **kwargs):
        raise requests.exceptions.ConnectionError("refused")

    monkeypatch.setattr(requests, "post", raise_conn)
    internal_url = "http://internal-api.local:8000/predict"
    value, err = request_prediction(internal_url, SAMPLE_PAYLOAD)
    assert value is None
    assert err is not None
    assert internal_url not in err
    assert "could not connect" in err.lower()


def test_request_prediction_request_exception(monkeypatch):
    def raise_req(*args, **kwargs):
        raise requests.exceptions.RequestException("generic error")

    monkeypatch.setattr(requests, "post", raise_req)
    value, err = request_prediction("http://localhost:8000/predict", SAMPLE_PAYLOAD)
    assert value is None
    assert err == "Failed to communicate with the prediction API."


def test_request_prediction_malformed_json(monkeypatch):
    class BadJsonResponse:
        status_code = 200

        def json(self):
            return {"unexpected": 1.0}

    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: BadJsonResponse())
    value, err = request_prediction("http://localhost:8000/predict", SAMPLE_PAYLOAD)
    assert value is None
    assert err == "Received an unexpected response format from the API."


def test_predict_local_success(tmp_path):
    X_fit = pd.DataFrame([SAMPLE_PAYLOAD, {**SAMPLE_PAYLOAD, "MedInc": 5.0}])
    model = Ridge().fit(X_fit, [2.0, 3.0])
    model_path = tmp_path / "model.joblib"
    joblib.dump(model, model_path)

    value, err = predict_local(SAMPLE_PAYLOAD, model_path=str(model_path))
    assert err is None
    assert isinstance(value, float)
    assert value > 0


def test_predict_local_missing_model(tmp_path):
    value, err = predict_local(SAMPLE_PAYLOAD, model_path=str(tmp_path / "missing.joblib"))
    assert value is None
    assert err is not None and "not found" in err.lower()
