import pytest
from httpx import ASGITransport, AsyncClient

from src.api import app


@pytest.fixture
def valid_payload():
    return {
        "MedInc": 3.5,
        "HouseAge": 28.0,
        "AveRooms": 5.0,
        "AveBedrms": 1.0,
        "Population": 1400.0,
        "AveOccup": 3.0,
        "Latitude": 35.0,
        "Longitude": -119.0,
    }


@pytest.mark.anyio
async def test_health_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "ok"


@pytest.mark.anyio
async def test_predict_valid_input(valid_payload):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/predict", json=valid_payload)
    # Model might not be loaded in test env → 503 or 200 are both acceptable
    assert response.status_code in (200, 503)
    if response.status_code == 200:
        data = response.json()
        assert "predicted_value" in data
        assert isinstance(data["predicted_value"], float)


@pytest.mark.anyio
async def test_predict_invalid_input():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/predict", json={"bad_field": 999})
    assert response.status_code == 422


@pytest.mark.anyio
async def test_predict_missing_fields(valid_payload):
    incomplete = {k: v for i, (k, v) in enumerate(valid_payload.items()) if i < 3}
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/predict", json=incomplete)
    assert response.status_code == 422


@pytest.mark.anyio
async def test_predict_success_with_loaded_model(monkeypatch, valid_payload):
    import src.api

    class DummyModel:
        def predict(self, df):
            return [2.5]

    monkeypatch.setattr(src.api, "model", DummyModel())
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/predict", json=valid_payload)
    assert response.status_code == 200
    assert response.json() == {"predicted_value": 2.5}


@pytest.mark.anyio
async def test_predict_model_not_loaded(monkeypatch, valid_payload):
    import src.api

    monkeypatch.setattr(src.api, "model", None)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/predict", json=valid_payload)
    assert response.status_code == 503
    assert response.json()["detail"] == "Model is not loaded."


@pytest.mark.anyio
@pytest.mark.parametrize(
    "overrides",
    [
        {"Population": -1.0},
        {"HouseAge": 0.0},
        {"Latitude": 95.0},
        {"Longitude": -185.0},
        {"MedInc": float("nan")},
        {"MedInc": float("inf")},
        {"unexpected_extra": 1.0},
    ],
)
async def test_predict_guardrails_reject_invalid_values(
    monkeypatch, valid_payload, overrides
):
    import json
    import math

    import src.api

    class DummyModel:
        def predict(self, df):
            return [1.0]

    monkeypatch.setattr(src.api, "model", DummyModel())
    payload = {**valid_payload, **overrides}
    # json.dumps handles NaN/Inf via allow_nan or raw string check if needed
    has_nonfinite = any(
        isinstance(v, float) and not math.isfinite(v) for v in payload.values()
    )
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        if has_nonfinite:
            body = json.dumps(payload, allow_nan=True)
            response = await client.post(
                "/predict",
                content=body,
                headers={"Content-Type": "application/json"},
            )
        else:
            response = await client.post("/predict", json=payload)
    assert response.status_code == 422


@pytest.mark.anyio
async def test_predict_internal_error_sanitized(monkeypatch, valid_payload):
    import src.api

    class FailingModel:
        def predict(self, df):
            raise ValueError("secret internal path /app/models/secret.pkl")

    monkeypatch.setattr(src.api, "model", FailingModel())
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/predict", json=valid_payload)
    assert response.status_code == 500
    assert "secret internal path" not in response.text
    assert response.json()["detail"] == "Prediction failed due to an internal error."


@pytest.mark.anyio
async def test_cors_headers_restricted():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        allowed_resp = await client.get(
            "/health", headers={"Origin": "http://localhost:8501"}
        )
        disallowed_resp = await client.get(
            "/health", headers={"Origin": "https://evil.example"}
        )
    assert (
        allowed_resp.headers.get("access-control-allow-origin")
        == "http://localhost:8501"
    )
    assert allowed_resp.headers.get("access-control-allow-credentials") != "true"
    assert "access-control-allow-origin" not in disallowed_resp.headers


def test_lifespan_loads_model_and_predicts(monkeypatch, tmp_path, valid_payload):
    import math

    import joblib
    import pandas as pd
    from fastapi.testclient import TestClient
    from sklearn.linear_model import Ridge

    import src.api

    X_fit = pd.DataFrame([valid_payload, {**valid_payload, "MedInc": 5.0}])
    y_fit = [2.0, 3.0]
    fitted = Ridge().fit(X_fit, y_fit)

    model_file = tmp_path / "model.joblib"
    joblib.dump(fitted, model_file)

    monkeypatch.setattr(src.api, "MODEL_PATH", str(model_file))
    monkeypatch.setattr(src.api, "model", None)

    with TestClient(app) as client:
        health_resp = client.get("/health")
        assert health_resp.status_code == 200
        assert health_resp.json() == {"status": "ok", "model_loaded": True}

        pred_resp = client.post("/predict", json=valid_payload)
        assert pred_resp.status_code == 200
        pred_val = pred_resp.json()["predicted_value"]
        assert isinstance(pred_val, float)
        assert math.isfinite(pred_val)
