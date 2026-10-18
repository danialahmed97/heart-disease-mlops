"""API tests using FastAPI's TestClient."""
import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # runs lifespan -> loads model
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True}


def test_model_info(client):
    r = client.get("/model-info")
    assert r.status_code == 200 and r.json()["model_name"] == "LogisticRegression"


def test_predict_valid(client, valid_payload):
    r = client.post("/predict", json=valid_payload)
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"prediction", "label", "confidence",
                         "probability_disease", "model_version"}
    assert body["prediction"] in (0, 1)
    assert 0.5 <= body["confidence"] <= 1


@pytest.mark.parametrize("field,value", [
    ("age", 300), ("sex", 2), ("cp", 0), ("thal", 5), ("oldpeak", -1), ("ca", 9),
])
def test_predict_out_of_range_422(client, valid_payload, field, value):
    valid_payload[field] = value
    assert client.post("/predict", json=valid_payload).status_code == 422


def test_predict_missing_field_422(client, valid_payload):
    del valid_payload["chol"]
    assert client.post("/predict", json=valid_payload).status_code == 422


def test_predict_wrong_type_422(client, valid_payload):
    valid_payload["age"] = "old"
    assert client.post("/predict", json=valid_payload).status_code == 422


def test_batch(client, valid_payload):
    r = client.post("/predict/batch", json={"records": [valid_payload] * 3})
    assert r.status_code == 200 and r.json()["count"] == 3


def test_batch_empty_422(client):
    assert client.post("/predict/batch", json={"records": []}).status_code == 422


def test_metrics_exposed(client, valid_payload):
    client.post("/predict", json=valid_payload)
    text = client.get("/metrics").text
    assert "heart_predictions_total" in text
    assert "heart_prediction_confidence_bucket" in text
    assert "http_request_duration_seconds" in text
