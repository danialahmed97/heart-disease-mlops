"""Deliberately failing test used to demonstrate CI failure handling."""
from fastapi.testclient import TestClient

from api.main import app


def test_prediction_range_wrong(valid_payload):
    with TestClient(app) as client:
        body = client.post("/predict", json=valid_payload).json()
    # Deliberate bug: a class label can only be 0 or 1, never 2.
    assert body["prediction"] == 2
