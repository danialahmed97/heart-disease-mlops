"""Tests for training utilities, prediction CLI and MLflow tracking."""
import json

import joblib
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src import predict, tracking, train
from src.config import RAW_FEATURES, TARGET


def test_candidates_cover_three_families():
    assert len(train.get_candidates()) == 3


def test_test_metrics_perfect():
    y = np.array([0, 1, 0, 1])
    m = train.test_metrics(y, y, y.astype(float))
    assert m["test_accuracy"] == 1 and m["test_roc_auc"] == 1


def test_select_best_prefers_recall_within_tolerance():
    results = [
        {"name": "A", "cv_roc_auc_mean": 0.90, "test_recall": 0.80},
        {"name": "B", "cv_roc_auc_mean": 0.898, "test_recall": 0.95},
        {"name": "C", "cv_roc_auc_mean": 0.85, "test_recall": 1.0},
    ]
    assert train.select_best(results)["name"] == "B"


def test_select_best_prefers_auc_when_clear():
    results = [
        {"name": "A", "cv_roc_auc_mean": 0.95, "test_recall": 0.7},
        {"name": "B", "cv_roc_auc_mean": 0.80, "test_recall": 0.99},
    ]
    assert train.select_best(results)["name"] == "A"


def test_saved_pipeline_predicts_valid_probabilities(model_path, frame):
    pipe = joblib.load(model_path)
    proba = pipe.predict_proba(frame[RAW_FEATURES])
    assert proba.shape == (len(frame), 2)
    assert np.allclose(proba.sum(axis=1), 1)


def test_predict_cli_json_and_csv(model_path, frame, tmp_path, valid_payload, capsys):
    jp = tmp_path / "p.json"
    jp.write_text(json.dumps(valid_payload))
    assert predict.main([str(jp), "--model", str(model_path)]) == 0
    assert json.loads(capsys.readouterr().out)[0]["label"] in {"Healthy", "Disease"}
    cp = tmp_path / "p.csv"
    frame[RAW_FEATURES].head(3).to_csv(cp, index=False)
    out = predict.predict(predict.load_input(cp), model_path)
    assert len(out) == 3


def test_predict_missing_columns_raises(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"age": 50}))
    with pytest.raises(ValueError):
        predict.load_input(p)


@pytest.fixture()
def tiny_training(tmp_path, monkeypatch, frame):
    """Run train.main on synthetic data, LR only, into tmp paths."""
    monkeypatch.setattr(train, "CLEAN_PATH", tmp_path / "clean.csv")
    frame.to_csv(tmp_path / "clean.csv", index=False)
    monkeypatch.setattr(train, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(train, "MODEL_PATH", tmp_path / "models" / "m.joblib")
    monkeypatch.setattr(train, "METADATA_PATH", tmp_path / "models" / "meta.json")
    monkeypatch.setattr(train, "DOCS_DIR", tmp_path / "docs")
    monkeypatch.setattr(tracking, "ROOT", tmp_path)
    monkeypatch.setattr(train, "get_candidates", lambda: {
        "LogisticRegression": (LogisticRegression(max_iter=500),
                               {"model__C": [0.1, 1]})})
    return tmp_path


def test_train_main_with_mlflow(tiny_training):
    out = train.main(use_mlflow=True)
    assert out["best"]["name"] == "LogisticRegression"
    meta = json.loads((tiny_training / "models" / "meta.json").read_text())
    assert meta["features"] == RAW_FEATURES and "sklearn_version" in meta
    assert (tiny_training / "docs" / "model_comparison.md").exists()
    assert (tiny_training / "mlruns").exists()


def test_train_main_without_mlflow(tiny_training):
    out = train.main(use_mlflow=False)
    assert "run_id" not in out["best"]
    assert TARGET not in out["split"][0].columns
