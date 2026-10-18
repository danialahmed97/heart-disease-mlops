"""Shared fixtures: a small trained model in a temp dir for API/model tests."""
import json
import os
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import RAW_FEATURES, TARGET  # noqa: E402
from src.features import build_pipeline  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
_TMP = Path(tempfile.mkdtemp(prefix="heart-test-"))


def synthetic_frame(n: int = 120, seed: int = 0) -> pd.DataFrame:
    """Synthetic data with the real schema/value ranges (no download needed)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": rng.integers(29, 78, n), "sex": rng.integers(0, 2, n),
        "cp": rng.integers(1, 5, n), "trestbps": rng.integers(94, 200, n),
        "chol": rng.integers(126, 400, n), "fbs": rng.integers(0, 2, n),
        "restecg": rng.integers(0, 3, n), "thalach": rng.integers(71, 202, n),
        "exang": rng.integers(0, 2, n), "oldpeak": rng.uniform(0, 6, n).round(1),
        "slope": rng.integers(1, 4, n), "ca": rng.integers(0, 4, n),
        "thal": rng.choice([3, 6, 7], n),
    }).astype("float64")
    score = df["oldpeak"] + df["ca"] + (df["cp"] == 4) - df["thalach"] / 100
    df[TARGET] = (score + rng.normal(0, 0.5, n) > score.median()).astype(int)
    return df


def _build_model_files() -> tuple[Path, Path]:
    df = synthetic_frame()
    pipe = build_pipeline(LogisticRegression(max_iter=1000, random_state=42))
    pipe.fit(df[RAW_FEATURES], df[TARGET])
    model_path, meta_path = _TMP / "model.joblib", _TMP / "metadata.json"
    joblib.dump(pipe, model_path)
    meta_path.write_text(json.dumps({
        "model_name": "LogisticRegression", "version": "test",
        "metrics": {}, "features": RAW_FEATURES}))
    return model_path, meta_path


_MODEL, _META = _build_model_files()
os.environ["MODEL_PATH"] = str(_MODEL)
os.environ["METADATA_PATH"] = str(_META)
os.environ["LOG_DIR"] = str(_TMP / "logs")


@pytest.fixture(scope="session")
def frame() -> pd.DataFrame:
    """Synthetic labelled frame."""
    return synthetic_frame()


@pytest.fixture(scope="session")
def model_path() -> Path:
    """Path of the temp trained pipeline."""
    return _MODEL


@pytest.fixture()
def valid_payload() -> dict:
    """A valid /predict request body."""
    return json.loads((ROOT / "sample_request.json").read_text())
