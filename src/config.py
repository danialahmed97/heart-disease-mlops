"""Shared constants and reproducibility helpers."""
import logging
import os
import random
from pathlib import Path

import numpy as np

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
CLEAN_PATH = ROOT / "data" / "processed" / "heart_clean.csv"
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "model.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"
FIGURES_DIR = ROOT / "reports" / "figures"
DOCS_DIR = ROOT / "docs"

TARGET = "num"
RAW_FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]
EXPERIMENT_NAME = "heart-disease-classification"
MODEL_NAME = "heart-disease-classifier"


def set_seeds(seed: int = SEED) -> None:
    """Seed python, numpy and hash randomisation for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure uniform console logging."""
    logging.basicConfig(
        level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
