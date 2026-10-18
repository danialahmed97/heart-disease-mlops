"""CLI: predict heart disease from a JSON or CSV file using the saved model."""
import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import MODEL_PATH, RAW_FEATURES, setup_logging  # noqa: E402

logger = logging.getLogger(__name__)


def load_input(path: Path) -> pd.DataFrame:
    """Read one record / list of records from JSON, or rows from CSV."""
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        data = json.loads(path.read_text())
        df = pd.DataFrame(data if isinstance(data, list) else [data])
    missing = [c for c in RAW_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df[RAW_FEATURES].astype("float64")


def predict(df: pd.DataFrame, model_path: Path = MODEL_PATH) -> pd.DataFrame:
    """Return a DataFrame with prediction, label and disease probability."""
    model = joblib.load(model_path)
    proba = model.predict_proba(df)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return pd.DataFrame({
        "prediction": pred,
        "label": ["Disease" if p else "Healthy" for p in pred],
        "probability_disease": proba.round(4),
    })


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSON or CSV file")
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    args = parser.parse_args(argv)
    setup_logging()
    result = predict(load_input(args.input), args.model)
    sys.stdout.write(result.to_json(orient="records", indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
