"""Download the UCI Cleveland Heart Disease dataset and save it as CSV."""
import io
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import requests

URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)
COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]
OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "heart.csv"

logger = logging.getLogger("download_data")


def fetch(url: str = URL, retries: int = 4, backoff: float = 2.0) -> str:
    """Fetch raw text from ``url`` with exponential-backoff retries."""
    for attempt in range(1, retries + 1):
        try:
            logger.info("Downloading %s (attempt %d/%d)", url, attempt, retries)
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as exc:
            logger.warning("Attempt %d failed: %s", attempt, exc)
            if attempt == retries:
                raise
            time.sleep(backoff ** attempt)
    raise RuntimeError("unreachable")  # pragma: no cover


def parse(text: str) -> pd.DataFrame:
    """Parse the headerless UCI text into a named DataFrame."""
    df = pd.read_csv(io.StringIO(text), header=None, names=COLUMNS, dtype=str)
    if df.shape[1] != len(COLUMNS):
        raise ValueError(f"Unexpected column count: {df.shape[1]}")
    return df


def main(out_path: Path = OUT_PATH) -> Path:
    """Download, name the columns and write the raw CSV."""
    df = parse(fetch())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info("Saved %d rows x %d cols to %s", *df.shape, out_path)
    return out_path


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    try:
        main()
    except Exception:
        logger.exception("Download failed")
        sys.exit(1)
