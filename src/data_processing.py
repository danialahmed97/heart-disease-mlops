"""Cleaning utilities for the Heart Disease UCI dataset."""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "heart.csv"
CLEAN_PATH = ROOT / "data" / "processed" / "heart_clean.csv"

FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]
TARGET = "num"
IMPUTE_MODE_COLS = ["ca", "thal"]
INT_COLS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "slope", "ca", "thal",
]


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Load the raw CSV, treating '?' as missing."""
    return pd.read_csv(path, na_values=["?"])


def replace_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Replace '?' placeholders with NaN and coerce everything to numeric."""
    out = df.replace("?", np.nan).copy()
    return out.apply(pd.to_numeric, errors="coerce")


def impute_mode(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """Fill NaNs in ``cols`` (default ca, thal) with each column's mode."""
    out = df.copy()
    for col in cols or IMPUTE_MODE_COLS:
        if out[col].isna().any():
            mode = out[col].mode().iloc[0]
            logger.info("Imputing %d NaN in %s with mode %s",
                        out[col].isna().sum(), col, mode)
            out[col] = out[col].fillna(mode)
    return out


def cast_types(df: pd.DataFrame) -> pd.DataFrame:
    """Cast discrete columns to int and continuous ones to float."""
    out = df.copy()
    for col in INT_COLS:
        out[col] = out[col].astype(int)
    out["oldpeak"] = out["oldpeak"].astype(float)
    return out


def binarize_target(df: pd.DataFrame) -> pd.DataFrame:
    """Convert multi-class ``num`` (0-4) into binary: any disease -> 1."""
    out = df.copy()
    out[TARGET] = (out[TARGET] > 0).astype(int)
    return out


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full cleaning pipeline on a raw DataFrame."""
    out = replace_missing(df)
    out = impute_mode(out)
    out = cast_types(out)
    out = binarize_target(out)
    before = len(out)
    out = out.drop_duplicates().reset_index(drop=True)
    logger.info("Dropped %d duplicate rows", before - len(out))
    return out


def main(raw: Path = RAW_PATH, out: Path = CLEAN_PATH) -> pd.DataFrame:
    """Load raw data, clean it, and persist the result."""
    df = clean(pd.read_csv(raw, dtype=str))
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    logger.info("Saved cleaned data %s to %s", df.shape, out)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
