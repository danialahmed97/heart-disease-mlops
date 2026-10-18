"""Feature engineering and preprocessing for the heart-disease model.

Engineered features (``ClinicalFeatureEngineer``) and their rationale:

* ``age_group`` - ordinal bucket (<40, 40-49, 50-59, 60+). Cardiac risk grows
  non-linearly with age decades, which a linear model cannot capture from raw
  age alone.
* ``chol_age_ratio`` - cholesterol / age. Normal-looking cholesterol is more
  alarming in a young patient than an elderly one; the ratio normalises it.
* ``heart_rate_reserve`` - (220 - age) - thalach. The gap between the
  age-predicted maximum heart rate and the achieved rate; a large reserve
  means the patient could not reach expected exertion (chronotropic
  incompetence), a known ischaemia marker.
"""
import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import RAW_FEATURES

logger = logging.getLogger(__name__)

ENGINEERED = ["age_group", "chol_age_ratio", "heart_rate_reserve"]
NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"] + ENGINEERED
ONEHOT = ["cp", "restecg", "slope", "thal", "ca"]
PASSTHROUGH = ["sex", "fbs", "exang"]
AGE_BINS = [-np.inf, 40, 50, 60, np.inf]


class ClinicalFeatureEngineer(BaseEstimator, TransformerMixin):
    """Add domain-driven features to the raw clinical DataFrame."""

    def fit(self, X: pd.DataFrame, y=None) -> "ClinicalFeatureEngineer":
        """Stateless transformer; records input columns."""
        self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of ``X`` with the engineered columns appended."""
        out = pd.DataFrame(X).copy()
        out["age_group"] = np.digitize(out["age"], AGE_BINS[1:-1], right=False)
        out["chol_age_ratio"] = out["chol"] / out["age"]
        out["heart_rate_reserve"] = (220 - out["age"]) - out["thalach"]
        return out


def build_preprocessor() -> ColumnTransformer:
    """Scale numerics, one-hot categoricals, pass binary flags through."""
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ONEHOT),
            ("bin", "passthrough", PASSTHROUGH),
        ]
    )


def build_pipeline(estimator) -> Pipeline:
    """Full pipeline: feature engineering -> preprocessing -> estimator."""
    return Pipeline(
        [
            ("features", ClinicalFeatureEngineer()),
            ("preprocess", build_preprocessor()),
            ("model", estimator),
        ]
    )


def transformed_feature_names(pipeline: Pipeline) -> list[str]:
    """Names of the columns reaching the final estimator."""
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    return [n.split("__", 1)[1] for n in names]


def input_columns() -> list[str]:
    """Raw input schema expected by the pipeline."""
    return list(RAW_FEATURES)
