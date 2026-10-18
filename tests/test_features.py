"""Tests for src.features."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src import features as ft
from src.config import RAW_FEATURES, TARGET


def test_engineered_columns_values(frame):
    out = ft.ClinicalFeatureEngineer().fit_transform(frame[RAW_FEATURES])
    assert set(ft.ENGINEERED) <= set(out.columns)
    row = out.iloc[0]
    assert np.isclose(row["chol_age_ratio"], frame.iloc[0]["chol"] / frame.iloc[0]["age"])
    assert np.isclose(row["heart_rate_reserve"],
                      220 - frame.iloc[0]["age"] - frame.iloc[0]["thalach"])


def test_age_group_buckets():
    df = pd.DataFrame({c: [1.0] * 4 for c in RAW_FEATURES})
    df["age"] = [35.0, 45.0, 55.0, 65.0]
    out = ft.ClinicalFeatureEngineer().fit_transform(df)
    assert out["age_group"].tolist() == [0, 1, 2, 3]


def test_transformer_does_not_mutate_input(frame):
    X = frame[RAW_FEATURES]
    before = X.copy()
    ft.ClinicalFeatureEngineer().fit_transform(X)
    pd.testing.assert_frame_equal(X, before)


def test_pipeline_output_shape_and_names(frame):
    pipe = ft.build_pipeline(LogisticRegression(max_iter=500))
    pipe.fit(frame[RAW_FEATURES], frame[TARGET])
    names = ft.transformed_feature_names(pipe)
    assert len(names) == pipe.named_steps["model"].coef_.shape[1]
    assert "heart_rate_reserve" in names and "sex" in names


def test_unknown_category_is_ignored(frame):
    pipe = ft.build_pipeline(LogisticRegression(max_iter=500))
    pipe.fit(frame[RAW_FEATURES], frame[TARGET])
    X = frame[RAW_FEATURES].head(2).copy()
    X["cp"] = 9.0
    assert pipe.predict_proba(X).shape == (2, 2)


def test_input_columns():
    assert ft.input_columns() == RAW_FEATURES
