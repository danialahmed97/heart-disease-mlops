"""Tests for src.data_processing."""
import numpy as np
import pandas as pd
import pytest

from src import data_processing as dp


@pytest.fixture()
def raw() -> pd.DataFrame:
    """Raw-style frame with '?' placeholders and a duplicate row."""
    rows = [
        ["63", "1", "1", "145", "233", "1", "2", "150", "0", "2.3", "3", "0.0", "6.0", "0"],
        ["67", "1", "4", "160", "286", "0", "2", "108", "1", "1.5", "2", "3.0", "3.0", "2"],
        ["41", "0", "2", "130", "204", "0", "2", "172", "0", "1.4", "1", "?", "3.0", "0"],
        ["56", "1", "2", "120", "236", "0", "0", "178", "0", "0.8", "1", "0.0", "?", "1"],
        ["63", "1", "1", "145", "233", "1", "2", "150", "0", "2.3", "3", "0.0", "6.0", "0"],
    ]
    cols = dp.FEATURES + [dp.TARGET]
    return pd.DataFrame(rows, columns=cols)


def test_replace_missing_converts_question_marks(raw):
    out = dp.replace_missing(raw)
    assert out["ca"].isna().sum() == 1
    assert out["thal"].isna().sum() == 1
    assert all(pd.api.types.is_numeric_dtype(t) for t in out.dtypes)


def test_impute_mode_fills_nans(raw):
    out = dp.impute_mode(dp.replace_missing(raw))
    assert out[["ca", "thal"]].isna().sum().sum() == 0
    assert out.loc[2, "ca"] == 0.0


def test_binarize_target():
    df = pd.DataFrame({dp.TARGET: [0, 1, 2, 3, 4]})
    assert dp.binarize_target(df)[dp.TARGET].tolist() == [0, 1, 1, 1, 1]


def test_cast_types(raw):
    out = dp.cast_types(dp.impute_mode(dp.replace_missing(raw)))
    assert out["age"].dtype == np.int64 and out["oldpeak"].dtype == float


def test_clean_pipeline_drops_duplicates(raw):
    out = dp.clean(raw)
    assert len(out) == 4
    assert out.isna().sum().sum() == 0
    assert set(out[dp.TARGET].unique()) <= {0, 1}


def test_main_roundtrip(raw, tmp_path):
    src, dst = tmp_path / "raw.csv", tmp_path / "out" / "clean.csv"
    raw.to_csv(src, index=False)
    df = dp.main(src, dst)
    assert dst.exists() and len(pd.read_csv(dst)) == len(df)


def test_load_raw_handles_question_marks(raw, tmp_path):
    p = tmp_path / "r.csv"
    raw.to_csv(p, index=False)
    assert dp.load_raw(p)["ca"].isna().sum() == 1
