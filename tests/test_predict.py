"""Smoke tests for inference using actual ApacheJIT schema-compatible rows."""

from pathlib import Path

import pandas as pd
import pytest

from src.predict import InferenceInputError, ReviewRadarPredictor


ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "apachejit_total.csv"


@pytest.fixture(scope="module")
def sample_frame() -> pd.DataFrame:
    return pd.read_csv(RAW_PATH, nrows=2).drop(columns=["buggy", "fix", "year", "author_date", "project"])


def test_one_row_prediction_returns_score_and_identifier(sample_frame: pd.DataFrame) -> None:
    predictor = ReviewRadarPredictor()
    result = predictor.predict_one(sample_frame.iloc[0])
    assert result["commit_id"] == sample_frame.iloc[0]["commit_id"]
    assert 0.0 <= result["risk_score"] <= 1.0
    assert isinstance(result["predicted_buggy"], bool)


def test_batch_prediction_preserves_identifiers(sample_frame: pd.DataFrame) -> None:
    predictor = ReviewRadarPredictor()
    result = predictor.predict(sample_frame)
    assert list(result["commit_id"]) == list(sample_frame["commit_id"])
    assert len(result) == 2
    assert result["risk_score"].between(0.0, 1.0).all()


def test_missing_and_unexpected_columns_are_rejected(sample_frame: pd.DataFrame) -> None:
    predictor = ReviewRadarPredictor()
    with pytest.raises(InferenceInputError, match="Missing required feature columns"):
        predictor.predict(sample_frame.drop(columns=["la"]))
    with pytest.raises(InferenceInputError, match="Unexpected input columns"):
        predictor.predict(sample_frame.assign(unexpected=1))

