"""Inference failure and schema-contract tests."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.predict import InferenceInputError, ReviewRadarPredictor


ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "apachejit_total.csv"


@pytest.fixture(scope="module")
def predictor() -> ReviewRadarPredictor:
    return ReviewRadarPredictor()


@pytest.fixture(scope="module")
def inputs(predictor: ReviewRadarPredictor) -> pd.DataFrame:
    raw = pd.read_csv(RAW_PATH, nrows=3)
    return raw[["commit_id", *predictor.feature_columns]].copy()


def test_reordered_schema_and_missing_numeric_value_are_supported(predictor: ReviewRadarPredictor, inputs: pd.DataFrame) -> None:
    reordered = inputs[["commit_id", *reversed(predictor.feature_columns)]].copy()
    reordered.loc[0, "la"] = np.nan
    result = predictor.predict(reordered)
    assert len(result) == 3
    assert result["commit_id"].tolist() == inputs["commit_id"].tolist()
    assert result["risk_score"].between(0, 1).all()


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda frame: frame.drop(columns=["la"]), "Missing required feature columns"),
        (lambda frame: frame.assign(extra_column=1), "Unexpected input columns"),
        (lambda frame: frame.assign(la="not-a-number"), "must contain numeric values"),
        (lambda frame: frame.assign(la=np.inf), "non-finite values"),
    ],
)
def test_invalid_input_columns_and_values_have_actionable_errors(
    predictor: ReviewRadarPredictor,
    inputs: pd.DataFrame,
    mutator,
    message: str,
) -> None:
    with pytest.raises(InferenceInputError, match=message):
        predictor.predict(mutator(inputs.copy()))


def test_empty_and_duplicate_columns_are_rejected(predictor: ReviewRadarPredictor, inputs: pd.DataFrame) -> None:
    with pytest.raises(InferenceInputError, match="input is empty"):
        predictor.predict(inputs.iloc[0:0])
    duplicate = inputs.copy()
    duplicate.insert(len(duplicate.columns), "la", duplicate["la"], allow_duplicates=True)
    with pytest.raises(InferenceInputError, match="duplicate columns"):
        predictor.predict(duplicate)


def test_missing_model_artifact_has_helpful_message(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Saved ReviewRadar model not found"):
        ReviewRadarPredictor(tmp_path / "missing.joblib", ROOT / "reports" / "evaluation_config.json")


def test_missing_configuration_artifact_has_helpful_message(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="evaluation configuration not found"):
        ReviewRadarPredictor(ROOT / "models" / "reviewradar_selected.joblib", tmp_path / "missing.json")

