"""Contract tests for data validation, temporal splitting, and preprocessing."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import DatasetValidationError, load_apachejit, validate_dataset
from src.pipeline import build_numeric_pipeline
from src.split import SAFE_NUMERIC_FEATURES, make_temporal_splits
from src.train import candidate_pipelines


ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "apachejit_total.csv"


def test_actual_dataset_loads_with_expected_contract() -> None:
    frame = load_apachejit(RAW_PATH)
    assert {"commit_id", "buggy", "author_date"}.issubset(frame.columns)
    assert frame["commit_id"].is_unique
    assert frame["buggy"].notna().all()
    assert frame[SAFE_NUMERIC_FEATURES].shape[1] == 12


@pytest.mark.parametrize("column", ["buggy", "author_date"])
def test_required_missing_values_are_rejected(column: str) -> None:
    frame = load_apachejit(RAW_PATH,).head(3).copy()
    if column == "buggy":
        frame["buggy"] = frame["buggy"].astype(object)
    frame.loc[frame.index[0], column] = np.nan
    with pytest.raises(DatasetValidationError, match=f"{column} contains missing values"):
        validate_dataset(frame)


def test_temporal_splits_are_disjoint_and_chronological() -> None:
    dates = pd.to_datetime(["2016-12-31", "2017-06-01", "2018-01-01"], utc=True)
    frame = pd.DataFrame(
        {
            "commit_id": ["a", "b", "c"],
            "buggy": [False, True, False],
            "author_date": (dates.view("int64") // 10**9).astype(int),
        }
    )
    splits = make_temporal_splits(frame)
    assert list(splits.train["commit_id"]) == ["a"]
    assert list(splits.validation["commit_id"]) == ["b"]
    assert list(splits.test["commit_id"]) == ["c"]
    assert set(splits.train.commit_id).isdisjoint(splits.validation.commit_id)
    assert set(splits.validation.commit_id).isdisjoint(splits.test.commit_id)


def test_pipeline_fits_imputation_and_scaling_on_fit_rows_only() -> None:
    train = pd.DataFrame({column: [1.0, 2.0, 100.0] for column in SAFE_NUMERIC_FEATURES})
    train.loc[0, "la"] = np.nan
    labels = [False, True, True]
    pipeline = build_numeric_pipeline(
        estimator=candidate_pipelines()["logistic_regression"][0].named_steps["estimator"],
        scale=True,
    )
    pipeline.fit(train, labels)
    imputer = pipeline.named_steps["preprocess"].named_transformers_["numeric"].named_steps["imputer"]
    assert imputer.statistics_[SAFE_NUMERIC_FEATURES.index("la")] == pytest.approx(51.0)
    assert list(pipeline.feature_names_in_) == SAFE_NUMERIC_FEATURES
    reordered = train[list(reversed(SAFE_NUMERIC_FEATURES))].copy()
    probabilities = pipeline.predict_proba(reordered)[:, 1]
    assert len(probabilities) == len(train)
    assert np.isfinite(probabilities).all()


def test_all_candidate_models_fit_inside_shared_pipeline() -> None:
    features = pd.DataFrame(
        {column: np.linspace(1.0, 6.0, 6) for column in SAFE_NUMERIC_FEATURES}
    )
    labels = [False, True, False, True, False, True]
    for name, (pipeline, _) in candidate_pipelines().items():
        pipeline.fit(features, labels)
        probabilities = pipeline.predict_proba(features)
        assert probabilities.shape == (6, 2), name
        assert np.isfinite(probabilities).all(), name


def test_feature_contract_excludes_target_identifiers_and_metadata() -> None:
    excluded = {"commit_id", "buggy", "fix", "year", "author_date", "project"}
    assert excluded.isdisjoint(SAFE_NUMERIC_FEATURES)

