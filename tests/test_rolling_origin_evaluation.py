import pandas as pd
import pytest

from src.pipeline import build_numeric_pipeline
from src.rolling_origin_evaluation import (
    DEFAULT_FUTURE_PATH,
    evaluate_frozen_holdout,
    evaluate_development,
    make_rolling_folds,
    metric_row,
    readiness_only,
    select_development_model,
    validate_later_dataset,
    validate_temporal_partition,
)
from sklearn.linear_model import LogisticRegression


def frame(rows: int = 6) -> pd.DataFrame:
    return pd.DataFrame({
        "commit_id": [f"c{i}" for i in range(rows)],
        "author_date": pd.date_range("2020-01-01", periods=rows, tz="UTC"),
        "project": ["p1", "p1", "p2", "p2", "p1", "p2"][:rows],
        "buggy": [False, True, False, True, False, True][:rows],
        "la": list(range(rows)),
    })


def test_temporal_ordering_and_fold_boundaries_are_strict() -> None:
    data = frame()
    folds = make_rolling_folds(data, [("f1", "2020-01-04T00:00:00Z", "2020-01-06T00:00:00Z")])
    assert len(folds) == 1
    assert folds[0].fit["author_date"].max() < folds[0].evaluation["author_date"].min()
    with pytest.raises(ValueError, match="strictly precede"):
        validate_temporal_partition(data.iloc[:4], data.iloc[3:])


def test_overlap_duplicates_labels_and_missingness_are_rejected() -> None:
    data = frame()
    duplicate_id = data.copy()
    duplicate_id.loc[3, "commit_id"] = duplicate_id.loc[0, "commit_id"]
    with pytest.raises(ValueError, match="overlap"):
        validate_temporal_partition(data.iloc[:3], duplicate_id.iloc[3:])
    duplicate_row = data.copy()
    duplicate_row.loc[1] = duplicate_row.loc[0]
    with pytest.raises(ValueError, match="unique"):
        validate_temporal_partition(duplicate_row.iloc[:3], duplicate_row.iloc[3:])
    invalid = data.copy()
    invalid["buggy"] = invalid["buggy"].astype(object)
    invalid.loc[0, "buggy"] = None
    with pytest.raises(ValueError, match="labels"):
        validate_temporal_partition(invalid.iloc[:3], data.iloc[3:])
    missing_project = data.copy()
    missing_project.loc[0, "project"] = None
    with pytest.raises(ValueError, match="project"):
        validate_temporal_partition(missing_project.iloc[:3], data.iloc[3:])


def test_pipeline_preprocessing_is_fit_on_training_rows_only() -> None:
    train = pd.DataFrame({"la": [1.0, 3.0], "ld": [2.0, 4.0]})
    labels = [False, True]
    pipeline = build_numeric_pipeline(LogisticRegression(max_iter=1000), ["la", "ld"], scale=True)
    pipeline.fit(train, labels)
    imputer = pipeline.named_steps["preprocess"].named_transformers_["numeric"].named_steps["imputer"]
    assert imputer.statistics_.tolist() == [2.0, 3.0]


def test_threshold_and_review_budgets_are_deterministic_and_label_safe() -> None:
    labels = pd.Series([True, False, True, False])
    scores = pd.Series([0.9, 0.8, 0.1, 0.1])
    row = metric_row(labels, scores, 0.5, dataset="toy", split="f", model="m")
    assert row["precision_at_5"] == 1.0
    assert row["recall_at_5"] == 0.5
    assert row["review_k_at_5"] == 1
    tied = metric_row(labels, pd.Series([0.5, 0.5, 0.1, 0.1]), 0.5, dataset="toy", split="f", model="m")
    assert tied["review_k_at_5"] == 1


def test_development_selection_is_separate_from_frozen_holdout_scoring() -> None:
    data = frame()
    folds = make_rolling_folds(data, [("dev_fold", "2020-01-04T00:00:00Z", "2020-01-06T00:00:00Z")])
    metrics = evaluate_development(folds, ["la"], "toy")
    selected, threshold = select_development_model(metrics)
    assert selected in {"dummy_prior", "logistic_regression", "random_forest", "hist_gradient_boosting"}
    result = evaluate_frozen_holdout(data.iloc[:5], data.iloc[5:], ["la"], dataset="toy", model_name="dummy_prior", threshold=threshold)
    assert set(result["split"]) == {"frozen_holdout"}
    assert set(result["project"]) == {"all_projects", "p2"}


def test_insufficient_positive_evidence_is_flagged() -> None:
    row = metric_row(pd.Series([True, False, False]), pd.Series([0.9, 0.2, 0.1]), 0.5, dataset="toy", split="f", model="m")
    assert row["insufficient_positive_evidence"] is True
    assert row["insufficient_negative_evidence"] is True


def test_missing_later_period_stops_without_evaluation(tmp_path) -> None:
    result = validate_later_dataset(tmp_path / "missing.csv", earlier_max_timestamp="2020-01-01T00:00:00Z")
    assert result["eligible"] is False
    assert "not found" in result["reason"]
    readiness = readiness_only(DEFAULT_FUTURE_PATH)
    assert readiness["evaluation_run"] is False
