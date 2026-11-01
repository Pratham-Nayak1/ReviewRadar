import pandas as pd
import pytest

from src.feature_signal_audit import (
    available_feature_families,
    project_asof_normalize,
    _validate_fold,
)


def toy_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "_row_id": ["a", "b", "c", "d"],
        "author_date": pd.date_range("2020-01-01", periods=4, tz="UTC"),
        "project": ["p", "p", "p", "p"],
        "buggy": [False, True, False, True],
        "la": [1.0, 3.0, 5.0, 100.0],
        "ld": [1.0, 1.0, 1.0, 1.0],
    })


def test_feature_family_membership_reports_unavailable_features() -> None:
    membership = available_feature_families("ApacheJIT", ["la", "ld", "ndev"])
    row = membership[membership["feature_family"] == "developer_experience_history"].iloc[0]
    assert bool(row["available"]) is True
    assert "aexp" in row["unavailable_features"]
    complexity = membership[membership["feature_family"] == "complexity_and_change_history"].iloc[0]
    assert bool(complexity["available"]) is False


def test_fold_validation_rejects_future_rows_in_training_and_overlap() -> None:
    data = toy_frame()
    with pytest.raises(ValueError, match="strictly precede"):
        _validate_fold(data.iloc[:3], data.iloc[2:], "author_date", "project", "buggy")
    overlapping = data.iloc[:3].copy()
    overlapping.loc[2, "_row_id"] = "a"
    with pytest.raises(ValueError, match="duplicate"):
        _validate_fold(overlapping, data.iloc[3:], "author_date", "project", "buggy")


def test_project_normalization_uses_history_not_future_value() -> None:
    data = toy_frame()
    fit = data.iloc[:3]
    evaluation = data.iloc[3:]
    _, normalized = project_asof_normalize(fit, evaluation, ["la", "ld"], project_col="project", timestamp_col="author_date")
    # The 100-valued future row is centered on the historical median 3, not
    # on a statistic containing the row itself or any later observation.
    assert normalized.loc[3, "la_project_delta"] == pytest.approx(97.0)


def test_normalization_is_reproducible_and_does_not_mutate_inputs() -> None:
    data = toy_frame()
    fit = data.iloc[:3].copy()
    evaluation = data.iloc[3:].copy()
    before_fit = fit.copy(deep=True)
    first = project_asof_normalize(fit, evaluation, ["la", "ld"], project_col="project", timestamp_col="author_date")
    second = project_asof_normalize(fit, evaluation, ["la", "ld"], project_col="project", timestamp_col="author_date")
    pd.testing.assert_frame_equal(first[0], second[0])
    pd.testing.assert_frame_equal(first[1], second[1])
    pd.testing.assert_frame_equal(fit, before_fit)
