import numpy as np
import pandas as pd
import pytest

from src.project_diagnosis import classification_metric_row, wilson_interval


def test_wilson_interval_handles_empty_and_zero_successes() -> None:
    assert np.isnan(wilson_interval(0, 0)[0])
    low, high = wilson_interval(0, 10)
    assert low == pytest.approx(0.0)
    assert 0 < high < 0.5


def test_project_metrics_include_denominators_and_expected_rates() -> None:
    labels = pd.Series([False, False, True, True])
    scores = pd.Series([0.9, 0.1, 0.8, 0.2])
    row = classification_metric_row(
        labels, scores, 0.5, model="rf", period="2017", project="p",
        comparison_group="individual_project",
    )
    assert (row["true_positive"], row["false_positive"], row["true_negative"], row["false_negative"]) == (1, 1, 1, 1)
    assert row["precision_denominator"] == 2
    assert row["recall_denominator"] == 2
    assert row["false_positive_rate_denominator"] == 2
    assert row["precision"] == pytest.approx(0.5)
    assert row["recall"] == pytest.approx(0.5)
    assert row["false_positive_rate"] == pytest.approx(0.5)
    assert row["precision_at_10"] == pytest.approx(0.0)
    assert row["review_k_at_10"] == 1


def test_all_negative_group_handles_undefined_positive_metrics_safely() -> None:
    labels = pd.Series([False, False, False])
    scores = pd.Series([0.1, 0.2, 0.3])
    row = classification_metric_row(
        labels, scores, 0.5, model="rf", period="2017", project="p",
        comparison_group="individual_project",
    )
    assert row["positive_count"] == 0
    assert row["recall"] == 0.0
    assert np.isnan(row["average_precision"])
