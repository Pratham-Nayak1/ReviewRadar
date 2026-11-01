import numpy as np
import pandas as pd
import pytest

from src.jit_defect4j_benchmark import FEATURES, metric_row, wilson


def test_metric_row_has_expected_confusion_matrix_and_budgets() -> None:
    labels = pd.Series([False, False, True, True])
    scores = pd.Series([0.9, 0.1, 0.8, 0.2])
    row = metric_row(labels, scores, 0.5, model="m", split="s", threshold_source="fixed")
    assert (row["true_negative"], row["false_positive"], row["false_negative"], row["true_positive"]) == (1, 1, 1, 1)
    assert row["precision"] == pytest.approx(0.5)
    assert row["recall"] == pytest.approx(0.5)
    assert row["precision_at_10"] == pytest.approx(0.0)
    assert row["review_k_at_10"] == 1


def test_wilson_and_zero_positive_group_are_safe() -> None:
    assert wilson(0, 0)[0] != wilson(0, 0)[0]
    row = metric_row(pd.Series([False, False]), pd.Series([0.2, 0.1]), 0.5, model="m", split="s", threshold_source="fixed")
    assert row["positive_count"] == 0
    assert row["recall"] == 0.0
    assert np.isnan(row["average_precision"])
    assert row["limited_group_warning"] is True


def test_feature_contract_is_numeric_expert_metrics_only() -> None:
    assert "project" not in FEATURES
    assert "commit_hash" not in FEATURES
    assert "is_buggy_commit" not in FEATURES
    assert "fix" not in FEATURES
