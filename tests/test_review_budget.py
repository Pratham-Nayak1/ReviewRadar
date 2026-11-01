"""Unit tests for score-only review-budget calculations."""

import numpy as np
import pandas as pd
import pytest

from src.review_budget import compare_review_strategies, rank_by_score, review_budget_metrics


def test_known_top_k_precision_and_recall() -> None:
    result = review_budget_metrics(
        scores=[0.90, 0.80, 0.70, 0.60],
        labels=[True, False, True, False],
        review_fraction=0.50,
    )
    assert result["k"] == 2
    assert result["selected_positive_count"] == 1
    assert result["precision_at_k"] == pytest.approx(0.5)
    assert result["recall_at_k"] == pytest.approx(0.5)


def test_ties_are_stable_and_labels_do_not_change_ranking() -> None:
    scores = [0.8, 0.8, 0.7]
    assert np.array_equal(rank_by_score(scores), np.array([0, 1, 2]))
    first_labels = review_budget_metrics(scores, [False, True, True], 1 / 3)
    second_labels = review_budget_metrics(scores, [True, False, True], 1 / 3)
    assert first_labels["k"] == second_labels["k"] == 1
    assert first_labels["selected_positive_count"] == 0
    assert second_labels["selected_positive_count"] == 1


def test_zero_positive_labels_have_defined_zero_recall() -> None:
    result = review_budget_metrics([0.2, 0.1], [False, False], 0.5)
    assert result["precision_at_k"] == 0.0
    assert result["recall_at_k"] == 0.0
    assert result["total_positive_count"] == 0


def test_random_simulation_is_reproducible_and_heuristic_is_optional() -> None:
    labels = pd.Series([True, False, True, False, False])
    first = compare_review_strategies(
        [0.9, 0.8, 0.7, 0.6, 0.5],
        labels,
        [0.2, 0.4],
        heuristic_scores=[0.1, 0.2, 0.9, 0.8, 0.7],
        heuristic_name="simple_heuristic",
        random_runs=200,
        random_state=17,
    )
    second = compare_review_strategies(
        [0.9, 0.8, 0.7, 0.6, 0.5],
        labels,
        [0.2, 0.4],
        heuristic_scores=[0.1, 0.2, 0.9, 0.8, 0.7],
        heuristic_name="simple_heuristic",
        random_runs=200,
        random_state=17,
    )
    pd.testing.assert_frame_equal(first, second)
    assert set(first["approach"]) == {"model", "random_selection", "simple_heuristic"}
    assert first.loc[first["approach"] == "simple_heuristic", "precision_at_k"].iloc[0] == pytest.approx(1.0)


def test_invalid_lengths_and_empty_scores_are_rejected_or_defined() -> None:
    with pytest.raises(ValueError, match="same number of rows"):
        review_budget_metrics([0.5], [True, False], 0.5)
    empty = review_budget_metrics([], [], 0.5)
    assert empty["k"] == 0
    assert empty["precision_at_k"] == empty["recall_at_k"] == 0.0

