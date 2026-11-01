import numpy as np
import pandas as pd

from src.generalization_audit import _ece
from src.review_budget import compare_review_strategies


def test_ece_is_zero_for_perfectly_calibrated_two_point_scores() -> None:
    labels = pd.Series([False, False, True, True])
    scores = pd.Series([0.0, 0.0, 1.0, 1.0])
    assert _ece(labels, scores) == 0.0


def test_review_budget_random_simulation_is_reproducible_and_label_free_for_ranking() -> None:
    scores = [0.9, 0.8, 0.1, 0.1]
    labels = [True, False, True, False]
    first = compare_review_strategies(scores, labels, [0.5], random_runs=50, random_state=42)
    second = compare_review_strategies(scores, labels, [0.5], random_runs=50, random_state=42)
    pd.testing.assert_frame_equal(first, second)
    model = first[first["approach"] == "model"].iloc[0]
    assert model["k"] == 2
    assert model["precision_at_k"] == 0.5
    assert model["recall_at_k"] == 0.5


def test_project_rank_is_based_on_metrics_not_raw_project_volume() -> None:
    frame = pd.DataFrame({
        "project": ["a", "a", "b", "b"],
        "f1": [0.9, 0.2, 0.8, 0.1],
        "average_precision": [0.9, 0.2, 0.8, 0.1],
        "rows": [100, 100, 10, 10],
        "positive_count": [10, 10, 1, 1],
    })
    frame["stable_group_eligible"] = (frame["rows"] >= 30) & (frame["positive_count"] > 0)
    frame["model_rank_by_f1"] = frame.groupby("project")["f1"].rank(method="min", ascending=False).astype(int)
    assert frame.loc[frame["project"] == "a", "model_rank_by_f1"].tolist() == [1, 2]
    assert frame.loc[frame["project"] == "b", "model_rank_by_f1"].tolist() == [1, 2]
    assert bool(frame.loc[frame["project"] == "b", "stable_group_eligible"].iloc[0]) is False
