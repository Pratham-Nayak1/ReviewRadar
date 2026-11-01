import pandas as pd
import pytest

from src.evaluate import choose_threshold_by_f1
from src.hgb_robustness import HGB_GRID, RANDOM_STATE, select_hgb_config, temporal_folds
from src.review_budget import rank_by_score, review_budget_metrics
from src.split import SAFE_NUMERIC_FEATURES


def make_year_frame() -> pd.DataFrame:
    rows = []
    for year in (2013, 2014, 2015, 2016):
        for index in range(3):
            row = {
                "commit_id": f"{year}-{index}",
                "buggy": bool((year + index) % 2),
                "author_date": int(pd.Timestamp(f"{year}-06-01", tz="UTC").timestamp()) + index,
                "project": f"project-{index}",
            }
            row.update({feature: float(index + 1) for feature in SAFE_NUMERIC_FEATURES})
            rows.append(row)
    return pd.DataFrame(rows)


def test_temporal_folds_are_expanding_and_disjoint() -> None:
    folds = temporal_folds(make_year_frame())
    assert [name for name, _, _ in folds] == ["2014", "2015", "2016"]
    for _, fit, validation in folds:
        assert set(fit.commit_id).isdisjoint(validation.commit_id)
        assert pd.to_datetime(fit.author_date, unit="s", utc=True).max() < pd.to_datetime(validation.author_date, unit="s", utc=True).min()
        assert len(fit) > 0 and len(validation) > 0


def test_hgb_grid_and_selection_are_reproducible() -> None:
    assert RANDOM_STATE == 42
    assert set(HGB_GRID) == {"hgb_existing", "hgb_shallow", "hgb_regularized", "hgb_slow_learning"}
    rows = []
    for config_name, mean_f1, mean_ap, threshold_std in (
        ("hgb_existing", 0.60, 0.63, 0.02),
        ("hgb_shallow", 0.61, 0.62, 0.01),
        ("hgb_regularized", 0.61, 0.64, 0.03),
        ("hgb_slow_learning", 0.61, 0.64, 0.01),
    ):
        for index, fold in enumerate(("2014", "2015", "2016")):
            threshold = 0.4 + threshold_std if config_name != "hgb_regularized" else 0.4 + (index * 0.03)
            rows.append({"status": "ok", "configuration_name": config_name, "fold": fold, "f1": mean_f1, "average_precision": mean_ap, "threshold": threshold})
    selected, aggregate = select_hgb_config(pd.DataFrame(rows))
    selected_again, aggregate_again = select_hgb_config(pd.DataFrame(rows))
    assert selected == "hgb_slow_learning"
    assert selected_again == selected
    pd.testing.assert_frame_equal(aggregate, aggregate_again)


def test_threshold_selection_returns_a_real_score_threshold() -> None:
    labels = pd.Series([False, True, True, False])
    scores = pd.Series([0.1, 0.4, 0.8, 0.9])
    threshold, best_f1 = choose_threshold_by_f1(labels, scores)
    assert threshold in set(scores)
    assert best_f1 == pytest.approx(0.8)


def test_review_budget_rounding_and_stable_ties() -> None:
    scores = [0.8, 0.8, 0.7, 0.1]
    assert list(rank_by_score(scores)) == [0, 1, 2, 3]
    result = review_budget_metrics(scores, [True, False, True, False], 0.5)
    assert result["k"] == 2
    assert result["precision_at_k"] == pytest.approx(0.5)
    assert result["recall_at_k"] == pytest.approx(0.5)
