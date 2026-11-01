import pandas as pd

from src.distribution_shift import analysis_periods, distribution_metrics, feature_availability_report
from src.split import SAFE_NUMERIC_FEATURES


def tiny_period_frame() -> pd.DataFrame:
    rows = []
    for year in (2013, 2014, 2015, 2016, 2017):
        for index in range(2):
            row = {
                "commit_id": f"{year}-{index}",
                "project": "apache/example",
                "buggy": bool(index),
                "author_date": int(pd.Timestamp(f"{year}-06-01", tz="UTC").timestamp()) + index,
            }
            row.update({feature: float(index + year) for feature in SAFE_NUMERIC_FEATURES})
            rows.append(row)
    return pd.DataFrame(rows)


def test_analysis_periods_excludes_final_test_and_preserves_temporal_order() -> None:
    periods = analysis_periods(tiny_period_frame())
    assert set(periods) == {"train", "2014", "2015", "2016", "2017_validation"}
    assert all(len(periods[name]) == 2 for name in ("2014", "2015", "2016", "2017_validation"))
    assert periods["2017_validation"]["commit_id"].tolist() == ["2017-0", "2017-1"]


def test_distribution_metrics_is_train_referenced_and_reports_missingness() -> None:
    periods = analysis_periods(tiny_period_frame())
    periods["2017_validation"].loc[0, "age"] = None
    metrics = distribution_metrics(periods)
    row = metrics.query("period == '2017_validation' and feature == 'age'").iloc[0]
    assert row["missing_count"] == 1
    assert row["missing_rate"] == 0.5
    assert metrics.query("period == 'train' and feature == 'age'").iloc[0]["standardized_mean_shift_vs_train"] == 0


def test_feature_availability_does_not_promote_target_derived_shortcuts() -> None:
    report = feature_availability_report()
    assert "Do not implement" in report
    assert "project prior prevalence" in report.lower()
    assert set(["buggy", "project", "commit_id"]).isdisjoint(SAFE_NUMERIC_FEATURES)
