import pandas as pd

from src.label_provenance import development_scope, filter_audit, parse_author_dates, project_label_coverage
from src.split import SAFE_NUMERIC_FEATURES


def make_frame() -> pd.DataFrame:
    rows = []
    for project, year, buggy in (
        ("apache/hadoop", 2016, True),
        ("apache/hadoop", 2017, False),
        ("apache/other", 2017, True),
        ("apache/other", 2018, True),
    ):
        row = {
            "commit_id": f"{project}-{year}-{buggy}", "project": project,
            "buggy": buggy, "year": year, "author_date": int(pd.Timestamp(f"{year}-06-01", tz="UTC").timestamp()),
        }
        row.update({feature: 1.0 for feature in SAFE_NUMERIC_FEATURES})
        rows.append(row)
    return pd.DataFrame(rows)


def test_parse_and_scope_exclude_final_period() -> None:
    frame = make_frame()
    dates = parse_author_dates(frame)
    scoped = development_scope(frame)
    assert dates.dt.year.tolist() == [2016, 2017, 2017, 2018]
    assert len(scoped) == 3
    assert scoped["commit_id"].str.contains("2018").sum() == 0


def test_coverage_counts_labels_by_training_and_validation_period() -> None:
    coverage = project_label_coverage(development_scope(make_frame()))
    hadoop_train = coverage.query("period == 'training_pre_2017' and project == 'apache/hadoop'").iloc[0]
    hadoop_validation = coverage.query("period == 'validation_2017' and project == 'apache/hadoop'").iloc[0]
    assert (hadoop_train["rows"], hadoop_train["buggy_true"], hadoop_train["buggy_false"]) == (1, 1, 0)
    assert (hadoop_validation["rows"], hadoop_validation["buggy_true"], hadoop_validation["buggy_false"]) == (1, 0, 1)


def test_integrity_audit_does_not_silently_deduplicate() -> None:
    integrity = filter_audit(development_scope(make_frame()))
    assert integrity.query("rule == 'unique_commit_id'").iloc[0]["excluded_rows"] == 0
    assert integrity.query("rule == 'exact_duplicate_rows'").iloc[0]["excluded_rows"] == 0
    assert integrity.query("rule == 'final_audit_scope'").iloc[0]["rows_before"] == 3
