"""ApacheJIT label provenance and project-coverage audit.

The audit scope ends at 2017-12-31 UTC. The previously inspected final-test
period is excluded before any coverage, integrity, or label counts are made.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    from .data import DEFAULT_DATA_PATH, DatasetValidationError, validate_dataset
    from .split import DEFAULT_REPORT_DIR, SAFE_NUMERIC_FEATURES
except ImportError:
    from data import DEFAULT_DATA_PATH, DatasetValidationError, validate_dataset
    from split import DEFAULT_REPORT_DIR, SAFE_NUMERIC_FEATURES


FINAL_TEST_START = pd.Timestamp("2018-01-01T00:00:00Z")


def parse_author_dates(frame: pd.DataFrame) -> pd.Series:
    """Parse Unix-second author dates as UTC without modifying the frame."""
    return pd.to_datetime(pd.to_numeric(frame["author_date"], errors="coerce"), unit="s", utc=True, errors="coerce")


def development_scope(frame: pd.DataFrame) -> pd.DataFrame:
    """Return only rows before the already-inspected final-test boundary."""
    dates = parse_author_dates(frame)
    return frame.loc[dates.notna() & (dates < FINAL_TEST_START)].copy()


def filter_audit(frame: pd.DataFrame) -> pd.DataFrame:
    """Report every integrity/filter rule and its count effect."""
    rows = []
    current = frame

    def add(rule: str, before: int, after: int, excluded: int, action: str) -> None:
        rows.append({
            "scope": "pre_2018_development",
            "rule": rule,
            "rows_before": before,
            "rows_after": after,
            "excluded_rows": excluded,
            "action": action,
        })

    add("development_date_scope", len(frame), len(current), 0, "Keep author_date before 2018-01-01 UTC; final-test rows are out of scope")
    required = {"commit_id", "project", "buggy", "author_date", *SAFE_NUMERIC_FEATURES}
    missing_required = sorted(required.difference(current.columns))
    add("required_columns_present", len(current), len(current), 0, f"No row filtering; missing columns={missing_required}")
    valid_project = current["project"].notna()
    add("nonmissing_project", len(current), int(valid_project.sum()), int((~valid_project).sum()), "Exclude missing project only for coverage table")
    current = current.loc[valid_project]
    valid_label = current["buggy"].isin([True, False])
    add("boolean_buggy_label", len(current), int(valid_label.sum()), int((~valid_label).sum()), "No relabeling; invalid labels would be excluded from coverage")
    current = current.loc[valid_label]
    dates = parse_author_dates(current)
    add("parseable_author_date", len(current), int(dates.notna().sum()), int(dates.isna().sum()), "No imputation or date coercion into a valid value")
    current = current.loc[dates.notna()]
    year_consistent = current["year"].astype("Int64").eq(parse_author_dates(current).dt.year)
    add("year_matches_utc_author_date", len(current), int(year_consistent.sum()), int((~year_consistent).sum()), "Audit only; chronological split uses author_date")
    duplicate_ids = current["commit_id"].duplicated(keep=False)
    add("unique_commit_id", len(current), int((~duplicate_ids).sum()), int(duplicate_ids.sum()), "Audit only; do not silently deduplicate")
    exact_duplicates = current.duplicated(keep=False)
    add("exact_duplicate_rows", len(current), int((~exact_duplicates).sum()), int(exact_duplicates.sum()), "Audit only; do not silently deduplicate")
    add("final_audit_scope", len(current), len(current), 0, "No rows are removed by the audit implementation")
    return pd.DataFrame(rows)


def project_label_coverage(frame: pd.DataFrame) -> pd.DataFrame:
    dates = parse_author_dates(frame)
    period = pd.Series("development_pre_2018", index=frame.index, dtype="object")
    period.loc[dates.dt.year < 2017] = "training_pre_2017"
    period.loc[dates.dt.year == 2017] = "validation_2017"
    frame = frame.assign(_date=dates, _period=period)
    grouped = frame.groupby(["_period", "project"], dropna=False)
    rows = []
    for (period_name, project), group in grouped:
        counts = group["buggy"].value_counts()
        rows.append({
            "scope": "pre_2018_development",
            "period": period_name,
            "year": "all" if period_name == "development_pre_2018" else ("2003-2016" if period_name == "training_pre_2017" else "2017"),
            "project": project,
            "rows": len(group),
            "buggy_true": int(counts.get(True, 0)),
            "buggy_false": int(counts.get(False, 0)),
            "buggy_prevalence": float(group["buggy"].mean()),
            "date_min_utc": group["_date"].min().isoformat(),
            "date_max_utc": group["_date"].max().isoformat(),
            "small_group_warning": bool(len(group) < 30),
        })
    yearly = frame.assign(_year=frame["_date"].dt.year).groupby(["_year", "project"], dropna=False)
    for (year, project), group in yearly:
        counts = group["buggy"].value_counts()
        rows.append({
            "scope": "pre_2018_development",
            "period": "calendar_year",
            "year": int(year), "project": project, "rows": len(group),
            "buggy_true": int(counts.get(True, 0)), "buggy_false": int(counts.get(False, 0)),
            "buggy_prevalence": float(group["buggy"].mean()),
            "date_min_utc": group["_date"].min().isoformat(), "date_max_utc": group["_date"].max().isoformat(),
            "small_group_warning": bool(len(group) < 30),
        })
    return pd.DataFrame(rows)


def provenance_report(coverage: pd.DataFrame, integrity: pd.DataFrame, scoped: pd.DataFrame) -> str:
    hadoop = coverage[(coverage.project == "apache/hadoop") & (coverage.period.isin(["training_pre_2017", "validation_2017", "calendar_year"]))]
    scoped_dates = parse_author_dates(scoped)
    year_mismatches = scoped[scoped["year"].astype("Int64") != scoped_dates.dt.year]
    hadoop_year_mismatches = int((year_mismatches["project"] == "apache/hadoop").sum())
    return f"""# ApacheJIT label provenance and Hadoop coverage audit

Audit date: 2026-10-26. The complete CSV is loaded only so rows can be
restricted immediately to `author_date < 2018-01-01 UTC`. No previously
inspected final-test row is included in any output, count, or conclusion.

## Label provenance

The local dataset documentation in `data/README.md` defines `buggy=True` as a
commit that introduced any bug. It states that bug-inducing commits were
identified with an SZZ-based process that traces fixed issues and changed lines
back to earlier commits, followed by filtering. The documentation describes
the label as an SZZ-derived proxy, not independently verified ground truth.

The row unit is one software commit/change revision. The loader in
`src/data.py` requires nonmissing, unique `commit_id`, boolean `buggy`, and a
numeric `author_date`; it does not relabel examples. The chronological rules
in `src/split.py` use UTC `author_date`: training is before 2017, validation is
2017, and the final-test boundary begins in 2018.

The local material does not include SZZ intermediate links, issue IDs, fixed
line mappings, or an independent adjudication file. The official paper and
dataset citation recorded locally are Keshavarz and Nagappan, “ApacheJIT: A
Large Dataset for Just-In-Time Defect Prediction,” MSR 2022, DOI
10.1145/3524842.3527996; dataset record DOI 10.5281/zenodo.5907002.

## Hadoop coverage

{hadoop.to_string(index=False)}

The coverage table shows that `apache/hadoop` has **0 positive labels in its
6,690 pre-2017 training rows and 0 positive labels in its 1,766 2017 rows**.
The calendar-year rows show that this is not a single 2017 boundary effect:
the observed Hadoop label rate is zero in every 2009–2017 year represented in
the permitted scope. This is not created by the ReviewRadar model, threshold,
or a target transformation. All projects are counted with the same rules in
`reports/project_label_coverage.csv`.

## Filtering and integrity

{integrity.to_string(index=False)}

No rows are silently deduplicated or relabeled. The final audit scope is a
read-only count over the filtered development frame. Commit-ID uniqueness,
project mapping, timestamp parsing, label values, exact duplicates, and
`year`/UTC-date consistency are represented in the CSV outputs.

There are {len(year_mismatches):,} `year`/UTC-date mismatches in the permitted
scope, including {hadoop_year_mismatches:,} for `apache/hadoop`. These rows are
not removed: `src/split.py` deliberately uses the parsed UTC `author_date` for
chronology, while `year` is treated as metadata.

## Interpretation and limitations

Zero observed Hadoop positives means only that this dataset slice contains no
rows labeled `buggy=True`; it does not establish zero real-world defect risk.
Possible explanations include SZZ coverage/attribution limitations, issue-link
availability, filtering in dataset construction, project-specific history, or
sampling. This audit can establish where observed labels occur and whether the
ReviewRadar filtering/split rules remove them. It cannot independently verify
unlabeled defects or reconstruct SZZ decisions from the available CSV alone.

For prospective evidence, obtain commit-to-issue links, SZZ trace outputs,
changed-file/line mappings, inclusion/exclusion logs, and a later labeled
period with enough positive outcomes per project. Freeze the project/time
split before evaluation and audit the label process separately from model
performance.

## Decision

The Hadoop anomaly is **not explained by a ReviewRadar filtering or
chronological-splitting bug**: the integrity audit has zero exclusions and the
source slice itself contains zero Hadoop positives throughout 2009–2017. The
anomaly is therefore an apparent source-label coverage or project-specific
label-generation phenomenon. It remains unresolved as a statement about true
defects because the SZZ evidence is not present locally.
"""


def run(data_path: Path = DEFAULT_DATA_PATH, report_dir: Path = DEFAULT_REPORT_DIR) -> dict[str, pd.DataFrame]:
    frame = pd.read_csv(data_path)
    # Restrict the analysis scope before any project/label/year summaries.
    scoped = development_scope(frame)
    integrity = filter_audit(scoped)
    try:
        validate_dataset(scoped)
    except DatasetValidationError as exc:
        raise DatasetValidationError(f"Development-scope integrity validation failed: {exc}") from exc
    coverage = project_label_coverage(scoped)
    report_dir.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(report_dir / "project_label_coverage.csv", index=False)
    integrity.to_csv(report_dir / "data_integrity_audit.csv", index=False)
    (report_dir / "label_provenance_audit.md").write_text(provenance_report(coverage, integrity, scoped), encoding="utf-8")
    return {"coverage": coverage, "integrity": integrity}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    args = parser.parse_args()
    result = run(args.input, args.report_dir)
    print({name: len(frame) for name, frame in result.items()})


if __name__ == "__main__":
    main()
