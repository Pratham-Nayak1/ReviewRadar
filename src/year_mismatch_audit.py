"""Audit ApacheJIT stored ``year`` versus UTC ``author_date`` year."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    from .data import DEFAULT_DATA_PATH
    from .label_provenance import FINAL_TEST_START, parse_author_dates
    from .split import DEFAULT_REPORT_DIR
except ImportError:
    from data import DEFAULT_DATA_PATH
    from label_provenance import FINAL_TEST_START, parse_author_dates
    from split import DEFAULT_REPORT_DIR


def run(data_path: Path = DEFAULT_DATA_PATH, report_dir: Path = DEFAULT_REPORT_DIR) -> pd.DataFrame:
    frame = pd.read_csv(data_path)
    dates = parse_author_dates(frame)
    in_scope = dates.notna() & (dates < FINAL_TEST_START)
    scoped = frame.loc[in_scope].copy()
    scoped["author_date_utc"] = dates.loc[in_scope].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    scoped["author_date_year_utc"] = dates.loc[in_scope].dt.year.astype("int64")
    scoped["year_difference"] = scoped["year"] - scoped["author_date_year_utc"]
    mismatches = scoped[scoped["year"] != scoped["author_date_year_utc"]].copy()

    sample = pd.concat([
        mismatches[mismatches["project"].eq("apache/hadoop")].sort_values(["author_date", "commit_id"]).head(50),
        mismatches[~mismatches["project"].eq("apache/hadoop")].sort_values(["author_date", "commit_id"]).head(50),
    ], ignore_index=True).drop_duplicates("commit_id")
    sample_columns = ["commit_id", "project", "buggy", "fix", "year", "author_date", "author_date_utc", "author_date_year_utc", "year_difference"]
    report_dir.mkdir(parents=True, exist_ok=True)
    sample[sample_columns].to_csv(report_dir / "year_date_mismatch_samples.csv", index=False)

    by_project = mismatches.groupby("project").agg(mismatched_rows=("commit_id", "size"), stored_year_min=("year", "min"), stored_year_max=("year", "max"), derived_year_min=("author_date_year_utc", "min"), derived_year_max=("author_date_year_utc", "max")).reset_index().sort_values("mismatched_rows", ascending=False)
    by_direction = mismatches.groupby("year_difference").size().rename("rows").reset_index()
    hadoop_years = scoped[scoped["project"].eq("apache/hadoop")].groupby(["author_date_year_utc", "year"]).size().rename("rows").reset_index()
    hadoop_mismatch_count = int(mismatches["project"].eq("apache/hadoop").sum())
    report = f"""# ApacheJIT year/date mismatch audit

Audit date: 2026-10-26. Only rows with parsed UTC `author_date` before
2018-01-01 were analyzed. The previously inspected final-test period was not
included in any output.

## Findings

The local documentation describes `year` as a calendar-year field but does not
specify whether it is derived from commit author time, committer time, a source
repository field, or another timezone. The raw CSV has no separate timezone,
committer timestamp, or provenance column for `year`.

There are **{len(mismatches):,} mismatches** in the permitted scope, including
**{hadoop_mismatch_count:,} Hadoop rows**. The mismatch is unresolved from the
local evidence; it is not automatically corrected or filtered.

`src/split.py` uses Unix-second `author_date`, parsed as UTC by
`src/data.py::author_dates_utc`, as the sole chronological authority. `year`
is not in `SAFE_NUMERIC_FEATURES`, is not used by the training pipeline, and
is not used for the temporal masks. The mismatch therefore does not alter the
earlier Hadoop label-coverage result, which was computed using `author_date`.

## Evidence tables

Mismatch counts by project:

{by_project.to_string(index=False)}

Mismatch direction (`stored year - UTC author-date year`):

{by_direction.to_string(index=False)}

Hadoop rows by parsed UTC year and stored `year`:

{hadoop_years.to_string(index=False)}

Representative raw values are in `reports/year_date_mismatch_samples.csv`.
They show timestamp parsing differences but do not identify the upstream
meaning of the stored field. No commit metadata beyond this CSV is available
locally to resolve timezone or source-field provenance.

## Integrity and decision

The mismatch is metadata inconsistency, not evidence that labels should be
changed. The existing chronological protocol remains internally consistent
because it uses `author_date` explicitly. Hadoop has zero observed positives
throughout its 2009–2017 `author_date` coverage, so correcting or ignoring
`year` would not create missing positive labels.

The stored `year` field should remain excluded from predictors and chronology
unless the dataset authors document its derivation. Any future reconciliation
requires the original construction scripts or commit metadata identifying the
source timestamp and timezone.
"""
    (report_dir / "year_date_mismatch_audit.md").write_text(report, encoding="utf-8")
    return mismatches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    args = parser.parse_args()
    print({"mismatches": len(run(args.input, args.report_dir))})


if __name__ == "__main__":
    main()
