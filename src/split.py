"""Leakage-safe temporal splitting and split-audit utilities for ReviewRadar."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

try:
    from .data import DEFAULT_DATA_PATH, author_dates_utc, load_apachejit
except ImportError:
    from data import DEFAULT_DATA_PATH, author_dates_utc, load_apachejit


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports"
DEFAULT_TRAIN_END = "2017-01-01T00:00:00Z"
DEFAULT_VALIDATION_END = "2018-01-01T00:00:00Z"
SAFE_NUMERIC_FEATURES = [
    "la", "ld", "nf", "nd", "ns", "ent", "ndev", "age", "nuc", "aexp", "arexp", "asexp"
]


@dataclass(frozen=True)
class TemporalSplits:
    """Three non-overlapping chronological dataframes."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    train_end: pd.Timestamp
    validation_end: pd.Timestamp


def make_temporal_splits(
    frame: pd.DataFrame,
    train_end: str | pd.Timestamp = DEFAULT_TRAIN_END,
    validation_end: str | pd.Timestamp = DEFAULT_VALIDATION_END,
) -> TemporalSplits:
    """Split by ``author_date`` using half-open UTC calendar intervals.

    The default intervals are:

    - train: dates before 2017-01-01 UTC
    - validation: 2017-01-01 through 2017-12-31 UTC
    - test: 2018-01-01 UTC onward

    No shuffling or random fallback is used. The input dataframe is not
    modified, and no target-derived or test-derived information is used.
    """

    dates = author_dates_utc(frame)
    train_end_ts = pd.Timestamp(train_end)
    validation_end_ts = pd.Timestamp(validation_end)
    if train_end_ts.tzinfo is None:
        train_end_ts = train_end_ts.tz_localize("UTC")
    else:
        train_end_ts = train_end_ts.tz_convert("UTC")
    if validation_end_ts.tzinfo is None:
        validation_end_ts = validation_end_ts.tz_localize("UTC")
    else:
        validation_end_ts = validation_end_ts.tz_convert("UTC")
    if train_end_ts >= validation_end_ts:
        raise ValueError("train_end must be earlier than validation_end")

    dated = frame.assign(_reviewradar_author_date=dates)
    masks = {
        "train": dated["_reviewradar_author_date"] < train_end_ts,
        "validation": (dated["_reviewradar_author_date"] >= train_end_ts)
        & (dated["_reviewradar_author_date"] < validation_end_ts),
        "test": dated["_reviewradar_author_date"] >= validation_end_ts,
    }
    assigned = sum(mask.astype(int) for mask in masks.values())
    if not bool((assigned == 1).all()):
        raise ValueError("Temporal split did not assign every row exactly once")

    def select(mask: pd.Series) -> pd.DataFrame:
        return (
            dated.loc[mask]
            .drop(columns=["_reviewradar_author_date"])
            .sort_values(["author_date", "commit_id"])
            .reset_index(drop=True)
        )

    return TemporalSplits(
        train=select(masks["train"]),
        validation=select(masks["validation"]),
        test=select(masks["test"]),
        train_end=train_end_ts,
        validation_end=validation_end_ts,
    )


def split_summary(splits: TemporalSplits) -> pd.DataFrame:
    """Return counts, date ranges, and class distributions for each split."""

    rows = []
    for name, part in (("train", splits.train), ("validation", splits.validation), ("test", splits.test)):
        dates = author_dates_utc(part)
        counts = part["buggy"].value_counts().to_dict()
        rows.append(
            {
                "split": name,
                "rows": len(part),
                "date_min_utc": dates.min().isoformat(),
                "date_max_utc": dates.max().isoformat(),
                "buggy_false": int(counts.get(False, 0)),
                "buggy_true": int(counts.get(True, 0)),
                "buggy_rate": float(part["buggy"].mean()),
                "projects": int(part["project"].nunique()) if "project" in part else None,
            }
        )
    return pd.DataFrame(rows)


def cross_split_audit(parts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Audit IDs and repeated safe-feature signatures across split artifacts."""

    feature_columns = ["project", *SAFE_NUMERIC_FEATURES]
    rows = []
    names = list(parts)
    for left_name, right_name in ((names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))):
        left = parts[left_name]
        right = parts[right_name]
        shared_ids = len(set(left["commit_id"]) & set(right["commit_id"]))
        available = [column for column in feature_columns if column in left.columns and column in right.columns]
        left_signatures = left[available].drop_duplicates()
        right_signatures = right[available].drop_duplicates()
        shared_signatures = len(left_signatures.merge(right_signatures, on=available, how="inner"))
        shared_projects = len(set(left["project"]) & set(right["project"])) if "project" in left else 0
        rows.append(
            {
                "left": left_name,
                "right": right_name,
                "shared_commit_ids": shared_ids,
                "shared_safe_feature_signatures": shared_signatures,
                "shared_projects": shared_projects,
            }
        )
    return pd.DataFrame(rows)


def official_split_summary(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize the authors' supplied split files when present."""

    filenames = ["apachejit_train.csv", "apachejit_test_large.csv", "apachejit_test_small.csv"]
    frames = {name.removesuffix(".csv"): load_apachejit(data_dir / name) for name in filenames}
    rows = []
    for name, frame in frames.items():
        dates = author_dates_utc(frame)
        counts = frame["buggy"].value_counts().to_dict()
        rows.append(
            {
                "official_split": name,
                "rows": len(frame),
                "date_min_utc": dates.min().isoformat(),
                "date_max_utc": dates.max().isoformat(),
                "year_min": int(frame["year"].min()),
                "year_max": int(frame["year"].max()),
                "buggy_false": int(counts.get(False, 0)),
                "buggy_true": int(counts.get(True, 0)),
                "buggy_rate": float(frame["buggy"].mean()),
                "date_year_mismatches": int((dates.dt.year != frame["year"]).sum()),
            }
        )
    audit = cross_split_audit(frames)
    train_cutoff = author_dates_utc(frames["apachejit_train"]).max()
    test_dates = author_dates_utc(frames["apachejit_test_large"])
    audit.attrs["train_author_date_cutoff"] = train_cutoff.isoformat()
    audit.attrs["test_large_on_or_before_train_cutoff"] = int((test_dates <= train_cutoff).sum())
    return pd.DataFrame(rows), audit


def build_protocol_report(
    total: pd.DataFrame,
    splits: TemporalSplits,
    official: pd.DataFrame | None = None,
    official_audit: pd.DataFrame | None = None,
) -> str:
    summary = split_summary(splits)
    lines = [
        "# ReviewRadar leakage-safe evaluation protocol",
        "",
        "## Decision",
        "",
        "Use an explicit `author_date` chronological split from `apachejit_total.csv`: train before 2017-01-01 UTC, validation during 2017, and final test from 2018-01-01 UTC onward. No random split is used.",
        "",
        "The authors' supplied train/test artifacts are retained for reference. They are disjoint by commit ID, and the official `year` values place the large test set in 2017–2019. However, the large test file contains rows whose actual `author_date` is on or before the training cutoff, so it is not a strictly later timestamp holdout. ReviewRadar therefore uses the explicit timestamp split below as its primary protocol.",
        "",
        "## ReviewRadar temporal split",
        "",
        "| Split | Rows | Date range UTC | Buggy=False | Buggy=True | Buggy rate | Projects |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.split} | {row.rows:,} | {row.date_min_utc} to {row.date_max_utc} | {row.buggy_false:,} | {row.buggy_true:,} | {row.buggy_rate:.2%} | {row.projects} |"
        )
    lines += [
        "",
        "The split is created by `src/split.py::make_temporal_splits`; it sorts by timestamp and commit ID only for deterministic output. The raw CSV is not modified and no split files are required for model training.",
        "",
        "## Official artifacts inspected",
        "",
    ]
    if official is not None:
        lines += [
            "| Artifact | Rows | author_date range UTC | `year` range | Buggy=False | Buggy=True | Buggy rate | Date/year mismatches |",
            "|---|---:|---|---|---:|---:|---:|---:|",
        ]
        for row in official.itertuples(index=False):
            lines.append(
                f"| {row.official_split} | {row.rows:,} | {row.date_min_utc} to {row.date_max_utc} | {row.year_min}–{row.year_max} | {row.buggy_false:,} | {row.buggy_true:,} | {row.buggy_rate:.2%} | {row.date_year_mismatches:,} |"
            )
        lines += ["", "Official artifact overlap audit:", "", "| Pair | Shared commit IDs | Shared safe-feature signatures | Shared projects |", "|---|---:|---:|---:|"]
        for row in official_audit.itertuples(index=False):
            lines.append(f"| {row.left} / {row.right} | {row.shared_commit_ids} | {row.shared_safe_feature_signatures} | {row.shared_projects} |")
        lines += [
            "",
            f"Official training `author_date` cutoff: `{official_audit.attrs.get('train_author_date_cutoff')}`.",
            f"Official large-test rows on or before that timestamp: **{official_audit.attrs.get('test_large_on_or_before_train_cutoff')}**.",
            "The official small test is a subset of the large test, so it must not be treated as an independent final test.",
            "",
        ]
    lines += [
        "## Leakage controls for modeling",
        "",
        "1. Keep `commit_id` out of features.",
        "2. Exclude `buggy` from features because it is the target.",
        "3. Exclude `fix` until its provenance proves it is available at commit time and not post-hoc.",
        "4. Exclude `year` and raw `author_date` from the default feature matrix; use `author_date` for ordering only.",
        "5. Start with the documented numeric change metrics (`la`, `ld`, `nf`, `nd`, `ns`, `ent`, `ndev`, `age`, `nuc`, `aexp`, `arexp`, `asexp`).",
        "6. Fit imputation, scaling, encoding, feature selection, and any resampling inside a training-only pipeline. Tune models and thresholds on train/validation only.",
        "7. Fit the selected pipeline once on train plus validation only after model and threshold decisions are frozen; evaluate the final test exactly once.",
        "8. Do not use the final-test metrics, distribution, or labels to revise the model, features, or threshold.",
        "",
        "## Related-record audit",
        "",
        "The total dataset has unique commit IDs and no exact duplicate rows. Official train and large-test artifacts share no commit IDs. They do share project names, which is expected because projects contribute changes across time; this is not row duplication but means cross-project generalization should be reported separately if pursued. Repeated safe-feature signatures are counted in the audit table and are not automatically treated as duplicates because different commits can legitimately have identical metric values.",
        "",
        "## Reproducibility",
        "",
        "Run `python -m src.split` from the project root. It writes `reports/split_summary.csv` and this report. The split function is reusable from Python and never silently falls back to a random split.",
        "",
    ]
    return "\n".join(lines)


def run_protocol(data_path: Path = DEFAULT_DATA_PATH, report_dir: Path = DEFAULT_REPORT_DIR) -> None:
    total = load_apachejit(data_path)
    splits = make_temporal_splits(total)
    report_dir.mkdir(parents=True, exist_ok=True)
    summary = split_summary(splits)
    summary.to_csv(report_dir / "split_summary.csv", index=False)
    data_dir = data_path.parent
    official = official_audit = None
    if all((data_dir / name).is_file() for name in ("apachejit_train.csv", "apachejit_test_large.csv", "apachejit_test_small.csv")):
        official, official_audit = official_split_summary(data_dir)
        official.to_csv(report_dir / "official_split_summary.csv", index=False)
        official_audit.to_csv(report_dir / "official_split_audit.csv", index=False)
    report = build_protocol_report(total, splits, official, official_audit)
    (report_dir / "evaluation_protocol.md").write_text(report, encoding="utf-8")
    print(report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    args = parser.parse_args()
    run_protocol(args.input, args.report_dir)


if __name__ == "__main__":
    main()

