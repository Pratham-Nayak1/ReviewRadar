"""Loading and basic validation for the ApacheJIT dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "apachejit_total.csv"
REQUIRED_COLUMNS = {"commit_id", "buggy", "author_date"}


class DatasetValidationError(ValueError):
    """Raised when the input file is not suitable for descriptive EDA."""


def load_apachejit(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load ApacheJIT and validate the minimum structural contract.

    The loader intentionally does not require an exact column list. If a
    future release adds columns, the EDA can inspect them rather than failing
    because of a hard-coded schema.
    """

    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"ApacheJIT CSV not found: {csv_path}")

    frame = pd.read_csv(csv_path)
    validate_dataset(frame, source=csv_path)
    return frame


def validate_dataset(frame: pd.DataFrame, source: str | Path = "dataframe") -> None:
    """Validate basic invariants without changing ``frame``."""

    if frame.empty:
        raise DatasetValidationError(f"Dataset is empty: {source}")

    missing_required = REQUIRED_COLUMNS.difference(frame.columns)
    if missing_required:
        raise DatasetValidationError(
            f"Missing required columns in {source}: {sorted(missing_required)}"
        )

    if frame["commit_id"].isna().any():
        raise DatasetValidationError("commit_id contains missing values")
    if frame["commit_id"].duplicated().any():
        duplicate_count = int(frame["commit_id"].duplicated().sum())
        raise DatasetValidationError(
            f"commit_id is not unique; duplicate rows after the first: {duplicate_count}"
        )

    if frame["buggy"].isna().any():
        raise DatasetValidationError("buggy contains missing values")
    target_values = set(frame["buggy"].unique().tolist())
    if not target_values.issubset({True, False}):
        raise DatasetValidationError(
            "buggy must contain only boolean values; "
            f"observed values: {sorted(target_values, key=str)}"
        )

    if frame["author_date"].isna().any():
        raise DatasetValidationError("author_date contains missing values")
    numeric_dates = pd.to_numeric(frame["author_date"], errors="coerce")
    if numeric_dates.isna().any():
        raise DatasetValidationError("author_date contains non-numeric values")


def dataset_profile(frame: pd.DataFrame) -> dict[str, Any]:
    """Return small, serializable descriptive summaries for reports."""

    missing = frame.isna().sum().astype(int)
    return {
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "column_names": [str(column) for column in frame.columns],
        "dtypes": {str(column): str(dtype) for column, dtype in frame.dtypes.items()},
        "missing_values": {str(column): int(value) for column, value in missing.items()},
        "duplicate_rows": int(frame.duplicated().sum()),
        "duplicate_commit_ids": int(frame["commit_id"].duplicated().sum()),
        "target_counts": {
            str(label): int(count)
            for label, count in frame["buggy"].value_counts(dropna=False).sort_index().items()
        },
    }


def author_dates_utc(frame: pd.DataFrame) -> pd.Series:
    """Convert the observed Unix-second author_date field to UTC datetimes."""

    return pd.to_datetime(frame["author_date"], unit="s", utc=True)

