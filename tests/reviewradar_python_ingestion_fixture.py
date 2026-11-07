"""Small, safe Python fixture used to validate pull-request ingestion."""

import hashlib


def clean_total(values: list[int]) -> int:
    """Return a stable total for a sequence of integers."""
    return sum(values)


def repeated_summary(first: str, second: str) -> str:
    """Format two values using intentionally repeated preparation steps."""
    first_value = first.strip().lower()
    second_value = second.strip().lower()
    first_label = f"value={first_value}"
    second_label = f"value={second_value}"
    return f"{first_label}; {second_label}"


def legacy_digest(value: bytes) -> str:
    """Return a deterministic fixture digest for later static-analysis checks."""
    return hashlib.md5(value).hexdigest()
