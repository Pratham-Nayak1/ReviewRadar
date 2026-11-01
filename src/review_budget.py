"""Review-budget analysis for score-based change prioritization.

The ranking is created from scores only. Labels are used only after the
ranking is fixed, to measure the observed benchmark outcome. Random-selection
rows are repeated simulations, not estimates of time or money saved.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd


POSITIVE_CLASS = "buggy=True"
DEFAULT_RANDOM_RUNS = 1000
DEFAULT_RANDOM_STATE = 42


def _as_numeric_scores(scores: Iterable[float] | pd.Series | np.ndarray) -> np.ndarray:
    values = pd.to_numeric(pd.Series(scores), errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("scores must contain only finite numeric values")
    return values


def _as_binary_labels(labels: Iterable[bool | int] | pd.Series | np.ndarray) -> np.ndarray:
    values = pd.Series(labels)
    if values.isna().any():
        raise ValueError("labels must not contain missing values")
    unique = set(values.tolist())
    if not unique.issubset({False, True, 0, 1}):
        raise ValueError("labels must contain only False/True or 0/1 values")
    return values.astype(bool).to_numpy()


def _validate_budget(review_fraction: float) -> float:
    fraction = float(review_fraction)
    if not 0 < fraction <= 1:
        raise ValueError("review_fraction must be in (0, 1]")
    return fraction


def rank_by_score(scores: Iterable[float] | pd.Series | np.ndarray) -> np.ndarray:
    """Return descending score positions, breaking ties by input order.

    Stable ordering makes ties reproducible and does not inspect labels.
    """

    values = _as_numeric_scores(scores)
    return np.argsort(-values, kind="mergesort")


def review_budget_metrics(
    scores: Iterable[float] | pd.Series | np.ndarray,
    labels: Iterable[bool | int] | pd.Series | np.ndarray,
    review_fraction: float,
    approach: str = "model",
) -> dict[str, Any]:
    """Measure precision@K and recall@K after score-only ranking."""

    score_values = _as_numeric_scores(scores)
    label_values = _as_binary_labels(labels)
    if len(score_values) != len(label_values):
        raise ValueError("scores and labels must have the same number of rows")
    fraction = _validate_budget(review_fraction)
    rows = len(score_values)
    k = 0 if rows == 0 else max(1, int(np.ceil(rows * fraction)))
    ranking = rank_by_score(score_values)
    selected = ranking[:k]
    selected_positive_count = int(label_values[selected].sum())
    total_positive_count = int(label_values.sum())
    return {
        "approach": approach,
        "rows": rows,
        "review_fraction": fraction,
        "k": k,
        "selected_positive_count": selected_positive_count,
        "total_positive_count": total_positive_count,
        "precision_at_k": float(selected_positive_count / k) if k else 0.0,
        "recall_at_k": float(selected_positive_count / total_positive_count) if total_positive_count else 0.0,
        "positive_class": POSITIVE_CLASS,
    }


def _random_metrics(
    labels: np.ndarray,
    review_fraction: float,
    random_runs: int,
    random_state: int,
) -> dict[str, Any]:
    if random_runs < 1:
        raise ValueError("random_runs must be at least 1")
    fraction = _validate_budget(review_fraction)
    rows = len(labels)
    k = 0 if rows == 0 else max(1, int(np.ceil(rows * fraction)))
    rng = np.random.default_rng(random_state)
    precisions: list[float] = []
    recalls: list[float] = []
    selected_counts: list[int] = []
    total_positive_count = int(labels.sum())
    for _ in range(random_runs):
        selected = rng.permutation(rows)[:k]
        selected_count = int(labels[selected].sum())
        selected_counts.append(selected_count)
        precisions.append(float(selected_count / k) if k else 0.0)
        recalls.append(float(selected_count / total_positive_count) if total_positive_count else 0.0)
    return {
        "approach": "random_selection",
        "rows": rows,
        "review_fraction": fraction,
        "k": k,
        "selected_positive_count": float(np.mean(selected_counts)) if selected_counts else 0.0,
        "total_positive_count": total_positive_count,
        "precision_at_k": float(np.mean(precisions)) if precisions else 0.0,
        "recall_at_k": float(np.mean(recalls)) if recalls else 0.0,
        "precision_at_k_std": float(np.std(precisions, ddof=1)) if len(precisions) > 1 else 0.0,
        "recall_at_k_std": float(np.std(recalls, ddof=1)) if len(recalls) > 1 else 0.0,
        "positive_class": POSITIVE_CLASS,
        "simulation_runs": random_runs,
        "random_seed": random_state,
    }


def compare_review_strategies(
    scores: Iterable[float] | pd.Series | np.ndarray,
    labels: Iterable[bool | int] | pd.Series | np.ndarray,
    review_fractions: Iterable[float],
    *,
    heuristic_scores: Iterable[float] | pd.Series | np.ndarray | None = None,
    heuristic_name: str = "heuristic",
    random_runs: int = DEFAULT_RANDOM_RUNS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> pd.DataFrame:
    """Compare model, random, and optional heuristic strategies.

    The caller should pass validation labels while developing a strategy and
    call this for held-out test labels only once after choices are frozen.
    """

    score_values = _as_numeric_scores(scores)
    label_values = _as_binary_labels(labels)
    if len(score_values) != len(label_values):
        raise ValueError("scores and labels must have the same number of rows")
    heuristic_values = None if heuristic_scores is None else _as_numeric_scores(heuristic_scores)
    if heuristic_values is not None and len(heuristic_values) != len(score_values):
        raise ValueError("heuristic_scores must have the same number of rows as scores")

    rows: list[dict[str, Any]] = []
    for fraction in review_fractions:
        rows.append(review_budget_metrics(score_values, label_values, fraction, "model"))
        random_row = _random_metrics(label_values, fraction, random_runs, random_state)
        rows.append(random_row)
        if heuristic_values is not None:
            rows.append(review_budget_metrics(heuristic_values, label_values, fraction, heuristic_name))
    return pd.DataFrame(rows)

