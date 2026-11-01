"""Guarded, reusable rolling-origin evaluation for a genuinely later dataset.

The default command performs readiness validation only. It refuses to use the
already-inspected ApacheJIT and JIT-Defect4J test periods as a new holdout.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score

try:
    from .evaluate import choose_threshold_by_f1
    from .pipeline import build_numeric_pipeline
    from .review_budget import review_budget_metrics
    from .train import positive_probability
except ImportError:
    from evaluate import choose_threshold_by_f1
    from pipeline import build_numeric_pipeline
    from review_budget import review_budget_metrics
    from train import positive_probability


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FUTURE_PATH = ROOT / "data" / "external" / "future_holdout.csv"
RANDOM_STATE = 42
BUDGETS = (0.05, 0.10, 0.20)
MIN_PERIOD_POSITIVES = 20
MIN_PERIOD_NEGATIVES = 20
MIN_PROJECT_ROWS = 30
MIN_PROJECT_POSITIVES = 5


def _timestamps(values: pd.Series) -> pd.Series:
    """Parse Unix-second columns or timezone-aware timestamp strings."""
    if pd.api.types.is_datetime64_any_dtype(values):
        return pd.to_datetime(values, errors="coerce", utc=True)
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().all():
        return pd.to_datetime(numeric, errors="coerce", unit="s", utc=True)
    return pd.to_datetime(values, errors="coerce", utc=True)


@dataclass(frozen=True)
class RollingFold:
    name: str
    fit: pd.DataFrame
    evaluation: pd.DataFrame


def validate_frame(
    frame: pd.DataFrame,
    *,
    id_col: str = "commit_id",
    timestamp_col: str = "author_date",
    project_col: str = "project",
    label_col: str = "buggy",
) -> dict[str, object]:
    """Validate integrity without changing the supplied frame."""
    required = {id_col, timestamp_col, project_col, label_col}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    if frame.empty:
        raise ValueError("dataset is empty")
    if frame[id_col].isna().any() or frame[id_col].duplicated().any():
        raise ValueError(f"{id_col} must be present and unique")
    if frame[project_col].isna().any():
        raise ValueError(f"{project_col} must not contain missing values")
    if frame.duplicated().any():
        raise ValueError("exact duplicate records are not allowed")
    timestamps = _timestamps(frame[timestamp_col])
    if timestamps.isna().any():
        raise ValueError(f"{timestamp_col} contains missing or invalid timestamps")
    labels = frame[label_col]
    if labels.isna().any() or not set(labels.unique()).issubset({True, False, 0, 1}):
        raise ValueError(f"{label_col} must contain only boolean/0/1 labels")
    counts = labels.astype(bool).value_counts().to_dict()
    return {
        "rows": len(frame),
        "date_min_utc": timestamps.min().isoformat(),
        "date_max_utc": timestamps.max().isoformat(),
        "positive_count": int(counts.get(True, 0)),
        "negative_count": int(counts.get(False, 0)),
        "positive_prevalence": float(labels.astype(bool).mean()),
        "projects": int(frame[project_col].nunique(dropna=False)),
        "missing_values": int(frame.isna().sum().sum()),
    }


def validate_temporal_partition(
    fit: pd.DataFrame,
    evaluation: pd.DataFrame,
    *,
    id_col: str = "commit_id",
    timestamp_col: str = "author_date",
    project_col: str = "project",
    label_col: str = "buggy",
) -> dict[str, object]:
    """Require strict chronology and report overlap/coverage checks."""
    fit_info = validate_frame(fit, id_col=id_col, timestamp_col=timestamp_col, project_col=project_col, label_col=label_col)
    eval_info = validate_frame(evaluation, id_col=id_col, timestamp_col=timestamp_col, project_col=project_col, label_col=label_col)
    shared_ids = set(fit[id_col]) & set(evaluation[id_col])
    fit_dates = _timestamps(fit[timestamp_col])
    eval_dates = _timestamps(evaluation[timestamp_col])
    if fit_dates.max() >= eval_dates.min():
        raise ValueError("training timestamps must strictly precede evaluation timestamps")
    if shared_ids:
        raise ValueError(f"commit-ID overlap detected: {len(shared_ids)}")
    return {
        "fit": fit_info,
        "evaluation": eval_info,
        "shared_commit_ids": len(shared_ids),
        "strictly_chronological": True,
    }


def make_rolling_folds(
    frame: pd.DataFrame,
    boundaries: Iterable[tuple[str, str, str]],
    *,
    id_col: str = "commit_id",
    timestamp_col: str = "author_date",
    project_col: str = "project",
    label_col: str = "buggy",
) -> list[RollingFold]:
    """Create expanding chronological folds from explicit UTC boundaries."""
    validate_frame(frame, id_col=id_col, timestamp_col=timestamp_col, project_col=project_col, label_col=label_col)
    dates = _timestamps(frame[timestamp_col])
    folds = []
    for name, evaluation_start, evaluation_end in boundaries:
        start = pd.Timestamp(evaluation_start)
        end = pd.Timestamp(evaluation_end)
        if start.tzinfo is None or end.tzinfo is None:
            raise ValueError("fold boundaries must include timezone information")
        if start >= end:
            raise ValueError(f"invalid boundary for {name}")
        fit = frame.loc[dates < start].copy()
        evaluation = frame.loc[(dates >= start) & (dates < end)].copy()
        if fit.empty or evaluation.empty:
            raise ValueError(f"fold {name} has empty fit or evaluation rows")
        validate_temporal_partition(fit, evaluation, id_col=id_col, timestamp_col=timestamp_col, project_col=project_col, label_col=label_col)
        folds.append(RollingFold(name, fit, evaluation))
    return folds


def validate_later_dataset(path: Path, *, earlier_max_timestamp: str | pd.Timestamp) -> dict[str, object]:
    """Readiness check for a new dataset, never falling back to existing tests."""
    if not path.is_file():
        return {"eligible": False, "reason": f"later-period dataset not found: {path}"}
    frame = pd.read_csv(path)
    info = validate_frame(frame)
    earlier = pd.Timestamp(earlier_max_timestamp)
    dates = _timestamps(frame["author_date"])
    if dates.min() <= earlier:
        return {"eligible": False, "reason": "later dataset is not strictly after the development cutoff", **info}
    return {"eligible": True, "reason": "eligible for explicit protocol review", **info}


def reference_models(features: list[str]) -> dict[str, object]:
    """Return only the preregistered reference configurations."""
    return {
        "dummy_prior": build_numeric_pipeline(DummyClassifier(strategy="prior"), features),
        "logistic_regression": build_numeric_pipeline(LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE, solver="lbfgs"), features, scale=True),
        "random_forest": build_numeric_pipeline(RandomForestClassifier(class_weight="balanced_subsample", min_samples_leaf=2, n_estimators=300, n_jobs=-1, random_state=RANDOM_STATE), features),
        "hist_gradient_boosting": build_numeric_pipeline(HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0, random_state=RANDOM_STATE), features),
    }


def metric_row(labels: pd.Series, scores: pd.Series, threshold: float, *, dataset: str, split: str, model: str, project: str = "all_projects") -> dict[str, object]:
    actual = labels.astype(bool).to_numpy()
    predicted = scores.to_numpy(dtype=float) >= threshold
    tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[False, True]).ravel()
    positive_count = int(actual.sum())
    row: dict[str, object] = {
        "dataset": dataset, "split": split, "model": model, "project": project, "rows": len(actual),
        "positive_count": positive_count, "negative_count": int(len(actual) - positive_count),
        "positive_prevalence": float(actual.mean()) if len(actual) else 0.0,
        "threshold": float(threshold), "true_negative": int(tn), "false_positive": int(fp), "false_negative": int(fn), "true_positive": int(tp),
        "precision": float(precision_score(actual, predicted, zero_division=0)),
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "f1": float(f1_score(actual, predicted, zero_division=0)),
        "average_precision": float(average_precision_score(actual, scores)) if positive_count else float("nan"),
        "insufficient_positive_evidence": positive_count < MIN_PERIOD_POSITIVES,
        "insufficient_negative_evidence": int(len(actual) - positive_count) < MIN_PERIOD_NEGATIVES,
    }
    for budget in BUDGETS:
        result = review_budget_metrics(scores, actual, budget)
        suffix = f"{int(budget * 100)}"
        row[f"precision_at_{suffix}"] = result["precision_at_k"]
        row[f"recall_at_{suffix}"] = result["recall_at_k"]
        row[f"review_k_at_{suffix}"] = result["k"]
    return row


def evaluate_development(folds: list[RollingFold], features: list[str], dataset: str) -> pd.DataFrame:
    """Fit and score only chronological development folds."""
    rows = []
    for fold in folds:
        for name, model in reference_models(features).items():
            model.fit(fold.fit[features], fold.fit["buggy"])
            scores = positive_probability(model, fold.evaluation[features])
            threshold, _ = choose_threshold_by_f1(fold.evaluation["buggy"], scores)
            rows.append(metric_row(fold.evaluation["buggy"], scores, threshold, dataset=dataset, split=fold.name, model=name))
    return pd.DataFrame(rows)


def select_development_model(development_metrics: pd.DataFrame) -> tuple[str, float]:
    """Freeze model and threshold policy from development rows only."""
    required = {"model", "average_precision", "f1", "threshold"}
    missing = required.difference(development_metrics.columns)
    if missing:
        raise ValueError(f"development metrics missing columns: {sorted(missing)}")
    summary = development_metrics.groupby("model", as_index=False).agg(
        mean_average_precision=("average_precision", "mean"),
        mean_f1=("f1", "mean"),
    )
    summary = summary.sort_values(["mean_average_precision", "mean_f1", "model"], ascending=[False, False, True], na_position="last")
    selected = str(summary.iloc[0]["model"])
    thresholds = development_metrics.loc[development_metrics["model"] == selected, "threshold"].dropna()
    if thresholds.empty:
        raise ValueError("selected development model has no usable thresholds")
    return selected, float(thresholds.median())


def evaluate_frozen_holdout(
    development: pd.DataFrame,
    holdout: pd.DataFrame,
    features: list[str],
    *,
    dataset: str,
    model_name: str,
    threshold: float,
) -> pd.DataFrame:
    """Evaluate one frozen choice on a strictly later holdout.

    The holdout is validated and scored, but never used to fit, select, or
    alter the threshold. This function is intentionally not called by the
    default readiness command.
    """
    validate_temporal_partition(development, holdout)
    models = reference_models(features)
    if model_name not in models:
        raise ValueError(f"unknown preregistered model: {model_name}")
    model = models[model_name]
    model.fit(development[features], development["buggy"])
    scores = positive_probability(model, holdout[features])
    rows = [metric_row(holdout["buggy"], scores, threshold, dataset=dataset, split="frozen_holdout", model=model_name)]
    for project, group in holdout.groupby("project", sort=True):
        rows.append(metric_row(group["buggy"], scores.loc[group.index], threshold, dataset=dataset, split="frozen_holdout", model=model_name, project=str(project)))
    return pd.DataFrame(rows)


def readiness_only(path: Path = DEFAULT_FUTURE_PATH) -> dict[str, object]:
    """Default safe entry point: no eligible future file means no evaluation."""
    result = validate_later_dataset(path, earlier_max_timestamp="2019-12-31T23:59:59Z")
    result["evaluation_run"] = False
    result["message"] = "No new holdout was evaluated; supply a separately reserved later dataset and manifest." if not result["eligible"] else "Dataset passed initial readiness checks; provide an explicit manifest before scoring."
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--future-data", type=Path, default=DEFAULT_FUTURE_PATH)
    args = parser.parse_args()
    print(readiness_only(args.future_data))


if __name__ == "__main__":
    main()
