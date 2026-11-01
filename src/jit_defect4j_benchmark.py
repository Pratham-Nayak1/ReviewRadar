"""Independent JIT-Defect4J commit-level benchmark."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, precision_recall_curve
from sklearn.model_selection import GroupKFold

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
DEFAULT_ROOT = ROOT / "data" / "external" / "raw" / "jit_smart_data" / "data" / "jitfine"
DEFAULT_REPORT_DIR = ROOT / "reports"
RANDOM_STATE = 42
MIN_PROJECT_ROWS = 30
FEATURES = ["la", "ld", "nf", "ns", "nd", "entropy", "ndev", "lt", "nuc", "age", "exp", "rexp", "sexp"]
BUDGETS = (0.05, 0.10, 0.20)


def load_split(root: Path, split: str) -> pd.DataFrame:
    frame = pd.read_pickle(root / f"features_{split}.pkl").copy()
    required = {"commit_hash", "project", "author_date_unix_timestamp", "is_buggy_commit", *FEATURES}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"JIT-Defect4J {split} is missing required columns: {missing}")
    labels = pd.to_numeric(frame["is_buggy_commit"], errors="coerce")
    if labels.isna().any() or not set(labels.unique()).issubset({0, 1}):
        raise ValueError(f"JIT-Defect4J {split} has invalid is_buggy_commit labels")
    frame["is_buggy_commit"] = labels.astype(bool)
    return frame


def make_models() -> dict[str, object]:
    return {
        "dummy_prior": build_numeric_pipeline(DummyClassifier(strategy="prior"), FEATURES),
        "logistic_balanced": build_numeric_pipeline(LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE), FEATURES, scale=True),
        "random_forest_balanced": build_numeric_pipeline(RandomForestClassifier(class_weight="balanced_subsample", n_estimators=100, min_samples_leaf=2, n_jobs=-1, random_state=RANDOM_STATE), FEATURES),
        "hist_gradient_boosting": build_numeric_pipeline(HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0, random_state=RANDOM_STATE), FEATURES),
    }


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return (float("nan"), float("nan"))
    p = successes / total
    d = 1 + z * z / total
    c = (p + z * z / (2 * total)) / d
    m = z * np.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / d
    return max(0.0, c - m), min(1.0, c + m)


def metric_row(labels: pd.Series, scores: pd.Series, threshold: float, *, model: str, split: str, threshold_source: str, project: str | None = None) -> dict:
    actual = labels.to_numpy(dtype=bool)
    predicted = scores.to_numpy(dtype=float) >= threshold
    tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[False, True]).ravel()
    precision_den = int(tp + fp); recall_den = int(tp + fn); fpr_den = int(fp + tn)
    p_ci = wilson(int(tp), precision_den); r_ci = wilson(int(tp), recall_den); f_ci = wilson(int(fp), fpr_den)
    row = {
        "model": model, "split": split, "project": project or "all_projects", "rows": len(actual),
        "positive_count": int(actual.sum()), "positive_prevalence": float(actual.mean()) if len(actual) else 0.0,
        "threshold": float(threshold), "threshold_source": threshold_source,
        "true_negative": int(tn), "false_positive": int(fp), "false_negative": int(fn), "true_positive": int(tp),
        "precision": float(precision_score(actual, predicted, zero_division=0)), "precision_denominator": precision_den,
        "precision_ci_low": p_ci[0], "precision_ci_high": p_ci[1],
        "recall": float(recall_score(actual, predicted, zero_division=0)), "recall_denominator": recall_den,
        "recall_ci_low": r_ci[0], "recall_ci_high": r_ci[1],
        "f1": float(f1_score(actual, predicted, zero_division=0)),
        "false_positive_rate": float(fp / fpr_den) if fpr_den else 0.0, "false_positive_rate_denominator": fpr_den,
        "false_positive_rate_ci_low": f_ci[0], "false_positive_rate_ci_high": f_ci[1],
        "average_precision": float(average_precision_score(actual, scores)) if actual.any() else float("nan"),
        "limited_group_warning": bool(len(actual) < MIN_PROJECT_ROWS or not actual.any()),
    }
    for budget in BUDGETS:
        result = review_budget_metrics(scores, actual, budget)
        suffix = f"{int(budget * 100)}"
        row[f"precision_at_{suffix}"] = result["precision_at_k"]
        row[f"recall_at_{suffix}"] = result["recall_at_k"]
        row[f"review_k_at_{suffix}"] = result["k"]
    return row


def fit_scores(pipeline, fit: pd.DataFrame, evaluation: pd.DataFrame) -> pd.Series:
    pipeline.fit(fit[FEATURES], fit["is_buggy_commit"])
    return positive_probability(pipeline, evaluation[FEATURES])


def safe_threshold(labels: pd.Series, scores: pd.Series) -> tuple[float, float]:
    """Select an F1 threshold, with an explicit safe result for degenerate folds."""
    if len(labels) == 0 or not labels.astype(bool).any() or labels.astype(bool).all():
        return 0.5, 0.0
    return choose_threshold_by_f1(labels, scores)


def pr_points(labels: pd.Series, scores: pd.Series) -> list[dict[str, float]]:
    """Return PR points without sklearn warnings for zero-positive folds."""
    actual = labels.astype(bool)
    if len(actual) == 0 or not actual.any():
        return [{"precision": 0.0, "recall": 0.0, "threshold": float("nan")}]
    precision, recall, thresholds = precision_recall_curve(actual, scores)
    return [
        {"precision": float(p), "recall": float(r), "threshold": float(thresholds[i]) if i < len(thresholds) else float("nan")}
        for i, (p, r) in enumerate(zip(precision, recall))
    ]


def run(root: Path = DEFAULT_ROOT, report_dir: Path = DEFAULT_REPORT_DIR) -> dict[str, pd.DataFrame]:
    train = load_split(root, "train")
    valid = load_split(root, "valid")
    test = load_split(root, "test")
    ids = {name: set(frame["commit_hash"]) for name, frame in [("train", train), ("valid", valid), ("test", test)]}
    if any(ids[a] & ids[b] for a, b in (("train", "valid"), ("train", "test"), ("valid", "test"))):
        raise ValueError("JIT-Defect4J official split commit IDs overlap")

    validation_rows = []; final_rows = []; pr_rows = []; frozen = {}; config_rows = []
    for name, pipeline in make_models().items():
        valid_scores = fit_scores(pipeline, train, valid)
        threshold, threshold_f1 = safe_threshold(valid["is_buggy_commit"], valid_scores)
        row = metric_row(valid["is_buggy_commit"], valid_scores, threshold, model=name, split="official_valid", threshold_source="validation_max_f1")
        row["validation_threshold_f1"] = threshold_f1
        validation_rows.append(row)
        frozen[name] = (pipeline, float(threshold))
        config_rows.append({
            "model": name,
            "features": json.dumps(FEATURES),
            "random_state": RANDOM_STATE,
            "estimator_parameters": repr(pipeline.named_steps["estimator"].get_params()),
            "validation_threshold": float(threshold),
            "validation_threshold_f1": float(threshold_f1),
            "status": "ok",
            "failure": "",
        })
        pr_rows.extend({"model": name, "split": "official_valid", "point": i, **point} for i, point in enumerate(pr_points(valid["is_buggy_commit"], valid_scores)))

    validation = pd.DataFrame(validation_rows)
    selected_name = str(validation.sort_values(["average_precision", "f1", "model"], ascending=[False, False, True]).iloc[0]["model"])
    for name, (pipeline, threshold) in frozen.items():
        test_scores = positive_probability(pipeline, test[FEATURES])
        final_rows.append(metric_row(test["is_buggy_commit"], test_scores, threshold, model=name, split="official_test", threshold_source="frozen_from_official_valid"))
        pr_rows.extend({"model": name, "split": "official_test", "point": i, **point} for i, point in enumerate(pr_points(test["is_buggy_commit"], test_scores)))

    temporal = pd.DataFrame(temporal_results(train))
    grouped = pd.DataFrame(project_holdout_results(pd.concat([train, valid], ignore_index=True)))
    report_dir.mkdir(parents=True, exist_ok=True)
    comparison = validation.merge(pd.DataFrame(final_rows), on="model", suffixes=("_valid", "_test"))
    comparison["selected_by_validation"] = comparison["model"].eq(selected_name)
    comparison.to_csv(report_dir / "jit_defect4j_model_comparison.csv", index=False)
    temporal.to_csv(report_dir / "jit_defect4j_temporal_results.csv", index=False)
    grouped.to_csv(report_dir / "jit_defect4j_project_holdout_results.csv", index=False)
    pd.DataFrame(config_rows).to_csv(report_dir / "jit_defect4j_model_configs.csv", index=False)
    pd.DataFrame(pr_rows).to_csv(report_dir / "jit_defect4j_pr_curve.csv", index=False)
    (report_dir / "jit_defect4j_benchmark_summary.md").write_text(build_summary(comparison, temporal, grouped, selected_name), encoding="utf-8")
    return {"comparison": comparison, "temporal": temporal, "project_holdout": grouped}


def temporal_results(train: pd.DataFrame) -> list[dict]:
    rows = []
    dates = pd.to_datetime(pd.to_numeric(train["author_date_unix_timestamp"], errors="coerce"), unit="s", utc=True)
    for year in (2011, 2012, 2013, 2014, 2015):
        fit = train.loc[dates.dt.year < year]
        valid = train.loc[dates.dt.year == year]
        if fit.empty or valid.empty:
            continue
        for name, pipeline in make_models().items():
            scores = fit_scores(pipeline, fit, valid)
            threshold, _ = safe_threshold(valid["is_buggy_commit"], scores)
            rows.append(metric_row(valid["is_buggy_commit"], scores, threshold, model=name, split=f"temporal_{year}", threshold_source="fold_validation_max_f1"))
    return rows


def project_holdout_results(development: pd.DataFrame) -> list[dict]:
    rows = []
    projects = sorted(development["project"].unique())
    splitter = GroupKFold(n_splits=len(projects))
    for fold, (fit_indices, holdout_indices) in enumerate(splitter.split(development, development["is_buggy_commit"], development["project"]), start=1):
        fit = development.iloc[fit_indices]; holdout = development.iloc[holdout_indices]
        project = str(holdout["project"].iloc[0])
        for name, pipeline in make_models().items():
            scores = fit_scores(pipeline, fit, holdout)
            rows.append(metric_row(holdout["is_buggy_commit"], scores, 0.5, model=name, split="leave_one_project_out", threshold_source="predeclared_0.5", project=project) | {"fold": fold})
    return rows


def build_summary(comparison: pd.DataFrame, temporal: pd.DataFrame, grouped: pd.DataFrame, selected_name: str) -> str:
    selected = comparison[comparison["model"] == selected_name].iloc[0]
    temporal_mean = temporal.groupby("model")["f1"].mean().sort_values(ascending=False).to_string() if not temporal.empty else "unavailable"
    grouped_mean = grouped.groupby("model")["f1"].mean().sort_values(ascending=False).to_string() if not grouped.empty else "unavailable"
    return f"""# JIT-Defect4J independent benchmark summary

The benchmark uses JIT-Fine feature tables from JIT-Defect4J via JIT-Smart.
Each row is a commit-level instance and the positive class is
`is_buggy_commit == 1`. Line-level files are not used here.

## Protocol

The provided train/valid/test commit hashes are disjoint, but timestamps
overlap; the official test is not a future-period holdout. Models and
thresholds are selected on official validation only. The final official test
metrics were scored once after selection and were not used to choose the
model. Temporal and project-held-out results are separate diagnostics.

## Selected model

Validation selected `{selected_name}` by average precision, with F1 as the
secondary criterion. Its official validation F1 was
`{selected['f1_valid']:.4f}` and official-test F1 was
`{selected['f1_test']:.4f}`. These are benchmark measurements, not evidence
of ApacheJIT or production generalization.

## Results

See `reports/jit_defect4j_model_comparison.csv` for full metrics, confusion
matrices, denominators, thresholds, and review-budget results. Temporal and
leave-one-project-out summaries are in their respective CSV files.

Mean temporal diagnostic F1:

```text
{temporal_mean}
```

Mean leave-one-project-out diagnostic F1:

```text
{grouped_mean}
```

Project folds with no positive labels or fewer than 30 rows are marked
`limited_group_warning`; their rates should not be interpreted as stable.

## Limitations

JIT-Defect4J differs from ApacheJIT in project universe, label artifact,
feature schema, and line-level task. The package does not include a complete
independent label-trace log, and license terms were not explicit in the
repository artifacts. Do not merge or claim cross-dataset generalization.
The official test is not temporally later than all training rows, so
prospective future performance remains unmeasured.

Reproduce with:

```powershell
.\\.venv\\Scripts\\python.exe -m src.jit_defect4j_benchmark
.\\.venv\\Scripts\\python.exe -m pytest -q
```
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    args = parser.parse_args()
    result = run(args.data_root, args.report_dir)
    print({name: len(frame) for name, frame in result.items()})


if __name__ == "__main__":
    main()
