"""Project-specific error diagnosis using training/development periods only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score

try:
    from .data import DEFAULT_DATA_PATH, author_dates_utc, load_apachejit
    from .evaluate import choose_threshold_by_f1
    from .hgb_robustness import HGB_GRID, make_random_forest
    from .pipeline import build_numeric_pipeline
    from .review_budget import review_budget_metrics
    from .split import DEFAULT_REPORT_DIR, SAFE_NUMERIC_FEATURES, make_temporal_splits
    from .train import positive_probability
except ImportError:
    from data import DEFAULT_DATA_PATH, author_dates_utc, load_apachejit
    from evaluate import choose_threshold_by_f1
    from hgb_robustness import HGB_GRID, make_random_forest
    from pipeline import build_numeric_pipeline
    from review_budget import review_budget_metrics
    from split import DEFAULT_REPORT_DIR, SAFE_NUMERIC_FEATURES, make_temporal_splits
    from train import positive_probability


RANDOM_STATE = 42
PERIODS = ("2014", "2015", "2016", "2017_validation")
MIN_PROJECT_ROWS = 30
THRESHOLD_GRID = (0.30, 0.38, 0.4402460058475116, 0.50, 0.60)


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Return a 95% Wilson interval for a binomial rate."""
    if total <= 0:
        return (float("nan"), float("nan"))
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = z * np.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def classification_metric_row(
    labels: pd.Series,
    scores: pd.Series,
    threshold: float,
    *,
    model: str,
    period: str,
    project: str,
    comparison_group: str,
) -> dict:
    """Calculate project metrics with explicit denominators and intervals."""
    predicted = scores.to_numpy() >= threshold
    actual = labels.to_numpy(dtype=bool)
    tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[False, True]).ravel()
    n = len(actual)
    precision_denominator = int(tp + fp)
    recall_denominator = int(tp + fn)
    fpr_denominator = int(fp + tn)
    precision = float(tp / precision_denominator) if precision_denominator else 0.0
    recall = float(tp / recall_denominator) if recall_denominator else 0.0
    fpr = float(fp / fpr_denominator) if fpr_denominator else 0.0
    precision_ci = wilson_interval(int(tp), precision_denominator)
    recall_ci = wilson_interval(int(tp), recall_denominator)
    fpr_ci = wilson_interval(int(fp), fpr_denominator)
    budget = review_budget_metrics(scores, actual, 0.10)
    return {
        "model": model, "period": period, "project": project,
        "comparison_group": comparison_group, "rows": n,
        "adequately_sized": n >= MIN_PROJECT_ROWS,
        "small_group_warning": n < MIN_PROJECT_ROWS,
        "positive_count": int(actual.sum()),
        "positive_prevalence": float(actual.mean()) if n else 0.0,
        "predicted_positive_count": int(predicted.sum()),
        "predicted_positive_rate": float(predicted.mean()) if n else 0.0,
        "true_negative": int(tn), "false_positive": int(fp),
        "false_negative": int(fn), "true_positive": int(tp),
        "precision": precision, "precision_denominator": precision_denominator,
        "precision_ci_low": precision_ci[0], "precision_ci_high": precision_ci[1],
        "recall": recall, "recall_denominator": recall_denominator,
        "recall_ci_low": recall_ci[0], "recall_ci_high": recall_ci[1],
        "f1": float(f1_score(actual, predicted, zero_division=0)),
        "false_positive_rate": fpr, "false_positive_rate_denominator": fpr_denominator,
        "false_positive_rate_ci_low": fpr_ci[0], "false_positive_rate_ci_high": fpr_ci[1],
        "average_precision": float(average_precision_score(actual, scores)) if actual.any() else float("nan"),
        "precision_at_10": budget["precision_at_k"], "recall_at_10": budget["recall_at_k"],
        "review_k_at_10": budget["k"], "threshold": float(threshold),
    }


def _fit_period_models(train: pd.DataFrame, validation: pd.DataFrame) -> dict[str, tuple[pd.Series, float]]:
    models = {
        "random_forest_current": make_random_forest(),
        "hist_gradient_boosting_selected": build_numeric_pipeline(
            HistGradientBoostingClassifier(random_state=RANDOM_STATE, **HGB_GRID["hgb_existing"])
        ),
    }
    output = {}
    for name, pipeline in models.items():
        pipeline.fit(train[SAFE_NUMERIC_FEATURES], train["buggy"])
        scores = positive_probability(pipeline, validation[SAFE_NUMERIC_FEATURES])
        threshold, _ = choose_threshold_by_f1(validation["buggy"], scores)
        output[name] = (scores, threshold)
    return output


def _period_frames(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    splits = make_temporal_splits(frame)
    train = splits.train
    years = author_dates_utc(train).dt.year
    result = {"train": train}
    for year in (2014, 2015, 2016):
        result[str(year)] = train.loc[years == year].reset_index(drop=True)
    result["2017_validation"] = splits.validation
    return result


def _profile_rows(validation: pd.DataFrame, scores: pd.Series, threshold: float, model: str, period: str) -> list[dict]:
    predicted = scores.to_numpy() >= threshold
    labels = validation["buggy"].to_numpy(dtype=bool)
    outcomes = np.select(
        [labels & predicted, ~labels & ~predicted, ~labels & predicted],
        ["true_positive", "true_negative", "false_positive"], default="false_negative"
    )
    scored = validation[SAFE_NUMERIC_FEATURES].copy()
    scored["label"] = labels
    scored["score"] = scores.to_numpy()
    scored["outcome"] = outcomes
    rows = []
    groups = {"apache/hadoop": scored[validation["project"].eq("apache/hadoop")], "other_projects": scored[~validation["project"].eq("apache/hadoop")]}
    for comparison_group, group in groups.items():
        for outcome in ("true_positive", "false_positive", "true_negative", "false_negative"):
            subset = group[group["outcome"] == outcome]
            if subset.empty:
                continue
            for feature in SAFE_NUMERIC_FEATURES:
                rows.append({
                    "model": model, "period": period, "comparison_group": comparison_group,
                    "outcome": outcome, "feature": feature, "rows": len(subset),
                    "feature_mean": float(subset[feature].mean()), "feature_median": float(subset[feature].median()),
                    "score_mean": float(subset["score"].mean()), "score_median": float(subset["score"].median()),
                    "score_q25": float(subset["score"].quantile(.25)), "score_q75": float(subset["score"].quantile(.75)),
                    "observed_positive_rate": float(subset["label"].mean()),
                })
    return rows


def run(data_path: Path = DEFAULT_DATA_PATH, report_dir: Path = DEFAULT_REPORT_DIR) -> dict[str, pd.DataFrame]:
    periods = _period_frames(load_apachejit(data_path))
    train = periods["train"]
    train_years = author_dates_utc(train).dt.year
    metric_rows = []
    profile_rows = []
    sensitivity_rows = []
    for period in PERIODS:
        validation = periods[period]
        fit = train if period == "2017_validation" else train.loc[train_years < int(period)].reset_index(drop=True)
        fitted = _fit_period_models(fit, validation)
        for model, (scores, selected_threshold) in fitted.items():
            project_values = sorted(validation["project"].dropna().unique())
            masks = {project: validation["project"].eq(project) for project in project_values}
            masks["other_projects"] = ~validation["project"].eq("apache/hadoop")
            for project, mask in list(masks.items()):
                if project == "other_projects":
                    group = "other_projects"
                else:
                    group = "apache/hadoop" if project == "apache/hadoop" else "individual_project"
                metric_rows.append(classification_metric_row(
                    validation.loc[mask, "buggy"], scores.loc[mask], selected_threshold,
                    model=model, period=period, project=project, comparison_group=group,
                ))
            profile_rows.extend(_profile_rows(validation, scores, selected_threshold, model, period))
            if model == "random_forest_current":
                for threshold in THRESHOLD_GRID:
                    for group_name, mask in {
                        "apache/hadoop": validation["project"].eq("apache/hadoop"),
                        "other_projects": ~validation["project"].eq("apache/hadoop"),
                    }.items():
                        row = classification_metric_row(
                            validation.loc[mask, "buggy"], scores.loc[mask], threshold,
                            model=model, period=period, project=group_name, comparison_group=group_name,
                        )
                        row["threshold_source"] = "predeclared_diagnostic_grid"
                        row["selected_fold_threshold"] = float(selected_threshold)
                        sensitivity_rows.append(row)
    metrics = pd.DataFrame(metric_rows)
    profiles = pd.DataFrame(profile_rows)
    sensitivity = pd.DataFrame(sensitivity_rows)
    report_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(report_dir / "project_error_metrics.csv", index=False)
    profiles.to_csv(report_dir / "hadoop_error_profiles.csv", index=False)
    sensitivity.to_csv(report_dir / "project_threshold_sensitivity.csv", index=False)
    config_path = report_dir / "evaluation_config.json"
    saved_threshold = json.loads(config_path.read_text(encoding="utf-8"))["selected_threshold"] if config_path.exists() else None
    _write_report(report_dir / "project_error_diagnosis.md", metrics, profiles, sensitivity, saved_threshold)
    return {"metrics": metrics, "profiles": profiles, "sensitivity": sensitivity}


def _write_report(path: Path, metrics: pd.DataFrame, profiles: pd.DataFrame, sensitivity: pd.DataFrame, saved_threshold: float | None) -> None:
    def table(frame: pd.DataFrame, columns: list[str], limit: int = 40) -> str:
        view = frame[columns].head(limit).copy()
        for column in view.columns:
            if pd.api.types.is_float_dtype(view[column]):
                view[column] = view[column].map(lambda value: f"{value:.4f}")
        return "\n".join([
            "| " + " | ".join(view.columns) + " |",
            "| " + " | ".join("---" for _ in view.columns) + " |",
            *["| " + " | ".join(str(value) for value in row) + " |" for row in view.itertuples(index=False, name=None)],
        ])

    hadoop = metrics[(metrics.project == "apache/hadoop") & metrics.period.eq("2017_validation")]
    hadoop_over_time = metrics[metrics.project.isin(["apache/hadoop", "other_projects"])]
    totals = metrics.groupby(["model", "period", "project"], as_index=False).agg(
        rows=("rows", "first"), positive_prevalence=("positive_prevalence", "first"),
        true_positive=("true_positive", "first"), false_positive=("false_positive", "first"),
        false_negative=("false_negative", "first"), true_negative=("true_negative", "first"),
        precision=("precision", "first"), recall=("recall", "first"), f1=("f1", "first"),
        false_positive_rate=("false_positive_rate", "first"), predicted_positive_rate=("predicted_positive_rate", "first"),
    )
    fp = metrics[metrics["false_positive"] > 0].groupby(["model", "period"], as_index=False)["false_positive"].sum()
    sensitivity = sensitivity[sensitivity.period.eq("2017_validation") & sensitivity.project.isin(["apache/hadoop", "other_projects"])]
    profile_focus = profiles[
        profiles.model.eq("random_forest_current")
        & profiles.period.eq("2017_validation")
        & profiles.feature.isin(["la", "ld", "ndev", "age", "nuc", "aexp"])
        & profiles.outcome.isin(["false_positive", "true_negative"])
    ]
    path.write_text(f"""# ReviewRadar project-specific error diagnosis

Audit date: 2026-10-26. This report uses only 2014–2017 development periods.
The complete CSV is loaded only to construct the established split; the
previously inspected final test period is not analyzed, scored, or used.

## Protocol confirmation

Each row is a commit/change revision and the positive class is `buggy=True`.
The saved Random Forest threshold is `{saved_threshold if saved_threshold is not None else 'not found'}`.
The project metrics use per-fold maximum-F1 thresholds selected on each
validation fold; this is separate from the saved threshold and is used to
make the rolling error diagnosis consistent with the existing research.

Rates include denominators in `reports/project_error_metrics.csv`. Wilson
95% intervals are provided for precision, recall, and false-positive rate.
Groups below {MIN_PROJECT_ROWS} rows are marked with a warning. F1 and
average precision are reported without invented intervals.

## Hadoop versus other projects in 2017 validation

{table(hadoop, ['model', 'project', 'rows', 'positive_prevalence', 'true_positive', 'false_positive', 'true_negative', 'false_negative', 'precision', 'recall', 'f1', 'false_positive_rate', 'predicted_positive_rate'], 10)}

For all periods and projects, use the complete metrics CSV rather than raw FP
counts. Error counts depend on project sample volume and prevalence.

Across periods, the Hadoop/other-project comparison is:

{table(hadoop_over_time, ['model', 'period', 'project', 'rows', 'positive_prevalence', 'precision', 'recall', 'f1', 'false_positive_rate', 'predicted_positive_rate'], 20)}

## Does Hadoop dominate because of volume?

{table(fp.sort_values('false_positive', ascending=False), ['model', 'period', 'false_positive'], 20)}

Hadoop's repeated false positives are evidence of a project-specific mismatch
in the current metric-only model, but volume and composition are both part of
the observation. The denominator-aware metrics and prevalence must be checked
before calling it a systematic weakness.

## Feature and score profiles

`reports/hadoop_error_profiles.csv` contains feature means/medians and score
quantiles for true positives, false positives, true negatives, and false
negatives for Hadoop and the pooled other-project comparison. These are
descriptive associations; they do not establish causation. Small subgroups
should be treated cautiously.

Selected Random Forest 2017 profiles for Hadoop and pooled other projects:

{table(profile_focus, ['comparison_group', 'outcome', 'feature', 'rows', 'feature_mean', 'feature_median', 'score_mean', 'score_median', 'score_q25', 'score_q75'], 40)}

## Threshold diagnostic

The fixed RF grid in `reports/project_threshold_sensitivity.csv` is a
predeclared historical diagnostic on validation folds. It does not alter the
saved model or threshold. Lower thresholds may raise recall while increasing
false positives; higher thresholds may reduce false positives while losing
defect recall. Ranking metrics and thresholded metrics are separate: the
continuous score ranking is unchanged by a threshold.

{table(sensitivity.sort_values(['project', 'threshold']), ['model', 'period', 'project', 'threshold', 'rows', 'precision', 'recall', 'f1', 'false_positive_rate', 'precision_at_10', 'recall_at_10'], 30)}

A project-specific threshold would require a deployment-time policy, enough
recent labeled examples per project, and leakage-safe calibration data. No
such rule is introduced here.

## Decision

The current evidence supports a combination: Hadoop's raw FP count is partly
a sample-volume/composition effect, while the repeated denominator-aware error
pattern is consistent with a project-specific model weakness. The evidence is
not sufficient to prove whether that weakness is caused by missing project
context, label process differences, score calibration, or another factor.

Additional data needed: a larger prospective labeled sample with stable
project identity, prediction-time change metrics, and enough outcomes per
project for predeclared calibration and threshold evaluation. The next
independent future holdout must remain untouched until all such choices are
frozen.
""", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    args = parser.parse_args()
    result = run(args.input, args.report_dir)
    print({name: len(frame) for name, frame in result.items()})


if __name__ == "__main__":
    main()
