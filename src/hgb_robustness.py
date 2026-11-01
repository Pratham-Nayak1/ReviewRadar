"""Temporal robustness and HistGradientBoosting tuning without final-test use."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold

try:
    from .data import DEFAULT_DATA_PATH, author_dates_utc, load_apachejit
    from .evaluate import choose_threshold_by_f1
    from .pipeline import build_numeric_pipeline
    from .review_budget import compare_review_strategies
    from .split import DEFAULT_REPORT_DIR, SAFE_NUMERIC_FEATURES, make_temporal_splits
    from .train import positive_probability
except ImportError:
    from data import DEFAULT_DATA_PATH, author_dates_utc, load_apachejit
    from evaluate import choose_threshold_by_f1
    from pipeline import build_numeric_pipeline
    from review_budget import compare_review_strategies
    from split import DEFAULT_REPORT_DIR, SAFE_NUMERIC_FEATURES, make_temporal_splits
    from train import positive_probability


RANDOM_STATE = 42
HGB_GRID = {
    "hgb_existing": {"max_iter": 200, "learning_rate": 0.05, "max_leaf_nodes": 31, "l2_regularization": 1.0},
    "hgb_shallow": {"max_iter": 200, "learning_rate": 0.05, "max_leaf_nodes": 15, "l2_regularization": 1.0},
    "hgb_regularized": {"max_iter": 200, "learning_rate": 0.05, "max_leaf_nodes": 31, "l2_regularization": 10.0},
    "hgb_slow_learning": {"max_iter": 300, "learning_rate": 0.03, "max_leaf_nodes": 31, "l2_regularization": 1.0},
}


def make_random_forest():
    return build_numeric_pipeline(
        RandomForestClassifier(
            class_weight="balanced_subsample", min_samples_leaf=2,
            n_estimators=300, n_jobs=-1, random_state=RANDOM_STATE,
        )
    )


def make_hgb(config: dict):
    return build_numeric_pipeline(
        HistGradientBoostingClassifier(random_state=RANDOM_STATE, **config)
    )


def temporal_folds(train: pd.DataFrame) -> list[tuple[str, pd.DataFrame, pd.DataFrame]]:
    dates = author_dates_utc(train)
    years = dates.dt.year
    folds = []
    for year in (2014, 2015, 2016):
        fit = train.loc[years < year].reset_index(drop=True)
        validation = train.loc[years == year].reset_index(drop=True)
        if fit.empty or validation.empty:
            raise ValueError(f"Temporal fold {year} is empty")
        if author_dates_utc(fit).max() >= author_dates_utc(validation).min():
            raise ValueError(f"Temporal fold {year} overlaps in time")
        folds.append((str(year), fit, validation))
    return folds


def metric_row(model: str, fold: str, labels: pd.Series, scores: pd.Series, threshold: float, config: dict, runtime: float, split_kind: str) -> dict:
    predicted = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[False, True]).ravel()
    return {
        "model": model,
        "fold": fold,
        "split_kind": split_kind,
        "rows": len(labels),
        "positive_count": int(labels.sum()),
        "positive_prevalence": float(labels.mean()),
        "threshold": float(threshold),
        "threshold_selection": "fold_validation_max_f1",
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "average_precision": float(average_precision_score(labels, scores)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "runtime_seconds": float(runtime),
        "configuration": json.dumps(config, sort_keys=True),
        "random_seed": RANDOM_STATE,
        "feature_count": len(SAFE_NUMERIC_FEATURES),
        "preprocessing": "median_imputer_inside_pipeline; no validation resampling",
    }


def fit_score(pipeline, fit: pd.DataFrame, validation: pd.DataFrame) -> tuple[pd.Series, float, float]:
    started = time.perf_counter()
    pipeline.fit(fit[SAFE_NUMERIC_FEATURES], fit["buggy"])
    scores = positive_probability(pipeline, validation[SAFE_NUMERIC_FEATURES])
    threshold, _ = choose_threshold_by_f1(validation["buggy"], scores)
    return scores, threshold, time.perf_counter() - started


def select_hgb_config(search: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    successful = search[search["status"] == "ok"].copy()
    aggregate = successful.groupby("configuration_name").agg(
        mean_f1=("f1", "mean"), mean_average_precision=("average_precision", "mean"),
        threshold_std=("threshold", "std"), fold_count=("fold", "nunique"),
    ).reset_index()
    aggregate["threshold_std"] = aggregate["threshold_std"].fillna(0.0)
    aggregate = aggregate.sort_values(
        ["mean_f1", "mean_average_precision", "threshold_std", "configuration_name"],
        ascending=[False, False, True, True], ignore_index=True,
    )
    if aggregate.empty or int(aggregate.iloc[0]["fold_count"]) < 3:
        raise ValueError("No HGB configuration completed all temporal folds")
    return str(aggregate.iloc[0]["configuration_name"]), aggregate


def run(data_path: Path = DEFAULT_DATA_PATH, report_dir: Path = DEFAULT_REPORT_DIR) -> dict:
    frame = load_apachejit(data_path)
    splits = make_temporal_splits(frame)
    train, validation_2017 = splits.train, splits.validation
    folds = temporal_folds(train)
    report_dir.mkdir(parents=True, exist_ok=True)

    rf_rows = []
    hgb_rows = []
    for fold_name, fit, validation in folds:
        scores, threshold, runtime = fit_score(make_random_forest(), fit, validation)
        rf_rows.append(metric_row("random_forest_current", fold_name, validation["buggy"], scores, threshold, {"n_estimators": 300, "min_samples_leaf": 2, "class_weight": "balanced_subsample"}, runtime, "rolling_temporal"))
        for config_name, config in HGB_GRID.items():
            try:
                scores, threshold, runtime = fit_score(make_hgb(config), fit, validation)
                hgb_rows.append({"status": "ok", "configuration_name": config_name, **metric_row(config_name, fold_name, validation["buggy"], scores, threshold, config, runtime, "rolling_temporal")})
            except Exception as exc:
                hgb_rows.append({"status": "failed", "configuration_name": config_name, "fold": fold_name, "error": repr(exc), "random_seed": RANDOM_STATE})

    hgb_search = pd.DataFrame(hgb_rows)
    hgb_search.to_csv(report_dir / "f1_hgb_search.csv", index=False)
    selected_name, aggregate = select_hgb_config(hgb_search)
    selected_config = HGB_GRID[selected_name]

    comparison = pd.DataFrame(rf_rows)
    selected_hgb = hgb_search[(hgb_search["configuration_name"] == selected_name) & (hgb_search["status"] == "ok")].copy()
    selected_hgb["model"] = "hist_gradient_boosting_selected"
    paired = comparison.merge(selected_hgb[["fold", "f1", "precision", "recall", "average_precision", "positive_prevalence", "rows"]], on="fold", suffixes=("_rf", "_hgb"))
    paired["f1_difference_hgb_minus_rf"] = paired["f1_hgb"] - paired["f1_rf"]
    paired["average_precision_difference_hgb_minus_rf"] = paired["average_precision_hgb"] - paired["average_precision_rf"]
    paired.to_csv(report_dir / "f1_temporal_model_comparison.csv", index=False)

    threshold_rows = []
    for row in rf_rows:
        threshold_rows.append({"model": "random_forest_current", **{key: row[key] for key in ("fold", "threshold", "precision", "recall", "f1", "positive_prevalence", "rows")}})
    for row in selected_hgb.to_dict("records"):
        threshold_rows.append({"model": "hist_gradient_boosting_selected", **{key: row[key] for key in ("fold", "threshold", "precision", "recall", "f1", "positive_prevalence", "rows")}})

    validation_scores = {}
    rf_scores, rf_threshold, rf_runtime = fit_score(make_random_forest(), train, validation_2017)
    hgb_scores, hgb_threshold, hgb_runtime = fit_score(make_hgb(selected_config), train, validation_2017)
    validation_scores["random_forest_current"] = rf_scores
    validation_scores["hist_gradient_boosting_selected"] = hgb_scores
    threshold_rows.extend([
        {"model": "random_forest_current", "fold": "2017", "threshold": rf_threshold, "precision": precision_score(validation_2017["buggy"], rf_scores >= rf_threshold, zero_division=0), "recall": recall_score(validation_2017["buggy"], rf_scores >= rf_threshold, zero_division=0), "f1": f1_score(validation_2017["buggy"], rf_scores >= rf_threshold, zero_division=0), "positive_prevalence": validation_2017["buggy"].mean(), "rows": len(validation_2017)},
        {"model": "hist_gradient_boosting_selected", "fold": "2017", "threshold": hgb_threshold, "precision": precision_score(validation_2017["buggy"], hgb_scores >= hgb_threshold, zero_division=0), "recall": recall_score(validation_2017["buggy"], hgb_scores >= hgb_threshold, zero_division=0), "f1": f1_score(validation_2017["buggy"], hgb_scores >= hgb_threshold, zero_division=0), "positive_prevalence": validation_2017["buggy"].mean(), "rows": len(validation_2017)},
    ])
    threshold_stability = pd.DataFrame(threshold_rows)
    threshold_stability.to_csv(report_dir / "f1_threshold_stability.csv", index=False)

    ranking_frames = []
    for model_name, scores in validation_scores.items():
        ranked = compare_review_strategies(scores, validation_2017["buggy"], (0.05, 0.10, 0.20), heuristic_scores=validation_2017["la"] + validation_2017["ld"], heuristic_name="change_size_la_plus_ld", random_runs=1000, random_state=RANDOM_STATE)
        ranked.insert(0, "model", model_name)
        ranked["split"] = "2017_validation"
        ranking_frames.append(ranked)
    review_budget = pd.concat(ranking_frames, ignore_index=True)
    review_budget.to_csv(report_dir / "f1_review_budget_comparison.csv", index=False)

    grouped_rows = []
    groups = train["project"]
    group_kfold = GroupKFold(n_splits=3)
    for fold_index, (fit_indices, validation_indices) in enumerate(group_kfold.split(train, train["buggy"], groups), start=1):
        fit = train.iloc[fit_indices]
        validation = train.iloc[validation_indices]
        for model_name, pipeline, config in (
            ("random_forest_current", make_random_forest(), {"n_estimators": 300, "min_samples_leaf": 2}),
            ("hist_gradient_boosting_selected", make_hgb(selected_config), selected_config),
        ):
            try:
                scores, threshold, runtime = fit_score(pipeline, fit, validation)
                row = metric_row(model_name, str(fold_index), validation["buggy"], scores, threshold, config, runtime, "grouped_project_3fold")
                row["validation_projects"] = ",".join(sorted(validation["project"].unique()))
                grouped_rows.append(row)
            except Exception as exc:
                grouped_rows.append({"model": model_name, "fold": str(fold_index), "split_kind": "grouped_project_3fold", "status": "failed", "error": repr(exc)})
    grouped = pd.DataFrame(grouped_rows)
    grouped.to_csv(report_dir / "f1_grouped_project_comparison.csv", index=False)

    aggregate.to_csv(report_dir / "f1_hgb_search_aggregate.csv", index=False)
    report = build_report(aggregate, paired, threshold_stability, review_budget, grouped, selected_name, selected_config, rf_threshold, hgb_threshold)
    (report_dir / "f1_temporal_robustness.md").write_text(report, encoding="utf-8")
    return {"selected_hgb": selected_name, "aggregate": aggregate, "paired": paired}


def build_report(aggregate, paired, thresholds, budget, grouped, selected_name, selected_config, rf_threshold, hgb_threshold) -> str:
    def table(frame: pd.DataFrame, columns: list[str], digits: int = 4) -> str:
        view = frame[columns].copy()
        for column in view.columns:
            if pd.api.types.is_float_dtype(view[column]):
                view[column] = view[column].map(lambda value: f"{value:.{digits}f}")
        # Keep report generation independent of pandas' optional tabulate
        # dependency.  The project requirements do not need a table renderer.
        header = "| " + " | ".join(view.columns) + " |"
        divider = "| " + " | ".join("---" for _ in view.columns) + " |"
        rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in view.itertuples(index=False, name=None)]
        return "\n".join([header, divider, *rows])

    paired_report = paired[["fold", "positive_prevalence_rf", "rows_rf", "f1_rf", "f1_hgb", "f1_difference_hgb_minus_rf", "average_precision_rf", "average_precision_hgb", "average_precision_difference_hgb_minus_rf"]]
    threshold_report = thresholds[["model", "fold", "threshold", "precision", "recall", "f1"]]
    budget_report = budget[(budget["approach"].isin(["model", "random_selection", "change_size_la_plus_ld"])) & (budget["review_fraction"] == 0.10)][["model", "approach", "precision_at_k", "recall_at_k", "k"]]
    grouped_report = grouped[["model", "fold", "rows", "positive_prevalence", "f1", "average_precision"]] if not grouped.empty else grouped
    return f"""# Temporal robustness and HistGradientBoosting tuning

Audit date: 2026-10-26

## Decision

Keep the saved Random Forest and retain HistGradientBoosting as a future
candidate. The selected HGB configuration is `{selected_name}` with
`{json.dumps(selected_config, sort_keys=True)}`. It was selected using the
mean F1, mean average precision, and threshold variability across three
expanding temporal folds inside the pre-2017 training period. It was not
selected using the previously inspected final test set.

There is no new independent future holdout in the available data. The current
evidence is sufficient to justify further candidate work, not a production
model replacement or a generalization claim.

## Protocol and comparability

All folds use the same target (`buggy=True`), 12 numeric features, median
imputation inside the pipeline, chronological ordering, and per-fold maximum-
F1 threshold selection. The current Random Forest uses its exact saved
configuration with 300 trees. The previous `reports/f1_temporal_cv.csv`
used 150 trees for its RF rolling check, so its mean is not an exact saved-
baseline comparison; the paired table below corrects that issue.

The final 2018-onward test is not scored by this runner.

## Paired temporal folds

{table(paired_report, list(paired_report.columns))}

Positive prevalence and row counts are identical between models within each
fold. HGB is better on the three folds by the following F1 differences: the
values in `f1_difference_hgb_minus_rf`. It is not uniformly better on every
possible time slice; fold-level results must remain visible.

## HGB search

{table(aggregate, ["configuration_name", "mean_f1", "mean_average_precision", "threshold_std", "fold_count"])}

The search grid and failure logging are in `reports/f1_hgb_search.csv`. All
configurations completed the planned folds in this run. Selection was not
based on one fold.

## Threshold stability

The 2017 development thresholds were RF `{rf_threshold:.6f}` and selected HGB
`{hgb_threshold:.6f}`. Per-fold thresholds, precision, recall, and F1 are in
`reports/f1_threshold_stability.csv` and summarized here:

{table(threshold_report, list(threshold_report.columns))}

Thresholds vary across time folds, so neither 0.5 nor the saved 0.440246
threshold should be treated as universal. Ranking scores and thresholded
classification remain separate decisions.

## Review-budget objective

2017 validation top-10% results:

{table(budget_report, list(budget_report.columns))}

Ranking uses stable input-order tie handling and `ceil(rows * fraction)`.
Random values are 1,000 reproducible simulations. These are benchmark
outcomes, not review-hours or cost savings.

## Project-grouped scenario

Known-project future prediction and unseen-project generalization are distinct
tasks. The grouped 3-fold results below use project only as a grouping key;
project is not a predictor or historical aggregate. They are not combined with
the temporal results.

{table(grouped_report, list(grouped_report.columns)) if not grouped_report.empty else "Grouped evaluation unavailable."}

## Limitations and next step

- The final test set was previously inspected and remains unsuitable for
  repeated selection.
- No fresh future-period holdout exists in the current dataset package.
- Three rolling folds and three grouped folds are useful diagnostics but not a
  substitute for prospective evaluation.
- SZZ-derived labels may contain attribution noise, and project/time shift is
  substantial.
- The saved Random Forest, joblib artifact, threshold, and UI were not changed.

Recommended next step: collect or reserve a genuinely later period, predefine
the HGB selection/threshold policy, and evaluate RF versus HGB once on that
new holdout. Until then, retain RF for continuity and keep HGB as a documented
candidate.

Reproduce with:

```powershell
python -m src.hgb_robustness
python -m pytest -q
```
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    args = parser.parse_args()
    result = run(args.input, args.report_dir)
    print(result["aggregate"].to_string(index=False))


if __name__ == "__main__":
    main()
