"""Leakage-safe ReviewRadar evaluation.

Validation is used for candidate comparison and threshold selection. The final
test set is touched only after those choices are frozen. The positive class is
``buggy=True``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

try:
    from .data import DEFAULT_DATA_PATH, load_apachejit
    from .review_budget import compare_review_strategies
    from .split import DEFAULT_REPORT_DIR, SAFE_NUMERIC_FEATURES, make_temporal_splits
    from .train import (
        DEFAULT_MODEL_DIR,
        DEFAULT_THRESHOLD,
        RANDOM_STATE,
        candidate_pipelines,
        positive_probability,
    )
except ImportError:
    from data import DEFAULT_DATA_PATH, load_apachejit
    from review_budget import compare_review_strategies
    from split import DEFAULT_REPORT_DIR, SAFE_NUMERIC_FEATURES, make_temporal_splits
    from train import (
        DEFAULT_MODEL_DIR,
        DEFAULT_THRESHOLD,
        RANDOM_STATE,
        candidate_pipelines,
        positive_probability,
    )


DEFAULT_REVIEW_FRACTION = 0.10
DEFAULT_REVIEW_FRACTIONS = (0.05, 0.10, 0.20)
DEFAULT_RANDOM_RUNS = 1000
POSITIVE_CLASS = True


def choose_threshold_by_f1(labels: pd.Series, probabilities: pd.Series) -> tuple[float, float]:
    """Choose the validation threshold with the highest F1.

    Ties are resolved by choosing the threshold closest to 0.5, then the
    lower threshold. If the validation data cannot produce a threshold, the
    fixed default threshold is returned.
    """

    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    if len(thresholds) == 0:
        return DEFAULT_THRESHOLD, float(f1_score(labels, probabilities >= DEFAULT_THRESHOLD, zero_division=0))
    f1_values = np.divide(
        2 * precision[:-1] * recall[:-1],
        precision[:-1] + recall[:-1],
        out=np.zeros_like(precision[:-1]),
        where=(precision[:-1] + recall[:-1]) != 0,
    )
    best_f1 = float(np.max(f1_values))
    candidates = np.flatnonzero(np.isclose(f1_values, best_f1))
    best_index = min(candidates, key=lambda index: (abs(float(thresholds[index]) - 0.5), float(thresholds[index])))
    return float(thresholds[best_index]), best_f1


def classification_metrics(
    labels: pd.Series,
    probabilities: pd.Series,
    threshold: float,
) -> tuple[dict[str, Any], dict[str, int]]:
    """Return threshold metrics and a labeled confusion matrix."""

    predictions = probabilities >= threshold
    matrix = confusion_matrix(labels, predictions, labels=[False, True])
    tn, fp, fn, tp = matrix.ravel()
    metrics = {
        "rows": int(len(labels)),
        "positive_class": "buggy=True",
        "threshold": float(threshold),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "pr_auc_average_precision": float(average_precision_score(labels, probabilities)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "predicted_positive_rate": float(predictions.mean()),
    }
    matrix_values = {"true_negative": int(tn), "false_positive": int(fp), "false_negative": int(fn), "true_positive": int(tp)}
    return metrics, matrix_values


def expected_calibration_error(labels: pd.Series, probabilities: pd.Series, bins: int = 10) -> float:
    """Calculate a simple equal-width expected calibration error."""

    labels_array = np.asarray(labels, dtype=float)
    probabilities_array = np.asarray(probabilities, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(labels_array)
    error = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        if index == bins - 1:
            mask = (probabilities_array >= lower) & (probabilities_array <= upper)
        else:
            mask = (probabilities_array >= lower) & (probabilities_array < upper)
        if not np.any(mask):
            continue
        error += float(mask.sum() / total) * abs(float(labels_array[mask].mean()) - float(probabilities_array[mask].mean()))
    return error


def calibration_rows(labels: pd.Series, probabilities: pd.Series, split: str, bins: int = 10) -> pd.DataFrame:
    fraction_true, mean_probability = calibration_curve(labels, probabilities, n_bins=bins, strategy="uniform")
    return pd.DataFrame(
        {
            "split": split,
            "bin": range(len(fraction_true)),
            "mean_predicted_probability": mean_probability,
            "fraction_buggy": fraction_true,
        }
    )


def plot_pr_curve(records: dict[str, tuple[pd.Series, pd.Series]], path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, (labels, probabilities) in records.items():
        precision, recall, _ = precision_recall_curve(labels, probabilities)
        ap = average_precision_score(labels, probabilities)
        ax.plot(recall, precision, label=f"{name} (AP={ap:.3f})")
    ax.set_title(title)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_calibration(labels: pd.Series, probabilities: pd.Series, path: Path, title: str) -> None:
    fraction_true, mean_probability = calibration_curve(labels, probabilities, n_bins=10, strategy="uniform")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    ax.plot(mean_probability, fraction_true, marker="o", label="ReviewRadar")
    ax.set_title(title)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed fraction buggy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_review_budget(metrics: pd.DataFrame, path: Path) -> None:
    pivot = metrics.pivot(index=["approach", "review_fraction"], columns="metric", values="value")
    pivot.index = [f"{approach} ({fraction:.0%})" for approach, fraction in pivot.index]
    ax = pivot[["precision_at_k", "recall_at_k"]].plot(kind="bar", figsize=(8, 5), color=["#4C78A8", "#E45756"])
    ax.set_title("Review prioritization at the configured budget")
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1)
    ax.legend(["Precision@K", "Recall@K"])
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def run_evaluation(
    data_path: Path = DEFAULT_DATA_PATH,
    report_dir: Path = DEFAULT_REPORT_DIR,
    model_dir: Path = DEFAULT_MODEL_DIR,
    review_fraction: float = DEFAULT_REVIEW_FRACTION,
) -> None:
    """Run validation selection, fixed-threshold test evaluation, and reports."""

    report_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = report_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    frame = load_apachejit(data_path)
    splits = make_temporal_splits(frame)
    train, validation, test = splits.train, splits.validation, splits.test

    # Candidate comparison: every candidate is fit on train only.
    candidate_validation: dict[str, tuple[pd.Series, pd.Series]] = {}
    candidate_rows: list[dict[str, Any]] = []
    fitted_candidates = {}
    for name, (pipeline, config) in candidate_pipelines().items():
        pipeline.fit(train[SAFE_NUMERIC_FEATURES], train["buggy"])
        probabilities = positive_probability(pipeline, validation[SAFE_NUMERIC_FEATURES])
        metrics, matrix = classification_metrics(validation["buggy"], probabilities, DEFAULT_THRESHOLD)
        candidate_validation[name] = (validation["buggy"], probabilities)
        fitted_candidates[name] = pipeline
        candidate_rows.append({"model": name, "configuration": json.dumps(config, sort_keys=True), "threshold_source": "fixed_default", **metrics, **matrix})

    comparison = pd.DataFrame(candidate_rows).sort_values(
        ["pr_auc_average_precision", "f1", "model"], ascending=[False, False, True], ignore_index=True
    )
    selected_name = str(comparison.iloc[0]["model"])
    selected_validation_probabilities = candidate_validation[selected_name][1]
    selected_threshold, selected_threshold_f1 = choose_threshold_by_f1(validation["buggy"], selected_validation_probabilities)
    tuned_validation_metrics, tuned_validation_matrix = classification_metrics(validation["buggy"], selected_validation_probabilities, selected_threshold)

    # Freeze model and threshold here. No test labels or scores are used below
    # to revise either decision.
    refit = pd.concat([train, validation], ignore_index=True)
    selected_pipeline, selected_config = candidate_pipelines()[selected_name]
    selected_pipeline.fit(refit[SAFE_NUMERIC_FEATURES], refit["buggy"])
    test_probabilities = positive_probability(selected_pipeline, test[SAFE_NUMERIC_FEATURES])
    test_metrics, test_matrix = classification_metrics(test["buggy"], test_probabilities, selected_threshold)

    classification_rows = comparison.to_dict("records")
    classification_rows.extend(
        [
            {"model": f"{selected_name}_tuned", "configuration": json.dumps(selected_config, sort_keys=True), "threshold_source": "validation_max_f1", **tuned_validation_metrics, **tuned_validation_matrix},
            {"model": f"{selected_name}_final_test", "configuration": json.dumps(selected_config, sort_keys=True), "threshold_source": "frozen_from_validation", **test_metrics, **test_matrix},
        ]
    )
    pd.DataFrame(classification_rows).to_csv(report_dir / "evaluation_classification_metrics.csv", index=False)
    pd.DataFrame(
        [
            {"split": "validation", "model": selected_name, "true_negative": tuned_validation_matrix["true_negative"], "false_positive": tuned_validation_matrix["false_positive"], "false_negative": tuned_validation_matrix["false_negative"], "true_positive": tuned_validation_matrix["true_positive"], "threshold": selected_threshold},
            {"split": "test", "model": selected_name, "true_negative": test_matrix["true_negative"], "false_positive": test_matrix["false_positive"], "false_negative": test_matrix["false_negative"], "true_positive": test_matrix["true_positive"], "threshold": selected_threshold},
        ]
    ).to_csv(report_dir / "evaluation_confusion_matrices.csv", index=False)

    budget_fractions = tuple(DEFAULT_REVIEW_FRACTIONS)
    validation_budget = compare_review_strategies(
        selected_validation_probabilities,
        validation["buggy"],
        budget_fractions,
        heuristic_scores=validation["la"] + validation["ld"],
        heuristic_name="change_size_la_plus_ld",
        random_runs=DEFAULT_RANDOM_RUNS,
        random_state=RANDOM_STATE,
    )
    test_budget = compare_review_strategies(
        test_probabilities,
        test["buggy"],
        budget_fractions,
        heuristic_scores=test["la"] + test["ld"],
        heuristic_name="change_size_la_plus_ld",
        random_runs=DEFAULT_RANDOM_RUNS,
        random_state=RANDOM_STATE,
    )
    validation_budget["split"] = "validation"
    test_budget["split"] = "test"
    budget_frame = pd.concat([validation_budget, test_budget], ignore_index=True)
    budget_frame.to_csv(report_dir / "evaluation_review_budget.csv", index=False)
    plot_budget = budget_frame.melt(id_vars=["split", "approach", "review_fraction"], value_vars=["precision_at_k", "recall_at_k"], var_name="metric", value_name="value")
    plot_review_budget(plot_budget[plot_budget["split"] == "validation"], figure_dir / "review_budget_validation.png")
    plot_review_budget(plot_budget[plot_budget["split"] == "test"], figure_dir / "review_budget_test.png")

    calibration_frame = pd.concat(
        [calibration_rows(validation["buggy"], selected_validation_probabilities, "validation"), calibration_rows(test["buggy"], test_probabilities, "test")],
        ignore_index=True,
    )
    calibration_frame.to_csv(report_dir / "evaluation_calibration_curve.csv", index=False)
    calibration_summary = pd.DataFrame(
        [
            {"split": "validation", "brier_score": tuned_validation_metrics["brier_score"], "expected_calibration_error": expected_calibration_error(validation["buggy"], selected_validation_probabilities)},
            {"split": "test", "brier_score": test_metrics["brier_score"], "expected_calibration_error": expected_calibration_error(test["buggy"], test_probabilities)},
        ]
    )
    calibration_summary.to_csv(report_dir / "evaluation_calibration_metrics.csv", index=False)

    plot_pr_curve(candidate_validation, figure_dir / "pr_curve_validation.png", "Validation precision-recall curves")
    plot_pr_curve({selected_name: (test["buggy"], test_probabilities)}, figure_dir / "pr_curve_test.png", "Held-out test precision-recall curve")
    plot_calibration(validation["buggy"], selected_validation_probabilities, figure_dir / "calibration_validation.png", "Validation calibration")
    plot_calibration(test["buggy"], test_probabilities, figure_dir / "calibration_test.png", "Held-out test calibration")

    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(selected_pipeline, model_dir / "reviewradar_selected.joblib")
    config = {
        "positive_class": "buggy=True",
        "review_fraction": review_fraction,
        "review_budgets": list(budget_fractions),
        "review_k_validation": int(np.ceil(len(validation) * review_fraction)),
        "review_k_test": int(np.ceil(len(test) * review_fraction)),
        "random_selection_simulations": DEFAULT_RANDOM_RUNS,
        "selected_model": selected_name,
        "selected_configuration": selected_config,
        "selection_metric": "validation_pr_auc_average_precision",
        "threshold_metric": "validation_f1",
        "selected_threshold": selected_threshold,
        "validation_threshold_f1": selected_threshold_f1,
        "random_seed": RANDOM_STATE,
        "feature_columns": SAFE_NUMERIC_FEATURES,
        "excluded_columns": ["commit_id", "buggy", "fix", "year", "author_date", "project"],
        "test_used_for_selection": False,
    }
    (report_dir / "evaluation_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    summary = [
        "# ReviewRadar evaluation summary",
        "",
        "Positive class: **`buggy=True`**.",
        "",
        f"Selected model: **`{selected_name}`** using validation PR-AUC. Threshold selected on validation by maximum F1: **{selected_threshold:.6f}**.",
        "",
        "The held-out test was evaluated once after the model and threshold were frozen. Test results are historical benchmark results on ApacheJIT, not evidence that the model prevents bugs or will generalize to other repositories.",
        "",
        "Undefined precision/recall/F1 cases are handled with `zero_division=0`. In particular, the DummyClassifier predicts no positive rows at threshold 0.5, so those three metrics are reported as 0; this does not mean the validation set contains no buggy changes.",
        "",
        f"Review-budget analysis reports top **{', '.join(f'{fraction:.0%}' for fraction in budget_fractions)}** of changes. Random selection is a reproducible {DEFAULT_RANDOM_RUNS}-run simulation (seed={RANDOM_STATE}); it is a benchmark comparison, not a measured estimate of money, bugs prevented, or review hours saved.",
        "",
        "See `evaluation_classification_metrics.csv`, `evaluation_review_budget.csv`, `evaluation_calibration_metrics.csv`, and `evaluation_config.json` for actual outputs.",
        "",
    ]
    (report_dir / "evaluation_summary.md").write_text("\n".join(summary), encoding="utf-8")
    print(json.dumps({"selected_model": selected_name, "selected_threshold": selected_threshold, "validation": tuned_validation_metrics, "test": test_metrics}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--review-fraction", type=float, default=DEFAULT_REVIEW_FRACTION)
    args = parser.parse_args()
    run_evaluation(args.input, args.report_dir, args.model_dir, args.review_fraction)


if __name__ == "__main__":
    main()

