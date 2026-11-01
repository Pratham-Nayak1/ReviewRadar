"""Validation-only F1/ranking research cycle for ReviewRadar.

The final test split is never scored or used for model/threshold development.
This runner uses the existing chronological train/validation protocol.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

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


def log1p_logistic() -> Pipeline:
    estimator = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE)
    base = build_numeric_pipeline(estimator, scale=False)
    return Pipeline(
        steps=[
            ("preprocess", base.named_steps["preprocess"]),
            ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scaler", StandardScaler()),
            ("estimator", estimator),
        ]
    )


def research_models() -> dict[str, tuple[Pipeline, dict]]:
    return {
        "dummy_prior": (build_numeric_pipeline(DummyClassifier(strategy="prior")), {"strategy": "prior"}),
        "logistic_balanced": (
            build_numeric_pipeline(LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE), scale=True),
            {"class_weight": "balanced", "max_iter": 2000, "scaling": "standard"},
        ),
        "logistic_unbalanced": (
            build_numeric_pipeline(LogisticRegression(class_weight=None, max_iter=2000, random_state=RANDOM_STATE), scale=True),
            {"class_weight": None, "max_iter": 2000, "scaling": "standard"},
        ),
        "logistic_log1p": (log1p_logistic(), {"class_weight": "balanced", "log1p": True, "scaling": "standard"}),
        "random_forest_current": (
            build_numeric_pipeline(RandomForestClassifier(class_weight="balanced_subsample", min_samples_leaf=2, n_estimators=300, n_jobs=-1, random_state=RANDOM_STATE)),
            {"class_weight": "balanced_subsample", "min_samples_leaf": 2, "n_estimators": 300},
        ),
        "random_forest_unbalanced": (
            build_numeric_pipeline(RandomForestClassifier(class_weight=None, min_samples_leaf=2, n_estimators=150, n_jobs=-1, random_state=RANDOM_STATE)),
            {"class_weight": None, "min_samples_leaf": 2, "n_estimators": 150},
        ),
        "random_forest_leaf1": (
            build_numeric_pipeline(RandomForestClassifier(class_weight="balanced_subsample", min_samples_leaf=1, n_estimators=150, n_jobs=-1, random_state=RANDOM_STATE)),
            {"class_weight": "balanced_subsample", "min_samples_leaf": 1, "n_estimators": 150},
        ),
        "random_forest_leaf5": (
            build_numeric_pipeline(RandomForestClassifier(class_weight="balanced_subsample", min_samples_leaf=5, n_estimators=150, n_jobs=-1, random_state=RANDOM_STATE)),
            {"class_weight": "balanced_subsample", "min_samples_leaf": 5, "n_estimators": 150},
        ),
        "extra_trees": (
            build_numeric_pipeline(ExtraTreesClassifier(class_weight="balanced", min_samples_leaf=2, n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE)),
            {"class_weight": "balanced", "min_samples_leaf": 2, "n_estimators": 200},
        ),
        "hist_gradient_boosting": (
            build_numeric_pipeline(HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0, random_state=RANDOM_STATE)),
            {"max_iter": 200, "learning_rate": 0.05, "max_leaf_nodes": 31, "l2_regularization": 1.0},
        ),
    }


def metric_row(name: str, labels: pd.Series, scores: pd.Series, threshold: float, config: dict, runtime_seconds: float) -> dict:
    predicted = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[False, True]).ravel()
    return {
        "experiment": name,
        "hypothesis": "Validation-only candidate comparison under the fixed chronological protocol",
        "split": "train_before_2017_validation_2017",
        "preprocessing": "median_imputer_inside_pipeline; scaling/model-specific",
        "configuration": json.dumps(config, sort_keys=True),
        "random_seed": RANDOM_STATE,
        "runtime_seconds": runtime_seconds,
        "threshold": float(threshold),
        "threshold_selection": "validation_max_f1",
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "average_precision": float(average_precision_score(labels, scores)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def run_research(data_path: Path = DEFAULT_DATA_PATH, report_dir: Path = DEFAULT_REPORT_DIR) -> pd.DataFrame:
    frame = load_apachejit(data_path)
    splits = make_temporal_splits(frame)
    train, validation = splits.train, splits.validation
    report_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = report_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    fitted: dict[str, tuple[pd.Series, float]] = {}
    rows = []
    for name, (pipeline, config) in research_models().items():
        started = time.perf_counter()
        pipeline.fit(train[SAFE_NUMERIC_FEATURES], train["buggy"])
        scores = positive_probability(pipeline, validation[SAFE_NUMERIC_FEATURES])
        threshold, _ = choose_threshold_by_f1(validation["buggy"], scores)
        runtime = time.perf_counter() - started
        rows.append(metric_row(name, validation["buggy"], scores, threshold, config, runtime))
        fitted[name] = (scores, threshold)

    results = pd.DataFrame(rows).sort_values(["f1", "average_precision", "experiment"], ascending=[False, False, True], ignore_index=True)
    results.to_csv(report_dir / "f1_experiments.csv", index=False)

    baseline_scores, _ = fitted["random_forest_current"]
    sweep_rows = []
    thresholds = sorted(set(np.round(np.arange(0.05, 0.951, 0.01), 6).tolist() + [0.5]))
    for threshold in thresholds:
        predicted = baseline_scores >= threshold
        tn, fp, fn, tp = confusion_matrix(validation["buggy"], predicted, labels=[False, True]).ravel()
        sweep_rows.append({
            "threshold": threshold,
            "precision": precision_score(validation["buggy"], predicted, zero_division=0),
            "recall": recall_score(validation["buggy"], predicted, zero_division=0),
            "f1": f1_score(validation["buggy"], predicted, zero_division=0),
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "true_positive": tp,
        })
    sweep = pd.DataFrame(sweep_rows)
    sweep.to_csv(report_dir / "f1_threshold_sweep.csv", index=False)
    selected_threshold = float(results.loc[results.experiment == "random_forest_current", "threshold"].iloc[0])
    fig, ax = plt.subplots(figsize=(8, 5))
    for metric, color in (("precision", "#4C78A8"), ("recall", "#F58518"), ("f1", "#54A24B")):
        ax.plot(sweep["threshold"], sweep[metric], label=metric, color=color)
    ax.axvline(selected_threshold, linestyle="--", color="#E45756", label=f"selected={selected_threshold:.3f}")
    ax.set(xlabel="Decision threshold", ylabel="Validation metric", title="Current Random Forest threshold trade-off")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "f1_threshold_sweep.png", dpi=160)
    plt.close(fig)

    ranking_frames = []
    for candidate_name in ("random_forest_current", "hist_gradient_boosting", "extra_trees"):
        candidate_scores, _ = fitted[candidate_name]
        candidate_ranking = compare_review_strategies(
            candidate_scores,
            validation["buggy"],
            (0.05, 0.10, 0.20),
            heuristic_scores=validation["la"] + validation["ld"],
            heuristic_name="change_size_la_plus_ld",
            random_runs=1000,
            random_state=RANDOM_STATE,
        )
        candidate_ranking.insert(0, "model", candidate_name)
        candidate_ranking["split"] = "validation"
        ranking_frames.append(candidate_ranking)
    pd.concat(ranking_frames, ignore_index=True).to_csv(report_dir / "f1_ranking_validation.csv", index=False)
    run_temporal_cv(train, report_dir)
    return results


def run_temporal_cv(train: pd.DataFrame, report_dir: Path) -> pd.DataFrame:
    """Assess two leading candidates on earlier rolling-origin folds only."""

    dates = author_dates_utc(train)
    years = dates.dt.year
    fold_rows = []
    for validation_year in (2014, 2015, 2016):
        fit = train[years < validation_year]
        fold = train[years == validation_year]
        if fit.empty or fold.empty:
            continue
        models = {
            "random_forest_current": build_numeric_pipeline(
                RandomForestClassifier(class_weight="balanced_subsample", min_samples_leaf=2, n_estimators=150, n_jobs=-1, random_state=RANDOM_STATE)
            ),
            "hist_gradient_boosting": build_numeric_pipeline(
                HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0, random_state=RANDOM_STATE)
            ),
        }
        for name, pipeline in models.items():
            pipeline.fit(fit[SAFE_NUMERIC_FEATURES], fit["buggy"])
            scores = positive_probability(pipeline, fold[SAFE_NUMERIC_FEATURES])
            threshold, _ = choose_threshold_by_f1(fold["buggy"], scores)
            row = metric_row(name, fold["buggy"], scores, threshold, {"rolling_validation_year": validation_year}, 0.0)
            row["fold_train_end_year"] = validation_year - 1
            row["fold_validation_year"] = validation_year
            row["fold_train_rows"] = len(fit)
            row["fold_validation_rows"] = len(fold)
            fold_rows.append(row)
    result = pd.DataFrame(fold_rows)
    result.to_csv(report_dir / "f1_temporal_cv.csv", index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    args = parser.parse_args()
    print(run_research(args.input, args.report_dir).to_string(index=False))


if __name__ == "__main__":
    main()
