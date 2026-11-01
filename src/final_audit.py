"""Read-only, validation-focused evidence audit for ReviewRadar.

This script reads the test split for prevalence/distribution/duplicate audits,
but never fits a model, selects a threshold, or scores predictions on it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

try:
    from .data import DEFAULT_DATA_PATH, load_apachejit
    from .evaluate import choose_threshold_by_f1
    from .pipeline import build_numeric_pipeline
    from .split import DEFAULT_REPORT_DIR, SAFE_NUMERIC_FEATURES, make_temporal_splits, split_summary
    from .train import positive_probability
except ImportError:
    from data import DEFAULT_DATA_PATH, load_apachejit
    from evaluate import choose_threshold_by_f1
    from pipeline import build_numeric_pipeline
    from split import DEFAULT_REPORT_DIR, SAFE_NUMERIC_FEATURES, make_temporal_splits, split_summary
    from train import positive_probability


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models" / "reviewradar_selected.joblib"
DEFAULT_CONFIG = ROOT / "reports" / "evaluation_config.json"


def _metric_row(name: str, labels: pd.Series, scores: pd.Series, threshold: float) -> dict:
    predicted = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[False, True]).ravel()
    return {
        "model": name,
        "threshold": float(threshold),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "average_precision": float(average_precision_score(labels, scores)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def run_audit(data_path: Path = DEFAULT_DATA_PATH, report_dir: Path = DEFAULT_REPORT_DIR) -> dict:
    frame = load_apachejit(data_path)
    splits = make_temporal_splits(frame)
    train, validation, test = splits.train, splits.validation, splits.test

    # Fit only on train and score only validation. This reconstructs the
    # candidate experiment without touching test predictions.
    candidates = {
        "dummy_prior": build_numeric_pipeline(DummyClassifier(strategy="prior")),
        "logistic_balanced": build_numeric_pipeline(LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42), scale=True),
        "rf_balanced_leaf5": build_numeric_pipeline(RandomForestClassifier(class_weight="balanced_subsample", min_samples_leaf=5, n_estimators=150, n_jobs=-1, random_state=42)),
    }
    validation_metrics = []
    for name, pipeline in candidates.items():
        pipeline.fit(train[SAFE_NUMERIC_FEATURES], train["buggy"])
        scores = positive_probability(pipeline, validation[SAFE_NUMERIC_FEATURES])
        threshold = 0.5
        if name == "rf_balanced_leaf5":
            threshold, _ = choose_threshold_by_f1(validation["buggy"], scores)
        validation_metrics.append(_metric_row(name, validation["buggy"], scores, threshold))

    # Error-pattern summary for the saved model's train-only validation fit.
    saved_rf = build_numeric_pipeline(
        RandomForestClassifier(
            class_weight="balanced_subsample", min_samples_leaf=2,
            n_estimators=300, n_jobs=-1, random_state=42,
        )
    )
    saved_rf.fit(train[SAFE_NUMERIC_FEATURES], train["buggy"])
    saved_scores = positive_probability(saved_rf, validation[SAFE_NUMERIC_FEATURES])
    saved_threshold = 0.4402460058475116
    validation_errors = validation[["project", "buggy"] + SAFE_NUMERIC_FEATURES].copy()
    validation_errors["score"] = saved_scores.to_numpy()
    validation_errors["predicted_buggy"] = validation_errors["score"] >= saved_threshold
    validation_errors["outcome"] = np.select(
        [validation_errors.buggy & validation_errors.predicted_buggy,
         ~validation_errors.buggy & ~validation_errors.predicted_buggy,
         ~validation_errors.buggy & validation_errors.predicted_buggy],
        ["true_positive", "true_negative", "false_positive"],
        default="false_negative",
    )
    error_group = validation_errors.groupby("outcome")[SAFE_NUMERIC_FEATURES].median().reset_index()
    error_counts = validation_errors["outcome"].value_counts().rename_axis("outcome").reset_index(name="rows")
    error_group = error_counts.merge(error_group, on="outcome")
    error_group.to_csv(report_dir / "final_audit_error_groups.csv", index=False)
    project_errors = validation_errors.groupby(["project", "outcome"]).size().rename("rows").reset_index()
    project_errors.to_csv(report_dir / "final_audit_project_errors.csv", index=False)

    feature_rows = []
    train_std = train[SAFE_NUMERIC_FEATURES].std(ddof=0).replace(0, np.nan)
    for feature in SAFE_NUMERIC_FEATURES:
        row = {
            "feature": feature,
            "train_mean": float(train[feature].mean()),
            "validation_mean": float(validation[feature].mean()),
            "test_mean": float(test[feature].mean()),
            "train_median": float(train[feature].median()),
            "validation_median": float(validation[feature].median()),
            "test_median": float(test[feature].median()),
            "validation_standardized_mean_shift": float((validation[feature].mean() - train[feature].mean()) / train_std[feature]),
            "test_standardized_mean_shift": float((test[feature].mean() - train[feature].mean()) / train_std[feature]),
        }
        feature_rows.append(row)

    project_rows = []
    for split_name, part in (("train", train), ("validation", validation), ("test", test)):
        grouped = part.groupby("project", dropna=False)["buggy"].agg(rows="size", buggy_rate="mean")
        for project, row in grouped.iterrows():
            project_rows.append({"split": split_name, "project": project, "rows": int(row["rows"]), "buggy_rate": float(row["buggy_rate"])})

    split_rows = split_summary(splits).to_dict("records")
    split_duplicate_rows = {
        name: {"exact_duplicate_rows": int(part.duplicated().sum()), "duplicate_commit_ids": int(part["commit_id"].duplicated().sum())}
        for name, part in (("train", train), ("validation", validation), ("test", test))
    }
    cross_split_ids = {
        "train_validation": len(set(train.commit_id) & set(validation.commit_id)),
        "train_test": len(set(train.commit_id) & set(test.commit_id)),
        "validation_test": len(set(validation.commit_id) & set(test.commit_id)),
    }
    pipeline = joblib.load(DEFAULT_MODEL)
    config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    pipeline_schema = list(getattr(pipeline, "feature_names_in_", []))
    metadata = {
        "rows": len(frame),
        "columns": len(frame.columns),
        "positive_class": "buggy=True",
        "feature_columns": SAFE_NUMERIC_FEATURES,
        "pipeline_schema_matches_config": pipeline_schema == config.get("feature_columns"),
        "pipeline_steps": list(getattr(pipeline, "named_steps", {}).keys()),
        "test_predictions_scored_by_this_script": False,
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(validation_metrics).to_csv(report_dir / "final_audit_validation_metrics.csv", index=False)
    pd.DataFrame(feature_rows).to_csv(report_dir / "final_audit_feature_shift.csv", index=False)
    pd.DataFrame(project_rows).to_csv(report_dir / "final_audit_project_shift.csv", index=False)
    summary = {
        "metadata": metadata,
        "split_summary": split_rows,
        "split_duplicates": split_duplicate_rows,
        "cross_split_commit_id_overlap": cross_split_ids,
        "validation_metrics": validation_metrics,
        "saved_model_validation_threshold": saved_threshold,
    }
    (report_dir / "final_audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    args = parser.parse_args()
    print(json.dumps(run_audit(args.input, args.report_dir), indent=2))


if __name__ == "__main__":
    main()
