"""Validation-only experiments for diagnosing ReviewRadar's error profile.

This module deliberately never loads the final test split.  It compares a
small, pre-declared set of alternatives on the existing train/validation
protocol, including threshold selection on validation only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

try:
    from .data import DEFAULT_DATA_PATH, load_apachejit
    from .pipeline import build_numeric_pipeline
    from .split import DEFAULT_REPORT_DIR, SAFE_NUMERIC_FEATURES, make_temporal_splits
    from .train import RANDOM_STATE, positive_probability
    from .evaluate import choose_threshold_by_f1
except ImportError:
    from data import DEFAULT_DATA_PATH, load_apachejit
    from pipeline import build_numeric_pipeline
    from split import DEFAULT_REPORT_DIR, SAFE_NUMERIC_FEATURES, make_temporal_splits
    from train import RANDOM_STATE, positive_probability
    from evaluate import choose_threshold_by_f1


def experiment_pipelines():
    """Return a deliberately small, reproducible validation experiment grid."""

    return {
        "logistic_unbalanced": build_numeric_pipeline(
            LogisticRegression(class_weight=None, max_iter=2000, random_state=RANDOM_STATE), scale=True
        ),
        "logistic_balanced": build_numeric_pipeline(
            LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE), scale=True
        ),
        "rf_unbalanced_leaf2": build_numeric_pipeline(
            RandomForestClassifier(
                class_weight=None, min_samples_leaf=2, n_estimators=150,
                n_jobs=-1, random_state=RANDOM_STATE
            ), scale=False
        ),
        "rf_balanced_leaf1": build_numeric_pipeline(
            RandomForestClassifier(
                class_weight="balanced_subsample", min_samples_leaf=1, n_estimators=150,
                n_jobs=-1, random_state=RANDOM_STATE
            ), scale=False
        ),
        "rf_balanced_leaf5": build_numeric_pipeline(
            RandomForestClassifier(
                class_weight="balanced_subsample", min_samples_leaf=5, n_estimators=150,
                n_jobs=-1, random_state=RANDOM_STATE
            ), scale=False
        ),
    }


def run_experiments(data_path: Path = DEFAULT_DATA_PATH, report_dir: Path = DEFAULT_REPORT_DIR) -> pd.DataFrame:
    frame = load_apachejit(data_path)
    splits = make_temporal_splits(frame)
    train, validation = splits.train, splits.validation
    rows = []
    for name, pipeline in experiment_pipelines().items():
        pipeline.fit(train[SAFE_NUMERIC_FEATURES], train["buggy"])
        scores = positive_probability(pipeline, validation[SAFE_NUMERIC_FEATURES])
        threshold, _ = choose_threshold_by_f1(validation["buggy"], scores)
        predictions = scores >= threshold
        rows.append(
            {
                "experiment": name,
                "threshold_selected_on": "validation",
                "threshold": threshold,
                "precision_at_threshold": precision_score(validation["buggy"], predictions, zero_division=0),
                "recall_at_threshold": recall_score(validation["buggy"], predictions, zero_division=0),
                "f1_at_threshold": f1_score(validation["buggy"], predictions, zero_division=0),
                "average_precision": average_precision_score(validation["buggy"], scores),
                "validation_rows": len(validation),
            }
        )
    result = pd.DataFrame(rows).sort_values(
        ["f1_at_threshold", "average_precision", "experiment"], ascending=[False, False, True], ignore_index=True
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(report_dir / "model_experiments.csv", index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    args = parser.parse_args()
    print(run_experiments(args.input, args.report_dir).to_string(index=False))


if __name__ == "__main__":
    main()
