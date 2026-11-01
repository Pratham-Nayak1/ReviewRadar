"""Curated, held-out examples and behavior-based explanations for the Demo Lab.

Examples are selected from the frozen chronological test split after model and
threshold decisions were made. They are never used to fit, select, or tune the
model. Explanations use a leave-one-feature-at-the-training-median sensitivity
calculation, not a causal claim.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from functools import lru_cache

import pandas as pd

from .data import DEFAULT_DATA_PATH, load_apachejit
from .predict import DEFAULT_CONFIG_PATH, DEFAULT_MODEL_PATH, ReviewRadarPredictor
from .split import SAFE_NUMERIC_FEATURES, make_temporal_splits


OUTCOME_LABELS = {
    "true_positive": "True positive",
    "true_negative": "True negative",
    "false_positive": "False positive",
    "false_negative": "False negative",
}


def _predict_score(predictor: ReviewRadarPredictor, frame: pd.DataFrame) -> float:
    return float(predictor.predict(frame[SAFE_NUMERIC_FEATURES]).iloc[0]["risk_score"])


def explain_record(
    predictor: ReviewRadarPredictor,
    record: pd.Series,
    training_medians: pd.Series,
    top_n: int = 5,
) -> list[dict[str, Any]]:
    """Return local score sensitivities relative to training medians.

    A positive delta means the observed value produced a higher score than the
    same record with that one feature replaced by its training median. This is
    an approximate behavioral explanation, not causation.
    """

    base_frame = record[SAFE_NUMERIC_FEATURES].to_frame().T
    base_score = _predict_score(predictor, base_frame)
    rows: list[dict[str, Any]] = []
    for feature in SAFE_NUMERIC_FEATURES:
        counterfactual = base_frame.copy()
        counterfactual.loc[counterfactual.index[0], feature] = training_medians[feature]
        counterfactual_score = _predict_score(predictor, counterfactual)
        delta = base_score - counterfactual_score
        rows.append(
            {
                "feature": feature,
                "observed_value": record[feature],
                "training_median": training_medians[feature],
                "score_delta_vs_median": delta,
                "direction": "higher score" if delta >= 0 else "lower score",
            }
        )
    return sorted(rows, key=lambda row: abs(float(row["score_delta_vs_median"])), reverse=True)[:top_n]


@lru_cache(maxsize=4)
def load_demo_cases(
    data_path: Path = DEFAULT_DATA_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> list[dict[str, Any]]:
    """Build one deterministic case for each available outcome category."""

    frame = load_apachejit(data_path)
    splits = make_temporal_splits(frame)
    predictor = ReviewRadarPredictor(model_path, config_path)
    test = splits.test.copy()
    predictions = predictor.predict(test[["commit_id", *SAFE_NUMERIC_FEATURES]])
    test["risk_score"] = predictions["risk_score"].to_numpy()
    threshold = float(predictor.threshold) if predictor.threshold is not None else 0.5
    test["predicted_buggy"] = test["risk_score"] >= threshold
    test["outcome_key"] = test.apply(
        lambda row: (
            "true_positive" if bool(row.buggy) and bool(row.predicted_buggy)
            else "true_negative" if not bool(row.buggy) and not bool(row.predicted_buggy)
            else "false_positive" if not bool(row.buggy) and bool(row.predicted_buggy)
            else "false_negative"
        ),
        axis=1,
    )
    training_medians = splits.train[SAFE_NUMERIC_FEATURES].median(numeric_only=True)
    selections = {
        "true_positive": test[test.outcome_key == "true_positive"].sort_values(["risk_score", "commit_id"], ascending=[False, True]),
        "true_negative": test[test.outcome_key == "true_negative"].sort_values(["risk_score", "commit_id"], ascending=[False, True]),
        "false_positive": test[test.outcome_key == "false_positive"].sort_values(["risk_score", "commit_id"], ascending=[False, True]),
        "false_negative": test[test.outcome_key == "false_negative"].sort_values(["risk_score", "commit_id"], ascending=[True, True]),
    }
    cases: list[dict[str, Any]] = []
    for outcome_key, candidates in selections.items():
        if candidates.empty:
            continue
        row = candidates.iloc[0]
        cases.append(
            {
                "case_id": f"apachejit-test-{outcome_key}-1",
                "source": "ApacheJIT held-out chronological test split",
                "commit_id": str(row["commit_id"]),
                "outcome_key": outcome_key,
                "outcome": OUTCOME_LABELS[outcome_key],
                "actual_buggy": bool(row["buggy"]),
                "predicted_buggy": bool(row["predicted_buggy"]),
                "risk_score": float(row["risk_score"]),
                "threshold": threshold,
                "project": str(row["project"]),
                "features": {feature: row[feature] for feature in SAFE_NUMERIC_FEATURES},
                "explanation": explain_record(predictor, row, training_medians),
            }
        )

    borderline = test.assign(distance=(test["risk_score"] - threshold).abs()).sort_values(["distance", "commit_id"])
    if not borderline.empty:
        row = borderline.iloc[0]
        cases.append(
            {
                "case_id": "apachejit-test-borderline-1",
                "source": "ApacheJIT held-out chronological test split",
                "commit_id": str(row["commit_id"]),
                "outcome_key": "borderline",
                "outcome": "Borderline case",
                "actual_buggy": bool(row["buggy"]),
                "predicted_buggy": bool(row["predicted_buggy"]),
                "risk_score": float(row["risk_score"]),
                "threshold": threshold,
                "project": str(row["project"]),
                "features": {feature: row[feature] for feature in SAFE_NUMERIC_FEATURES},
                "explanation": explain_record(predictor, row, training_medians),
            }
        )
    return cases
