"""Train and compare the first ReviewRadar models.

The workflow is deliberately validation-only for model selection:

1. Load the complete ApacheJIT file.
2. Create the approved author_date train/validation/test split.
3. Fit each candidate on train only and score validation only.
4. Select by validation PR-AUC, with validation F1 as a deterministic tie-break.
5. Refit the selected pipeline on train + validation, never on final test rows.

The final test split is not loaded into the training workflow's feature or
label calculations. It remains reserved for a later, one-time evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

try:
    from .data import DEFAULT_DATA_PATH, load_apachejit
    from .pipeline import build_numeric_pipeline
    from .split import DEFAULT_REPORT_DIR, SAFE_NUMERIC_FEATURES, make_temporal_splits
except ImportError:
    from data import DEFAULT_DATA_PATH, load_apachejit
    from pipeline import build_numeric_pipeline
    from split import DEFAULT_REPORT_DIR, SAFE_NUMERIC_FEATURES, make_temporal_splits


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"
TARGET = "buggy"
RANDOM_STATE = 42
DEFAULT_THRESHOLD = 0.5


def candidate_pipelines() -> dict[str, tuple[Pipeline, dict[str, Any]]]:
    """Return reproducible candidate pipelines and their recorded configs."""

    candidates: dict[str, tuple[Pipeline, dict[str, Any]]] = {
        "dummy_prior": (
            build_numeric_pipeline(DummyClassifier(strategy="prior"), scale=False),
            {"strategy": "prior"},
        ),
        "logistic_regression": (
            build_numeric_pipeline(
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                    solver="lbfgs",
                ),
                scale=True,
            ),
            {
                "class_weight": "balanced",
                "max_iter": 2000,
                "random_state": RANDOM_STATE,
                "solver": "lbfgs",
                "scaling": "standard_scaler",
            },
        ),
        "random_forest": (
            build_numeric_pipeline(
                RandomForestClassifier(
                    class_weight="balanced_subsample",
                    min_samples_leaf=2,
                    n_estimators=300,
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
                scale=False,
            ),
            {
                "class_weight": "balanced_subsample",
                "min_samples_leaf": 2,
                "n_estimators": 300,
                "n_jobs": -1,
                "random_state": RANDOM_STATE,
            },
        ),
    }
    return candidates


def positive_probability(model: Pipeline, features: pd.DataFrame) -> pd.Series:
    """Return probability for the True/bug-inducing class."""

    probabilities = model.predict_proba(features)
    classes = list(model.named_steps["estimator"].classes_)
    try:
        positive_index = classes.index(True)
    except ValueError as exc:
        raise ValueError(f"Estimator does not expose the True class: {classes}") from exc
    return pd.Series(probabilities[:, positive_index], index=features.index, name="risk_probability")


def validation_metrics(model: Pipeline, validation: pd.DataFrame) -> dict[str, float]:
    """Calculate metrics at the fixed 0.5 threshold and PR-AUC on validation."""

    features = validation[SAFE_NUMERIC_FEATURES]
    labels = validation[TARGET]
    probabilities = positive_probability(model, features)
    predictions = probabilities >= DEFAULT_THRESHOLD
    return {
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "predicted_positive_rate": float(predictions.mean()),
    }


def train_and_compare(
    data_path: Path = DEFAULT_DATA_PATH,
    report_dir: Path = DEFAULT_REPORT_DIR,
    model_dir: Path = DEFAULT_MODEL_DIR,
) -> pd.DataFrame:
    """Train candidates, select on validation, and save the selected pipeline."""

    frame = load_apachejit(data_path)
    temporal = make_temporal_splits(frame)
    train = temporal.train
    validation = temporal.validation

    records: list[dict[str, Any]] = []
    fitted_candidates: dict[str, Pipeline] = {}
    for model_name, (pipeline, config) in candidate_pipelines().items():
        pipeline.fit(train[SAFE_NUMERIC_FEATURES], train[TARGET])
        fitted_candidates[model_name] = pipeline
        metrics = validation_metrics(pipeline, validation)
        records.append(
            {
                "model": model_name,
                "configuration": json.dumps(config, sort_keys=True),
                "feature_count": len(SAFE_NUMERIC_FEATURES),
                "features": ",".join(SAFE_NUMERIC_FEATURES),
                "train_rows": len(train),
                "validation_rows": len(validation),
                "threshold": DEFAULT_THRESHOLD,
                "selection_metric": "validation_pr_auc",
                "selected": False,
                **{f"validation_{key}": value for key, value in metrics.items()},
            }
        )

    comparison = pd.DataFrame(records).sort_values(
        by=["validation_pr_auc", "validation_f1", "model"],
        ascending=[False, False, True],
        ignore_index=True,
    )
    selected_name = str(comparison.iloc[0]["model"])
    comparison.loc[comparison["model"] == selected_name, "selected"] = True

    # Final model fitting uses only train + validation. The final test is not
    # loaded, scored, or used to select the model or threshold.
    refit_data = pd.concat([train, validation], ignore_index=True)
    selected_pipeline, _ = candidate_pipelines()[selected_name]
    selected_pipeline.fit(refit_data[SAFE_NUMERIC_FEATURES], refit_data[TARGET])
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "reviewradar_selected.joblib"
    joblib.dump(selected_pipeline, model_path)

    report_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(report_dir / "model_comparison.csv", index=False)
    summary = [
        "# ReviewRadar training summary",
        "",
        f"Selected model: **`{selected_name}`**",
        "",
        "Selection used validation PR-AUC, with validation F1 and model name as deterministic tie-breakers. The final test split was not used.",
        "",
        f"The selected pipeline was refit on **{len(refit_data):,}** train-plus-validation rows and saved to `{model_path.as_posix()}`.",
        "",
        "Features used:",
        "",
        *[f"- `{feature}`" for feature in SAFE_NUMERIC_FEATURES],
        "",
        "`commit_id`, `buggy`, `fix`, `year`, `author_date`, and `project` were not used as predictors. `project` was excluded deliberately to avoid repository-identity shortcuts; this first model is based only on change metrics.",
        "",
        "Run with:",
        "",
        "```powershell",
        "python -m src.train",
        "```",
        "",
    ]
    (report_dir / "training_summary.md").write_text("\n".join(summary), encoding="utf-8")
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    args = parser.parse_args()
    comparison = train_and_compare(args.input, args.report_dir, args.model_dir)
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()

