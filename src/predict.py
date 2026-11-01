"""Reusable inference for the saved ReviewRadar pipeline.

This module never retrains. It loads the persisted scikit-learn pipeline and
the evaluation configuration produced by ``src.evaluate``.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "reviewradar_selected.joblib"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "reports" / "evaluation_config.json"
DISPLAY_IDENTIFIER = "commit_id"


class InferenceInputError(ValueError):
    """Raised when incoming prediction data is not schema-compatible."""


class ReviewRadarPredictor:
    """Load and use the persisted ReviewRadar model without fitting."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
    ) -> None:
        self.model_path = Path(model_path)
        self.config_path = Path(config_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Saved ReviewRadar model not found: {self.model_path}")
        if not self.config_path.is_file():
            raise FileNotFoundError(f"ReviewRadar evaluation configuration not found: {self.config_path}")

        self.pipeline = joblib.load(self.model_path)
        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.feature_columns = list(self.config.get("feature_columns", []))
        if not self.feature_columns:
            raise InferenceInputError("Evaluation configuration has no feature_columns")
        self.positive_class = self.config.get("positive_class")
        if self.positive_class != "buggy=True":
            raise InferenceInputError(
                f"Expected positive_class='buggy=True'; found {self.positive_class!r}"
            )
        self.threshold = self.config.get("selected_threshold")
        self._validate_loaded_pipeline()

    def _validate_loaded_pipeline(self) -> None:
        if not hasattr(self.pipeline, "predict_proba"):
            raise InferenceInputError("Saved model does not provide predict_proba()")
        if not hasattr(self.pipeline, "named_steps") or "preprocess" not in self.pipeline.named_steps:
            raise InferenceInputError("Saved model is not the expected preprocessing pipeline")

        fitted_features = getattr(self.pipeline, "feature_names_in_", None)
        if fitted_features is not None and list(fitted_features) != self.feature_columns:
            raise InferenceInputError(
                "Saved pipeline feature schema does not match evaluation_config.json: "
                f"pipeline={list(fitted_features)!r}, config={self.feature_columns!r}"
            )
        estimator = self.pipeline.named_steps.get("estimator")
        classes = getattr(estimator, "classes_", None)
        if classes is None or not any(bool(value) for value in classes):
            raise InferenceInputError("Saved estimator does not expose the True positive class")

    def _as_frame(self, data: pd.DataFrame | pd.Series | Mapping[str, Any]) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            frame = data.copy()
        elif isinstance(data, pd.Series):
            frame = data.to_frame().T
        elif isinstance(data, Mapping):
            frame = pd.DataFrame([dict(data)])
        else:
            raise TypeError("Prediction input must be a pandas DataFrame, Series, or mapping")
        if frame.empty:
            raise InferenceInputError("Prediction input is empty; provide at least one row")
        if frame.columns.duplicated().any():
            duplicates = frame.columns[frame.columns.duplicated()].tolist()
            raise InferenceInputError(f"Prediction input contains duplicate columns: {duplicates}")
        return frame

    def _validate_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        required = set(self.feature_columns)
        missing_columns = sorted(required.difference(frame.columns))
        if missing_columns:
            raise InferenceInputError(
                "Missing required feature columns: " + ", ".join(missing_columns)
            )

        allowed_columns = required | {DISPLAY_IDENTIFIER}
        unexpected_columns = sorted(set(frame.columns).difference(allowed_columns))
        if unexpected_columns:
            raise InferenceInputError(
                "Unexpected input columns: "
                + ", ".join(unexpected_columns)
                + ". Provide only the trained features and optional commit_id."
            )

        for column in self.feature_columns:
            series = frame[column]
            converted = pd.to_numeric(series, errors="coerce")
            invalid_values = converted.isna() & series.notna()
            if invalid_values.any():
                examples = series.loc[invalid_values].astype(str).head(3).tolist()
                raise InferenceInputError(
                    f"Feature '{column}' must contain numeric values; invalid examples: {examples}"
                )
            frame[column] = converted
            non_missing = converted.dropna().to_numpy()
            if not np.isfinite(non_missing).all():
                raise InferenceInputError(
                    f"Feature '{column}' contains non-finite values; use finite numbers or missing values"
                )
        return frame

    def _positive_scores(self, features: pd.DataFrame) -> np.ndarray:
        probabilities = self.pipeline.predict_proba(features)
        classes = list(self.pipeline.named_steps["estimator"].classes_)
        positive_indices = [index for index, value in enumerate(classes) if bool(value) is True]
        if len(positive_indices) != 1:
            raise InferenceInputError(f"Expected exactly one True class; estimator classes are {classes!r}")
        return probabilities[:, positive_indices[0]]

    def predict(self, data: pd.DataFrame | pd.Series | Mapping[str, Any]) -> pd.DataFrame:
        """Return display ID, positive score, prediction, and configured category."""

        frame = self._validate_frame(self._as_frame(data))
        scores = self._positive_scores(frame[self.feature_columns])
        result = pd.DataFrame({"risk_score": scores}, index=frame.index)
        if DISPLAY_IDENTIFIER in frame.columns:
            result.insert(0, DISPLAY_IDENTIFIER, frame[DISPLAY_IDENTIFIER].to_numpy())

        if self.threshold is not None:
            try:
                threshold = float(self.threshold)
            except (TypeError, ValueError) as exc:
                raise InferenceInputError("Configured selected_threshold is not numeric") from exc
            result["predicted_buggy"] = scores >= threshold

        categories = self.config.get("risk_categories")
        if categories:
            result["risk_category"] = [self._category_for_score(float(score), categories) for score in scores]
        return result.reset_index(drop=True)

    @staticmethod
    def _category_for_score(score: float, categories: list[dict[str, Any]]) -> str:
        for category in categories:
            lower = float(category["min_score"])
            upper = float(category["max_score"])
            if lower <= score <= upper:
                return str(category["name"])
        raise InferenceInputError(f"Score {score} does not fit any configured risk category")

    def predict_one(self, record: Mapping[str, Any] | pd.Series) -> dict[str, Any]:
        """Predict one row and return a plain dictionary for UI/API use."""

        result = self.predict(record)
        return result.iloc[0].to_dict()

    def predict_csv(self, input_path: str | Path, output_path: str | Path | None = None) -> pd.DataFrame:
        """Predict a schema-compatible batch CSV and optionally save results."""

        csv_path = Path(input_path)
        if not csv_path.is_file():
            raise FileNotFoundError(f"Prediction input CSV not found: {csv_path}")
        result = self.predict(pd.read_csv(csv_path))
        if output_path is not None:
            result.to_csv(output_path, index=False)
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Schema-compatible batch CSV")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV for predictions")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    predictor = ReviewRadarPredictor(args.model, args.config)
    result = predictor.predict_csv(args.input, args.output)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()

