"""Training-only scikit-learn preprocessing pipeline construction."""

from __future__ import annotations

from typing import Sequence

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from .split import SAFE_NUMERIC_FEATURES
except ImportError:
    from split import SAFE_NUMERIC_FEATURES


def build_numeric_pipeline(
    estimator,
    feature_columns: Sequence[str] = SAFE_NUMERIC_FEATURES,
    *,
    scale: bool = False,
) -> Pipeline:
    """Build a pipeline whose preprocessing learns only during ``fit``.

    Call ``pipeline.fit(train[features], train[target])``. Do not fit this
    object on validation or final-test rows. The feature list is a documented
    schema decision, not a statistic learned from the final test set.
    """

    preprocessing_steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        preprocessing_steps.append(("scaler", StandardScaler()))
    numeric_preprocessing = Pipeline(steps=preprocessing_steps)
    preprocessor = ColumnTransformer(
        transformers=[("numeric", numeric_preprocessing, list(feature_columns))],
        remainder="drop",
    )
    return Pipeline(
        steps=[("preprocess", preprocessor), ("estimator", estimator)]
    )

