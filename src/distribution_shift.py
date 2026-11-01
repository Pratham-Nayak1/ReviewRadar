"""Leakage-safe distribution-shift and feature-availability audit.

Only pre-2018 training/development rows are used. The previously inspected
final test period is not loaded, scored, or tuned here.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

try:
    from .data import DEFAULT_DATA_PATH, author_dates_utc, load_apachejit
    from .evaluate import choose_threshold_by_f1
    from .hgb_robustness import HGB_GRID, make_random_forest
    from .pipeline import build_numeric_pipeline
    from .review_budget import POSITIVE_CLASS
    from .split import DEFAULT_REPORT_DIR, SAFE_NUMERIC_FEATURES, make_temporal_splits
    from .train import positive_probability
except ImportError:
    from data import DEFAULT_DATA_PATH, author_dates_utc, load_apachejit
    from evaluate import choose_threshold_by_f1
    from hgb_robustness import HGB_GRID, make_random_forest
    from pipeline import build_numeric_pipeline
    from review_budget import POSITIVE_CLASS
    from split import DEFAULT_REPORT_DIR, SAFE_NUMERIC_FEATURES, make_temporal_splits
    from train import positive_probability


RANDOM_STATE = 42
PERIODS = ("2014", "2015", "2016", "2017_validation")


def analysis_periods(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return the training reference and development periods only."""
    splits = make_temporal_splits(frame)
    train = splits.train
    years = author_dates_utc(train).dt.year
    result = {"train": train}
    for year in (2014, 2015, 2016):
        result[str(year)] = train.loc[years == year].reset_index(drop=True)
    result["2017_validation"] = splits.validation
    return result


def distribution_metrics(periods: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Summarize feature distributions and train-relative standardized shifts."""
    reference = periods["train"]
    reference_std = reference[SAFE_NUMERIC_FEATURES].std(ddof=0).replace(0, np.nan)
    rows = []
    for period, frame in periods.items():
        for feature in SAFE_NUMERIC_FEATURES:
            values = pd.to_numeric(frame[feature], errors="coerce")
            rows.append({
                "table_type": "feature_distribution", "period": period,
                "rows": len(frame), "feature": feature,
                "mean": float(values.mean()), "median": float(values.median()),
                "q25": float(values.quantile(.25)), "q75": float(values.quantile(.75)),
                "q95": float(values.quantile(.95)),
                "missing_count": int(values.isna().sum()),
                "missing_rate": float(values.isna().mean()),
                "standardized_mean_shift_vs_train": float(
                    (values.mean() - reference[feature].mean()) / reference_std[feature]
                ) if pd.notna(reference_std[feature]) else 0.0,
            })
    return pd.DataFrame(rows)


def project_metrics(periods: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for period, frame in periods.items():
        grouped = frame.groupby("project", dropna=False)["buggy"].agg(
            rows="size", positive_count="sum", positive_prevalence="mean"
        )
        for project, values in grouped.iterrows():
            rows.append({
                "table_type": "project_composition", "period": period,
                "project": project, "rows": int(values["rows"]),
                "project_share": float(values["rows"] / len(frame)),
                "positive_count": int(values["positive_count"]),
                "positive_prevalence": float(values["positive_prevalence"]),
                "small_group_warning": bool(values["rows"] < 30),
            })
    return pd.DataFrame(rows)


def _outcomes(labels: pd.Series, scores: pd.Series, threshold: float) -> np.ndarray:
    actual = labels.to_numpy(dtype=bool)
    predicted = scores.to_numpy() >= threshold
    return np.select(
        [actual & predicted, ~actual & ~predicted, ~actual & predicted],
        ["true_positive", "true_negative", "false_positive"],
        default="false_negative",
    )


def temporal_error_analysis(periods: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Report model outcomes by temporal fold and project."""
    train = periods["train"]
    train_years = author_dates_utc(train).dt.year
    rows = []
    for period in PERIODS:
        validation = periods[period]
        fit = train if period == "2017_validation" else train.loc[train_years < int(period)].reset_index(drop=True)
        models = {
            "random_forest_current": make_random_forest(),
            "hist_gradient_boosting_selected": build_numeric_pipeline(
                HistGradientBoostingClassifier(random_state=RANDOM_STATE, **HGB_GRID["hgb_existing"])
            ),
        }
        for model, pipeline in models.items():
            pipeline.fit(fit[SAFE_NUMERIC_FEATURES], fit["buggy"])
            scores = positive_probability(pipeline, validation[SAFE_NUMERIC_FEATURES])
            threshold, _ = choose_threshold_by_f1(validation["buggy"], scores)
            scored = validation[["project", "buggy"]].copy()
            scored["outcome"] = _outcomes(validation["buggy"], scores, threshold)
            grouped = scored.groupby(["project", "outcome"], dropna=False).size().rename("outcome_count").reset_index()
            totals = scored.groupby("project", dropna=False).agg(
                project_rows=("outcome", "size"), project_positive_count=("buggy", "sum"),
                project_positive_prevalence=("buggy", "mean"),
            ).reset_index()
            for record in grouped.merge(totals, on="project").to_dict("records"):
                rows.append({
                    "model": model, "period": period, "project": record["project"],
                    "outcome": record["outcome"], "outcome_count": int(record["outcome_count"]),
                    "project_rows": int(record["project_rows"]),
                    "outcome_rate_within_project_period": float(record["outcome_count"] / record["project_rows"]),
                    "project_positive_count": int(record["project_positive_count"]),
                    "project_positive_prevalence": float(record["project_positive_prevalence"]),
                    "small_group_warning": bool(record["project_rows"] < 30),
                    "threshold": float(threshold), "fit_rows": len(fit),
                })
    return pd.DataFrame(rows)


def feature_availability_report() -> str:
    return """# Feature availability audit

ApacheJIT rows are software commit/change revisions. The positive class is
`buggy=True`, documented as a commit that introduced a bug; the label is an
SZZ-derived proxy. Prediction is assumed to occur when current change metrics
are available, before later bug reports or fix attribution.

The current model uses `la`, `ld`, `nf`, `nd`, `ns`, `ent`, `ndev`, `age`,
`nuc`, `aexp`, `arexp`, and `asexp`. `commit_id` is display-only; `project`,
`fix`, `year`, and `author_date` are excluded from prediction.

| Candidate signal | Required source fields | Available safely? | Decision |
|---|---|---|---|
| Prior file-level defect history | File identifiers and earlier labels | No file IDs or historical labels | Do not implement |
| Rolling developer experience | Developer identity and as-of history | No developer ID; experience aggregates already exist | No new feature |
| Project prior prevalence | Project and earlier target labels | Target-derived shortcut | Do not implement |
| Future bug/fix indicators | Later issue/fix data | Not prediction-time safe; `fix` is unresolved | Exclude |

No valid additional feature family is available in the current schema and
documentation. Therefore `feature_ablation_results.csv` is intentionally not
created. Feature associations and importances are descriptive model behavior,
not evidence of causation.
"""


def _markdown_table(frame: pd.DataFrame, columns: list[str], limit: int = 30) -> str:
    view = frame[columns].head(limit).copy()
    for column in view.columns:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(lambda value: f"{value:.4f}")
    header = "| " + " | ".join(view.columns) + " |"
    divider = "| " + " | ".join("---" for _ in view.columns) + " |"
    body = ["| " + " | ".join(str(value) for value in row) + " |" for row in view.itertuples(index=False, name=None)]
    return "\n".join([header, divider, *body])


def build_report(distribution: pd.DataFrame, projects: pd.DataFrame, errors: pd.DataFrame) -> str:
    focus = distribution[distribution.feature.isin(["aexp", "ndev", "age", "nuc"])]
    totals = errors.groupby(["model", "period", "outcome"], as_index=False)["outcome_count"].sum()
    counts = projects.groupby("period", as_index=False).agg(rows=("rows", "sum"), positive_count=("positive_count", "sum"))
    counts["positive_prevalence"] = counts["positive_count"] / counts["rows"]
    return f"""# ReviewRadar distribution-shift analysis

Audit date: 2026-10-26. Only pre-2018 training/development rows are used;
the previously inspected final test period is not read or scored.

The positive class is `buggy=True`, and one row is one commit/change revision.
The pre-2017 training set is the reference distribution. 2014–2016 are
rolling validation periods and 2017 is the existing development validation
period. Standardized mean shift is `(period mean - train mean) / train
standard deviation`; it is descriptive evidence of covariate shift, not a
causal explanation.

## Counts and prevalence

{_markdown_table(counts, ["period", "rows", "positive_count", "positive_prevalence"], 10)}

All feature summaries and project composition are in
`reports/distribution_shift_metrics.csv`. The prioritized features are shown
below; the same file contains all 12 model features.

{_markdown_table(focus, ["period", "feature", "mean", "median", "q25", "q75", "missing_rate", "standardized_mean_shift_vs_train"], 40)}

The raw documentation reports no missing cells; missingness is checked per
period rather than assumed. Project rows include prevalence and a small-group
warning. Changes in project mix and prevalence can affect observed metrics.

## Error analysis

`reports/temporal_error_analysis.csv` reports false positives and false
negatives with project-period denominators for both models. Models are fit only
on rows earlier than each validation period, and thresholds are selected on
that validation period. Summary counts:

{_markdown_table(totals, ["model", "period", "outcome", "outcome_count"], 100)}

These are descriptive error concentrations, not evidence that project or any
feature causes defects. SZZ label noise and changing project practices remain
possible explanations not resolved by this audit.

## Feature value decision

The raw schema has no file IDs, developer IDs, prior defect labels, or as-of
issue history. Existing experience fields are already included, while `fix`
is undocumented and potentially post-hoc. No valid additional feature family
was available, so no proxy feature or ablation was invented.

## Decision

Document the observed temporal and project distribution changes and retain the
current Random Forest reference unchanged. A genuinely new future-period
holdout is required before confirming any improvement. The next evaluation
rules are recorded in `reports/next_evaluation_protocol.md`.
"""


def run(data_path: Path = DEFAULT_DATA_PATH, report_dir: Path = DEFAULT_REPORT_DIR) -> dict[str, pd.DataFrame]:
    periods = analysis_periods(load_apachejit(data_path))
    distribution = distribution_metrics(periods)
    projects = project_metrics(periods)
    errors = temporal_error_analysis(periods)
    report_dir.mkdir(parents=True, exist_ok=True)
    pd.concat([distribution, projects], ignore_index=True, sort=False).to_csv(report_dir / "distribution_shift_metrics.csv", index=False)
    errors.to_csv(report_dir / "temporal_error_analysis.csv", index=False)
    (report_dir / "feature_availability_audit.md").write_text(feature_availability_report(), encoding="utf-8")
    (report_dir / "distribution_shift_analysis.md").write_text(build_report(distribution, projects, errors), encoding="utf-8")
    return {"distribution": distribution, "projects": projects, "errors": errors}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    args = parser.parse_args()
    result = run(args.input, args.report_dir)
    print({name: len(frame) for name, frame in result.items()})


if __name__ == "__main__":
    main()
