"""Audit cross-project generalization and review prioritization.

This module consumes the existing benchmark artifacts and, for the JIT-Smart
validation audit, refits only on the supplied training split. It never loads
JIT-Defect4J official-test labels and does not change any saved artifact.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve

try:
    from .jit_defect4j_benchmark import (
        DEFAULT_ROOT as JIT_ROOT,
        FEATURES,
        fit_scores,
        load_split,
        make_models,
    )
    from .review_budget import compare_review_strategies
except ImportError:
    from jit_defect4j_benchmark import DEFAULT_ROOT as JIT_ROOT, FEATURES, fit_scores, load_split, make_models
    from review_budget import compare_review_strategies


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"
RANDOM_STATE = 42
BUDGETS = (0.05, 0.10, 0.20)


def _ece(labels: pd.Series, scores: pd.Series, bins: int = 10) -> float:
    """Equal-width expected calibration error with empty-bin safety."""
    y = labels.astype(bool).to_numpy()
    p = pd.to_numeric(scores).to_numpy(dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y)
    if total == 0:
        return float("nan")
    error = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (p >= left) & (p <= right if right == 1.0 else p < right)
        if mask.any():
            error += float(mask.mean()) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return error


def calibration_rows(labels: pd.Series, scores: pd.Series, dataset: str, split: str, model: str) -> list[dict]:
    if len(labels) == 0:
        return []
    fraction_true, mean_probability = calibration_curve(labels.astype(bool), scores, n_bins=10, strategy="uniform")
    return [
        {
            "dataset": dataset,
            "split": split,
            "model": model,
            "brier_score": float(brier_score_loss(labels.astype(bool), scores)),
            "expected_calibration_error": float(_ece(labels, scores)),
            "calibration_bins": len(fraction_true),
        }
    ]


def project_level_generalization(report_dir: Path = REPORT_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Annotate existing JIT-Defect4J LOPO rows and bootstrap project means."""
    source = pd.read_csv(report_dir / "jit_defect4j_project_holdout_results.csv")
    source["stable_group_eligible"] = (source["rows"] >= 30) & (source["positive_count"] > 0)
    source["model_rank_by_f1"] = source.groupby("project")["f1"].rank(method="min", ascending=False).astype(int)
    source["model_rank_by_average_precision"] = source.groupby("project")["average_precision"].rank(method="min", ascending=False).astype("Int64")

    rng = np.random.default_rng(RANDOM_STATE)
    summary_rows: list[dict] = []
    for model, frame in source.groupby("model", sort=True):
        eligible = frame[frame["stable_group_eligible"]]
        values = eligible["f1"].to_numpy(dtype=float)
        ap_values = eligible["average_precision"].dropna().to_numpy(dtype=float)
        if len(values):
            samples = values[rng.integers(0, len(values), size=(2000, len(values)))]
            f1_mean = float(values.mean())
            f1_low, f1_high = np.quantile(samples.mean(axis=1), [0.025, 0.975])
        else:
            f1_mean = f1_low = f1_high = float("nan")
        if len(ap_values):
            samples = ap_values[rng.integers(0, len(ap_values), size=(2000, len(ap_values)))]
            ap_mean = float(ap_values.mean())
            ap_low, ap_high = np.quantile(samples.mean(axis=1), [0.025, 0.975])
        else:
            ap_mean = ap_low = ap_high = float("nan")
        summary_rows.append({
            "dataset": "JIT-Defect4J",
            "result_type": "project_bootstrap_summary",
            "model": model,
            "eligible_projects": int(eligible["project"].nunique()),
            "all_projects": int(frame["project"].nunique()),
            "mean_f1": f1_mean,
            "f1_bootstrap_ci_low": float(f1_low),
            "f1_bootstrap_ci_high": float(f1_high),
            "mean_average_precision": ap_mean,
            "average_precision_bootstrap_ci_low": float(ap_low),
            "average_precision_bootstrap_ci_high": float(ap_high),
        })
    summary = pd.DataFrame(summary_rows)
    output = source.copy()
    output.insert(0, "dataset", "JIT-Defect4J")
    output.insert(1, "result_type", "project_fold")
    output = output.merge(
        summary[["model", "mean_f1", "f1_bootstrap_ci_low", "f1_bootstrap_ci_high", "mean_average_precision", "average_precision_bootstrap_ci_low", "average_precision_bootstrap_ci_high"]],
        on="model",
        how="left",
    )
    output.to_csv(report_dir / "project_level_generalization.csv", index=False)
    return output, summary


def review_budget_audit(report_dir: Path = REPORT_DIR, jit_root: Path = JIT_ROOT) -> pd.DataFrame:
    """Write validation-only ranking comparisons for both benchmarks."""
    rows: list[pd.DataFrame] = []

    apache = pd.read_csv(report_dir / "evaluation_review_budget.csv")
    rows.append(apache[apache["split"] == "validation"].assign(dataset="ApacheJIT", source="existing_validation_report"))

    train = load_split(jit_root, "train")
    valid = load_split(jit_root, "valid")
    for name, pipeline in make_models().items():
        scores = fit_scores(pipeline, train, valid)
        heuristic = pd.to_numeric(valid["la"], errors="coerce") + pd.to_numeric(valid["ld"], errors="coerce")
        result = compare_review_strategies(scores, valid["is_buggy_commit"], BUDGETS, heuristic_scores=heuristic, heuristic_name="change_size_la_plus_ld", random_runs=1000, random_state=RANDOM_STATE)
        result.insert(0, "dataset", "JIT-Defect4J")
        result.insert(1, "model", name)
        result["split"] = "official_valid"
        result["source"] = "train_fit_validation_scored"
        rows.append(result)
    output = pd.concat(rows, ignore_index=True, sort=False)
    output.to_csv(report_dir / "review_budget_audit.csv", index=False)
    return output


def evaluation_integrity_report(report_dir: Path = REPORT_DIR) -> str:
    """Build a source-backed integrity report without accessing final test labels."""
    return """# Generalization audit: evaluation integrity

## Verified ApacheJIT facts

- `src/split.py::make_temporal_splits` creates train before 2017-01-01 UTC,
  validation during 2017, and test from 2018-01-01 onward. It does not use a
  random fallback.
- `src/train.py` fits each candidate on training rows only and selects by
  validation average precision. `src/evaluate.py` selects the threshold on
  validation F1, refits the selected pipeline on train plus validation, and
  scores the final test afterward.
- The saved feature contract excludes commit IDs, target, fix, year, date,
  and project. Imputation and scaling are inside scikit-learn pipelines.
- The temporal split has the same 15 projects in train, validation, and test;
  that is within-project future performance, not unseen-project performance.
- The official ApacheJIT artifacts have zero train/large-test commit overlap,
  but repeated feature signatures exist and the official small test overlaps
  the large test. The project uses the explicit temporal split as primary.

## Verified JIT-Defect4J facts

- `src/jit_defect4j_benchmark.py` reads disjoint supplied commit-hash splits,
  fits on official train, selects model and threshold on official validation,
  then scores official test with frozen choices. It excludes identifiers,
  project, timestamps, text, `fix`, and the target.
- JIT-Smart train/validation/test timestamps overlap and all 21 projects occur
  in all three splits. The supplied test is therefore not a future-period
  holdout. The separate leave-one-project-out results use train plus
  validation and a fixed threshold of 0.5, without official-test labels.
- Median imputation and Logistic scaling are inside the shared pipeline. The
  model configuration report records seed 42, features, parameters,
  validation thresholds, and status.

## What artifacts cannot prove

- Source code and reports cannot prove that no one ever manually inspected a
  final-test result outside the recorded workflow. The audit found no test
  labels used in model/threshold selection in the inspected implementations,
  but execution-history claims are limited to available artifacts.
- Commit-ID disjointness does not rule out related commits, repeated files,
  or repository history dependence. The available reports do not contain a
  complete file-level ancestry audit for both datasets.
- Project-grouped and temporal estimates are correlated and have small-group
  uncertainty. They should not be pooled as if they were independent IID
  observations.

## Calibration interpretation

Calibration diagnostics in the new audit use only validation predictions from
training-fitted pipelines for JIT-Defect4J. Existing ApacheJIT validation
calibration results are reproduced from its saved evaluation report. These
scores are not a calibration layer and do not alter the saved model or
threshold. Until calibration is separately validated under the intended
deployment distribution, risk scores should be interpreted primarily as
ranking scores, not literal probabilities.
"""


def run(report_dir: Path = REPORT_DIR, jit_root: Path = JIT_ROOT) -> dict[str, object]:
    report_dir.mkdir(parents=True, exist_ok=True)
    project_rows, project_summary = project_level_generalization(report_dir)
    budget = review_budget_audit(report_dir, jit_root)

    calibration = [
        {"dataset": "ApacheJIT", "split": "validation", "model": "random_forest", "brier_score": 0.15655115640476713, "expected_calibration_error": 0.09614678865113804, "calibration_bins": 10, "source": "evaluation_calibration_metrics.csv"},
    ]
    train = load_split(jit_root, "train")
    valid = load_split(jit_root, "valid")
    for name in ("random_forest_balanced", "hist_gradient_boosting"):
        scores = fit_scores(make_models()[name], train, valid)
        calibration.extend({**row, "source": "train_fit_validation_scored"} for row in calibration_rows(valid["is_buggy_commit"], scores, "JIT-Defect4J", "official_valid", name))
    pd.DataFrame(calibration).to_csv(report_dir / "calibration_audit.csv", index=False)

    report = evaluation_integrity_report(report_dir)
    report += "\n## Project-level uncertainty\n\n"
    report += "The project-level CSV contains per-project confusion metrics and model ranks. Bootstrap intervals in the table resample projects, not individual commits, using seed 42 and 2,000 resamples. They describe variability across the observed project set; they are not guarantees for unseen repositories.\n"
    report += "\n## Output references\n\n"
    report += "- `reports/project_level_generalization.csv` contains the 21 JIT-Defect4J leave-one-project-out folds for each model, with small-group flags and within-project ranks.\n"
    report += "- `reports/review_budget_audit.csv` contains ApacheJIT validation rows from the existing report and new JIT-Defect4J official-validation rankings, including reproducible random simulations and the change-size heuristic.\n"
    report += "- `reports/calibration_audit.csv` contains validation-only JIT-Defect4J calibration diagnostics and the existing ApacheJIT validation diagnostic.\n"
    (report_dir / "evaluation_integrity_audit.md").write_text(report, encoding="utf-8")

    selected = budget[(budget["dataset"] == "JIT-Defect4J") & (budget["model"] == "hist_gradient_boosting") & (budget["approach"] == "model")]
    apache_valid = budget[(budget["dataset"] == "ApacheJIT") & (budget["approach"] == "model")]
    summary = """# ReviewRadar generalization audit

This report compares ApacheJIT and JIT-Defect4J as separate benchmark tasks.
No datasets were merged and no saved model or UI artifact was changed.

## Generalization decision

ApacheJIT measures within-project chronological transfer: all 15 projects
occur in each temporal split. JIT-Defect4J's supplied split measures
disjoint-commit benchmark performance with overlapping time ranges and all 21
projects represented in each split. Its leave-one-project-out results are the
separate unseen-project diagnostic.

The observed gap between supplied/validation results, chronological folds,
and project holdout results is real in the recorded metrics, but no single
cause is proven. Plausible contributors include project composition, temporal
feature/label shift, and project-specific conventions. The project-held-out
results show materially different model rankings, so within-project results
should not be presented as evidence of unseen-project generalization.

## Review prioritization

Ranking results are reported at 5%, 10%, and 20% budgets. The ranking is
created from scores only; labels are used afterward for measurement. Random
rows are reproducible simulations (seed 42, 1,000 runs), not operational
benefit estimates. ApacheJIT rows are validation-only from the existing
evaluation report. New JIT-Defect4J rows are scored on official validation
after fitting on official training only. The official-test ranking is not
used here for selection.

## Calibration

Calibration is diagnostic only. Validation Brier/ECE values are shown in
`reports/calibration_audit.csv`; no calibration model was fitted and no saved
threshold was changed. Because both datasets exhibit distribution and project
shift, scores should not be called calibrated risk probabilities without a
deployment-specific calibration study.

## Limitations and next experiment

Project folds are correlated through shared repositories and historical
development processes. Small or zero-positive project groups are flagged and
their precision/recall estimates are unstable or undefined. A useful next
experiment is a pre-registered rolling-origin, project-stratified evaluation
with a genuinely later collection period and enough positives per project,
using ranking metrics and fixed review budgets as primary outcomes. It should
be run once without using the current benchmark test labels for tuning.
"""
    project_table = project_summary[["model", "mean_f1", "f1_bootstrap_ci_low", "f1_bootstrap_ci_high", "mean_average_precision", "average_precision_bootstrap_ci_low", "average_precision_bootstrap_ci_high"]].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    jit_budget = selected[["review_fraction", "precision_at_k", "recall_at_k"]].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    apache_budget = apache_valid[["approach", "review_fraction", "precision_at_k", "recall_at_k"]].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    summary += f"""
## Measured project-holdout results

The following are means across the 21 JIT-Defect4J project folds. The 95%
bootstrap intervals resample projects, not commits:

```text
{project_table}
```

The fold-level CSV contains rows, positive counts, confusion matrices, rates,
and model ranks for every project. These are diagnostic estimates, not a
claim about arbitrary unseen projects.

## Measured validation review budgets

JIT-Defect4J HistGradientBoosting was fit on official train and measured on
official validation. Its score-only ranking produced:

```text
{jit_budget}
```

ApacheJIT's existing selected-Random-Forest validation rows were:

```text
{apache_budget}
```

Both tables are benchmark observations. Random-selection comparisons in the
CSV are simulations and do not imply review-hours, cost, or bugs prevented.

## Reproduction

```powershell
.\\.venv\\Scripts\\python.exe -m src.generalization_audit
.\\.venv\\Scripts\\python.exe -m pytest -q tests/test_generalization_audit.py
.\\.venv\\Scripts\\python.exe -m pytest -q
```
"""
    (report_dir / "generalization_audit.md").write_text(summary, encoding="utf-8")
    return {"project": project_rows, "project_summary": project_summary, "budget": budget}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--jit-root", type=Path, default=JIT_ROOT)
    args = parser.parse_args()
    result = run(args.report_dir, args.jit_root)
    print({name: len(value) if hasattr(value, "__len__") else value for name, value in result.items()})


if __name__ == "__main__":
    main()
