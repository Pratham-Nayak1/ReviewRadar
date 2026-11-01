"""Controlled feature-signal and baseline comparison on development data only.

This experiment never loads or scores the previously inspected ApacheJIT test
period or JIT-Defect4J supplied test split. It writes new reports only.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score

try:
    from .data import DEFAULT_DATA_PATH, load_apachejit
    from .evaluate import choose_threshold_by_f1
    from .jit_defect4j_benchmark import DEFAULT_ROOT as JIT_ROOT, load_split
    from .pipeline import build_numeric_pipeline
    from .review_budget import review_budget_metrics
    from .split import make_temporal_splits
    from .train import positive_probability
except ImportError:
    from data import DEFAULT_DATA_PATH, load_apachejit
    from evaluate import choose_threshold_by_f1
    from jit_defect4j_benchmark import DEFAULT_ROOT as JIT_ROOT, load_split
    from pipeline import build_numeric_pipeline
    from review_budget import review_budget_metrics
    from split import make_temporal_splits
    from train import positive_probability


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"
RANDOM_STATE = 42
BUDGETS = (0.05, 0.10, 0.20)


@dataclass(frozen=True)
class Fold:
    dataset: str
    name: str
    fit: pd.DataFrame
    evaluation: pd.DataFrame
    timestamp_col: str
    project_col: str
    label_col: str


FEATURE_FAMILIES = {
    "ApacheJIT": {
        "change_size_churn": ["la", "ld", "nf", "nd", "ns"],
        "developer_experience_history": ["ndev", "aexp", "arexp", "asexp"],
        "complexity_and_change_history": ["ent", "age", "nuc"],
        "all_permitted": ["la", "ld", "nf", "nd", "ns", "ent", "ndev", "age", "nuc", "aexp", "arexp", "asexp"],
    },
    "JIT-Defect4J": {
        "change_size_churn": ["la", "ld", "nf", "nd", "ns"],
        "developer_experience_history": ["ndev", "exp", "rexp", "sexp"],
        "complexity_and_change_history": ["entropy", "lt", "age", "nuc"],
        "all_permitted": ["la", "ld", "nf", "ns", "nd", "entropy", "ndev", "lt", "nuc", "age", "exp", "rexp", "sexp"],
    },
}


def available_feature_families(dataset: str, columns: list[str]) -> pd.DataFrame:
    """Return membership and explicitly mark unavailable fields."""
    rows = []
    for family, requested in FEATURE_FAMILIES[dataset].items():
        present = [feature for feature in requested if feature in columns]
        missing = [feature for feature in requested if feature not in columns]
        rows.append({"dataset": dataset, "feature_family": family, "available_features": ",".join(present), "unavailable_features": ",".join(missing), "available": bool(present)})
    return pd.DataFrame(rows)


def _dates(frame: pd.DataFrame, column: str) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(frame[column]):
        return pd.to_datetime(frame[column], errors="coerce", utc=True)
    numeric = pd.to_numeric(frame[column], errors="coerce")
    if numeric.notna().all():
        return pd.to_datetime(numeric, unit="s", utc=True)
    return pd.to_datetime(frame[column], errors="coerce", utc=True)


def _validate_fold(fit: pd.DataFrame, evaluation: pd.DataFrame, timestamp_col: str, project_col: str, label_col: str) -> None:
    fit_dates = _dates(fit, timestamp_col)
    evaluation_dates = _dates(evaluation, timestamp_col)
    if fit_dates.max() >= evaluation_dates.min():
        raise ValueError("fold training timestamps must strictly precede evaluation timestamps")
    if set(fit["_row_id"]) & set(evaluation["_row_id"]):
        raise ValueError("fold rows overlap")
    if fit["_row_id"].duplicated().any() or evaluation["_row_id"].duplicated().any():
        raise ValueError("duplicate fold rows")
    if fit[label_col].isna().any() or evaluation[label_col].isna().any():
        raise ValueError("fold labels must be available")
    if fit[project_col].isna().any() or evaluation[project_col].isna().any():
        raise ValueError("fold projects must be available")


def _safe_threshold(labels: pd.Series, scores: pd.Series) -> float:
    """Avoid undefined PR-threshold warnings on zero-positive folds."""
    values = labels.astype(bool)
    if len(values) == 0 or not values.any() or values.all():
        return 0.5
    threshold, _ = choose_threshold_by_f1(labels, scores)
    return float(threshold)


def apache_folds(path: Path = DEFAULT_DATA_PATH) -> list[Fold]:
    """Use the latest two development folds; never load the 2018+ test rows."""
    frame = load_apachejit(path).copy()
    dates = _dates(frame, "author_date")
    frame["_row_id"] = frame["commit_id"].astype(str)
    folds = []
    for year in (2016, 2017):
        fit = frame.loc[dates.dt.year < year].copy()
        evaluation = frame.loc[dates.dt.year == year].copy()
        if fit.empty or evaluation.empty:
            continue
        _validate_fold(fit, evaluation, "author_date", "project", "buggy")
        folds.append(Fold("ApacheJIT", f"rolling_{year}", fit, evaluation, "author_date", "project", "buggy"))
    return folds


def jit_defect4j_folds(root: Path = JIT_ROOT) -> list[Fold]:
    """Use the latest two existing train-only chronological folds."""
    frame = load_split(root, "train").copy()
    frame["_row_id"] = frame["commit_hash"].astype(str)
    dates = _dates(frame, "author_date_unix_timestamp")
    folds = []
    for year in (2014, 2015):
        fit = frame.loc[dates.dt.year < year].copy()
        evaluation = frame.loc[dates.dt.year == year].copy()
        if fit.empty or evaluation.empty:
            continue
        _validate_fold(fit, evaluation, "author_date_unix_timestamp", "project", "is_buggy_commit")
        folds.append(Fold("JIT-Defect4J", f"rolling_{year}", fit, evaluation, "author_date_unix_timestamp", "project", "is_buggy_commit"))
    return folds


def reference_models(dataset: str, features: list[str], *, rf_trees: int | None = None) -> dict[str, object]:
    rf_trees = (300 if dataset == "ApacheJIT" else 100) if rf_trees is None else rf_trees
    return {
        "dummy_prior": build_numeric_pipeline(DummyClassifier(strategy="prior"), features),
        "logistic_regression": build_numeric_pipeline(LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE, solver="lbfgs"), features, scale=True),
        "random_forest": build_numeric_pipeline(RandomForestClassifier(class_weight="balanced_subsample", min_samples_leaf=2, n_estimators=rf_trees, n_jobs=-1, random_state=RANDOM_STATE), features),
        "hist_gradient_boosting": build_numeric_pipeline(HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0, random_state=RANDOM_STATE), features),
    }


def project_asof_normalize(fit: pd.DataFrame, evaluation: pd.DataFrame, features: list[str], *, project_col: str, timestamp_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create historical project-centered values using no future rows.

    Fit rows use only values preceding each row; evaluation rows use all fit
    rows from the same project. A global prior history is used when a project
    has no prior value. The transform is deliberately outside the estimator
    but receives only the current fold's fit/evaluation frames.
    """
    fit_sorted = fit.assign(_parsed_date=_dates(fit, timestamp_col)).sort_values(["_parsed_date", "_row_id"])
    eval_sorted = evaluation.assign(_parsed_date=_dates(evaluation, timestamp_col)).sort_values(["_parsed_date", "_row_id"])
    history: dict[str, list[float]] = {}
    global_history: dict[str, list[float]] = {feature: [] for feature in features}

    def transform(frame: pd.DataFrame, update: bool) -> pd.DataFrame:
        output = pd.DataFrame(index=frame.index)
        for index, row in frame.iterrows():
            project = str(row[project_col])
            project_values = history.setdefault(project, {feature: [] for feature in features})
            for feature in features:
                values = project_values[feature] or global_history[feature]
                current = pd.to_numeric(pd.Series([row[feature]]), errors="coerce").iloc[0]
                if not values or pd.isna(current):
                    output.loc[index, f"{feature}_project_delta"] = 0.0 if pd.isna(current) else float(current)
                else:
                    output.loc[index, f"{feature}_project_delta"] = float(current) - float(np.median(values))
            if update:
                for feature in features:
                    current = pd.to_numeric(pd.Series([row[feature]]), errors="coerce").iloc[0]
                    if pd.notna(current):
                        project_values[feature].append(float(current))
                        global_history[feature].append(float(current))
        return output.sort_index()

    fit_output = transform(fit_sorted, update=True)
    eval_output = transform(eval_sorted, update=False)
    return fit_output, eval_output


def _metric_rows(labels: pd.Series, scores: pd.Series, threshold: float, *, dataset: str, family: str, model: str, split: str, project: str, fit_rows: int, configuration_variant: str) -> dict[str, object]:
    actual = labels.astype(bool).to_numpy()
    predicted = scores.to_numpy(dtype=float) >= threshold
    tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[False, True]).ravel()
    positives = int(actual.sum())
    row: dict[str, object] = {
        "dataset": dataset, "feature_family": family, "model": model, "configuration_variant": configuration_variant, "split": split, "project": project,
        "fit_rows": fit_rows, "evaluation_rows": len(actual), "positive_count": positives, "negative_count": int(len(actual) - positives),
        "positive_prevalence": float(actual.mean()) if len(actual) else 0.0, "threshold": float(threshold),
        "true_negative": int(tn), "false_positive": int(fp), "false_negative": int(fn), "true_positive": int(tp),
        "precision": float(precision_score(actual, predicted, zero_division=0)), "recall": float(recall_score(actual, predicted, zero_division=0)),
        "f1": float(f1_score(actual, predicted, zero_division=0)), "average_precision": float(average_precision_score(actual, scores)) if positives else float("nan"),
        "insufficient_evidence": bool(positives < 20 or (len(actual) - positives) < 20),
    }
    for budget in BUDGETS:
        budget_metrics = review_budget_metrics(scores, actual, budget)
        suffix = int(budget * 100)
        row[f"precision_at_{suffix}"] = budget_metrics["precision_at_k"]
        row[f"recall_at_{suffix}"] = budget_metrics["recall_at_k"]
        row[f"review_k_at_{suffix}"] = budget_metrics["k"]
    return row


def evaluate_family(fold: Fold, family: str, features: list[str], *, exact_baseline: bool = False) -> list[dict[str, object]]:
    fit = fold.fit
    evaluation = fold.evaluation
    if family == "project_normalized_all":
        fit_x, eval_x = project_asof_normalize(fit, evaluation, features, project_col=fold.project_col, timestamp_col=fold.timestamp_col)
        model_features = list(fit_x.columns)
    else:
        fit_x, eval_x, model_features = fit[features], evaluation[features], features
    rows: list[dict[str, object]] = []
    rf_trees = None if exact_baseline else 100
    configuration_variant = "registered_baseline" if exact_baseline else "ablation_rf_100"
    for model_name, model in reference_models(fold.dataset, model_features, rf_trees=rf_trees).items():
        model.fit(fit_x, fit[fold.label_col])
        scores = positive_probability(model, eval_x)
        threshold = _safe_threshold(evaluation[fold.label_col], scores)
        rows.append(_metric_rows(evaluation[fold.label_col], scores, threshold, dataset=fold.dataset, family=family, model=model_name, split=fold.name, project="all_projects", fit_rows=len(fit), configuration_variant=configuration_variant))
        for project, group in evaluation.groupby(fold.project_col, sort=True):
            rows.append(_metric_rows(group[fold.label_col], scores.loc[group.index], threshold, dataset=fold.dataset, family=family, model=model_name, split=fold.name, project=str(project), fit_rows=len(fit), configuration_variant=configuration_variant))
    return rows


def run(data_path: Path = DEFAULT_DATA_PATH, jit_root: Path = JIT_ROOT, report_dir: Path = REPORT_DIR) -> dict[str, pd.DataFrame]:
    report_dir.mkdir(parents=True, exist_ok=True)
    folds = apache_folds(data_path) + jit_defect4j_folds(jit_root)
    rows = []
    family_rows = []
    for fold in folds:
        columns = list(fold.fit.columns)
        family_table = available_feature_families(fold.dataset, columns)
        family_rows.append(family_table)
        for family, requested in FEATURE_FAMILIES[fold.dataset].items():
            features = [feature for feature in requested if feature in columns]
            if features:
                rows.extend(evaluate_family(fold, family, features, exact_baseline=(family == "all_permitted")))
                if family == "all_permitted":
                    heuristic_scores = pd.to_numeric(fold.evaluation["la"], errors="coerce") + pd.to_numeric(fold.evaluation["ld"], errors="coerce")
                    threshold = _safe_threshold(fold.evaluation[fold.label_col], heuristic_scores)
                    family_rows[-1] = family_rows[-1]
                    rows.append(_metric_rows(fold.evaluation[fold.label_col], heuristic_scores, threshold, dataset=fold.dataset, family="all_permitted", model="change_size_la_plus_ld", split=fold.name, project="all_projects", fit_rows=len(fold.fit), configuration_variant="heuristic_no_fit"))
        all_features = [feature for feature in FEATURE_FAMILIES[fold.dataset]["all_permitted"] if feature in columns]
        rows.extend(evaluate_family(fold, "project_normalized_all", all_features))
    result = pd.DataFrame(rows)
    result.to_csv(report_dir / "feature_ablation_results.csv", index=False)

    baselines = result[(result["feature_family"] == "all_permitted") & (result["project"] == "all_projects")].copy()
    baselines.to_csv(report_dir / "simple_baseline_results.csv", index=False)
    membership = pd.concat(family_rows, ignore_index=True).drop_duplicates().reset_index(drop=True)
    membership.to_csv(report_dir / "feature_family_membership.csv", index=False)
    summary = build_report(result, membership)
    (report_dir / "feature_signal_audit.md").write_text(summary, encoding="utf-8")
    return {"ablation": result, "baselines": baselines, "membership": membership}


def build_report(result: pd.DataFrame, membership: pd.DataFrame) -> str:
    all_rows = result[(result["project"] == "all_projects")]
    summary = all_rows.groupby(["dataset", "feature_family", "model"], as_index=False).agg(mean_average_precision=("average_precision", "mean"), mean_f1=("f1", "mean"), mean_precision_at_10=("precision_at_10", "mean"), mean_recall_at_10=("recall_at_10", "mean"), folds=("split", "nunique"))
    summary = summary.sort_values(["dataset", "feature_family", "mean_average_precision", "mean_f1"], ascending=[True, True, False, False])
    lines = [
        "# ReviewRadar feature-signal and baseline audit",
        "",
        "This experiment uses ApacheJIT rolling folds for 2016 and 2017 and",
        "JIT-Defect4J train-only rolling folds for 2014 and 2015. It does not load or score",
        "the previously inspected ApacheJIT 2018–2019 test period or the",
        "JIT-Defect4J supplied test split. Datasets remain separate.",
        "",
        "## Protocol and leakage controls",
        "",
        "Each row is fitted on earlier timestamps and evaluated on a later fold.",
        "Imputation, scaling, and estimators are inside training-only pipelines.",
        "The all_permitted baseline uses the registered ApacheJIT 300-tree RF",
        "and registered JIT-Defect4J 100-tree RF. To keep the broader ablation",
        "tractable, non-baseline RF rows use the registered 100-tree variant and",
        "are marked `ablation_rf_100` in the CSV; they are not silently compared",
        "as identical configurations.",
        "Thresholds are selected on each development fold by maximum F1; review",
        "ranking uses continuous scores with stable ties and ceil-based budgets.",
        "Project normalization uses only prior same-project values for fit rows",
        "and all fit-row project history for later evaluation rows.",
        "",
        "## Feature availability",
        "",
        "The detailed membership table records every requested and unavailable",
        "feature. ApacheJIT has no explicit file identifier or file-level defect",
        "history; its `fileschanged` path list is excluded. JIT-Defect4J has no",
        "file identifier or independently auditable file-level history in the",
        "commit feature table. Developer-history metrics are used only when they",
        "are already present in the source schema.",
        "",
        "## Mean development-fold results",
        "",
        "The table below is descriptive; it does not establish future-period",
        "generalization or causality. Full fold and project rows are in the CSV.",
        "",
        "```text",
        summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"),
        "```",
        "",
        "## Project-aware normalization",
        "",
        "A leakage-safe as-of normalization family was implemented and evaluated",
        "project, timestamp, and existing numeric metrics. It is not a project",
        "identifier predictor and it does not use labels. Its fold/project",
        "results are included as `project_normalized_all`. It does not silently",
        "treat project composition as a causal signal.",
        "",
        "## Recommendation",
        "",
        "Treat a feature family as promising only if its ranking metrics improve",
        "consistently across development folds and do not depend on one project.",
        "No result here is production evidence. A genuinely later holdout remains",
        "required before adopting a new feature family or claiming temporal",
        "generalization. Historical project normalization is implementable, but",
        "its operational value remains unmeasured until that later evaluation.",
        "",
        "Reproduce with `.",
        "\\.venv\\Scripts\\python.exe -m src.feature_signal_audit`.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--jit-root", type=Path, default=JIT_ROOT)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    args = parser.parse_args()
    result = run(args.data_path, args.jit_root, args.report_dir)
    print({name: len(frame) for name, frame in result.items()})


if __name__ == "__main__":
    main()
