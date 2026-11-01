"""Generate descriptive EDA outputs for the ApacheJIT raw dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

try:
    from .data import DEFAULT_DATA_PATH, author_dates_utc, dataset_profile, load_apachejit
except ImportError:
    from data import DEFAULT_DATA_PATH, author_dates_utc, dataset_profile, load_apachejit


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports"
TARGET = "buggy"
METRIC_COLUMNS = [
    "la", "ld", "nf", "nd", "ns", "ent", "ndev", "age", "nuc", "aexp", "arexp", "asexp"
]


def write_class_balance(frame: pd.DataFrame, path: Path) -> None:
    counts = frame[TARGET].value_counts().reindex([False, True], fill_value=0)
    labels = ["Clean (False)", "Bug-inducing (True)"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, counts.values, color=["#4C78A8", "#E45756"])
    ax.set_title("ApacheJIT target class distribution")
    ax.set_ylabel("Commit count")
    ax.tick_params(axis="x", rotation=10)
    for bar, count in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, count, f"{count:,}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_feature_distributions(frame: pd.DataFrame, path: Path) -> None:
    available = [column for column in METRIC_COLUMNS if column in frame.columns]
    n_columns = 3
    n_rows = (len(available) + n_columns - 1) // n_columns
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(14, 3.2 * n_rows))
    axes = list(axes.flat) if hasattr(axes, "flat") else [axes]
    for ax, column in zip(axes, available):
        sns.histplot(
            data=frame,
            x=column,
            hue=TARGET,
            bins=40,
            stat="density",
            common_norm=False,
            element="step",
            fill=False,
            palette={False: "#4C78A8", True: "#E45756"},
            ax=ax,
        )
        ax.set_title(column)
        ax.set_ylabel("Density")
    for ax in axes[len(available):]:
        ax.set_visible(False)
    fig.suptitle("ApacheJIT change-metric distributions by target", y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_date_coverage(frame: pd.DataFrame, path: Path) -> None:
    dates = author_dates_utc(frame)
    by_year = dates.dt.year.value_counts().sort_index()
    target_by_year = frame.assign(author_year=dates.dt.year).groupby("author_year")[TARGET].mean()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].bar(by_year.index.astype(str), by_year.values, color="#72B7B2")
    axes[0].set_title("Commit rows by author year")
    axes[0].set_xlabel("Year")
    axes[0].set_ylabel("Commit count")
    axes[0].tick_params(axis="x", rotation=60)
    axes[1].plot(target_by_year.index, target_by_year.values, marker="o", color="#E45756")
    axes[1].set_title("Observed bug-inducing rate by author year")
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("Fraction buggy")
    axes[1].set_ylim(0, 1)
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_missingness(frame: pd.DataFrame, path: Path) -> None:
    missing = frame.isna().sum()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(missing.index, missing.values, color="#B279A2")
    ax.set_title("Missing values by column")
    ax.set_ylabel("Missing cells")
    ax.tick_params(axis="x", rotation=70)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_suspicious_features(frame: pd.DataFrame, path: Path) -> dict[str, pd.DataFrame]:
    grouped: dict[str, pd.DataFrame] = {}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    if "fix" in frame.columns:
        fix_summary = frame.groupby("fix", dropna=False)[TARGET].agg(["count", "mean"])
        grouped["fix"] = fix_summary
        axes[0].bar(fix_summary.index.astype(str), fix_summary["mean"], color=["#4C78A8", "#E45756"])
        axes[0].set_title("Buggy rate by auxiliary fix flag")
        axes[0].set_xlabel("fix")
        axes[0].set_ylabel("Fraction buggy")
        axes[0].set_ylim(0, 1)
    else:
        axes[0].set_visible(False)
    dates = author_dates_utc(frame)
    year_summary = frame.assign(author_year=dates.dt.year).groupby("author_year")[TARGET].agg(["count", "mean"])
    grouped["year"] = year_summary
    axes[1].plot(year_summary.index, year_summary["mean"], marker="o", color="#F58518")
    axes[1].set_title("Buggy rate by author year")
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("Fraction buggy")
    axes[1].set_ylim(0, 1)
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return grouped


def build_report(frame: pd.DataFrame, grouped: dict[str, pd.DataFrame], source: Path) -> str:
    profile = dataset_profile(frame)
    dates = author_dates_utc(frame)
    target_counts = frame[TARGET].value_counts().reindex([False, True], fill_value=0)
    target_rate = float(frame[TARGET].mean())
    metrics = [column for column in METRIC_COLUMNS if column in frame.columns]
    correlations = frame[metrics + [TARGET]].corr(numeric_only=True)[TARGET].drop(TARGET)
    correlations = correlations.reindex(correlations.abs().sort_values(ascending=False).index)

    lines = [
        "# ReviewRadar ApacheJIT EDA report",
        "",
        f"Source inspected: `{source.as_posix()}`",
        "",
        "This report is descriptive only. It does not impute, encode, scale, select features for a final model, or use a final test set.",
        "",
        "## Dataset summary",
        "",
        f"- Rows: **{profile['rows']:,}**",
        f"- Columns: **{profile['columns']}**",
        f"- Exact duplicate rows: **{profile['duplicate_rows']}**",
        f"- Duplicate `commit_id` rows after the first occurrence: **{profile['duplicate_commit_ids']}**",
        f"- Author-date coverage: **{dates.min().isoformat()}** to **{dates.max().isoformat()}**",
        f"- Missing cells: **{sum(profile['missing_values'].values()):,}**",
        "",
        "## Target distribution",
        "",
        f"`buggy=False`: **{target_counts[False]:,}** ({(1 - target_rate):.2%})",
        f"`buggy=True`: **{target_counts[True]:,}** ({target_rate:.2%})",
        "",
        "![Class balance](figures/class_balance.png)",
        "",
        "## Column types and missingness",
        "",
        "| Column | dtype | missing | unique non-null |",
        "|---|---:|---:|---:|",
    ]
    for column in frame.columns:
        lines.append(
            f"| `{column}` | `{frame[column].dtype}` | {int(frame[column].isna().sum())} | {int(frame[column].nunique(dropna=True))} |"
        )
    lines += [
        "",
        "![Missingness](figures/missingness.png)",
        "",
        "## Date coverage",
        "",
        "`author_date` is a complete Unix-seconds timestamp field and is suitable for chronological ordering. `year` is redundant with that timestamp and should not be treated as an independent source of time information without a reason.",
        "",
        "![Date coverage](figures/date_coverage.png)",
        "",
        "## Feature distributions",
        "",
        f"The observed numeric change metrics are: {', '.join(f'`{column}`' for column in metrics)}.",
        "",
        "![Feature distributions](figures/feature_distributions.png)",
        "",
        "These correlations are descriptive only, not model selection and not predictive performance.",
        "",
        "| Metric | Pearson correlation with `buggy` |",
        "|---|---:|",
    ]
    for column, value in correlations.items():
        lines.append(f"| `{column}` | {value:.4f} |")
    lines += [
        "",
        "## Suspicious features and leakage review",
        "",
        "- `commit_id` is unique and should remain an identifier, not a predictor.",
        "- `buggy` is the target and must not appear in features.",
        "- `fix` is an auxiliary boolean field. Its observed relationship with the target is shown below, but its exact provenance is unresolved. Exclude it until documentation confirms it is known at commit time and is not post-hoc information.",
        "- `author_date` and `year` describe time. Use `author_date` for chronological evaluation; do not let future timestamps or future-derived aggregates enter training features.",
        "- `project` is a repository label, not an individual-row identifier. It may be useful for grouped analysis, but project-based generalization must be decided explicitly before modeling.",
        "- The remaining change metrics are candidate predictors, subject to a later time-aware training/evaluation design.",
        "",
        "![Suspicious feature review](figures/suspicious_features.png)",
        "",
    ]
    if "fix" in grouped:
        fix_table = grouped["fix"].rename(columns={"count": "rows", "mean": "buggy_rate"})
        lines += [
            "### Observed `fix` relationship",
            "",
            "| fix | rows | buggy_rate |",
            "|---:|---:|---:|",
        ]
        for value, row in fix_table.iterrows():
            lines.append(f"| {value} | {int(row['rows'])} | {row['buggy_rate']:.4f} |")
        lines.append("")
    lines += [
        "The `fix` relationship is an observational warning, not proof of leakage. The source documentation must be checked before deciding whether it can be used.",
        "",
        "## Reproducibility",
        "",
        "Run `python -m src.eda` from the project root. The script reads only the raw CSV and writes charts in `reports/figures/` plus this report. It does not alter `data/raw/apachejit_total.csv`.",
        "",
    ]
    return "\n".join(lines)


def run_eda(input_path: Path, report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = report_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    frame = load_apachejit(input_path)
    sns.set_theme(style="whitegrid")
    write_class_balance(frame, figure_dir / "class_balance.png")
    write_feature_distributions(frame, figure_dir / "feature_distributions.png")
    write_date_coverage(frame, figure_dir / "date_coverage.png")
    write_missingness(frame, figure_dir / "missingness.png")
    grouped = write_suspicious_features(frame, figure_dir / "suspicious_features.png")
    (report_dir / "eda_report.md").write_text(build_report(frame, grouped, input_path), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    args = parser.parse_args()
    run_eda(args.input, args.report_dir)
    print(f"EDA report written to {args.report_dir / 'eda_report.md'}")
    print(f"Charts written to {args.report_dir / 'figures'}")


if __name__ == "__main__":
    main()

