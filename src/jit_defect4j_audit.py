"""Generate schema and coverage reports for the acquired JIT-Defect4J data."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    from .jit_defect4j_benchmark import DEFAULT_ROOT, FEATURES, load_split
except ImportError:
    from jit_defect4j_benchmark import DEFAULT_ROOT, FEATURES, load_split


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"


def run(root: Path = DEFAULT_ROOT, report_dir: Path = REPORT_DIR) -> dict[str, pd.DataFrame]:
    frames = {split: load_split(root, split) for split in ("train", "valid", "test")}
    schema_rows = []
    role_map = {feature: "prediction-time numeric expert feature" for feature in FEATURES}
    role_map.update({"is_buggy_commit": "target", "commit_hash": "identifier", "project": "group key; excluded predictor", "author_date_unix_timestamp": "chronology only", "author_date": "metadata; excluded predictor", "parent_hashes": "identifier metadata", "author_name": "identity; excluded predictor", "author_email": "identity; excluded predictor", "commit_message": "text; excluded predictor", "fileschanged": "path list; excluded predictor", "classification": "auxiliary metadata; excluded", "fix": "timing/provenance unresolved; excluded"})
    frame = frames["train"]
    for column in frame.columns:
        values = frame[column]
        schema_rows.append({
            "column": column, "observed_dtype": str(values.dtype), "train_missing": int(values.isna().sum()),
            "train_unique": int(values.nunique(dropna=False)), "train_example": str(values.iloc[0])[:160],
            "role": role_map.get(column, "not used"), "used_by_benchmark": column in FEATURES,
        })
    schema = pd.DataFrame(schema_rows)

    coverage_rows = []
    for split, frame in frames.items():
        dates = pd.to_datetime(pd.to_numeric(frame["author_date_unix_timestamp"], errors="coerce"), unit="s", utc=True)
        for (project, year), group in frame.assign(_year=dates.dt.year).groupby(["project", "_year"]):
            coverage_rows.append({
                "split": split, "project": project, "year": int(year), "rows": len(group),
                "positive_count": int(group["is_buggy_commit"].sum()), "negative_count": int((~group["is_buggy_commit"]).sum()),
                "positive_prevalence": float(group["is_buggy_commit"].mean()), "date_min_utc": dates.loc[group.index].min().isoformat(),
                "date_max_utc": dates.loc[group.index].max().isoformat(), "small_group_warning": bool(len(group) < 30),
            })
    coverage = pd.DataFrame(coverage_rows)
    overlap_rows = []
    id_sets = {split: set(frame["commit_hash"]) for split, frame in frames.items()}
    for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
        overlap_rows.append({"check": f"{left}_{right}_commit_overlap", "value": len(id_sets[left] & id_sets[right])})

    report_dir.mkdir(parents=True, exist_ok=True)
    schema.to_csv(report_dir / "jit_defect4j_schema.csv", index=False)
    coverage.to_csv(report_dir / "jit_defect4j_project_coverage.csv", index=False)
    summary = """# JIT-Defect4J data audit

The acquired JIT-Fine feature tables contain commit-level rows with 25 columns
and a separate line-level annotation artifact. This audit uses only the
commit-level feature tables. The target is `is_buggy_commit == 1`; it is not
renamed or inferred from another field.

Observed rows are 16,374 train, 5,465 validation, and 5,480 test, with 21
projects in every split. Commit hashes are unique within each split and the
three official split ID intersections are zero. The label prevalences are
8.49%, 8.55%, and 8.67% respectively. Exact schema roles and missingness are
in `reports/jit_defect4j_schema.csv`; project/year counts are in
`reports/jit_defect4j_project_coverage.csv`.

The supplied splits are not chronological: the train/valid/test timestamp
ranges overlap. They are therefore suitable as a disjoint benchmark split but
not as a future-period evaluation. The package README identifies both
commit-level and line-level tasks; this benchmark keeps them separate.

The inspected package does not include a complete independent SZZ/revert or
issue-link trace for every commit label. The README and source package provide
the benchmark/reproduction artifacts, but label provenance and license terms
remain limitations. No records were deduplicated, relabeled, or merged.
"""
    (report_dir / "jit_defect4j_data_audit.md").write_text(summary + "\n\nSplit overlap checks:\n\n" + pd.DataFrame(overlap_rows).to_string(index=False), encoding="utf-8")
    return {"schema": schema, "coverage": coverage, "overlap": pd.DataFrame(overlap_rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    args = parser.parse_args()
    result = run(args.data_root, args.report_dir)
    print({name: len(frame) for name, frame in result.items()})


if __name__ == "__main__":
    main()
