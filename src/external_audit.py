"""Generate reproducible inventories and separate audits for external datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import zipfile

import pandas as pd

try:
    from .split import DEFAULT_REPORT_DIR
except ImportError:
    from split import DEFAULT_REPORT_DIR


ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "external" / "raw"
OFFICIAL_APACHE = RAW / "apachejit_official"


def md5(path: pathlib.Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def apache_inventory() -> pd.DataFrame:
    rows = []
    files = ["apachejit_total.csv", "apachejit_train.csv", "apachejit_test_large.csv", "apachejit_test_small.csv"]
    for name in files:
        for source, path in [("ReviewRadar local", ROOT / "data" / "raw" / name), ("official GitHub", OFFICIAL_APACHE / name)]:
            frame = pd.read_csv(path)
            dates = pd.to_datetime(frame["author_date"], unit="s", utc=True)
            rows.append({
                "dataset": "ApacheJIT", "artifact": name, "source": source,
                "source_url": "https://github.com/hosseinkshvrz/apachejit/tree/master/dataset" if source == "official GitHub" else "local existing artifact",
                "local_path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
                "md5": md5(path), "rows": len(frame), "columns": len(frame.columns),
                "commit_id_unique": frame["commit_id"].is_unique,
                "buggy_true": int(frame["buggy"].sum()), "buggy_false": int((~frame["buggy"]).sum()),
                "date_min_utc": dates.min().isoformat(), "date_max_utc": dates.max().isoformat(),
                "license_or_terms": "Zenodo/GitHub rights not explicit in inspected metadata",
            })
    return pd.DataFrame(rows)


def redef_summary() -> pd.DataFrame:
    root = RAW / "redef_repo" / "ReDef-main" / "dataset"
    rows = []
    for path in sorted(root.glob("*.jsonl")):
        records = [json.loads(line) for line in path.open(encoding="utf-8")]
        frame = pd.DataFrame(records)
        dates = pd.to_datetime(frame["date"], errors="coerce", utc=True)
        rows.append({
            "dataset": "ReDef", "artifact": path.name, "rows": len(frame),
            "defective_modification_1": int((frame["defective_modification"] == 1).sum()),
            "defective_modification_0": int((frame["defective_modification"] == 0).sum()),
            "projects": frame["project"].nunique(), "commit_unique": frame["commit"].is_unique,
            "date_min_utc": dates.min().isoformat(), "date_max_utc": dates.max().isoformat(),
            "columns": ";".join(frame.columns), "md5": md5(path),
        })
    return pd.DataFrame(rows)


def jit_smart_summary() -> pd.DataFrame:
    root = RAW / "jit_smart_data" / "data" / "jitfine"
    rows = []
    for split in ("train", "valid", "test"):
        frame = pd.read_pickle(root / f"features_{split}.pkl")
        timestamps = pd.to_numeric(frame["author_date_unix_timestamp"], errors="coerce")
        dates = pd.to_datetime(timestamps, unit="s", errors="coerce", utc=True)
        labels = pd.to_numeric(frame["is_buggy_commit"], errors="coerce")
        rows.append({
            "dataset": "JIT-Defect4J via JIT-Smart", "artifact": f"features_{split}.pkl",
            "rows": len(frame), "positive_count": int((labels == 1).sum()),
            "negative_count": int((labels == 0).sum()), "missing_label_count": int(labels.isna().sum()),
            "projects": frame["project"].nunique(), "commit_unique": frame["commit_hash"].is_unique,
            "date_parseable_count": int(dates.notna().sum()), "date_min_utc": dates.min().isoformat(),
            "date_max_utc": dates.max().isoformat(), "columns": ";".join(frame.columns),
        })
    return pd.DataFrame(rows)


def write_reports(report_dir: pathlib.Path = DEFAULT_REPORT_DIR) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    inv = apache_inventory()
    inv.to_csv(report_dir / "apachejit_artifact_inventory.csv", index=False)
    inv.to_csv(ROOT / "data" / "external" / "apachejit_artifact_inventory.csv", index=False)
    redef = redef_summary(); redef.to_csv(report_dir / "redef_schema_summary.csv", index=False)
    jit = jit_smart_summary(); jit.to_csv(report_dir / "jit_smart_schema_summary.csv", index=False)

    total_equal = inv[(inv.artifact == "apachejit_total.csv") & (inv.source == "ReviewRadar local")]["md5"].iloc[0] == inv[(inv.artifact == "apachejit_total.csv") & (inv.source == "official GitHub")]["md5"].iloc[0]
    reconciliation = f"""# ApacheJIT source reconciliation

The official ApacheJIT repository is `https://github.com/hosseinkshvrz/apachejit`.
The Zenodo v2 record is `https://zenodo.org/records/5907847`; the prior Zenodo
record used by the project documentation is `https://zenodo.org/records/5907002`.
The official repository README describes four artifacts: complete total,
balanced 2003–2016 train, unbalanced later-period large test, and a smaller
test subset. It also describes fixed-issue linking, changed-line tracing,
PyDriller collection, and four GumTree filters.

## Reconciliation result

`apachejit_total.csv` is byte-identical to the downloaded official GitHub
artifact: `{total_equal}`. The local train and test artifacts have different
CSV byte hashes from the current GitHub copies, but their schemas, row counts,
commit-ID sets, labels, timestamps, and numeric values match after parsing.
The exact hashes and comparison fields are in
`reports/apachejit_artifact_inventory.csv`.

The current ApacheJIT total therefore is not a different data version that
explains Hadoop's zero labels. The supplied train/test files differ only in
serialization/representation at the value level observed here. No source
artifact or preprocessing log supplied an explanation for project-specific
zero label coverage.

## Provenance limitations

The local and downloaded package contains the construction scripts and
notebooks, but no independently auditable SZZ trace table, complete issue-link
join, changed-line mapping export, or per-row inclusion/exclusion log was
available in the inspected artifacts. The `buggy` field remains an SZZ-derived
proxy. Dataset rights were not explicit in the inspected Zenodo/GitHub
metadata; do not redistribute raw files without confirming terms.
"""
    (report_dir / "apachejit_source_reconciliation.md").write_text(reconciliation, encoding="utf-8")

    (report_dir / "dataset_landscape.md").write_text("""# Dataset landscape

| Dataset | Unit | Labels | Scope | Compatibility |
|---|---|---|---|---|
| ApacheJIT | commit/change | SZZ-derived `buggy` | 15 Apache projects, 2003–2019 | Current commit-level benchmark |
| ReDef | function-level modification | revert-anchored defective/clean | 22 C/C++ projects, 13,432 rows | Separate function/code benchmark |
| JIT-Defect4J via JIT-Smart | commit and line-level artifacts | `is_buggy_commit` plus line labels | 21 Java projects; commit splits in extracted data | Separate commit-level benchmark; feature/schema harmonization required |
| PROMISE/BPD Zenodo 4625975 | release/class-level | defect datasets | 36 datasets, 13 Java systems | Separate release-level benchmark |

ReDef was downloaded and inspected from its official repository. JIT-Smart's
official `data/data.zip` was downloaded and inspected. PROMISE/BPD was
researched but not downloaded because it is a release/class-level collection
and is not a compatible JIT commit benchmark. No datasets are merged.

License/usage terms were not explicit in the inspected ReDef and JIT-Smart
repository artifacts or ApacheJIT metadata; confirm with maintainers before
redistribution. The source URLs, versions, local paths, checksums, and status
are recorded in the CSV logs.
""", encoding="utf-8")

    matrix = pd.DataFrame([
        {"dataset": "ApacheJIT", "source_url": "https://github.com/hosseinkshvrz/apachejit", "unit": "commit/change", "label": "buggy; SZZ-derived", "timestamp": "author_date", "project": "yes", "license_status": "unclear", "compatibility": "current benchmark", "download_status": "downloaded and reconciled"},
        {"dataset": "ReDef", "source_url": "https://github.com/waroad/ReDef", "unit": "function-level modification", "label": "revert-anchored defective_modification", "timestamp": "date", "project": "yes", "license_status": "no license file observed", "compatibility": "separate task", "download_status": "downloaded and inspected"},
        {"dataset": "JIT-Defect4J via JIT-Smart", "source_url": "https://github.com/JIT-A/JIT-Smart", "unit": "commit and line-level", "label": "is_buggy_commit plus line labels", "timestamp": "author_date_unix_timestamp", "project": "yes", "license_status": "no license file observed", "compatibility": "separate benchmark; possible future harmonization", "download_status": "downloaded and inspected"},
        {"dataset": "PROMISE/BPD", "source_url": "https://zenodo.org/records/4625975", "unit": "release/class", "label": "dataset-specific defect labels", "timestamp": "not established", "project": "system/version", "license_status": "rights not explicit", "compatibility": "separate release benchmark", "download_status": "researched; not downloaded"},
    ])
    matrix.to_csv(report_dir / "dataset_compatibility_matrix.csv", index=False)
    downloads = pd.DataFrame([
        {"dataset": "ApacheJIT official GitHub", "source_url": "https://github.com/hosseinkshvrz/apachejit", "version_or_ref": "master inspected 2026-10-26", "local_path": "data/external/raw/apachejit_official", "status": "downloaded", "checksum": "total MD5 98e102741020b3b13798d1e10a350689; other hashes in artifact inventory", "license_terms": "unclear"},
        {"dataset": "ReDef", "source_url": "https://github.com/waroad/ReDef", "version_or_ref": "main archive inspected 2026-10-26", "local_path": "data/external/raw/redef_repo", "status": "downloaded", "checksum": "JSONL MD5s in redef_schema_summary.csv", "license_terms": "no license file observed"},
        {"dataset": "JIT-Smart/JIT-Defect4J", "source_url": "https://github.com/JIT-A/JIT-Smart", "version_or_ref": "master data/data.zip inspected 2026-10-26", "local_path": "data/external/raw/jit_smart_data", "status": "downloaded", "checksum": "archive MD5 5e9010d948e078cc7b6c6eaf138134bc", "license_terms": "no license file observed"},
        {"dataset": "PROMISE/BPD", "source_url": "https://zenodo.org/records/4625975", "version_or_ref": "v1", "local_path": "not downloaded", "status": "researched only", "checksum": "not applicable", "license_terms": "rights not explicit"},
    ])
    downloads.to_csv(report_dir / "dataset_download_log.csv", index=False)
    registry = downloads.rename(columns={"dataset": "name", "version_or_ref": "version", "checksum": "checksum_or_hash"})
    registry.to_csv(ROOT / "data" / "external" / "dataset_registry.csv", index=False)

    (report_dir / "redef_audit.md").write_text("""# ReDef external dataset audit

The official ReDef repository describes 3,164 defective and 10,268 clean
function-level modifications across 22 C/C++ projects. The downloaded splits
contain 10,744 train, 1,344 validation, and 1,344 test rows; observed labels
are 0/1 in `defective_modification`. Each row includes a commit hash, date,
project, file path, function name, before/after function text, revert metadata,
confidence, and explanation.

The repository describes defective instances as anchored to developer reverts
and ambiguous cases filtered with GPT-4o-assisted voting. This is a different
label event and prediction unit from ApacheJIT. The data includes explicit
function/code content and is not compatible with the current 12-metric model.
No license file was present in the downloaded repository; usage terms remain
unresolved. The dataset is retained as a separate benchmark only.
""", encoding="utf-8")
    (report_dir / "jit_smart_audit.md").write_text("""# JIT-Defect4J / JIT-Smart external dataset audit

The official JIT-Smart README identifies JIT-Defect4J as a large-scale
commit-level and line-level dataset covering 21 Java projects. The downloaded
`data/data.zip` contains multiple representation-specific pickles. The
JIT-Fine feature tables contain 16,374 train, 5,465 validation, and 5,480 test
commit rows with 25 columns, including `commit_hash`, project, author
timestamp, change metrics, `is_buggy_commit`, and line-level auxiliary data.

The extracted feature tables have 21 projects, unique commit hashes within
each split, parseable Unix timestamps, and labels in the `is_buggy_commit`
field. The exact counts, date ranges, and schema are in
`reports/jit_smart_schema_summary.csv`.

This is a promising separate commit-level benchmark because it has commit IDs,
projects, timestamps, and expert metrics, but it is not merged with ApacheJIT:
the project universe, label construction, metric names/types, and line-level
task differ. The README points to JIT-Defect4J and reproduction code but does
not provide a complete independently auditable label-generation trace in the
downloaded data. No license file was observed; confirm terms before
redistribution.
""", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=pathlib.Path, default=DEFAULT_REPORT_DIR)
    args = parser.parse_args()
    write_reports(args.report_dir)
    print("external reports written")


if __name__ == "__main__":
    main()
