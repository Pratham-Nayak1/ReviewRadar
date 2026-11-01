# ApacheJIT data

## Source and download

ReviewRadar uses the complete ApacheJIT CSV for initial inspection:

- Official Zenodo record: <https://zenodo.org/records/5907002>
- Direct file download, verified from the Zenodo record: <https://zenodo.org/records/5907002/files/apachejit_total.csv?download=1>
- Official author repository and replication package: <https://github.com/hosseinkshvrz/apachejit>
- Zenodo file name: `apachejit_total.csv`
- Zenodo-reported MD5: `98e102741020b3b13798d1e10a350689`
- Local path: `data/raw/apachejit_total.csv`

The raw file was downloaded into `data/raw/` and has not been modified. The repository `.gitignore` excludes local raw and processed datasets. Processed data, if created later, must go under `data/processed/` and must never overwrite the raw file.

The authors' supplied split artifacts are also kept unchanged under `data/raw/` for protocol inspection: `apachejit_train.csv`, `apachejit_test_large.csv`, and `apachejit_test_small.csv`. They were obtained from the official author repository. The small test file is a subset of the large test file and is not an independent evaluation set.

The official Zenodo record also lists `apachejit_train.csv`, `apachejit_test_large.csv`, and `apachejit_test_small.csv`. Those are pre-made subsets. This project starts from `apachejit_total.csv` so that its chronological evaluation protocol can be designed and checked explicitly.

## Citation and usage information

Recommended citation:

> Hossein Keshavarz and Meiyappan Nagappan. “ApacheJIT: A Large Dataset for Just-In-Time Defect Prediction.” Proceedings of the 19th International Conference on Mining Software Repositories (MSR ’22), 2022, pp. 191–195. DOI: <https://doi.org/10.1145/3524842.3527996>.

Dataset record: Keshavarz, Hossein; Nagappan, Meiyappan (2022), “ApacheJIT: A Large Dataset for Just-In-Time Defect Prediction,” Zenodo, DOI: <https://doi.org/10.5281/zenodo.5907002>.

The Zenodo record page inspected for this project does not display a specific license or clear reuse terms in its Rights section. Therefore, do not assume an open-data license or redistribute the CSV. Keep the source attribution above and check with the authors/record maintainers before publishing derived data or redistributing the original file. The Apache repositories represented in the data have their own project licenses; those do not automatically establish a license for this compiled dataset.

## What one row represents

Each row represents one software commit/change revision. The official description calls the rows commits, and the paper describes ApacheJIT as a collection of software changes/revisions for just-in-time defect prediction. `commit_id` is the commit identifier.

## Observed file profile

Inspection of the downloaded file on 2026-10-22 found:

- 106,674 rows and 18 columns
- 15 project values
- 17 `year` values, covering 2003–2019
- `author_date` is present for every row and is stored as Unix epoch seconds; parsed UTC range is 2003-09-11 14:54:46 through 2019-12-26 17:39:53
- 0 missing fields in the observed CSV
- 0 duplicate `commit_id` values
- 0 exact duplicate rows
- 28,239 `buggy=True` rows (26.47%) and 78,435 `buggy=False` rows (73.53%)
- `fix=True` occurs in 10,690 rows (10.02%); its exact modeling meaning is not sufficiently documented in the source description, so it is excluded from candidate predictors pending provenance confirmation

The first three observed rows are commit-level records from `apache/groovy`, with `buggy=False`, `fix=False`, `year=2003`, Unix timestamps in `author_date`, and numeric change metrics.

## Schema

Types below are inferred from the complete CSV values. `author_date` is read as an integer initially and should be converted to a datetime feature only in a later, explicit preprocessing step.

| Column | Observed type | Meaning / modeling note |
|---|---|---|
| `commit_id` | string | Commit identifier; identifier only, not a predictor. |
| `project` | string | Source repository/project; categorical grouping variable and possible cross-project split key. |
| `buggy` | boolean | Target label. The official description says it indicates whether the commit introduced any bug. |
| `fix` | boolean | Auxiliary flag observed in the file; exact documented meaning is unresolved. Exclude until confirmed. |
| `year` | integer | Calendar year; derived/redundant with `author_date`, so treat cautiously. |
| `author_date` | integer | Commit author timestamp in Unix seconds; primary chronological ordering field. |
| `la` | integer | Lines added. |
| `ld` | integer | Lines deleted. |
| `nf` | integer | Files touched. |
| `nd` | integer | Directories touched. |
| `ns` | integer | Subsystems touched. |
| `ent` | float | Change entropy. |
| `ndev` | float | Number of distinct developers who touched the changed files. |
| `age` | float | Average time since the last change to the changed files. |
| `nuc` | float | Number of unique changes in the changed files. |
| `aexp` | integer | Author experience. |
| `arexp` | float | Author recent experience. |
| `asexp` | float | Author subsystem experience. |

The metric names and meanings follow the paper’s description of the common JIT change metrics. The file itself does not contain expanded descriptions for each abbreviation.

## Target and label provenance

Use `buggy` as the target. The authors state that `buggy` marks whether a commit introduced any bug. The paper explains that bug-inducing commits were identified through an SZZ-based process that traces fixed issues and changed lines back to earlier commits, followed by filtering. This means the label is an SZZ-derived proxy, not an independently verified ground-truth label; later results should state that limitation.

The row unit is a commit/change revision, not an issue, bug-fix event, file, or individual line.

## Candidate predictors and leakage review

Initial candidate predictors are the change metrics `la`, `ld`, `nf`, `nd`, `ns`, `ent`, `ndev`, `age`, `nuc`, `aexp`, `arexp`, and `asexp`. `project` may be used as a categorical predictor only if the split design makes its interpretation clear.

Exclude `commit_id` because it is an identifier. Exclude `buggy` because it is the target. Exclude `fix` for now because it is an auxiliary flag whose exact semantics and timing are not established; if it indicates a fixing commit or post-hoc information, using it could leak label information. Treat `year` as redundant with `author_date` and avoid using both without a clear rationale. Use only information available at commit time, and do not derive features from future rows or later bug reports.

## Uncertainty and next checks

- The source documentation clearly defines `buggy`, the commit-level row unit, the time coverage, and the broad metric family.
- The source description does not clearly define the `fix` column in the material inspected. It must be traced through the replication code or author documentation before any decision to use it.
- The Zenodo page does not expose a specific license in the inspected Rights section. Treat reuse rights as unresolved and preserve attribution.
- The reported metrics above are dataset inspection results only. Predictive performance is documented separately in the root `README.md` and generated reports; no performance claim should be made beyond those measured artifacts.

