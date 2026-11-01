# ReviewRadar

ReviewRadar is an educational, explainable classical-machine-learning project for prioritizing historical software changes for review. It predicts whether a commit/change is labeled as bug-inducing in ApacheJIT and ranks incoming changes by a model risk score.

The intended users are software-engineering students, researchers, and engineers learning how to build a leakage-aware defect-risk benchmark. It is not a production defect gate, an automated approval system, or a replacement for code review.

## What ReviewRadar predicts

The positive class is `buggy=True`: ApacheJIT documents this as a commit that introduced any bug. The model uses 12 numeric change metrics available in the project’s training schema:

`la`, `ld`, `nf`, `nd`, `ns`, `ent`, `ndev`, `age`, `nuc`, `aexp`, `arexp`, and `asexp`.

The saved Random Forest produces a score used to rank changes. The score is not presented as a calibrated probability. The optional `predicted_buggy` flag uses a threshold selected on validation data, but neither a high score nor a flag proves that an individual change is buggy. The model does not predict bug severity, time to repair, review effort, money saved, or bugs prevented.

## Dataset

ReviewRadar uses the complete ApacheJIT dataset:

- Official Zenodo record: <https://zenodo.org/records/5907002>
- Verified direct file: <https://zenodo.org/records/5907002/files/apachejit_total.csv?download=1>
- Author replication repository: <https://github.com/hosseinkshvrz/apachejit>
- Expected file: `data/raw/apachejit_total.csv`
- Verified MD5: `98e102741020b3b13798d1e10a350689`

Recommended citation:

> Hossein Keshavarz and Meiyappan Nagappan. “ApacheJIT: A Large Dataset for Just-In-Time Defect Prediction.” Proceedings of the 19th International Conference on Mining Software Repositories (MSR ’22), 2022, pp. 191–195. DOI: <https://doi.org/10.1145/3524842.3527996>.

Dataset record citation: Keshavarz, Hossein; Nagappan, Meiyappan (2022), “ApacheJIT: A Large Dataset for Just-In-Time Defect Prediction,” Zenodo. DOI: <https://doi.org/10.5281/zenodo.5907002>.

The inspected Zenodo Rights section did not specify a clear dataset license. Do not assume that the compiled CSV is freely redistributable. Preserve attribution and check the Zenodo record or authors before publishing or redistributing the raw file. Licenses of the Apache repositories represented in the data do not automatically license the compiled dataset.

### Observed dataset profile

The local file inspected for this project contains 106,674 rows and 18 columns, covering 15 projects and 2003–2019. Every row represents one software commit/change revision; `commit_id` is its identifier. It is not a row for an issue, file, line, or bug-fix event.

The observed target distribution is 78,435 `buggy=False` rows (73.53%) and 28,239 `buggy=True` rows (26.47%). `author_date` is Unix epoch seconds and is complete in the inspected file. The source documentation describes `buggy` as indicating whether the commit introduced a bug and explains that bug-inducing commits were identified with an SZZ-based process. This is a derived proxy label, not independently verified ground truth.

`commit_id`, `project`, `fix`, `year`, and `author_date` are retained in the raw data for inspection. They are not predictors in the current model. `fix` is excluded because its exact timing and meaning were not sufficiently documented; using post-hoc information could leak the target. `author_date` is used for chronological ordering only.

Full schema and source notes are in [data/README.md](data/README.md).

## Architecture

```text
ApacheJIT CSV
    |
    +--> src/data.py       loading and schema validation
    +--> src/eda.py        descriptive EDA and charts
    +--> src/split.py      chronological train/validation/test split
    +--> src/pipeline.py   training-only imputation/scaling pipeline
    +--> src/train.py      model comparison and saved pipeline
    +--> src/evaluate.py   validation selection and one-time test evaluation
    +--> src/predict.py    reusable batch/one-row inference
    +--> src/review_budget.py  precision@K, recall@K, and baselines
    +--> src/demo.py       held-out Demo Lab cases and local score sensitivities
    |
    +--> app.py            Streamlit prediction and reporting interface
```

Important directories and files:

```text
data/raw/                 unchanged ApacheJIT files
data/README.md            source, schema, citation, and usage notes
src/                      reusable data, modeling, evaluation, and inference code
tests/                    pytest tests for contracts and edge cases
models/                   generated joblib model artifact
reports/                  generated summaries, CSV metrics, and figures
app.py                    Streamlit interface
requirements.txt          dependency ranges
```

Raw data, generated models, reports/figures, and local environments are excluded by `.gitignore`.

## Installation

From the project root, create a fresh environment. Windows PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell activation is unavailable, use the environment interpreter directly:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

macOS/Linux equivalent:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

## Dataset preparation

Download the file from the official Zenodo record or its verified direct file. On Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force data\raw | Out-Null
Invoke-WebRequest -Uri "https://zenodo.org/records/5907002/files/apachejit_total.csv?download=1" -OutFile data\raw\apachejit_total.csv
Get-FileHash data\raw\apachejit_total.csv -Algorithm MD5
```

The hash should be `98E102741020B3B13798D1E10A350689`. Do not edit the raw CSV. The supplied author artifacts, if downloaded for protocol inspection, are `apachejit_train.csv`, `apachejit_test_large.csv`, and `apachejit_test_small.csv`; the small test file is a subset of the large test and is not an independent test set.

Inspect and validate the dataset:

```powershell
python -m src.eda
python -m src.split
```

These commands write descriptive reports under `reports/` and leave the raw file unchanged.

## Training and evaluation

Train the DummyClassifier, Logistic Regression, and Random Forest candidates:

```powershell
python -m src.train
```

This writes `models/reviewradar_selected.joblib` and `reports/model_comparison.csv`. The selected model is chosen using validation PR-AUC; the final test labels are not used for selection.

Run evaluation after training:

```powershell
python -m src.evaluate
```

This writes classification, confusion-matrix, calibration, review-budget, configuration, and figure artifacts under `reports/`. The final test is evaluated once after the model and threshold are frozen.

Run the controlled validation-only experiment log:

```powershell
python -m src.experiments
```

This compares a small pre-declared grid on train/validation only and writes
`reports/model_experiments.csv`. It does not replace the saved model or tune on
the held-out test set. The pre-change reference is in
[`reports/baseline.md`](reports/baseline.md).

Run the complete test suite:

```powershell
python -m pytest -q
```

The latest verified run in this workspace passed **56 tests**. This is a reproducibility record, not a claim about statistical model quality.

## Chronological evaluation and leakage prevention

The approved split uses `author_date` in UTC:

| Split | Rows | Date range | Buggy=False | Buggy=True | Buggy rate |
|---|---:|---|---:|---:|---:|
| Train | 76,832 | 2003-09-11 to 2016-12-31 | 54,365 | 22,467 | 29.24% |
| Validation | 10,525 | 2017-01-01 to 2017-12-31 | 7,730 | 2,795 | 26.56% |
| Test | 19,317 | 2018-01-01 to 2019-12-26 | 16,340 | 2,977 | 15.41% |

No random fallback is used. Candidate pipelines are fit on training rows only. Imputation and scaling are inside scikit-learn pipelines. Model selection and threshold tuning use validation data only. The selected pipeline is refit on train plus validation after those decisions, then evaluated on the held-out test with the frozen threshold.

Identifiers and target/metadata columns are excluded from the feature matrix. Inference validates the same 12 named feature columns and preserves `commit_id` only for display.

## Measured model results

The following candidate results are from `reports/model_comparison.csv`, measured on the 2017 validation split at the fixed 0.5 threshold. `PR-AUC` is average precision for `buggy=True`.

| Model | Precision | Recall | F1 | PR-AUC |
|---|---:|---:|---:|---:|
| DummyClassifier (prior) | 0.0000 | 0.0000 | 0.0000 | 0.2656 |
| Logistic Regression | 0.4683 | 0.6401 | 0.5409 | 0.5544 |
| Random Forest | 0.5677 | 0.6132 | 0.5896 | 0.6233 |

Random Forest was selected by validation PR-AUC. The threshold was then tuned on validation by maximum F1 to `0.440246`. With that frozen threshold:

| Split | Precision | Recall | F1 | PR-AUC | Brier score |
|---|---:|---:|---:|---:|---:|
| Validation, tuned | 0.5327 | 0.6902 | 0.6013 | 0.6233 | 0.1566 |
| Held-out test, frozen | 0.3430 | 0.6664 | 0.4529 | 0.4138 | 0.1482 |

These are historical ApacheJIT benchmark measurements. They are not guarantees for new repositories or future data, and they do not show that ReviewRadar prevents bugs.

## Review-budget analysis

`src/review_budget.py` ranks by score only, then uses known labels only to measure evaluation outcomes. It reports precision@K among selected changes and recall@K among all positive examples. Ties are resolved stably by input order. The random baseline is a reproducible 1,000-run simulation with seed 42. The `la + ld` heuristic ranks by change size.

The measured results below are from `reports/evaluation_review_budget.csv`:

| Split | Budget | Approach | K | Precision@K | Recall@K |
|---|---:|---|---:|---:|---:|
| Validation | 5% | Model | 527 | 0.8330 | 0.1571 |
| Validation | 5% | Random simulation mean | 527 | 0.2643 | 0.0498 |
| Validation | 5% | `la + ld` | 527 | 0.7135 | 0.1345 |
| Validation | 10% | Model | 1,053 | 0.7407 | 0.2791 |
| Validation | 10% | Random simulation mean | 1,053 | 0.2650 | 0.0999 |
| Validation | 10% | `la + ld` | 1,053 | 0.6562 | 0.2472 |
| Validation | 20% | Model | 2,105 | 0.6380 | 0.4805 |
| Validation | 20% | Random simulation mean | 2,105 | 0.2655 | 0.1999 |
| Validation | 20% | `la + ld` | 2,105 | 0.5810 | 0.4376 |
| Test | 5% | Model | 966 | 0.5300 | 0.1720 |
| Test | 5% | Random simulation mean | 966 | 0.1541 | 0.0500 |
| Test | 5% | `la + ld` | 966 | 0.5062 | 0.1643 |
| Test | 10% | Model | 1,932 | 0.4674 | 0.3033 |
| Test | 10% | Random simulation mean | 1,932 | 0.1541 | 0.1000 |
| Test | 10% | `la + ld` | 1,932 | 0.4581 | 0.2973 |
| Test | 20% | Model | 3,864 | 0.3996 | 0.5186 |
| Test | 20% | Random simulation mean | 3,864 | 0.1541 | 0.2000 |
| Test | 20% | `la + ld` | 3,864 | 0.3773 | 0.4898 |

These values describe historical label enrichment in a ranked benchmark. They do not measure review hours saved, cost savings, bugs prevented, or the causal effect of reviewing a selected change. Operational benefits are **pending** because no review-time, cost, intervention, or production-outcome data is available.

## Inference and Streamlit app

Score a schema-compatible CSV without retraining:

```powershell
python -m src.predict --input path\to\prediction_input.csv --output reports\predictions.csv
```

The input must contain the 12 trained numeric columns and may contain `commit_id`. Target and metadata columns are intentionally rejected. The output contains the identifier when supplied, `risk_score`, and the configured threshold flag.

The model artifact and evaluation configuration are generated locally and are
ignored by Git. A fresh clone must create them before launching the app:

```powershell
python -m src.train
python -m src.evaluate
```

`src.train` writes `models/reviewradar_selected.joblib`. `src.evaluate` writes
`reports/evaluation_config.json` and the evaluation outputs used by the UI.
Run these commands after downloading the ApacheJIT CSV. They do not modify the
raw dataset.

Launch the interface:

```powershell
python -m streamlit run app.py
```

Open the local URL printed by Streamlit, normally `http://localhost:8501`. The interface has Overview, Analyze Changes, Model Performance, Demo Lab, and About/Methodology workspaces. The default sample mode uses 50 actual ApacheJIT rows with target and metadata columns removed. Analyze Changes provides CSV validation, configurable review budget and score filters, ranked results, a change-metric detail view, global Random Forest feature-importance context, and CSV export. Demo Lab shows real examples from the frozen held-out test split, including true/false positives and negatives when available. Its local explanation uses one-feature-at-a-time replacement with a training median; this is behavior evidence, not causation.

### Capturing real screenshots

The repository includes a real local screenshot at
[`docs/images/reviewradar-overview.png`](docs/images/reviewradar-overview.png).
It was captured from the Streamlit app running locally; it is not a hosted
deployment or a synthetic result. To capture additional screenshots:

1. Run `python -m streamlit run app.py` after training and evaluation.
2. Open the printed local URL in a browser.
3. Capture the “Predict & prioritize” page in Sample data mode with the ranked queue visible.
4. Select “Model performance” in the sidebar and capture the saved benchmark tables/plot.
5. Store reviewed screenshots under `docs/images/` and link them with relative paths.

Any screenshot should show the actual local app state and should not be described as a deployment URL or production result.

## Limitations and responsible use

- ApacheJIT labels are SZZ-derived proxies and may include attribution noise.
- The positive rate falls from 29.24% in training to 15.41% in the held-out test, demonstrating dataset shift within this benchmark.
- ApacheJIT covers historical Apache projects; performance may not transfer to another language, repository, workflow, or time period.
- The current model uses only numeric change metrics and excludes project identity, code content, issue semantics, and review context.
- Scores are ranking signals, not calibrated probabilities. Calibration diagnostics are reported, but no calibrated-risk guarantee is made.
- A top-ranked change is not definitely buggy, and a low-ranked change is not definitely safe.
- The current evaluation does not measure intervention outcomes, reviewer workload, costs, or production defects after deployment.
- The current Streamlit app is a local demonstration. It has no authentication, database, deployment hardening, or audit trail.

Use ReviewRadar to support human review prioritization and retrospective analysis. Keep normal engineering review, testing, and ownership decisions in place, and avoid using the score as the sole basis for blocking releases or assigning blame.
