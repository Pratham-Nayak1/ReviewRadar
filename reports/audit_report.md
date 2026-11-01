# ReviewRadar audit report

Audit date: 2026-10-25

This report records checks performed against the current local workspace, the
actual ApacheJIT files, the saved artifacts, the pytest suite, and a running
Streamlit instance. Results below are measured observations, not deployment
claims.

## Executive summary

- The project has a usable end-to-end path: load ApacheJIT, create a timestamp
  split, fit a scikit-learn pipeline, evaluate it, load it for inference, and
  display ranked review candidates in Streamlit.
- The primary model is a Random Forest using 12 numeric change metrics. The
  positive class is `buggy=True`.
- The final test result is materially worse than validation: validation F1 is
  0.6013 after validation threshold selection, while the frozen test F1 is
  0.4529. Test average precision is 0.4138.
- The strongest evidence is for prioritization/ranking, not calibrated risk
  probabilities. The app labels scores accordingly.
- No credential-like strings or personal absolute paths were found in project
  text outside ignored data/model/cache locations. The workspace is not a Git
  repository yet, so Git history/status checks remain pending until the user
  initializes one.

## Repository and artifact audit

The workspace contains `app.py`, `src/`, `tests/`, `data/`, `models/`,
`reports/`, `README.md`, `data/README.md`, `requirements.txt`, and
`.gitignore`. The raw CSV files and the 515.63 MB joblib model are ignored by
default; they are reproducibly generated or downloaded instead of being
silently committed. `reports/` contains small reviewed CSV/Markdown/JSON
outputs, while generated figures are ignored.

The scan found no credential patterns, private-key headers, `C:\\Users\\...`
paths, or `D:\\Radar` paths in project text. This is a static scan, not a
secret-management guarantee.

## Dataset and leakage controls

The raw ApacheJIT total file has 106,674 commit rows, 18 columns, no missing
cells, no exact duplicate rows, and no duplicate `commit_id` values. Its
observed `author_date` coverage is 2003-09-11 through 2019-12-26 UTC. The
target is the documented SZZ-derived `buggy` label.

The modeling feature list is explicitly limited to the 12 numeric change
metrics in `src/split.py`. `commit_id` is retained only for display;
`buggy` is the target; `fix` is excluded because its timing/meaning is not
resolved; `year` and `author_date` are used for temporal protocol rather than
as predictors; and `project` is excluded from the first model to avoid
repository-identity shortcuts.

The official train/test artifacts were inspected. They have no shared commit
IDs, but the large official test artifact contains rows at or before the
training timestamp cutoff and the small test is a subset of the large test.
ReviewRadar therefore uses the explicit chronological split documented in
`reports/evaluation_protocol.md`.

The split is:

| split | rows | UTC period | buggy=False | buggy=True |
|---|---:|---|---:|---:|
| train | 76,832 | through 2016-12-31 | 54,365 | 22,467 |
| validation | 10,525 | 2017 | 7,730 | 2,795 |
| test | 19,317 | 2018-2019 | 16,340 | 2,977 |

All imputation/scaling is inside the scikit-learn pipeline. Model and
threshold selection use validation only. The saved model is refit on
train+validation after those decisions and the held-out test is evaluated
once in `src/evaluate.py`.

## Model and evaluation findings

Validation candidate results at threshold 0.5:

| model | precision | recall | F1 | average precision |
|---|---:|---:|---:|---:|
| DummyClassifier prior | 0.0000 | 0.0000 | 0.0000 | 0.2656 |
| Logistic Regression | 0.4683 | 0.6401 | 0.5409 | 0.5544 |
| Random Forest | 0.5677 | 0.6132 | 0.5896 | 0.6233 |

The Random Forest was selected by validation average precision. Its
validation-selected F1 threshold is 0.440246. At that threshold, validation
precision/recall/F1 are 0.5327/0.6902/0.6013. The frozen test result is
0.3430/0.6664/0.4529, with average precision 0.4138 and Brier score 0.1482.

The decrease is consistent with temporal dataset shift: test prevalence is
15.41%, versus 26.56% in validation, and the model was not recalibrated on
test labels. It is not evidence that the model prevents bugs.

## Review-budget findings

At the 10% review budget, the selected model achieved validation
Precision@K 0.7407 and Recall@K 0.2791; on test these were 0.4674 and
0.3033. The reproducible random-selection simulation on test was
Precision@K 0.1541 and Recall@K 0.1000. The `la + ld` heuristic was 0.4581
and 0.2973 on test.

These are historical benchmark outcomes. They do not measure money saved,
bugs prevented, or review hours saved. Operational value would require a
prospective study with an agreed review process and costs.

## Application audit

The app was run at `http://localhost:8501` and inspected in a browser.

- Sample mode loaded 50 legitimate schema-compatible ApacheJIT rows.
- Ranked queue scores, identifiers, available metrics, and global feature
  importance context rendered.
- Changing the review budget changed the queue size; a real browser action
  from 20% to 21% changed the budget size from 10 to 11 rows.
- The threshold filter changed the displayed sample queue from 11 rows to 2.
- The model-performance page displayed saved validation/test metrics,
  review-budget plots/tables, calibration diagnostics, and confusion matrices.
- The About page stated the score is not a definitive defect diagnosis.
- A presentation bug was found and fixed: the overview budget card now shows
  the selected sidebar budget instead of always showing the configured 10%.

The browser audit did not submit a real uploaded file. Upload schema paths are
covered by pytest and `src/predict.py`; a future UI smoke test could exercise
file selection separately.

## Tests and remaining limitations

The complete pytest command currently passes: **56 passed in 67.37 seconds**.
Streamlit `AppTest` checks also completed with zero prediction-page and
performance-page exceptions, and the live HTTP endpoint returned status 200.

Remaining limitations:

- The label is SZZ-derived and can contain attribution noise.
- The final test prevalence is substantially lower than validation, so a
  threshold selected for validation is not guaranteed to transfer.
- The score is not calibrated; Brier/calibration diagnostics are shown to
  prevent probability misinterpretation, not to claim calibration.
- Project-level and future cross-project generalization were not the primary
  protocol.
- A trained joblib is too large for a normal lightweight GitHub commit and
  raw data reuse rights are unresolved in the inspected Zenodo record.
- No Git repository exists yet. Initialize/version the project only after
  reviewing the ignored artifact policy.
