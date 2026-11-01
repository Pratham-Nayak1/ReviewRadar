# ReviewRadar UI QA

Audit date: 2026-10-26

## Rendered pages checked

The local Streamlit app was served at `http://localhost:8501` and inspected
through a browser after the redesign.

- **Overview:** rendered the 50-row deterministic sample batch, actual live
  sample counts, threshold-flagged count, median score, histogram, and queue
  preview.
- **Analyze Changes:** rendered sample mode, input preview, review-budget
  control, threshold filter, minimum-score control, ranked queue, detail view,
  global feature context, and ranked-queue download control. Ranks now follow
  descending score while source row indices remain stable for detail lookup.
- **Model Performance:** rendered validation candidate comparison, frozen
  validation/test results, held-out PR curve, budget comparison, calibration
  diagnostics, and confusion-matrix table. The page states that changing the
  displayed budget does not retrain or retune the model.
- **Demo Lab:** rendered all five available case choices: true positive, true
  negative, false positive, false negative, and borderline. Switching from the
  true-positive case to the false-negative case updated the identifier, score,
  ground truth, prediction, features, and local sensitivity table.
- **About / Methodology:** documents the target, score interpretation,
  chronology, leakage controls, and responsible-use limitations.

## Automated checks

- Existing regression tests: **56 passed**.
- Demo Lab tests: **2 passed**.
- The app loaded through Streamlit `AppTest` during QA without a prediction
  or performance exception in the prior application harness.
- Live HTTP endpoint returned status 200 during the audit.

The AppTest loop for every new workspace exceeded the local command time
window while the held-out Demo Lab was first materialized, so the browser
rendering checks above are the evidence for every page. The Matplotlib cache
warning seen in bare-mode diagnostics is environmental and did not prevent
the app or tests from running.

## Not browser-tested

File chooser upload was not automated in this pass. CSV schema and malformed
input paths remain covered by the inference tests in `tests/test_predict.py`
and `tests/test_predict_failures.py`.
