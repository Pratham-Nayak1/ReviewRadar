# ReviewRadar baseline

Recorded before the premium UI and Demo Lab changes on 2026-10-26.

## Application baseline

The existing Streamlit app had three pages: prediction/prioritization,
model performance, and About. Prediction mode supported deterministic sample
data, CSV upload, a review-budget slider, a threshold filter, a minimum-score
filter, ranked results, selected-change metrics, and global Random Forest
feature importance context.

The baseline app was served locally at `http://localhost:8501`. The endpoint
returned HTTP 200. Streamlit `AppTest` reported zero exceptions on the
prediction and model-performance pages.

## Data and protocol baseline

- Dataset: ApacheJIT `apachejit_total.csv`
- Rows/columns: 106,674 / 18
- Row unit: software commit/change revision
- Positive class: `buggy=True`
- Class counts: 28,239 positive and 78,435 negative
- Predictors: `la`, `ld`, `nf`, `nd`, `ns`, `ent`, `ndev`, `age`, `nuc`,
  `aexp`, `arexp`, `asexp`
- Excluded: `commit_id`, `buggy`, `fix`, `year`, `author_date`, `project`
- Split: train through 2016, validation during 2017, test during 2018-2019
- Split sizes: 76,832 / 10,525 / 19,317
- Selection: validation average precision
- Threshold: validation maximum-F1 threshold `0.4402460058`

## Baseline model results

Validation candidate metrics at threshold 0.5:

| model | precision | recall | F1 | average precision |
|---|---:|---:|---:|---:|
| DummyClassifier prior | 0.0000 | 0.0000 | 0.0000 | 0.2656 |
| Logistic Regression | 0.4683 | 0.6401 | 0.5409 | 0.5544 |
| Random Forest | 0.5677 | 0.6132 | 0.5896 | 0.6233 |

Frozen selected Random Forest:

| split | precision | recall | F1 | average precision | Brier |
|---|---:|---:|---:|---:|---:|
| validation, tuned threshold | 0.5327 | 0.6902 | 0.6013 | 0.6233 | 0.1566 |
| held-out test, frozen threshold | 0.3430 | 0.6664 | 0.4529 | 0.4138 | 0.1482 |

These are historical benchmark measurements. They do not establish a
production probability, cost saving, review-hour saving, or bug-prevention
effect.

## Reproduction commands

```powershell
python -m src.train
python -m src.evaluate
python -m pytest -q
python -m streamlit run app.py
```

Source artifacts: `reports/model_comparison.csv`,
`reports/evaluation_classification_metrics.csv`,
`reports/evaluation_config.json`, and `reports/evaluation_review_budget.csv`.
