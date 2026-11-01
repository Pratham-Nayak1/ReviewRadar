# ReviewRadar final ML readiness audit

Audit date: 2026-10-26

This is an evidence audit of the existing model, protocol, experiment logs,
saved pipeline, and Demo Lab. It does not redesign the UI, replace the saved
model, or tune against the final test set.

## Readiness conclusion

The project is reproducible as an educational historical benchmark and the
inference pipeline is internally consistent. It is not ready to be described
as a production defect predictor. The main validation-to-test gap is supported
by a substantial temporal/base-rate and project-composition shift, with
additional feature-distribution changes. The available evidence does not
support claiming a reliable F1 improvement from the `min_samples_leaf=5`
experiment.

## 1. Reconstructed protocols

### Saved baseline/selected model

The original candidate comparison in `src/train.py` fits each candidate on the
training split only and measures validation performance at threshold 0.5. The
three candidates are:

- `DummyClassifier(strategy="prior")`
- class-balanced Logistic Regression with median imputation and StandardScaler
- class-balanced-subset Random Forest with median imputation and no scaling

All use the same 12 features, target, chronological split, and scikit-learn
pipeline construction. The Random Forest is selected by validation average
precision, with validation F1/model name as deterministic tie-breakers. The
selected pipeline is refit on train+validation and saved. `src/evaluate.py`
then selects the threshold on validation by maximum F1 and evaluates the
refit model once on the held-out test.

### Leaf-5 candidate experiment

`src/experiments.py` uses the same target, positive class, 12 features,
chronological split, median-imputation pipeline, and validation-only
threshold selection for every experiment. The `rf_balanced_leaf5` candidate
changes only the Random Forest configuration to
`min_samples_leaf=5`, `n_estimators=150`, with the same
`balanced_subsample`, random seed, and parallelism. Its validation threshold
was selected by the same maximum-F1 rule.

Therefore the validation comparison between the saved Random Forest and the
leaf-5 candidate is protocol-compatible. The candidate is not the saved
artifact. Its previously recorded test check was performed after validation
selection, not used to choose the candidate, and is retained only as an
experiment log.

## 2. Data, prevalence, and split evidence

The documented target is `buggy=True`, meaning the ApacheJIT label indicates
that a commit introduced a bug. It is an SZZ-derived proxy label. Each row is
a software commit/change revision.

| split | rows | UTC coverage | buggy=False | buggy=True | positive rate |
|---|---:|---|---:|---:|---:|
| train | 76,832 | 2003-09-11 to 2016-12-31 | 54,365 | 22,467 | 29.24% |
| validation | 10,525 | 2017 | 7,730 | 2,795 | 26.56% |
| test | 19,317 | 2018-01-01 to 2019-12-26 | 16,340 | 2,977 | 15.41% |

The test positive rate is 11.14 percentage points lower than validation and
13.83 points lower than training. At a fixed operating threshold, that alone
can reduce precision and F1 even when ranking quality is partly retained.

The split audit found zero exact duplicate rows and zero duplicate commit IDs
within each split. Commit-ID overlap is zero for train/validation,
train/test, and validation/test. The official supplied artifacts were also
checked previously; the official small test is a subset of the large test and
is not treated as an independent test set.

The split remains chronological and the model does not use `author_date` or
`year` as predictors. No random fallback is present.

## 3. Metrics and confusion matrices

All metrics below use positive class `buggy=True`.

### Reference baselines and leaf-5 validation candidate

These are train-only fits evaluated on the 2017 validation split. Dummy and
Logistic Regression are reference baselines; the leaf-5 Random Forest is the
controlled candidate experiment.

| model | threshold | precision | recall | F1 | average precision | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dummy prior | 0.500000 | 0.0000 | 0.0000 | 0.0000 | 0.2656 | 7,730 | 0 | 2,795 | 0 |
| Logistic balanced | 0.500000 | 0.4683 | 0.6401 | 0.5409 | 0.5544 | 5,699 | 2,031 | 1,006 | 1,789 |
| Saved RF, leaf=2 | 0.440246 | 0.5327 | 0.6902 | 0.6013 | 0.6233 | 6,038 | 1,692 | 866 | 1,929 |
| RF candidate, leaf=5 | 0.488876 | 0.5226 | 0.7152 | 0.6039 | 0.6257 | 5,904 | 1,826 | 796 | 1,999 |

The Dummy prior is an appropriate minimal reference because it uses the class
prior and no feature signal. Since the training/validation prior is below
0.5, its default threshold predicts no positive rows; its zero precision,
recall, and F1 are expected at that threshold. Its average precision is the
validation prevalence, as expected for a non-informative score. A constant-
positive or stratified dummy could be an additional sensitivity baseline, but
its absence does not invalidate the current comparison.

### Frozen test result for the saved model

The saved leaf-2 Random Forest was refit on train+validation and evaluated
once with the validation-selected threshold:

| model/split | threshold | precision | recall | F1 | average precision | Brier | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Saved RF, held-out test | 0.440246 | 0.3430 | 0.6664 | 0.4529 | 0.4138 | 0.1482 | 12,539 | 3,801 | 993 | 1,984 |

### Leaf-5 candidate test record

The experiment log records the candidate's single frozen test check as
precision 0.3350, recall 0.7007, F1 0.4533, average precision 0.4207, and
Brier score 0.1686 at threshold 0.488876. Its test confusion matrix was not
written to a report artifact. It cannot be reconstructed from those aggregate
metrics alone, and this audit deliberately does not rescore the final test
just to create a missing table. That is a documentation gap, not a reason to
invent counts.

## 4. Time, project, and feature shift

The validation and test periods are different calendar periods by design. All
15 projects appear in all three splits, but their proportions and label rates
change materially. Examples:

| project | validation rows/rate | test rows/rate |
|---|---:|---:|
| apache/camel | 2,132 / 9.19% | 4,297 / 3.42% |
| apache/flink | 1,198 / 38.98% | 4,210 / 19.24% |
| apache/ignite | 1,653 / 28.98% | 1,407 / 38.66% |
| apache/hbase | 856 / 34.58% | 1,282 / 22.15% |
| apache/kafka | 637 / 55.42% | 1,003 / 29.31% |
| apache/hadoop | 1,766 / 0.00% | 3,508 / 0.00% |

The model excludes `project`, so it cannot adjust directly for project-level
base-rate differences. This is a defensible leakage-control choice, but it
also makes project-mix shift a plausible contributor to the precision drop.

Feature means also shift relative to training. Standardized mean shifts from
training to test include `aexp` +0.300, `ndev` +0.294, `age` +0.266,
`nuc` -0.192, and `arexp` +0.067. Validation shifts include `ndev` +0.360
and `age` +0.305. Medians move as well, for example training/test `age`
10.67/25.00 and `asexp` 49.00/14.00. These are distributional evidence;
they do not identify a causal source of the performance gap.

## 5. False-positive and false-negative analysis

For the saved Random Forest's train-only validation fit at threshold 0.440246:

| outcome | rows | median la | median ld | median nf | median ent | median age | median ndev | median nuc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| true negative | 6,038 | 13.0 | 4.0 | 2.0 | 0.656 | 17.0 | 4.0 | 13.5 |
| true positive | 1,929 | 366.0 | 86.0 | 11.0 | 2.554 | 31.6 | 2.86 | 23.8 |
| false positive | 1,692 | 212.0 | 41.0 | 8.0 | 2.142 | 35.0 | 3.0 | 18.6 |
| false negative | 866 | 38.5 | 8.0 | 3.0 | 0.975 | 16.0 | 5.0 | 22.0 |

Supported observations:

- True positives generally have much larger change size and entropy than true
  negatives in this validation sample.
- False positives also have relatively large changes and entropy, so the
  available metrics can trigger on change magnitude without the historical
  label being positive.
- False negatives are often smaller changes than true positives, which shows
  that some positive labels are not separated by size/entropy signals.
- `apache/hadoop` contributed 604 validation false positives while its observed
  validation and test positive rate was 0%. This is strong evidence that
  project-specific prevalence matters for the current metric-only model.

Hypotheses not established by this audit include whether code semantics,
review context, SZZ attribution noise, project conventions, or temporal label
process changes caused individual errors. The feature summaries cannot prove
those explanations.

## 6. Leakage and preprocessing audit

The saved configuration names the same 12 feature columns used by the saved
pipeline, and the audit verified `pipeline.feature_names_in_` matches the
configuration. The saved pipeline steps are `preprocess` and `estimator`.
Median imputation is inside the pipeline; Logistic Regression additionally
uses StandardScaler. No preprocessing object is fitted outside the training
pipeline.

Excluded columns are `commit_id`, `buggy`, `fix`, `year`, `author_date`, and
`project`. `commit_id` is display-only. `fix` remains excluded because its
provenance/timing is unresolved. No duplicate IDs, exact duplicates, or
cross-split commit overlap were found. The audit found no evidence of target
leakage or feature-order mismatch.

## 7. Additional experiments and decision

The controlled validation-only grid found:

| experiment | validation F1 | validation average precision |
|---|---:|---:|
| RF balanced, leaf=5 | 0.6039 | 0.6257 |
| RF unbalanced, leaf=2 | 0.6002 | 0.6200 |
| RF balanced, leaf=1 | 0.5962 | 0.6170 |
| Logistic balanced | 0.5469 | 0.5544 |
| Logistic unbalanced | 0.5417 | 0.5491 |

The leaf-5 validation result is only a small improvement over the saved
leaf-2 model's validation F1 0.6013 and does not establish generalization.
Its recorded test F1 0.4533 is effectively the same as the saved model's
0.4529, while its Brier score is worse. No additional model or threshold is
supported strongly enough to replace the saved artifact. Further work should
use rolling-origin validation or a pre-registered project/time generalization
study rather than repeatedly checking the final test.

## 8. Demo Lab verification

`src/demo.py` loads the raw dataset, applies the same chronological split,
loads the saved joblib pipeline and evaluation configuration, and computes
each case score through `ReviewRadarPredictor`. The threshold is read from
`evaluation_config.json`; it is not invented in the UI.

Programmatic verification produced five real cases from the held-out test
split:

| case | score | threshold | actual | predicted | explanation rows |
|---|---:|---:|---:|---:|---:|
| true positive | 0.983870 | 0.440246 | True | True | 5 |
| true negative | 0.439974 | 0.440246 | False | False | 5 |
| false positive | 0.968802 | 0.440246 | False | True | 5 |
| false negative | 0.008123 | 0.440246 | True | False | 5 |
| borderline | 0.440256 | 0.440246 | False | True | 5 |

All are real ApacheJIT held-out records; none is synthetic. The borderline
case is intentionally an additional category, not a replacement for one of
the four confusion-matrix outcomes. Explanations are one-feature-at-a-time
training-median score sensitivities. They are generated from model behavior,
not an LLM, and are explicitly not causal explanations.

## 9. What is working and what remains

Working and verified:

- Actual ApacheJIT schema and target are enforced.
- Chronological split and positive-class definition are consistent.
- Training preprocessing is pipeline-bound.
- Saved model schema matches evaluation configuration.
- Dummy and Logistic references are reproducible.
- Validation-only experiments are logged.
- Demo Lab categories and scores are programmatically consistent.
- The current complete suite contains 56 tests and passes in the verified
  workspace run; the app compiles and browser/AppTest checks were previously
  completed.

Remaining work/limitations:

- The leaf-5 candidate test confusion matrix was not saved; do not claim it.
- Only one temporal holdout is available for the current benchmark. Rolling or
  project-grouped validation would better characterize generalization.
- The SZZ-derived label can contain attribution noise and is not independent
  ground truth.
- No calibration layer is fitted; scores must remain ranking signals.
- No prospective review intervention, reviewer workload, cost, or production
  outcome study exists.

## Reproduction commands

Run from `D:\Radar`:

```powershell
python -m src.eda
python -m src.split
python -m src.train
python -m src.evaluate
python -m src.experiments
python -m src.final_audit
python -m pytest -q
python -m pytest -q tests/test_demo.py
python -m streamlit run app.py
```

`src.final_audit` reads the final test rows for prevalence, distribution, and
duplicate audits but deliberately reports
`test_predictions_scored_by_this_script: false`. It fits and scores only
train/validation models for its reconstructed validation tables.
