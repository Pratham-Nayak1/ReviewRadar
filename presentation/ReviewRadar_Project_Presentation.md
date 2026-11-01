# ReviewRadar Project Presentation

## Slide 1 ReviewRadar

Classical machine learning for software-change defect-risk ranking

Evidence-based project audit, benchmarks, and research limits

Speaker note: This is a ranking demonstrator, not a bug oracle or a production deployment.

## Slide 2 The problem

- Review capacity is limited.
- Historical change metrics can support a prioritization score.
- The output ranks changes for attention.
- It does not replace reviewer judgment or prove a defect.

## Slide 3 Intended workflow

1. Upload a schema-compatible change CSV.
2. Validate the input schema.
3. Score each change with the saved pipeline.
4. Rank changes by score under a review budget.
5. Inspect available metrics and model context.

## Slide 4 System architecture

See `presentation/architecture.mmd`.

Speaker note: `src/predict.py` is the inference boundary. `app.py` does not duplicate model logic.

## Slide 5 Data sources

| Dataset | Unit | Positive class | Evaluation meaning |
|---|---|---|---|
| ApacheJIT | Commit/change | `buggy=True` | Primary ReviewRadar benchmark |
| JIT-Defect4J | Commit | `is_buggy_commit=1` | Separate independent benchmark |
| ReDef | Function modification | Revert-anchored label | Separate task, not merged |

## Slide 6 ApacheJIT schema and split

- 106,674 rows in the local total CSV.
- 12 permitted numeric predictors.
- Train before 2017, validation in 2017, test from 2018 onward.
- Validation prevalence 26.56%; test prevalence 15.41%.
- `author_date` controls chronology. `year` is not trusted as the split field.

## Slide 7 Leakage safeguards

- IDs, project, target, dates, and unresolved post-hoc fields excluded.
- Imputation and scaling fit inside pipelines.
- Threshold selected on validation only.
- Final test used once after decisions freeze.
- Supplied JIT-Defect4J test is not a future holdout.

## Slide 8 ApacheJIT benchmark

| Split | Precision | Recall | F1 | AP |
|---|---:|---:|---:|---:|
| Validation | 0.5327 | 0.6902 | 0.6013 | 0.6233 |
| Frozen test | 0.3430 | 0.6664 | 0.4529 | 0.4138 |

Speaker note: The prevalence shift and feature/project shift limit the interpretation.

## Slide 9 JIT-Defect4J benchmark

| Model | Validation F1 | Validation AP | Supplied-test F1 |
|---|---:|---:|---:|
| Dummy | 0.1575 | 0.0855 | 0.1595 |
| Logistic | 0.3286 | 0.2564 | 0.2329 |
| Random Forest | 0.3366 | 0.2651 | 0.2516 |
| HistGradientBoosting | 0.3425 | 0.2789 | 0.2575 |

Speaker note: Do not compare these raw scores directly with ApacheJIT.

## Slide 10 Cross-project generalization

- JIT-Defect4J leave-one-project-out mean F1:
  - Logistic Regression 0.2559
  - Random Forest 0.0946
  - HistGradientBoosting 0.0596
- Model ranking changes by deployment setting.
- ApacheJIT measures within-project temporal transfer, not unseen projects.

## Slide 11 Feature-signal experiment

| Dataset/family | Model | Mean AP | Mean F1 |
|---|---|---:|---:|
| Apache all permitted | RF | 0.6268 | 0.6096 |
| Apache project-normalized | RF | 0.6870 | 0.6433 |
| JIT all permitted | RF | 0.3003 | 0.2174 |
| JIT project-normalized | RF | 0.1811 | 0.1395 |

Speaker note: Two development folds only. This is a candidate signal, not an adoption decision.

## Slide 12 Review prioritization

- ApacheJIT validation RF at top 10%: precision 0.7407, recall 0.2791.
- JIT-Defect4J validation HGB at top 10%: precision 0.3108, recall 0.3640.
- Random ranking is a simulation baseline.
- No hours, money, or bugs prevented are measured.

## Slide 13 Calibration and score meaning

- ApacheJIT validation Brier 0.1566, ECE 0.0961.
- ApacheJIT test ECE 0.1756.
- The UI should treat scores mainly as ranking signals.
- No calibration layer is saved.

## Slide 14 Product status

Implemented: CSV validation, saved-pipeline inference, ranking, filters, explanations, performance page, demo cases.

Missing: GitHub integration, live metric extraction, prospective review study, later labeled holdout.

## Slide 15 Reproducibility and tests

- Commands use `.venv\Scripts\python.exe`.
- Reports record source files, split rules, and seeds.
- Full suite: 56 passed after the final audit additions.
- Large local datasets are ignored and must be acquired separately.

## Slide 16 Conclusion

- ReviewRadar is a credible research demonstrator.
- Historical ranking signal exists, but it varies by dataset, time, and project.
- Project normalization is promising on ApacheJIT development folds only.
- A genuinely later holdout is required before stronger claims.

