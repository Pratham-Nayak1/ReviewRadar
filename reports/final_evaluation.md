# Final evaluation record

## Selection protocol

The final selected artifact remains the required Random Forest candidate from
the original comparison. Model selection used validation average precision;
the operating threshold was selected on validation by maximum F1. The model
was then refit on train+validation and evaluated once on the held-out 2018-
2019 test split.

Positive class: `buggy=True`.

## Measured results

| split | precision | recall | F1 | average precision | Brier |
|---|---:|---:|---:|---:|---:|
| validation, threshold 0.440246 | 0.5327 | 0.6902 | 0.6013 | 0.6233 | 0.1566 |
| held-out test, same frozen threshold | 0.3430 | 0.6664 | 0.4529 | 0.4138 | 0.1482 |

Held-out test confusion matrix: TN 12,539; FP 3,801; FN 993; TP 1,984.

At the test top-10% review budget, the model selected 1,932 of 19,317
changes, with Precision@K 0.4674 and Recall@K 0.3033. The corresponding
random-selection simulation mean was 0.1541 and 0.1000.

These results are historical ApacheJIT benchmark measurements. They do not
measure intervention effects, review hours, money, or bugs prevented.

## Reproduction

```powershell
python -m src.train
python -m src.evaluate
```

The source-of-truth CSV reports are `reports/evaluation_classification_metrics.csv`,
`reports/evaluation_confusion_matrices.csv`, and
`reports/evaluation_review_budget.csv`.
