# ReviewRadar evaluation summary

Positive class: **`buggy=True`**.

Selected model: **`random_forest`** using validation PR-AUC. Threshold selected on validation by maximum F1: **0.440246**.

The held-out test was evaluated once after the model and threshold were frozen. Test results are historical benchmark results on ApacheJIT, not evidence that the model prevents bugs or will generalize to other repositories.

Undefined precision/recall/F1 cases are handled with `zero_division=0`. In particular, the DummyClassifier predicts no positive rows at threshold 0.5, so those three metrics are reported as 0; this does not mean the validation set contains no buggy changes.

Review-budget analysis reports top **5%, 10%, 20%** of changes. Random selection is a reproducible 1000-run simulation (seed=42); it is a benchmark comparison, not a measured estimate of money, bugs prevented, or review hours saved.

See `evaluation_classification_metrics.csv`, `evaluation_review_budget.csv`, `evaluation_calibration_metrics.csv`, and `evaluation_config.json` for actual outputs.
