# ReviewRadar controlled model experiments

The experiment runner is `python -m src.experiments`. It uses the existing
chronological train/validation split only. It does not load the final test
split, create an artifact, or tune on test labels. Thresholds in this table
were chosen independently on validation by maximum F1.

| experiment | threshold | precision | recall | F1 | average precision |
|---|---:|---:|---:|---:|---:|
| RF balanced, leaf=5 | 0.488876 | 0.5226 | 0.7152 | 0.6039 | 0.6257 |
| RF unbalanced, leaf=2 | 0.397889 | 0.5222 | 0.7055 | 0.6002 | 0.6200 |
| RF balanced, leaf=1 | 0.420794 | 0.5539 | 0.6454 | 0.5962 | 0.6170 |
| Logistic balanced | 0.436973 | 0.4346 | 0.7374 | 0.5469 | 0.5544 |
| Logistic unbalanced | 0.268748 | 0.4484 | 0.6841 | 0.5417 | 0.5491 |

The current saved model was selected earlier by validation average precision
from the required Dummy/Logistic/Random Forest comparison and remains the
production artifact. The leaf=5 variant was checked once on the held-out test
after validation selection: precision 0.3350, recall 0.7007, F1 0.4533,
average precision 0.4207, Brier score 0.1686. This is only a diagnostic
comparison; it does not justify repeated test-driven tuning.

Interpretation: the leaf=5 variant slightly improves the measured test F1
relative to the saved model's 0.4529, but its test average precision is only
0.4207 and its Brier score is worse. The change is not a decisive improvement
and would require a fresh, pre-declared evaluation cycle before adoption.
