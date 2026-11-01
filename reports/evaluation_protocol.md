# ReviewRadar leakage-safe evaluation protocol

## Decision

Use an explicit `author_date` chronological split from `apachejit_total.csv`: train before 2017-01-01 UTC, validation during 2017, and final test from 2018-01-01 UTC onward. No random split is used.

The authors' supplied train/test artifacts are retained for reference. They are disjoint by commit ID, and the official `year` values place the large test set in 2017–2019. However, the large test file contains rows whose actual `author_date` is on or before the training cutoff, so it is not a strictly later timestamp holdout. ReviewRadar therefore uses the explicit timestamp split below as its primary protocol.

## ReviewRadar temporal split

| Split | Rows | Date range UTC | Buggy=False | Buggy=True | Buggy rate | Projects |
|---|---:|---|---:|---:|---:|---:|
| train | 76,832 | 2003-09-11T14:54:46+00:00 to 2016-12-31T10:46:02+00:00 | 54,365 | 22,467 | 29.24% | 15 |
| validation | 10,525 | 2017-01-01T07:30:34+00:00 to 2017-12-31T23:00:41+00:00 | 7,730 | 2,795 | 26.56% | 15 |
| test | 19,317 | 2018-01-01T13:07:32+00:00 to 2019-12-26T17:39:53+00:00 | 16,340 | 2,977 | 15.41% | 15 |

The split is created by `src/split.py::make_temporal_splits`; it sorts by timestamp and commit ID only for deterministic output. The raw CSV is not modified and no split files are required for model training.

## Official artifacts inspected

| Artifact | Rows | author_date range UTC | `year` range | Buggy=False | Buggy=True | Buggy rate | Date/year mismatches |
|---|---:|---|---|---:|---:|---:|---:|
| apachejit_train | 44,834 | 2003-09-11T14:55:37+00:00 to 2016-12-31T10:46:02+00:00 | 2003–2016 | 22,413 | 22,421 | 50.01% | 196 |
| apachejit_test_large | 30,111 | 2013-08-21T08:18:00+00:00 to 2019-12-26T17:39:53+00:00 | 2017–2019 | 24,293 | 5,818 | 19.32% | 961 |
| apachejit_test_small | 7,526 | 2015-06-15T18:04:02+00:00 to 2019-12-26T11:24:06+00:00 | 2017–2019 | 6,078 | 1,448 | 19.24% | 254 |

Official artifact overlap audit:

| Pair | Shared commit IDs | Shared safe-feature signatures | Shared projects |
|---|---:|---:|---:|
| apachejit_train / apachejit_test_large | 0 | 5 | 15 |
| apachejit_train / apachejit_test_small | 0 | 3 | 15 |
| apachejit_test_large / apachejit_test_small | 7526 | 7401 | 15 |

Official training `author_date` cutoff: `2016-12-31T10:46:02+00:00`.
Official large-test rows on or before that timestamp: **269**.
The official small test is a subset of the large test, so it must not be treated as an independent final test.

## Leakage controls for modeling

1. Keep `commit_id` out of features.
2. Exclude `buggy` from features because it is the target.
3. Exclude `fix` until its provenance proves it is available at commit time and not post-hoc.
4. Exclude `year` and raw `author_date` from the default feature matrix; use `author_date` for ordering only.
5. Start with the documented numeric change metrics (`la`, `ld`, `nf`, `nd`, `ns`, `ent`, `ndev`, `age`, `nuc`, `aexp`, `arexp`, `asexp`).
6. Fit imputation, scaling, encoding, feature selection, and any resampling inside a training-only pipeline. Tune models and thresholds on train/validation only.
7. Fit the selected pipeline once on train plus validation only after model and threshold decisions are frozen; evaluate the final test exactly once.
8. Do not use the final-test metrics, distribution, or labels to revise the model, features, or threshold.

## Related-record audit

The total dataset has unique commit IDs and no exact duplicate rows. Official train and large-test artifacts share no commit IDs. They do share project names, which is expected because projects contribute changes across time; this is not row duplication but means cross-project generalization should be reported separately if pursued. Repeated safe-feature signatures are counted in the audit table and are not automatically treated as duplicates because different commits can legitimately have identical metric values.

## Reproducibility

Run `python -m src.split` from the project root. It writes `reports/split_summary.csv` and this report. The split function is reusable from Python and never silently falls back to a random split.
