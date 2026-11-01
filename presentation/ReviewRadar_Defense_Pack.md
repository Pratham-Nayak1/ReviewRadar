# ReviewRadar Defense Pack

Answers below distinguish measured evidence from interpretation. Repository
references point to source files and reports rather than unsupported claims.

## Problem and motivation

### 1. What problem does ReviewRadar solve?

Answer: It ranks software changes for review prioritization when review capacity is limited.

Technical: The model consumes change-time metrics and returns a positive-class score.

Evidence: `app.py`, `src/predict.py`, `reports/evaluation_protocol.md`.

Caveat: No prospective review outcome has been measured.

### 2. Why not rely on ordinary code review?

Answer: The tool is intended to help order reviewer attention, not replace code review.

Technical: A budgeted ranking can surface higher-scoring changes earlier, while reviewers still inspect the code.

Evidence: `src/review_budget.py`, `app.py`.

Caveat: Ranking usefulness is shown only on historical labels.

### 3. Who should use it?

Answer: Researchers, students, and engineers exploring historical change prioritization.

Technical: The current UI accepts CSVs and does not integrate with GitHub or CI.

Evidence: `app.py`, `README.md`.

Caveat: Production teams need an integration and prospective evaluation.

## Architecture and implementation

### 4. What happens from CSV upload to result?

Answer: The app validates columns, loads the saved pipeline, scores rows, and ranks them.

Technical: `src/predict.py` is the inference layer and preserves display identifiers without passing them to the estimator.

Evidence: `src/predict.py`, `app.py`.

Caveat: The score is not a proof of a bug.

### 5. Why use a saved pipeline?

Answer: To keep preprocessing identical between training and inference.

Technical: Imputation and scaling are pipeline steps fitted during training.

Evidence: `src/pipeline.py`, `src/train.py`.

Caveat: A saved artifact still needs version and checksum management in deployment.

### 6. What model is saved?

Answer: The existing saved artifact is a balanced-subset Random Forest trained on ApacheJIT.

Technical: It uses 300 trees, minimum leaf size 2, seed 42, and the 12 permitted numeric metrics.

Evidence: `reports/evaluation_config.json`, `src/train.py`.

Caveat: The new feature experiment did not replace it.

### 7. Why classical machine learning?

Answer: The project prioritizes interpretability, small infrastructure, and reproducible tabular baselines.

Technical: Logistic Regression, Random Forest, HGB, and Dummy provide complementary references.

Evidence: `src/train.py`, `src/jit_defect4j_benchmark.py`.

Caveat: More complex models were not justified by this scope.

## Data and labels

### 8. What is one row?

Answer: One software commit or change revision in ApacheJIT and JIT-Defect4J.

Technical: The ID fields are `commit_id` and `commit_hash` respectively.

Evidence: `src/data.py`, `reports/jit_defect4j_data_audit.md`.

Caveat: ReDef is a different function-level task.

### 9. What is the positive label?

Answer: ApacheJIT uses `buggy=True`; JIT-Defect4J uses `is_buggy_commit=1`.

Technical: These are dataset-specific historical labels and are not interchangeable.

Evidence: `reports/label_provenance_audit.md`, `reports/jit_defect4j_evaluation_protocol.md`.

Caveat: SZZ and issue-link limitations can introduce label noise.

### 10. Why are the datasets not merged?

Answer: They differ in projects, labels, feature schemas, and split semantics.

Technical: Cross-dataset comparison is descriptive, not a pooled training task.

Evidence: `reports/dataset_compatibility_matrix.csv`, `reports/complete_project_audit.md`.

Caveat: A future harmonization study would need a separate protocol.

## Evaluation and leakage

### 11. How was ApacheJIT split?

Answer: By `author_date`: train before 2017, validation in 2017, test from 2018 onward.

Technical: The split is chronological and avoids random fallback.

Evidence: `src/split.py`, `reports/evaluation_protocol.md`.

Caveat: It is future within represented projects, not unseen-project evaluation.

### 12. Why is F1 lower on test?

Answer: The test prevalence is much lower and the feature/project distributions shift.

Technical: Validation F1 is 0.6013 at the selected threshold; test F1 is 0.4529.

Evidence: `reports/final_evaluation.md`, `reports/distribution_shift_analysis.md`.

Caveat: The artifacts do not prove one single causal reason.

### 13. Was the threshold tuned on test?

Answer: No in the recorded evaluation code.

Technical: Threshold selection uses validation maximum F1, then the selected pipeline is refit before one test score.

Evidence: `src/evaluate.py`, `reports/evaluation_config.json`.

Caveat: Code cannot prove what someone may have inspected manually outside the recorded workflow.

### 14. What prevents preprocessing leakage?

Answer: Preprocessing is inside scikit-learn pipelines and fitted on fit rows.

Technical: Median imputation and StandardScaler are pipeline steps.

Evidence: `src/pipeline.py`, `tests/test_data_split_pipeline.py`.

Caveat: New feature transformations need the same discipline.

### 15. What is the Dummy baseline for?

Answer: It establishes the score obtained without feature signal.

Technical: `DummyClassifier(strategy="prior")` gives a prevalence-based reference.

Evidence: `src/train.py`, `src/jit_defect4j_benchmark.py`.

Caveat: Threshold metrics can be zero when the dummy predicts no positives.

## Generalization and models

### 16. Why was Random Forest retained?

Answer: It remains the saved application model because it won the original ApacheJIT validation AP comparison and was frozen before test evaluation.

Technical: The later HGB experiments did not justify replacing the saved artifact.

Evidence: `reports/model_comparison.csv`, `reports/final_readiness.md`.

Caveat: This is a historical selection decision, not proof RF is universally best.

### 17. Why does Logistic Regression generalize better across projects in JIT-Defect4J?

Answer: It had the highest mean leave-one-project-out F1 in the recorded diagnostic.

Technical: Its simpler linear structure may be less sensitive to project-specific feature patterns, but this is a hypothesis.

Evidence: `reports/jit_defect4j_project_holdout_results.csv`, `reports/generalization_audit.md`.

Caveat: The diagnostic uses correlated project folds and does not prove causality.

### 18. What did the feature experiment improve?

Answer: Project-as-of normalization improved ApacheJIT development-fold ranking metrics, but not JIT-Defect4J RF metrics.

Technical: Apache RF AP rose from 0.6268 to 0.6870 across two folds; JIT RF fell from 0.3003 to 0.1811.

Evidence: `reports/feature_ablation_results.csv`, `reports/feature_signal_audit.md`.

Caveat: Two folds are insufficient for adoption or future generalization claims.

### 19. Is project normalization safe?

Answer: The implemented experiment uses only prior same-project feature values and does not use labels.

Technical: Evaluation rows use fit-history medians; fit rows are processed in timestamp order.

Evidence: `src/feature_signal_audit.py`, `tests/test_feature_signal_audit.py`.

Caveat: Operational availability and stability need a genuinely later evaluation.

## Review prioritization

### 20. What does Precision@10% mean?

Answer: Among the top 10% of changes ranked by score, it is the observed positive-label fraction.

Technical: K uses ceiling and ties preserve input order.

Evidence: `src/review_budget.py`.

Caveat: Historical precision does not equal time saved.

### 21. What did ApacheJIT show at 10%?

Answer: Validation Random Forest precision was 0.7407 and recall was 0.2791.

Technical: This uses the 2017 validation partition and the saved score ranking.

Evidence: `reports/evaluation_review_budget.csv`.

Caveat: The test result is lower and the operating environment may differ.

### 22. Does the model save review hours?

Answer: The project does not measure that.

Technical: Random selection is a simulated benchmark comparison only.

Evidence: `src/review_budget.py`, `reports/evaluation_summary.md`.

Caveat: A prospective controlled review study is required.

## Product limitations

### 23. Is the score a probability?

Answer: It should be treated primarily as a ranking score.

Technical: Calibration diagnostics differ between ApacheJIT validation and test, and no calibration layer is saved.

Evidence: `reports/evaluation_calibration_metrics.csv`.

Caveat: A deployment-specific calibration study could change that interpretation.

### 24. What does the UI explain?

Answer: It shows available change metrics and model-behavior explanations.

Technical: These explanations are sensitivity/context summaries rather than causal feature effects.

Evidence: `src/predict.py`, `src/demo.py`, `app.py`.

Caveat: Explanations do not prove why a defect occurred.

### 25. What is missing for GitHub workflow?

Answer: Secure repository integration, metric extraction, asynchronous scoring, and monitoring.

Technical: The current app accepts CSV files and has no GitHub connector.

Evidence: `app.py`, `README.md`.

Caveat: Each addition would require a new security and evaluation review.

## Future work and integrity

### 26. Why is JIT-Defect4J test not a future holdout?

Answer: Its train, validation, and test timestamps overlap.

Technical: Commit IDs are disjoint, but chronology is not strictly ordered.

Evidence: `reports/jit_defect4j_evaluation_protocol.md`.

Caveat: It remains a useful supplied benchmark, just not prospective evidence.

### 27. What is the rolling-origin readiness status?

Answer: The workflow exists, but no eligible later dataset is available.

Technical: The default command stops at a missing future dataset path.

Evidence: `src/rolling_origin_evaluation.py`, `reports/rolling_origin_data_requirements.md`.

Caveat: Future evaluation requires a new immutable labeled collection.

### 28. What does zero-positive Hadoop mean?

Answer: It means no positive labels were observed in that analyzed slice.

Technical: It does not establish zero true defect risk.

Evidence: `reports/label_provenance_audit.md`, `reports/project_error_diagnosis.md`.

Caveat: Label coverage and SZZ limitations remain unresolved.

### 29. What would invalidate the current results?

Answer: A schema, label, duplicate, or chronology error would require freezing the claim and rerunning from the corrected provenance.

Technical: Existing reports record checksums, schemas, split definitions, and overlap audits.

Evidence: `reports/data_integrity_audit.csv`, `reports/evaluation_protocol.md`.

Caveat: License and external source changes still need maintainer confirmation.

### 30. What is the most important limitation?

Answer: There is no genuinely later independent holdout available for the next research question.

Technical: Existing test periods have already been inspected or overlap timestamps.

Evidence: `reports/rolling_origin_protocol.md`, `reports/rolling_origin_data_requirements.md`.

Caveat: Historical evidence remains useful for method development, not deployment guarantees.

### 31. What would you do next?

Answer: Collect and reserve a later labeled period, then run the fixed rolling-origin protocol once.

Technical: Freeze model set, feature contract, threshold policy, budgets, and reporting before scoring.

Evidence: `reports/rolling_origin_protocol.md`.

Caveat: The new period must meet provenance and positive-count requirements.

### 32. What ten concepts must I understand?

Answer: Prediction unit, positive class, prevalence, average precision, F1, threshold selection, chronological split, preprocessing leakage, project shift, and review-budget recall.

Technical: These concepts explain why results can change without a code bug and why ranking scores are not causal claims.

Evidence: `reports/evaluation_protocol.md`, `reports/generalization_audit.md`.

Caveat: Be precise about which dataset and split each number comes from.
