from pathlib import Path

import pandas as pd

from src.demo import load_demo_cases


ROOT = Path(__file__).resolve().parents[1]


def test_demo_cases_are_real_held_out_records_with_consistent_outcomes() -> None:
    cases = load_demo_cases()
    assert cases
    outcome_keys = {case["outcome_key"] for case in cases}
    assert {"true_positive", "true_negative", "false_positive", "false_negative"}.issubset(outcome_keys)
    for case in cases:
        assert case["source"] == "ApacheJIT held-out chronological test split"
        assert case["case_id"].startswith("apachejit-test-")
        assert len(case["features"]) == 12
        if case["outcome_key"] != "borderline":
            assert case["actual_buggy"] == (case["outcome_key"] in {"true_positive", "false_negative"})
            assert case["predicted_buggy"] == (case["outcome_key"] in {"true_positive", "false_positive"})
        assert len(case["explanation"]) == 5
        assert pd.notna(case["risk_score"])


def test_demo_explanations_are_sorted_by_absolute_score_sensitivity() -> None:
    cases = load_demo_cases()
    for case in cases:
        magnitudes = [abs(float(row["score_delta_vs_median"])) for row in case["explanation"]]
        assert magnitudes == sorted(magnitudes, reverse=True)
