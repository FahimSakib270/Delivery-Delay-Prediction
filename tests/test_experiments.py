import numpy as np
import pandas as pd
import pytest
from src.experiments import supplementary_experiments


def fixture_frames():
    validation = pd.DataFrame({"probability": np.linspace(0, 1, 100),
                               "is_late": [0] * 50 + [0, 1] * 25})
    test = pd.DataFrame({"probability": np.linspace(0, 1, 80),
                        "is_late": [0, 1] * 40,
                        "lead_time_days": [5] * 8 + [20] * 72})
    return validation, test


def test_test_labels_do_not_fit_calibration_or_select_model_topups(cfg):
    validation, test = fixture_frames()
    first, policies, scores = supplementary_experiments(validation, test, cfg, 0.63)
    altered = test.copy()
    altered["is_late"] = 1 - altered["is_late"]
    second, altered_policies, altered_scores = supplementary_experiments(validation, altered, cfg, 0.63)
    np.testing.assert_array_equal(scores, altered_scores)
    np.testing.assert_array_equal(policies.flagged, altered_policies.flagged)
    assert first["calibration_fit_split"] == "validation_only"
    assert first["status"] == "exploratory_retrospective_not_confirmatory"


def test_rule_costs_are_computed_from_observed_confusion_counts(cfg):
    validation, test = fixture_frames()
    _, policies, _ = supplementary_experiments(validation, test, cfg, 0.63)
    rule = policies.set_index("policy").loc["promise_le_7_days"]
    assert rule["flagged"] == 8 and rule["tp"] == 4 and rule["fp"] == 4
    assert rule["fn"] == 36
    assert rule.hypothetical_cost_per_1000 == pytest.approx((4 * 4 + 36 * 25) / 80 * 1000)


def test_isotonic_ties_have_recomputed_ranking_metrics(cfg):
    validation, test = fixture_frames()
    summary, _, _ = supplementary_experiments(validation, test, cfg, 0.63)
    assert np.isfinite(summary["calibrated_test"]["average_precision"])
    assert np.isfinite(summary["calibrated_test"]["roc_auc"])
    assert "ties" in summary["ranking_note"]
