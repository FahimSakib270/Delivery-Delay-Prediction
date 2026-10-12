"""Tests for :mod:`src.evaluation`.

The metric layer is where a project can quietly fool itself, so the tests pin down
the definitions rather than the numbers: what counts as a false positive, that the
threshold is chosen on one split and applied to another, and that a single-class draw
makes a ranking metric ``nan`` instead of raising.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_prep import TARGET_COLUMN
from src.evaluation import (
    add_analysis_buckets,
    bootstrap_ci,
    calibration_table,
    compute_metrics,
    cost_curve,
    cost_optimal_threshold,
    error_examples,
    evaluate_split,
    expected_calibration_error,
    expected_cost,
    permutation_importance_table,
    predict_positive_proba,
    select_threshold,
    subgroup_report,
    threshold_grid,
    threshold_table,
)


# --------------------------------------------------------------------------- metrics
def test_compute_metrics_counts_every_cell_of_the_confusion_matrix():
    metrics = compute_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], threshold=0.5)
    assert (metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"]) == (2, 0, 0, 2)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["prevalence"] == pytest.approx(0.5)
    assert metrics["average_precision"] == pytest.approx(1.0)
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["n"] == 4 and metrics["n_late"] == 2


def test_raising_the_threshold_only_removes_flags():
    probabilities = [0.1, 0.2, 0.8, 0.9]
    row = compute_metrics([0, 0, 1, 1], probabilities, threshold=0.85)
    assert (row["tp"], row["fp"], row["fn"]) == (1, 0, 1)
    assert row["recall"] == pytest.approx(0.5)
    assert row["f1"] == pytest.approx(2 / 3)
    assert row["predicted_positive_rate"] == pytest.approx(0.25)


def test_a_single_class_sample_yields_nan_ranking_metrics_instead_of_raising():
    """Nothing here may raise: a fold with no delays is a normal accident.

    scikit-learn's conventions are kept rather than overridden: ``roc_auc_score`` is
    ``nan`` (undefined), while ``average_precision_score`` returns ``0.0`` for a sample
    with no positive class. The report quotes the counts next to the metrics so a
    ``0.0`` is never mistaken for a measured failure.
    """
    metrics = compute_metrics([0, 0, 0], [0.1, 0.2, 0.3], threshold=0.5)
    assert metrics["n_late"] == 0
    assert np.isnan(metrics["roc_auc"])
    assert metrics["average_precision"] == 0.0
    assert metrics["recall"] == 0.0 and metrics["precision"] == 0.0
    assert metrics["specificity"] == pytest.approx(1.0)


def test_precision_lift_is_reported_against_the_base_rate():
    # 4 rows, 2 late: flagging exactly the two late orders gives 2x lift.
    metrics = compute_metrics([0, 0, 1, 1], [0.1, 0.2, 0.9, 0.9], threshold=0.5)
    assert metrics["precision_lift_over_prevalence"] == pytest.approx(2.0)


def test_threshold_grid_is_unique_sorted_and_inside_the_unit_interval():
    grid = threshold_grid()
    assert np.all(np.diff(grid) > 0)
    assert len(grid) == len(np.unique(grid))
    assert grid.min() > 0.0 and grid.max() < 1.0


# --------------------------------------------------------------------------- selection
def test_threshold_selection_maximises_f1_on_the_split_it_is_given():
    y_true = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
    y_prob = [0.05, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.7, 0.85, 0.95]
    threshold, table, info = select_threshold(y_true, y_prob, objective="max_f1")

    assert info["objective"] == "max_f1" and info["fallback"] is False
    assert table["f1"].max() == pytest.approx(info["selected"]["f1"])
    assert threshold == pytest.approx(float(info["selected"]["threshold"]))
    assert 0.0 < threshold < 1.0


def test_recall_at_precision_falls_back_and_says_so_when_the_floor_is_unreachable():
    rng = np.random.default_rng(3)
    y_true = rng.integers(0, 2, 400)
    y_prob = rng.random(400)  # pure noise: high precision is impossible
    threshold, _table, info = select_threshold(
        y_true, y_prob, objective="recall_at_prec", min_precision_floor=0.999
    )
    assert info["fallback"] is True
    assert "fell back" in info["note"]
    assert 0.0 < threshold < 1.0


def test_an_unknown_threshold_objective_is_rejected():
    with pytest.raises(ValueError):
        select_threshold([0, 1], [0.2, 0.8], objective="max_profit")


def test_threshold_table_covers_the_grid():
    table = threshold_table([0, 1, 0, 1], [0.2, 0.8, 0.4, 0.6])
    assert len(table) == len(threshold_grid())
    assert {"threshold", "precision", "recall", "f1"}.issubset(table.columns)


# --------------------------------------------------------------------------- uncertainty
def test_bootstrap_intervals_are_deterministic_and_bracketed():
    y_true = [0] * 80 + [1] * 20
    y_prob = np.linspace(0.01, 0.99, 100)
    intervals = bootstrap_ci(y_true, y_prob, threshold=0.5, n_boot=60, seed=1)

    assert intervals["average_precision"]["n_effective"] == 60
    for name, block in intervals.items():
        assert block["lower"] <= block["upper"], f"{name}: inverted interval"
        assert np.isfinite(block["point"])
    again = bootstrap_ci(y_true, y_prob, threshold=0.5, n_boot=60, seed=1)
    assert again["average_precision"]["lower"] == pytest.approx(intervals["average_precision"]["lower"])


def test_bootstrap_returns_nothing_for_an_empty_sample():
    assert bootstrap_ci([], [], n_boot=10) == {}


# --------------------------------------------------------------------------- cost
def test_expected_cost_arithmetic_is_explicit():
    # 2 late orders out of 4; flagging one of them correctly.
    panel = expected_cost([1, 1, 0, 0], [1, 0, 0, 0], cost_false_negative=25.0, cost_false_positive=4.0)
    assert (panel["true_positives"], panel["false_negatives"], panel["false_positives"]) == (1, 1, 0)
    assert panel["cost_total"] == pytest.approx(25.0)
    assert panel["cost_per_1000_orders"] == pytest.approx(6250.0)
    assert panel["cost_if_never_flagging_per_1000"] == pytest.approx(12500.0)
    assert panel["saving_vs_never_flagging_per_1000"] == pytest.approx(6250.0)


def test_flagging_everything_has_a_worse_cost_than_the_optimal_threshold():
    y_true = [0] * 60 + [1] * 40
    y_prob = np.concatenate([np.linspace(0.01, 0.6, 60), np.linspace(0.4, 0.99, 40)])
    threshold, curve = cost_optimal_threshold(y_true, y_prob, cost_false_negative=25.0,
                                              cost_false_positive=4.0)

    flag_all = expected_cost(y_true, [1] * 100, 25.0, 4.0)
    best = curve.loc[curve["threshold"] == threshold].iloc[0]
    assert best["cost_per_1000_orders"] <= flag_all["cost_per_1000_orders"]
    assert best["cost_per_1000_orders"] == pytest.approx(curve["cost_per_1000_orders"].min())


def test_cost_curve_has_one_row_per_threshold():
    curve = cost_curve([0, 1, 0, 1], [0.2, 0.8, 0.4, 0.6])
    assert len(curve) == len(threshold_grid())
    assert curve["cost_per_1000_orders"].ge(0).all()


# --------------------------------------------------------------------------- calibration
def test_calibration_error_is_near_zero_when_probabilities_are_honest():
    rng = np.random.default_rng(0)
    probability = 0.2
    y_true = (rng.random(4000) < probability).astype(int)  # observed rate ~= 0.2
    table = calibration_table(y_true, np.full(4000, probability))
    assert expected_calibration_error(table) < 0.02


def test_calibration_error_is_large_when_probabilities_are_overconfident():
    y_true = np.zeros(1000, dtype=int)  # nothing is ever late ...
    table = calibration_table(y_true, np.full(1000, 0.9))  # ... but we say 90%
    assert expected_calibration_error(table) > 0.8


def test_calibration_error_of_an_empty_table_is_nan_not_zero():
    """``nan`` is the honest answer for \"not computed\"; zero would read as \"perfect\"."""
    assert np.isnan(expected_calibration_error(pd.DataFrame()))


def test_calibration_table_reports_one_row_per_non_empty_bin():
    table = calibration_table([0, 1, 0, 1], [0.1, 0.2, 0.8, 0.9], n_bins=10)
    assert {"bin", "n", "mean_predicted", "observed_rate", "abs_gap"}.issubset(table.columns)
    assert table["n"].sum() == 4
    assert table["abs_gap"].ge(0).all()


# --------------------------------------------------------------------------- subgroups
def test_analysis_buckets_use_the_documented_boundaries():
    frame = pd.DataFrame({
        "lead_time_days": [0, 7, 8, 15, 16, 25, 26, 90],
        "n_items": [1, 1, 2, 5, 1, 3, 1, 2],
        "same_state": [1, 0, 1, 1, 0, 1, 1, 1],
        "purchase_month": [1, 11, 12, 6, 6, 6, 6, 6],
    })
    bucketed = add_analysis_buckets(frame)
    assert bucketed["lead_time_bucket"].tolist() == [
        "<=7 days", "<=7 days", "8-15 days", "8-15 days",
        "16-25 days", "16-25 days", ">25 days", ">25 days",
    ]
    assert bucketed["items_bucket"].tolist() == ["1 item", "1 item", "2+ items", "2+ items",
                                                 "1 item", "2+ items", "1 item", "2+ items"]
    assert bucketed["route_bucket"].tolist()[0] == "same-state seller"
    assert bucketed["route_bucket"].tolist()[1] == "cross-state seller"
    assert bucketed["month_bucket"].tolist()[0] == "1"


def test_analysis_buckets_are_built_from_booking_time_fields_only():
    """Slicing the errors must not be a back door for the outcome."""
    frame = pd.DataFrame({
        "lead_time_days": [10.0], "n_items": [2], "same_state": [1], "purchase_month": [12],
    })
    bucketed = add_analysis_buckets(frame)
    for column in ("lead_time_bucket", "items_bucket", "route_bucket", "month_bucket"):
        assert column in bucketed.columns
    assert not any("deliver" in column for column in bucketed.columns if column not in frame.columns)


def test_subgroup_report_flags_small_segments_instead_of_quoting_them(prepared):
    probabilities = np.full(len(prepared), 0.5)
    report = subgroup_report(prepared, probabilities, threshold=0.5, min_n=10_000)
    assert not report.empty
    assert report["below_min_n"].all(), "a segment smaller than min_n must be flagged"
    assert {"dimension", "level", "n", "recall", "precision", "below_min_n"}.issubset(report.columns)


def test_subgroup_report_keeps_only_the_largest_levels_per_dimension(prepared):
    probabilities = np.random.default_rng(0).random(len(prepared))
    report = subgroup_report(prepared, probabilities, threshold=0.5, min_n=0)
    assert report.groupby("dimension")["n"].count().max() <= 12
    assert report.groupby("dimension")["level"].nunique().eq(report.groupby("dimension").size()).all()


# --------------------------------------------------------------------------- errors and importance
def test_error_examples_separate_false_positives_from_false_negatives(prepared):
    probabilities = np.zeros(len(prepared))
    frame = prepared.copy().reset_index(drop=True)  # never mutate the shared fixture
    probabilities[:5] = 0.9   # flagged
    frame[TARGET_COLUMN] = 0                 # default: on time
    frame.loc[:4, TARGET_COLUMN] = 0         # flagged but on time -> false positive
    frame.loc[5:9, TARGET_COLUMN] = 1        # not flagged and late -> false negative
    frame.loc[10:19, TARGET_COLUMN] = 0      # not flagged, on time -> true negative

    examples = error_examples(frame, probabilities, threshold=0.5, n=10)
    assert set(examples) == {"false_positives", "false_negatives"}
    assert len(examples["false_positives"]) == 5
    assert (examples["false_positives"][TARGET_COLUMN] == 0).all()
    assert (examples["false_positives"]["probability"] >= 0.5).all()
    assert len(examples["false_negatives"]) == 5
    assert (examples["false_negatives"][TARGET_COLUMN] == 1).all()
    assert (examples["false_negatives"]["probability"] < 0.5).all()


def test_evaluate_split_returns_metrics_and_probabilities_for_one_frame(prepared, cfg):
    from src.features import build_model_pipeline
    from src.data_prep import MODEL_FEATURES

    subset = prepared.head(500)
    pipeline = build_model_pipeline("logistic_regression", cfg, seed=0)
    pipeline.fit(subset[MODEL_FEATURES], subset[TARGET_COLUMN].astype(int))

    metrics, probabilities = evaluate_split(pipeline, subset, MODEL_FEATURES, threshold=0.5)
    assert len(probabilities) == len(subset)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    assert metrics["n"] == len(subset)
    assert metrics["threshold"] == 0.5


def test_permutation_importance_names_the_features_it_permuted(prepared, cfg):
    from src.data_prep import MODEL_FEATURES
    from src.features import build_model_pipeline

    subset = prepared.head(400)
    pipeline = build_model_pipeline("logistic_regression", cfg, seed=0)
    pipeline.fit(subset[MODEL_FEATURES], subset[TARGET_COLUMN].astype(int))

    table = permutation_importance_table(
        pipeline, subset, subset[TARGET_COLUMN].astype(int), MODEL_FEATURES, n_repeats=2, seed=0
    )
    assert sorted(table["feature"]) == sorted(MODEL_FEATURES)
    assert table["importance_mean"].is_monotonic_decreasing
    assert table["importance_std"].ge(0).all()


def test_positive_probabilities_are_extracted_from_a_two_class_model(prepared, cfg):
    from src.data_prep import MODEL_FEATURES
    from src.features import build_model_pipeline

    subset = prepared.head(300)
    pipeline = build_model_pipeline("logistic_regression", cfg, seed=0)
    pipeline.fit(subset[MODEL_FEATURES], subset[TARGET_COLUMN].astype(int))
    probabilities = predict_positive_proba(pipeline, subset[MODEL_FEATURES])
    assert probabilities.shape == (len(subset),)
    assert probabilities.min() >= 0.0 and probabilities.max() <= 1.0
