"""Tests for the pure helpers inside the Streamlit app.

The app module is imported by file location so that the tests need no Streamlit
runtime: ``streamlit`` is imported at module import time, but nothing inside the
helpers touches session state. Every number the demo shows is produced by one of
these functions, which is exactly why they are testable.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_prep import CATEGORICAL_FEATURES, MODEL_FEATURES, NUMERIC_FEATURES

APP_PATH = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


@pytest.fixture(scope="module")
def app():
    spec = importlib.util.spec_from_file_location("profast_streamlit_app", APP_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- features
def test_assemble_features_builds_a_single_row_in_model_order(app):
    values = {name: 1 for name in MODEL_FEATURES}
    values["customer_state"] = "SP"
    frame = app.assemble_features(values)

    assert isinstance(frame, pd.DataFrame)
    assert list(frame.columns) == list(MODEL_FEATURES)
    assert len(frame) == 1
    assert frame.loc[0, "customer_state"] == "SP"


def test_assemble_features_refuses_to_silently_impute_a_missing_input(app):
    values = {name: 1 for name in MODEL_FEATURES}
    values.pop("lead_time_days")
    with pytest.raises(KeyError) as excinfo:
        app.assemble_features(values)
    assert "lead_time_days" in str(excinfo.value)


def test_every_model_feature_has_a_help_string_and_a_fallback(app):
    assert set(app.FEATURE_HELP) == set(MODEL_FEATURES)
    assert set(app.FALLBACK_NUMERIC) == set(NUMERIC_FEATURES)
    assert set(app.FALLBACK_CATEGORICAL) == set(CATEGORICAL_FEATURES)


# --------------------------------------------------------------------------- economics
def test_break_even_probability_follows_the_cost_ratio(app):
    # p* = c_fp / (c_fp + c_fn): the cost of an unnecessary contact vs the cost of a delay.
    assert app.break_even_probability(25.0, 4.0) == pytest.approx(4.0 / 29.0)
    assert app.break_even_probability(10.0, 10.0) == pytest.approx(0.5)
    assert app.break_even_probability(25.0, 0.0) == pytest.approx(0.0)
    assert np.isnan(app.break_even_probability(0.0, 0.0)), "no cost information => no answer"


def test_flagging_value_says_flag_only_above_the_break_even_point(app):
    cheap = app.flagging_value(0.95, cost_false_negative=25.0, cost_false_positive=4.0)
    assert cheap["expected_cost_if_flag"] < cheap["expected_cost_if_ignore"]

    safe = app.flagging_value(0.02, cost_false_negative=25.0, cost_false_positive=4.0)
    assert safe["expected_cost_if_ignore"] < safe["expected_cost_if_flag"]
    assert set(safe) >= {"probability", "break_even_probability", "expected_cost_if_ignore",
                         "expected_cost_if_flag"}


def test_flagging_value_never_reports_a_saving_when_costs_are_zero(app):
    """The demo must not invent business value out of an empty cost assumption."""
    panel = app.flagging_value(0.5, cost_false_negative=0.0, cost_false_positive=0.0)
    assert panel["expected_cost_if_ignore"] == 0.0
    assert panel["expected_cost_if_flag"] == 0.0


# --------------------------------------------------------------------------- widgets
def test_numeric_widget_specs_fall_back_when_the_dataset_is_missing(app):
    specs = app.numeric_widget_specs(None)
    assert set(specs) == set(NUMERIC_FEATURES)
    for lo, hi, default in specs.values():
        assert lo < hi
        assert lo <= default <= hi


def test_numeric_widget_specs_use_the_dataset_percentiles(app):
    frame = pd.DataFrame({
        "lead_time_days": list(range(1, 101)),
        "n_items": [1] * 90 + [5] * 10,
        "customer_state": ["SP"] * 100,
    })
    specs = app.numeric_widget_specs(frame)

    lo, hi, default = specs["lead_time_days"]
    # 1st/99th percentile of 1..100 is about 2 and 99, and the median about 50.
    assert 1.0 <= lo <= 3.0 and 98.0 <= hi <= 100.0
    assert 45.0 <= default <= 55.0
    # Integer features default to a whole number so the widget can be a stepper.
    assert float(specs["n_items"][2]).is_integer()


def test_numeric_widget_specs_widen_a_degenerate_range(app):
    frame = pd.DataFrame({name: [3.0] * 50 for name in NUMERIC_FEATURES})
    specs = app.numeric_widget_specs(frame)
    for lo, hi, default in specs.values():
        assert hi > lo, "a zero-width slider would be unusable"


def test_categorical_widget_options_offer_a_sentinel_for_unseen_levels(app):
    frame = pd.DataFrame({
        "customer_state": ["SP"] * 60 + ["RJ"] * 30 + [f"S{i:02d}" for i in range(50)],
        "seller_state": ["SP"] * 140,
        "product_category": ["toys"] * 140,
    })
    options = app.categorical_widget_options(frame)

    assert set(options) == set(CATEGORICAL_FEATURES)
    assert "SP" in options["customer_state"] and "RJ" in options["customer_state"]
    assert app.OTHER_LABEL in options["customer_state"], (
        "with >40 levels the long tail must be exposed as one honest 'rare' choice"
    )
    assert len(options["customer_state"]) <= 41


def test_categorical_widget_options_fall_back_without_data(app):
    options = app.categorical_widget_options(None)
    assert set(options) == set(CATEGORICAL_FEATURES)
    for levels in options.values():
        assert levels and all(isinstance(level, str) for level in levels)


# --------------------------------------------------------------------------- presentation
def test_risk_band_moves_up_as_the_probability_rises(app):
    threshold = 0.74
    probabilities = [0.01, 0.4, 0.74, 0.8, 0.99]
    bands = [app.risk_band(p, threshold) for p in probabilities]
    ranks = {"very low": 0, "low": 1, "elevated": 2, "high": 3}
    assert [ranks[band] for band in bands] == sorted(ranks[band] for band in bands)
    assert bands[0] == "very low"
    # At or above the frozen threshold the order is flagged, and the band says "high".
    assert app.risk_band(threshold, threshold) == "high"
    assert app.risk_band(0.999, threshold) == "high"


def test_risk_band_has_an_elevated_band_only_when_the_threshold_is_below_a_half(app):
    """``elevated`` means \"above the threshold, not obviously high\"."""
    assert app.risk_band(0.35, 0.30) == "elevated"
    assert app.risk_band(0.31, 0.30) == "elevated"
    assert app.risk_band(0.55, 0.30) == "high"
    assert app.risk_band(0.10, 0.30) == "very low"


def test_risk_band_is_safe_for_a_threshold_of_zero(app):
    assert app.risk_band(0.0, 0.0) in {"low", "elevated", "high", "very low"}


def test_format_currency_is_two_decimals_with_thousands_separators(app):
    assert app.format_currency(1234.5) == "1,234.50"
    assert app.format_currency(0.0) == "0.00"
