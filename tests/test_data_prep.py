"""Tests for data preparation: the joins, the label and — most importantly — the
leakage guards.

The single most valuable test in this file is
:func:`test_no_outcome_column_can_reach_the_model`. A booking-time model is only
honest if every input is knowable at the booking instant; when a delivered-date
column leaks in, the metric jumps and nothing else notices.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import data_prep
from src.data_prep import (
    ANALYSIS_COLUMNS,
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    PROHIBITED_MODEL_COLUMNS,
    TARGET_COLUMN,
)


# --------------------------------------------------------------------------- leakage
def test_no_outcome_column_can_reach_the_model():
    overlap = set(MODEL_FEATURES) & set(PROHIBITED_MODEL_COLUMNS)
    assert not overlap, f"outcome/identifier columns leak into the model: {sorted(overlap)}"


def test_no_delivery_timestamp_is_used_as_a_feature():
    for name in MODEL_FEATURES:
        assert "delivered" not in name
        assert "promise" not in name and "estimated" not in name
    # `lead_time_days` is the *promise* made at booking, not the realised delivery.
    assert "lead_time_days" in MODEL_FEATURES


def test_the_feature_schema_is_consistent():
    assert MODEL_FEATURES == NUMERIC_FEATURES + CATEGORICAL_FEATURES
    assert TARGET_COLUMN == "is_late"
    assert len(set(MODEL_FEATURES)) == len(MODEL_FEATURES), "duplicate feature name"
    assert TARGET_COLUMN not in MODEL_FEATURES
    assert "delay_days" not in MODEL_FEATURES
    # Analysis columns are the raw Olist columns the split, the maturity mask and the
    # error analysis need. ``promised_date``/``delivered_date`` are derived from them by
    # ``add_target`` and are kept in the persisted table as well.
    for column in ("order_purchase_timestamp", "order_estimated_delivery_date",
                   "order_delivered_customer_date"):
        assert column in ANALYSIS_COLUMNS


# --------------------------------------------------------------------------- cleaning
def test_clean_orders_parses_timestamps_and_keeps_only_delivered_orders(synthetic_tables, cfg):
    stats: dict = {}
    cleaned = data_prep.clean_orders(synthetic_tables["orders"], stats)

    assert pd.api.types.is_datetime64_any_dtype(cleaned["order_purchase_timestamp"])
    assert cleaned["order_purchase_timestamp"].notna().all()
    assert set(cleaned["order_status"]).issubset({"delivered"})
    assert cleaned["order_id"].is_unique
    assert stats["raw_orders"] >= len(cleaned)
    assert stats["usable_orders"] == len(cleaned)


def test_add_target_is_a_calendar_date_comparison(synthetic_tables, cfg):
    stats: dict = {}
    cleaned = data_prep.clean_orders(synthetic_tables["orders"], stats)
    labelled = data_prep.add_target(cleaned, label_definition="date_based")

    promised = pd.to_datetime(labelled["order_estimated_delivery_date"]).dt.normalize()
    delivered = pd.to_datetime(labelled["order_delivered_customer_date"]).dt.normalize()
    expected = (delivered > promised).astype(int)

    assert labelled[TARGET_COLUMN].isin([0, 1]).all()
    assert labelled[TARGET_COLUMN].to_numpy().tolist() == expected.to_numpy().tolist()
    # Delivering exactly on the promised day is ON TIME, by definition.
    on_time = delivered == promised
    assert (labelled.loc[on_time, TARGET_COLUMN] == 0).all()
    assert "delay_days" in labelled.columns


def test_timestamp_based_label_is_at_least_as_strict(synthetic_tables):
    stats: dict = {}
    cleaned = data_prep.clean_orders(synthetic_tables["orders"], stats)
    labelled = data_prep.add_target(cleaned, label_definition="timestamp_based")
    date_late = data_prep.add_target(cleaned, label_definition="date_based")[TARGET_COLUMN]

    assert labelled["is_late_ts"].sum() >= date_late.sum(), (
        "a same-day-but-later-clock delivery is on time by date and late by timestamp"
    )


# --------------------------------------------------------------------------- joins
def test_item_aggregation_produces_one_row_per_order(synthetic_tables):
    stats: dict = {}
    agg = data_prep.aggregate_items(
        items=synthetic_tables["items"],
        products=synthetic_tables["products"],
        sellers=synthetic_tables["sellers"],
        translation=synthetic_tables.get("translation"),
        stats=stats,
    )
    assert agg["order_id"].is_unique
    assert (agg["n_items"] >= 1).all()
    assert (agg["n_products"] <= agg["n_items"]).all()
    assert (agg["n_sellers"] <= agg["n_products"]).all()
    assert (agg["price_sum"] > 0).all()
    assert (agg["freight_sum"] >= 0).all()
    assert stats["multi_item_orders"] <= len(agg)
