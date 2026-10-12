"""Tests for the chronological split and the label-maturity filter.

The split is where look-ahead bias would enter the project. A random split would let
the model train on orders booked *after* the ones it is scored on; a naive split would
let it train on orders whose outcome was not yet known when the next period began.
Both are tested explicitly.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data_prep import TARGET_COLUMN
from src.splitting import (
    SPLIT_NAMES,
    _cut_points,
    apply_label_maturity,
    chronological_split,
    split_and_filter,
)


def test_cut_points_follow_the_configured_fractions(cfg):
    n_train, n_val_end = _cut_points(1000, cfg)
    assert n_train == 700
    assert n_val_end == 850


def test_cut_points_reject_fractions_that_do_not_sum_to_one(cfg):
    broken = {**cfg, "split": {"train_frac": 0.6, "val_frac": 0.2, "test_frac": 0.3}}
    with pytest.raises(ValueError):
        _cut_points(100, broken)


def test_chronological_split_never_mixes_past_and_future(prepared, cfg):
    frames, info = chronological_split(prepared, cfg)

    train_end = frames["train"]["order_purchase_timestamp"].max()
    val_start = frames["validation"]["order_purchase_timestamp"].min()
    val_end = frames["validation"]["order_purchase_timestamp"].max()
    test_start = frames["test"]["order_purchase_timestamp"].min()

    assert train_end <= val_start, "a training order is booked after the validation period starts"
    assert val_end <= test_start, "a validation order is booked after the test period starts"
    # Ties on the cut instant are possible (many orders share a timestamp); the split
    # records them instead of pretending they cannot happen.
    assert isinstance(info["boundary_ties"]["train_val_shared_instant"], bool)
    assert info["periods"]["train_end"] <= info["periods"]["validation_start"]


def test_chronological_split_sizes_and_coverage(prepared, cfg):
    frames, info = chronological_split(prepared, cfg)
    total = sum(len(frames[name]) for name in SPLIT_NAMES)
    assert total == len(prepared), "every prepared order must land in exactly one fold"
    assert abs(info["folds"]["train"]["share"] - cfg["split"]["train_frac"]) < 0.02
    for name in SPLIT_NAMES:
        assert len(frames[name]) > 0
        assert frames[name][TARGET_COLUMN].isin([0, 1]).all()


def test_chronological_split_requires_a_labelled_frame(prepared, cfg):
    with pytest.raises(KeyError):
        chronological_split(prepared.drop(columns=[TARGET_COLUMN]), cfg)


def test_chronological_split_refuses_a_dataset_that_is_too_small(cfg):
    tiny = pd.DataFrame({
        "order_purchase_timestamp": pd.date_range("2018-01-01", periods=5, freq="D"),
        TARGET_COLUMN: [0, 1, 0, 0, 1],
        "delivered_date": pd.date_range("2018-01-03", periods=5, freq="D"),
    })
    with pytest.raises(ValueError):
        chronological_split(tiny, cfg)


def test_label_maturity_drops_only_unobservable_training_rows(prepared, cfg):
    frames, _ = chronological_split(prepared, cfg)
    filtered, info = apply_label_maturity(frames, cfg)

    horizon = filtered["validation"]["order_purchase_timestamp"].min()
    assert (filtered["train"]["delivered_date"] < horizon).all(), (
        "a training row is kept whose delivery was not yet known when validation began"
    )
    assert len(filtered["train"]) <= len(frames["train"])
    assert info["train"]["n_dropped"] == len(frames["train"]) - len(filtered["train"])
    # The test period is never filtered: its whole purpose is to represent the future.
    assert len(filtered["test"]) == len(frames["test"])
    assert info["test"]["n_rows"] == len(frames["test"])


def test_label_maturity_can_be_switched_off(cfg, prepared):
    frames, _ = chronological_split(prepared, cfg)
    off = {**cfg, "label_maturity": {"enabled": False}}
    unfiltered, info = apply_label_maturity(frames, off)
    assert info["enabled"] is False
    assert len(unfiltered["train"]) == len(frames["train"])


def test_split_and_filter_reports_every_fold(prepared, cfg):
    frames, info = split_and_filter(prepared, cfg)
    assert set(frames) == set(SPLIT_NAMES)
    assert set(info["folds_after_maturity"]) == set(SPLIT_NAMES)
    for name in SPLIT_NAMES:
        assert info["folds_after_maturity"][name]["n_rows"] == len(frames[name])
        assert 0.0 <= info["folds_after_maturity"][name]["late_rate"] <= 1.0
    assert info["label_maturity"]["enabled"] is True


def test_same_timestamp_groups_never_cross_outer_boundaries(cfg):
    timestamps = pd.Series(pd.date_range("2018-01-01", periods=12, freq="D")).repeat(3).reset_index(drop=True)
    frame = pd.DataFrame({"order_purchase_timestamp": timestamps, TARGET_COLUMN: [0, 1, 0] * 12})
    frames, info = chronological_split(frame, cfg)
    assert not any(info["boundary_ties"].values())
    groups = [set(frames[name].order_purchase_timestamp) for name in SPLIT_NAMES]
    assert groups[0].isdisjoint(groups[1]) and groups[1].isdisjoint(groups[2])
    assert sum(map(len, frames.values())) == len(frame)


def test_maturity_uses_actual_delivery_time_not_midnight(cfg):
    horizon = pd.Timestamp("2018-02-01 12:00")
    frames = {
        "train": pd.DataFrame({"order_purchase_timestamp": [horizon - pd.Timedelta(days=2)] * 2,
            "delivered_date": [horizon.normalize()] * 2,
            "order_delivered_customer_date": [horizon - pd.Timedelta(hours=1), horizon + pd.Timedelta(hours=1)]}),
        "validation": pd.DataFrame({"order_purchase_timestamp": [horizon],
            "order_delivered_customer_date": [horizon + pd.Timedelta(hours=2)]}),
        "test": pd.DataFrame({"order_purchase_timestamp": [horizon + pd.Timedelta(days=2)],
            "order_delivered_customer_date": [horizon + pd.Timedelta(days=3)]}),
    }
    filtered, info = apply_label_maturity(frames, cfg)
    assert len(filtered["train"]) == 1
    assert filtered["train"].order_delivered_customer_date.iloc[0] < horizon
    assert info["train"]["n_dropped"] == 1


def test_inner_cv_excludes_unobserved_labels_and_shared_booking_instants():
    from src.splitting import mature_cv_splits
    booking = pd.Series(pd.date_range("2018-01-01 12:00", periods=40, freq="D")).repeat(2).reset_index(drop=True)
    delivery = booking + pd.Timedelta(hours=2)
    delivery.iloc[0] = booking.iloc[-1] + pd.Timedelta(days=10)
    frame = pd.DataFrame({"order_purchase_timestamp": booking,
                          "order_delivered_customer_date": delivery,
                          TARGET_COLUMN: [0, 1] * 40})
    folds, audit = mature_cv_splits(frame, n_splits=3)
    for (train, val), record in zip(folds, audit):
        horizon = booking.iloc[val].min()
        assert (delivery.iloc[train] < horizon).all()
        assert booking.iloc[train].max() < horizon
        assert 0 not in train
        assert set(booking.iloc[train]).isdisjoint(set(booking.iloc[val]))
        assert record["dropped_immature"] >= 1
