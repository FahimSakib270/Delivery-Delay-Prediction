"""Chronological train/validation/test splitting with label-maturity discipline.

Two rules keep this honest:

1. **Time order is never shuffled.** The split is a cut on
   ``order_purchase_timestamp``, so every training order was booked before every
   validation order, which in turn was booked before every test order.
2. **A label must have been observable.** Training only on orders whose delivery
   was already known when the next period started removes the look-ahead bias
   that a naive "all rows are labelled today" split would introduce.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

from .config import load_config
from .data_prep import TARGET_COLUMN
from .utils import get_logger

logger = get_logger("splitting")

SPLIT_NAMES = ("train", "validation", "test")


def _cut_points(n: int, cfg: Dict[str, Any]) -> Tuple[int, int]:
    """Return the two index cut points implied by the configured fractions."""
    train_frac = float(cfg["split"]["train_frac"])
    val_frac = float(cfg["split"]["val_frac"])
    test_frac = float(cfg["split"]["test_frac"])
    total = train_frac + val_frac + test_frac
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"split fractions must sum to 1.0, got {total}")

    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    # Guarantee non-empty folds even on a tiny dataset (the unit tests use 40 rows).
    n_train = max(1, min(n_train, n - 2))
    n_val = max(1, min(n_val, n - n_train - 1))
    return n_train, n_train + n_val


def chronological_split(
    df: pd.DataFrame,
    cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    """Split the prepared frame into train/validation/test periods by booking time."""
    cfg = cfg or load_config()
    if TARGET_COLUMN not in df.columns:
        raise KeyError(f"expected a {TARGET_COLUMN!r} column; run data preparation first")

    ordered = df.sort_values("order_purchase_timestamp", kind="stable").reset_index(drop=True)
    n = len(ordered)
    if n < 10:
        raise ValueError(f"need at least 10 orders to build three folds, got {n}")

    if ordered["order_purchase_timestamp"].isna().any():
        raise ValueError("booking timestamps must not be missing")
    desired_train, desired_val_end = _cut_points(n, cfg)
    # Cut only between distinct booking timestamps, reserving a group for each
    # remaining period. No order at a shared instant crosses a boundary.
    boundaries = np.flatnonzero(
        ordered["order_purchase_timestamp"].ne(ordered["order_purchase_timestamp"].shift())
    )[1:]
    if len(boundaries) < 2:
        raise ValueError("need at least three distinct booking timestamps")
    first_candidates = boundaries[:-1]
    n_train = int(first_candidates[np.argmin(abs(first_candidates - desired_train))])
    second_candidates = boundaries[boundaries > n_train]
    n_val_end = int(second_candidates[np.argmin(abs(second_candidates - desired_val_end))])
    frames = {
        "train": ordered.iloc[:n_train].copy(),
        "validation": ordered.iloc[n_train:n_val_end].copy(),
        "test": ordered.iloc[n_val_end:].copy(),
    }
    return frames, _describe_periods(frames, n)


def _describe_periods(frames: Dict[str, pd.DataFrame], n: int) -> Dict[str, Any]:
    """Record fold sizes, date ranges and the boundary-safety evidence."""
    info: Dict[str, Any] = {"n_total": int(n), "folds": {}}
    for name in SPLIT_NAMES:
        frame = frames[name]
        ts = frame["order_purchase_timestamp"]
        info["folds"][name] = {
            "n_rows": int(len(frame)),
            "share": round(len(frame) / n, 6),
            "booking_start": ts.min().isoformat(),
            "booking_end": ts.max().isoformat(),
            "late_rate": float(frame[TARGET_COLUMN].mean()),
        }

    train_end = frames["train"]["order_purchase_timestamp"].max()
    val_start = frames["validation"]["order_purchase_timestamp"].min()
    val_end = frames["validation"]["order_purchase_timestamp"].max()
    test_start = frames["test"]["order_purchase_timestamp"].min()

    if train_end >= val_start or val_end >= test_start:
        raise AssertionError("chronological split produced overlapping periods")
    # Ties on the cut instant are unavoidable when many orders share a timestamp;
    # they are recorded explicitly rather than silently ignored.
    info["boundary_ties"] = {
        "train_val_shared_instant": bool(train_end == val_start),
        "val_test_shared_instant": bool(val_end == test_start),
    }
    info["periods"] = {
        "train_end": train_end.isoformat(),
        "validation_start": val_start.isoformat(),
        "validation_end": val_end.isoformat(),
        "test_start": test_start.isoformat(),
    }
    return info


def apply_label_maturity(
    frames: Dict[str, pd.DataFrame],
    cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    """Drop rows whose outcome could not have been observed before the next period.

    * training rows must have been delivered strictly before the validation period began;
    * validation rows must have been delivered strictly before the test period began;
    * test rows are kept as-is (they represent the future we are predicting into), but
      immature ones are counted so the limitation is stated explicitly rather than hidden.
    """
    cfg = cfg or load_config()
    info: Dict[str, Any] = {"enabled": bool(cfg["label_maturity"]["enabled"])}
    if not info["enabled"]:
        info["note"] = "label maturity filter disabled in config"
        return frames, info

    out = {k: v.copy() for k, v in frames.items()}
    horizons = {
        "train": out["validation"]["order_purchase_timestamp"].min(),
        "validation": out["test"]["order_purchase_timestamp"].min(),
    }
    info["observation_horizons"] = {k: v.isoformat() for k, v in horizons.items()}

    for name, horizon in horizons.items():
        before = len(out[name])
        keep = pd.to_datetime(out[name]["order_delivered_customer_date"]) < horizon
        out[name] = out[name].loc[keep].reset_index(drop=True)
        info[name] = {
            "horizon": horizon.isoformat(),
            "n_before": int(before),
            "n_after": int(len(out[name])),
            "n_dropped": int(before - len(out[name])),
            "pct_dropped": round(100.0 * (before - len(out[name])) / max(1, before), 4),
        }
        logger.info("label-maturity: %s %d -> %d rows (horizon %s)",
                    name, before, len(out[name]), horizon.date())

    # Test immaturity is reported, never filtered: requiring the test label to be
    # observable before scoring would be circular.
    last_booking = max(f["order_purchase_timestamp"].max() for f in frames.values())
    test = out["test"]
    info["test"] = {
        "n_rows": int(len(test)),
        "n_delivered_after_observation_window": int(
            (pd.to_datetime(test["order_delivered_customer_date"]) > last_booking).sum()
        ),
        "note": ("test labels are all treated as observable; the count above flags rows whose "
                 "delivery falls after the last booking instant present in the dataset"),
    }
    return out, info


def mature_cv_splits(frame: pd.DataFrame, n_splits: int = 3):
    """Expanding-window CV with strict booking boundaries and observable labels.

    Return positional indices for sklearn plus an audit trail. Validation labels
    are retrospective evaluation outcomes; training labels must be known before
    the fold's first prediction. The caller's outer maturity filter limits all
    these observations to the outer model-selection horizon.
    """
    booking = pd.to_datetime(frame["order_purchase_timestamp"]).reset_index(drop=True)
    delivery = pd.to_datetime(frame["order_delivered_customer_date"]).reset_index(drop=True)
    if booking.isna().any() or delivery.isna().any():
        raise ValueError("CV requires complete booking and delivery timestamps")
    if not booking.is_monotonic_increasing:
        raise ValueError("CV rows must be sorted by booking timestamp")
    unique_times = booking.drop_duplicates().reset_index(drop=True)
    folds, audit = [], []
    for fold, (train_groups, val_groups) in enumerate(
        TimeSeriesSplit(n_splits=n_splits).split(unique_times), start=1
    ):
        horizon = unique_times.iloc[val_groups[0]]
        train_mask = booking.isin(unique_times.iloc[train_groups])
        val_mask = booking.isin(unique_times.iloc[val_groups])
        mature_mask = train_mask & (delivery < horizon)
        train_idx = np.flatnonzero(mature_mask.to_numpy())
        val_idx = np.flatnonzero(val_mask.to_numpy())
        if len(train_idx) == 0 or frame.iloc[train_idx][TARGET_COLUMN].nunique() < 2:
            raise ValueError(f"CV fold {fold} has insufficient mature training classes")
        if frame.iloc[val_idx][TARGET_COLUMN].nunique() < 2:
            raise ValueError(f"CV fold {fold} has insufficient validation classes")
        folds.append((train_idx, val_idx))
        audit.append({
            "fold": fold, "validation_start": horizon.isoformat(),
            "train_rows_before_maturity": int(train_mask.sum()),
            "train_rows": len(train_idx), "validation_rows": len(val_idx),
            "dropped_immature": int(train_mask.sum()) - len(train_idx),
            "latest_training_delivery": delivery.iloc[train_idx].max().isoformat(),
            "latest_training_booking": booking.iloc[train_idx].max().isoformat(),
        })
    return folds, audit


def split_and_filter(
    df: pd.DataFrame,
    cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    """Convenience wrapper: chronological split followed by the maturity filter."""
    frames, split_info = chronological_split(df, cfg)
    frames, maturity_info = apply_label_maturity(frames, cfg)
    info: Dict[str, Any] = {"split": split_info, "label_maturity": maturity_info,
                            "folds_after_maturity": {}}
    for name in SPLIT_NAMES:
        frame = frames[name]
        info["folds_after_maturity"][name] = {
            "n_rows": int(len(frame)),
            "late_rate": float(frame[TARGET_COLUMN].mean()) if len(frame) else float("nan"),
        }
    return frames, info
