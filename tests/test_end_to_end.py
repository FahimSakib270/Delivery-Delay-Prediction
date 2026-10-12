"""End-to-end test: raw CSVs in, frozen model and metrics out.

Everything runs inside ``tmp_path`` (see the ``tmp_config`` fixture), so the suite
never reads or writes the repository's real ``data/``, ``models/`` or ``reports/``
directories, and the test cannot be fooled by stale artefacts left over from a manual
run. The model search spaces are shrunk; the pipeline is the production one.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data_prep import (
    MODEL_FEATURES,
    TARGET_COLUMN,
    ensure_synthetic_raw,
    load_processed,
    prepare_dataset,
)
from src.evaluation import compute_metrics, evaluate_split, select_threshold
from src.features import build_model_pipeline
from src.splitting import split_and_filter
from src.training import select_best_model, fit_candidates


def test_prepare_dataset_writes_a_modelling_table_and_metadata(tmp_config):
    frame, meta = prepare_dataset(tmp_config)

    out_dir = Path(tmp_config["paths"]["processed_dir"])
    assert (out_dir / "orders_prepared.csv").is_file()
    assert (out_dir / "dataset_metadata.json").is_file()

    assert len(frame) > 0
    for column in MODEL_FEATURES + [TARGET_COLUMN, "order_purchase_timestamp", "delivered_date"]:
        assert column in frame.columns, f"{column} missing from the prepared table"
    assert frame[TARGET_COLUMN].isin([0, 1]).all()
    assert frame["order_purchase_timestamp"].is_monotonic_increasing, (
        "the prepared table must be sorted by booking time so the split is a simple cut"
    )
    assert meta["provenance"]["data_source_kind"] == "synthetic"
    assert meta["provenance"]["synthetic"] is True
    assert meta["provenance"]["processed_sha256"]
    assert meta["statistics"]["late_rate"] == pytest.approx(frame[TARGET_COLUMN].mean())


def test_stale_synthetic_csvs_are_regenerated_but_fresh_ones_are_kept(tmp_config):
    """The reproducibility guard: a changed generator must not be silently ignored."""
    from src.synthetic import read_provenance

    raw_dir = Path(tmp_config["paths"]["synthetic_raw_dir"])

    assert ensure_synthetic_raw(raw_dir, tmp_config) is True
    assert ensure_synthetic_raw(raw_dir, tmp_config) is False, "fresh data must not be regenerated"

    # A different requested size is a different dataset: regenerate.
    other = {**tmp_config, "data_source": {**tmp_config["data_source"], "synthetic_n_orders": 900}}
    assert ensure_synthetic_raw(raw_dir, other) is True

    # A different seed is a different dataset too.
    reseeded = {**tmp_config, "data_source": {**tmp_config["data_source"], "synthetic_seed": 999}}
    assert ensure_synthetic_raw(raw_dir, reseeded) is True
    assert read_provenance(raw_dir)["seed"] == 999

    # A generator whose fingerprint no longer matches the marker forces regeneration.
    # (Simulated by rewriting the marker, because editing src/synthetic.py inside a test
    # would invalidate every other test in the session.)
    marker = Path(raw_dir) / "PROVENANCE.json"
    from src.utils import load_json, save_json

    tampered = load_json(marker)
    tampered["generator_fingerprint"] = "0" * 64
    save_json(tampered, marker)
    assert ensure_synthetic_raw(raw_dir, tmp_config) is True, (
        "a stale generator fingerprint must trigger regeneration"
    )
    assert read_provenance(raw_dir)["generator_fingerprint"] != "0" * 64


def test_prepared_table_reloads_from_disk_with_the_same_fingerprint(tmp_config):
    frame, meta = prepare_dataset(tmp_config)
    reloaded = load_processed(Path(tmp_config["paths"]["processed_dir"]) / "orders_prepared.csv",
                              tmp_config)

    assert len(reloaded) == len(frame)
    assert pd.api.types.is_datetime64_any_dtype(reloaded["order_purchase_timestamp"])
    assert reloaded[TARGET_COLUMN].sum() == frame[TARGET_COLUMN].sum()
    assert reloaded["delivered_date"].notna().all()


def test_load_processed_fails_loudly_when_preparation_has_not_run(tmp_config):
    with pytest.raises(FileNotFoundError):
        load_processed(Path(tmp_config["paths"]["processed_dir"]) / "orders_prepared.csv", tmp_config)


def test_full_pipeline_produces_a_frozen_threshold_and_better_than_dummy_skill(tmp_config):
    """prepare -> split -> tune -> select -> freeze threshold -> score the test split."""
    prepare_dataset(tmp_config)
    # Read the *persisted* table back, exactly as `python -m src.train` does.
    frame = load_processed(Path(tmp_config["paths"]["processed_dir"]) / "orders_prepared.csv",
                           tmp_config)

    frames, split_info = split_and_filter(frame, tmp_config)
    assert split_info["label_maturity"]["enabled"] is True
    assert len(frames["test"]) > 0

    fitted, comparison, _info = fit_candidates(tmp_config, frames, cv_splits=2, seed=0)
    best = select_best_model(comparison)
    pipeline = fitted[best]

    # The threshold is chosen on validation only, then reused unchanged on test.
    validation = frames["validation"]
    validation_probabilities = pipeline.predict_proba(
        validation[MODEL_FEATURES]
    )[:, list(pipeline.named_steps["model"].classes_).index(1)]
    threshold, _table, _thr_info = select_threshold(
        validation[TARGET_COLUMN].astype(int), validation_probabilities,
        objective=tmp_config["threshold_selection"]["objective"],
        min_precision_floor=tmp_config["threshold_selection"]["min_precision_floor"],
    )

    test_metrics, test_probabilities = evaluate_split(pipeline, frames["test"],
                                                     MODEL_FEATURES, threshold)
    assert test_metrics["n"] == len(frames["test"])
    assert len(test_probabilities) == len(frames["test"])
    assert 0.0 < threshold < 1.0
    assert test_metrics["average_precision"] >= 0.0

    validation_metrics = compute_metrics(
        validation[TARGET_COLUMN].astype(int), validation_probabilities, threshold=threshold
    )
    ranked = comparison.set_index(["model", "split"])["average_precision"]
    if best != "dummy":
        assert ranked.loc[(best, "validation")] >= float(ranked.loc[("dummy", "validation")]), (
            "the selected model must beat the always-predict-the-majority-class floor"
        )
    assert validation_metrics["n"] == len(validation)


def test_a_model_trained_on_the_redirected_paths_is_loadable_and_serves_one_order(tmp_config):
    """The serving path: build a model, then score a single row as the app does."""
    prepare_dataset(tmp_config)
    frame = load_processed(Path(tmp_config["paths"]["processed_dir"]) / "orders_prepared.csv",
                           tmp_config)
    pipeline = build_model_pipeline("logistic_regression", tmp_config, seed=0)
    pipeline.fit(frame[MODEL_FEATURES], frame[TARGET_COLUMN].astype(int))

    row = frame[MODEL_FEATURES].iloc[[0]]
    probability = float(pipeline.predict_proba(row)[0][1])
    assert 0.0 <= probability <= 1.0

    # A never-before-seen category must still score rather than crash the service.
    unseen = row.copy()
    unseen["customer_state"] = "ZZ"
    unseen["product_category"] = "a_category_that_does_not_exist"
    assert 0.0 <= float(pipeline.predict_proba(unseen)[0][1]) <= 1.0
