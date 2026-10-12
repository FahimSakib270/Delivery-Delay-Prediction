"""Tests for the feature layer: the audit table, the input frame and the pipeline.

Two things are being defended here. First, that the audit documents *every* feature
the model sees (an undocumented feature is an unreviewed feature). Second, that the
preprocessing survives the two situations that kill a demo in production: a level the
encoder has never seen, and a missing value.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import features
from src.data_prep import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)
from src.evaluation import predict_positive_proba


# --------------------------------------------------------------------------- audit
def test_every_model_feature_is_documented_in_the_audit():
    documented = {row["Feature"] for row in features.FEATURE_AUDIT}
    missing = [name for name in MODEL_FEATURES if name not in documented]
    assert not missing, f"undocumented model inputs: {missing}"


def test_every_feature_marked_as_used_really_is_a_model_input():
    used = {row["Feature"] for row in features.FEATURE_AUDIT if row["Used?"] == "Yes"}
    assert used == set(MODEL_FEATURES), (
        "the audit and the feature list disagree: "
        f"only-in-audit={sorted(used - set(MODEL_FEATURES))}, "
        f"only-in-code={sorted(set(MODEL_FEATURES) - used)}"
    )


def test_every_used_feature_is_available_at_the_booking_instant():
    offenders = [
        row["Feature"] for row in features.FEATURE_AUDIT
        if row["Used?"] == "Yes" and row["Available at booking?"] != "Yes"
    ]
    assert not offenders, f"features used but not knowable at booking: {offenders}"


def test_feature_audit_table_has_the_documented_columns():
    table = features.feature_audit_table()
    assert list(table.columns) == ["Feature", "Meaning", "Source", "Available at booking?",
                                   "Used?", "Reason"]
    assert len(table) == len(features.FEATURE_AUDIT)
    assert table["Reason"].str.len().gt(0).all(), "every decision needs a reason"


# --------------------------------------------------------------------------- frame
def test_feature_frame_returns_exactly_the_model_columns_in_order(prepared):
    frame = features.feature_frame(prepared)
    assert list(frame.columns) == list(MODEL_FEATURES)
    assert len(frame) == len(prepared)


def test_feature_frame_accepts_an_explicit_subset(prepared):
    subset = ["lead_time_days", "customer_state"]
    frame = features.feature_frame(prepared, features=subset)
    assert list(frame.columns) == subset


def test_feature_frame_raises_for_a_missing_column(prepared):
    with pytest.raises(KeyError):
        features.feature_frame(prepared, features=["lead_time_days", "does_not_exist"])


def test_feature_frame_refuses_an_unsupported_payment_feature_request(prepared, cfg):
    cfg_payments = {**cfg, "features": {**cfg["features"], "include_payment_features": True}}
    with pytest.raises(NotImplementedError):
        features.feature_frame(prepared, cfg=cfg_payments)


# --------------------------------------------------------------------------- preprocessor
def test_preprocessor_survives_categories_never_seen_during_training():
    """Production scores orders from categories the encoder did not learn."""
    train = pd.DataFrame({
        "lead_time_days": [5.0, 9.0, 12.0, 20.0],
        "customer_state": ["SP", "RJ", "SP", "MG"],
        "seller_state": ["SP", "SP", "RJ", "MG"],
        "product_category": ["toys", "toys", "auto", "auto"],
    })
    unseen = pd.DataFrame({
        "lead_time_days": [8.0],
        "customer_state": ["AC"],          # never seen
        "seller_state": ["AP"],            # never seen
        "product_category": ["brand_new"],  # never seen
    })
    preprocessor = features.build_preprocessor(
        _tiny_cfg(),
        numeric_features=["lead_time_days"],
        categorical_features=["customer_state", "seller_state", "product_category"],
    )
    transformed = preprocessor.fit(train).transform(unseen)
    assert np.asarray(transformed).shape[0] == 1
    assert np.isfinite(np.asarray(transformed, dtype=float)).all()


def test_preprocessor_imputes_missing_numeric_values():
    train = pd.DataFrame({"lead_time_days": [5.0, 9.0, 12.0, 20.0], "customer_state": ["SP"] * 4})
    with_gap = train.copy()
    with_gap.loc[0, "lead_time_days"] = np.nan
    preprocessor = features.build_preprocessor(
        _tiny_cfg(), numeric_features=["lead_time_days"], categorical_features=["customer_state"]
    )
    preprocessor.fit(train)
    assert np.isfinite(np.asarray(preprocessor.transform(with_gap), dtype=float)).all()


def _tiny_cfg() -> dict:
    """Minimal feature config: the encoder knobs the preprocessor reads."""
    return {"features": {"min_category_frequency": 1, "max_category_levels": 20}}


# --------------------------------------------------------------------------- pipelines
def test_model_param_grid_only_searches_knobs_with_several_candidates(cfg):
    grid = features.model_param_grid(cfg, "logistic_regression")
    assert grid == {"model__C": [0.1, 1.0]}, "a single-valued knob is a fixed parameter, not a search axis"

    forest = features.model_param_grid(cfg, "random_forest")
    assert set(forest) == {"model__max_depth", "model__min_samples_leaf"}
    assert all(key.startswith("model__") for key in forest)

    assert features.model_param_grid(cfg, "dummy") == {}


def test_build_model_pipeline_nests_the_model_under_the_preprocessor(cfg):
    pipeline = features.build_model_pipeline("logistic_regression", cfg)
    assert list(pipeline.named_steps) == ["preprocessor", "model"]

    # `transformers` is the public, pre-fit list of (name, transformer, columns) triples,
    # so the wiring can be inspected without training anything.
    def numeric_steps(built):
        preprocessor = built.named_steps["preprocessor"]
        by_name = {name: transformer for name, transformer, _columns in preprocessor.transformers}
        return list(by_name["numeric"].named_steps)

    assert "scaler" in numeric_steps(pipeline), "the linear model needs standardised inputs"
    assert "scaler" not in numeric_steps(features.build_model_pipeline("random_forest", cfg)), (
        "scaling a tree model is pointless work"
    )


def test_build_model_pipeline_rejects_an_unknown_model(cfg):
    """An unknown name must fail loudly rather than silently fall back to something.

    The config lookup happens first, so a name that is not in ``modeling`` raises a
    ``KeyError``; a name that is configured but not implemented reaches the explicit
    ``ValueError``. Both are acceptable, silence is not.
    """
    with pytest.raises((KeyError, ValueError)):
        features.build_model_pipeline("gradient_boosting", cfg)


def test_transformed_feature_names_describe_what_the_model_actually_sees(prepared, cfg):
    """After imputation and one-hot encoding the columns are no longer the raw ones.

    Reporting an importance for a raw feature name would be wrong, so the pipeline must
    be able to say what its design matrix contains.
    """
    pipeline = features.build_model_pipeline("logistic_regression", cfg, seed=0)
    pipeline.fit(prepared[MODEL_FEATURES], prepared[TARGET_COLUMN].astype(int))
    names = features.transformed_feature_names(pipeline)

    assert "lead_time_days" in names, "passthrough numeric features keep their name"
    assert any(name.startswith("customer_state_") for name in names), "categoricals are one-hot encoded"
    assert len(names) > len(MODEL_FEATURES), "one-hot encoding must widen the matrix"
    assert len(set(names)) == len(names), "duplicate design-matrix column names"


def test_every_candidate_learns_a_separable_signal(cfg):
    """A smoke test for all three candidates on a small, separable problem.

    The frame carries every configured model input (the pipeline selects them by name),
    but only ``lead_time_days`` carries signal: the target is "the promise was tight".
    """
    rng = np.random.default_rng(0)
    n = 300
    tight = rng.integers(2, 40, n).astype(float)
    X = pd.DataFrame({name: rng.random(n) for name in NUMERIC_FEATURES})
    X["lead_time_days"] = tight
    for name in CATEGORICAL_FEATURES:
        X[name] = rng.choice(["SP", "RJ", "MG"], n)
    y = pd.Series((tight < 14).astype(int))  # late iff the promise was tight

    for model_name in ("dummy", "logistic_regression", "random_forest"):
        pipeline = features.build_model_pipeline(model_name, cfg)
        pipeline.fit(X[MODEL_FEATURES], y)
        probabilities = predict_positive_proba(pipeline, X[MODEL_FEATURES])
        assert len(probabilities) == n
        assert np.isfinite(probabilities).all()
        assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_dummy_classifier_that_never_predicts_the_positive_class_returns_zeros():
    """Regression test: ``DummyClassifier`` only exposes the class it saw.

    Indexing ``predict_proba`` at column 1 crashed on this, which took the whole CV
    loop down. The helper must return a probability vector anyway.
    """
    from sklearn.dummy import DummyClassifier
    from sklearn.pipeline import Pipeline

    X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    y = pd.Series([0, 0, 0])
    model = Pipeline([("model", DummyClassifier(strategy="most_frequent"))]).fit(X, y)
    probabilities = predict_positive_proba(model, X)
    assert probabilities.shape == (3,)
    assert np.all(probabilities == 0.0)
