"""Tests for the training layer: candidate fitting, selection and the persisted bundle.

``train_and_evaluate`` is deliberately *not* re-run here with the shipped search spaces
(400 trees x 4 configurations x 3 folds is minutes of compute, and the suite should
stay fast). Instead the pieces are tested directly on a small frame, plus one full
end-to-end run in :mod:`tests.test_end_to_end` with a shrunken search space.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_prep import MODEL_FEATURES, TARGET_COLUMN
from src.training import (
    CANDIDATE_MODELS,
    PRIMARY_SELECTION_METRIC,
    fit_candidates,
    select_best_model,
    tune_model,
)


@pytest.fixture(scope="module")
def small_frames(synthetic_tables, cfg, as_production_frame):
    """A 3-way chronological split of a small prepared frame, built once.

    The frame is round-tripped through CSV before splitting so that the dtypes match
    what ``python -m src.prepare`` writes and ``python -m src.train`` reads.
    """
    from src.data_prep import build_order_dataset
    from src.splitting import split_and_filter

    stats: dict = {}
    prepared = as_production_frame(build_order_dataset(synthetic_tables, cfg, stats))
    frames, _info = split_and_filter(prepared, cfg)
    return frames


@pytest.fixture(scope="module")
def cheap_cfg(cfg):
    """The real config with a search space a laptop can fit in seconds.

    Each model keeps at least one searched axis (two values) so the grid-search code
    path is exercised rather than always falling into the "nothing to search" branch.
    """
    cheap = {**cfg, "modeling": {**cfg["modeling"]}}
    cheap["modeling"]["random_forest"] = {
        "n_estimators": 25, "max_depth": [6], "min_samples_leaf": [2, 5],
        "max_features": ["sqrt"], "class_weight": "balanced_subsample", "n_jobs": 1,
    }
    cheap["modeling"]["logistic_regression"] = {"C": [0.1, 1.0], "max_iter": 1000,
                                                "class_weight": "balanced"}
    return cheap


def test_tune_model_cross_validates_the_untuned_baseline_as_well(cheap_cfg, small_frames):
    """The dummy floor must be measured on the same footing as the real candidates.

    It used to be reported without a CV score, which made the baseline look better
    than it was: a ``nan`` in a comparison table reads as \"not evaluated\", not \"no
    skill\".
    """
    train = small_frames["train"]
    pipeline, info = tune_model(
        "dummy", train[MODEL_FEATURES], train[TARGET_COLUMN].astype(int), cheap_cfg, cv_splits=2
    )
    assert info["grid"] == {}
    assert "best_cv_score" in info and np.isfinite(info["best_cv_score"])
    assert len(info["cv_scores"]) == 2
    assert info["seconds"] >= 0.0
    assert hasattr(pipeline, "predict_proba")


def test_tune_model_scores_every_grid_point_for_the_linear_model(cheap_cfg, small_frames):
    train = small_frames["train"]
    _pipeline, info = tune_model(
        "logistic_regression", train[MODEL_FEATURES], train[TARGET_COLUMN].astype(int),
        cheap_cfg, cv_splits=2,
    )
    assert set(info["best_params"]) == {"model__C"}
    table = info["search_table"]
    assert len(table) >= 1
    assert "mean_test_score" in table.columns
    assert np.isfinite(info["best_cv_score"])


def test_fit_candidates_covers_every_candidate_and_every_split(cheap_cfg, small_frames):
    fitted, comparison, info = fit_candidates(cheap_cfg, small_frames, cv_splits=2, seed=0)

    assert set(fitted) == set(CANDIDATE_MODELS)
    assert set(comparison["model"]) == set(CANDIDATE_MODELS)
    assert set(comparison["split"]) == {"train", "validation", "test"}
    assert len(comparison) == len(CANDIDATE_MODELS) * 3
    assert set(info) == set(CANDIDATE_MODELS)
    for column in ("average_precision", "roc_auc", "recall", "precision"):
        assert comparison[column].notna().any()


def test_fit_candidates_never_scores_the_unfitted_test_split_higher_than_train(cheap_cfg, small_frames):
    """A sanity floor: the model cannot be *better* on a later period than on the past.

    If it is, the pipeline is leaking (for example by shuffling before splitting).
    """
    _fitted, comparison, _info = fit_candidates(cheap_cfg, small_frames, cv_splits=2, seed=0)
    comparison = comparison.set_index(["model", "split"])["average_precision"]
    for model in CANDIDATE_MODELS:
        train_score = comparison.loc[(model, "train")]
        test_score = comparison.loc[(model, "test")]
        if np.isfinite(train_score) and np.isfinite(test_score):
            assert test_score <= train_score + 0.15, (
                f"{model}: test AP {test_score:.3f} is implausibly better than train AP "
                f"{train_score:.3f} - suspect a leak"
            )


def test_select_best_model_uses_validation_average_precision_not_test():
    comparison = pd.DataFrame([
        {"model": "dummy", "split": "validation", "average_precision": 0.05, "brier": 0.05},
        {"model": "logistic_regression", "split": "validation", "average_precision": 0.20, "brier": 0.10},
        {"model": "random_forest", "split": "validation", "average_precision": 0.18, "brier": 0.10},
        # The forest looks better on the test split; selecting on test would be cheating.
        {"model": "logistic_regression", "split": "test", "average_precision": 0.21, "brier": 0.10},
        {"model": "random_forest", "split": "test", "average_precision": 0.30, "brier": 0.10},
    ])
    assert select_best_model(comparison) == "logistic_regression"
    assert PRIMARY_SELECTION_METRIC == "average_precision"


def test_ties_on_average_precision_are_broken_by_calibration():
    """Two models with identical ranking quality: prefer the better-calibrated one."""
    comparison = pd.DataFrame([
        {"model": "overconfident", "split": "validation", "average_precision": 0.20, "brier": 0.12},
        {"model": "calibrated", "split": "validation", "average_precision": 0.20, "brier": 0.09},
    ])
    assert select_best_model(comparison) == "calibrated"


def test_select_best_model_needs_validation_rows():
    comparison = pd.DataFrame([{"model": "dummy", "split": "test", "average_precision": 0.1,
                                "brier": 0.1}])
    with pytest.raises(ValueError):
        select_best_model(comparison)
