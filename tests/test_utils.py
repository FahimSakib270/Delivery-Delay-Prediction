"""Tests for :mod:`src.utils` and :mod:`src.config`.

These two modules are load-bearing for reproducibility: the fingerprints they
produce are what let the pipeline detect that its inputs changed, and the config
loader is what keeps every tunable decision in one reviewable place.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import config as config_mod
from src import utils


# --------------------------------------------------------------------------- basics
def test_get_logger_is_configured_once_and_does_not_touch_the_root_logger():
    first = utils.get_logger("unit_test_logger")
    second = utils.get_logger("unit_test_logger")
    assert first is second, "a second call must not stack another handler"
    assert len(first.handlers) == 1
    assert first.propagate is False, "a library should not reconfigure the caller's logging"


def test_set_seed_makes_python_and_numpy_deterministic():
    import random

    utils.set_seed(1234)
    first = (random.random(), float(np.random.rand()))
    utils.set_seed(1234)
    second = (random.random(), float(np.random.rand()))
    assert first == second


def test_ensure_dir_creates_nested_paths_and_is_idempotent(tmp_path):
    target = tmp_path / "a" / "b" / "c"
    assert utils.ensure_dir(target) == target
    assert target.is_dir()
    assert utils.ensure_dir(target) == target  # second call must not raise


# --------------------------------------------------------------------------- json IO
def test_save_json_converts_numpy_and_non_finite_values(tmp_path):
    path = tmp_path / "nested" / "meta.json"
    payload = {
        "int": np.int64(3),
        "float": np.float64(1.5),
        "bool": np.bool_(True),
        "timestamp": pd.Timestamp("2018-01-02 03:04:05"),
        "array": np.array([1, 2]),
        "nan": float("nan"),
        "path": Path("data") / "x.csv",
        "tuple": (1, 2),
    }
    utils.save_json(payload, path)

    assert path.exists()
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["int"] == 3 and isinstance(loaded["int"], int)
    assert loaded["float"] == 1.5
    assert loaded["bool"] is True
    assert loaded["timestamp"] == "2018-01-02T03:04:05"
    assert loaded["array"] == [1, 2]
    assert loaded["nan"] is None, "NaN is not valid JSON; it must become null"
    assert loaded["tuple"] == [1, 2]
    assert loaded["path"].endswith("x.csv")


def test_load_json_round_trips_through_save_json(tmp_path):
    payload = {"a": [1, 2, {"b": "c"}], "d": None}
    path = utils.save_json(payload, tmp_path / "p.json")
    assert utils.load_json(path) == payload


# --------------------------------------------------------------------------- hashing
def test_file_sha256_returns_none_for_missing_and_detects_edits(tmp_path):
    assert utils.file_sha256(tmp_path / "nope.csv") is None
    target = tmp_path / "f.txt"
    target.write_text("hello", encoding="utf-8")
    first = utils.file_sha256(target)
    assert first is not None and len(first) == 64
    target.write_text("hello!", encoding="utf-8")
    assert utils.file_sha256(target) != first


def test_dataframe_fingerprint_ignores_row_order_but_not_content():
    df = pd.DataFrame({"order_id": ["a", "b"], "is_late": [0, 1]})
    same_content = df.iloc[::-1]
    changed = df.copy()
    changed.loc[0, "is_late"] = 1

    assert utils.dataframe_fingerprint(df) == utils.dataframe_fingerprint(same_content)
    assert utils.dataframe_fingerprint(df) != utils.dataframe_fingerprint(changed)


def test_dataframe_fingerprint_is_subset_sensitive_and_handles_missing_values():
    df = pd.DataFrame({"a": [1, None], "b": [2, 3]})
    assert utils.dataframe_fingerprint(df, ["a"]) != utils.dataframe_fingerprint(df, ["b"])
    # A null must hash deterministically (it is replaced by a sentinel, not dropped).
    assert utils.dataframe_fingerprint(df, ["a"]) == utils.dataframe_fingerprint(df, ["a"])


def test_environment_info_records_the_libraries_that_change_results():
    info = utils.environment_info()
    for key in ("python", "platform", "numpy", "pandas", "scikit_learn"):
        assert info.get(key), f"{key} missing from environment info"


def test_timer_measures_a_real_interval():
    with utils.Timer() as timer:
        sum(range(1000))
    assert 0.0 <= timer.seconds < 5.0


# --------------------------------------------------------------------------- config
def test_load_config_merges_user_file_over_defaults(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text(
        "split:\n  train_frac: 0.5\n  val_frac: 0.25\n  test_frac: 0.25\n", encoding="utf-8"
    )
    loaded = config_mod.load_config(path)
    assert loaded["split"]["train_frac"] == 0.5
    # Keys absent from the file still come from DEFAULTS: a config file is a patch,
    # not a replacement, so adding a key to DEFAULTS never breaks existing configs.
    assert "modeling" in loaded
    assert isinstance(loaded["project"]["random_seed"], int)


def test_load_config_raises_for_an_explicit_missing_path(tmp_path):
    with pytest.raises(FileNotFoundError):
        config_mod.load_config(tmp_path / "absent.yaml")


def test_config_override_does_not_mutate_the_defaults(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("modeling:\n  random_forest:\n    n_estimators: 7\n", encoding="utf-8")
    loaded = config_mod.load_config(path)
    loaded["modeling"]["random_forest"]["n_estimators"] = 999
    assert config_mod.DEFAULTS["modeling"]["random_forest"]["n_estimators"] != 999


def test_resolve_returns_absolute_paths_and_can_demand_existence(cfg, tmp_path):
    reports = config_mod.resolve(cfg, "paths", "reports_dir")
    assert reports.is_absolute()
    assert reports.name == "reports"

    cfg_with_missing = {"paths": {"x": str(tmp_path / "missing")}}
    with pytest.raises(FileNotFoundError):
        config_mod.resolve(cfg_with_missing, "paths", "x", must_exist=True)


def test_project_path_is_anchored_at_the_repository_root():
    assert config_mod.project_path("config", "config.yaml").is_file()


def test_shipped_config_is_self_consistent(cfg):
    split = cfg["split"]
    assert abs(split["train_frac"] + split["val_frac"] + split["test_frac"] - 1.0) < 1e-9
    assert split["train_frac"] > split["val_frac"] > 0
    assert cfg["business_case"]["cost_false_negative"] > cfg["business_case"]["cost_false_positive"] > 0
    assert cfg["label"]["primary"] in {"date_based", "timestamp_based"}
    assert cfg["modeling"]["dummy"]["strategy"] == "most_frequent"
