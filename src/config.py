"""Configuration loading.

The project keeps every tunable decision in ``config/config.yaml`` so that a run
is fully described by (code commit, resolved config, dataset fingerprint).

Only one small third-party dependency is introduced here (PyYAML); JSON is the
fallback so the package still imports if PyYAML is unavailable.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict

# Repository root = the directory that contains this package's parent folder.
ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT_DIR / "config" / "config.yaml"

DEFAULTS: Dict[str, Any] = {
    "project": {"name": "ProFast Delivery Delay Prediction", "version": "1.1.0", "random_seed": 42},
    "paths": {
        "raw_dir": "data/raw",
        "synthetic_raw_dir": "data/raw/synthetic",
        "processed_dir": "data/processed",
        "models_dir": "models",
        "reports_dir": "reports",
        "figures_dir": "reports/figures",
    },
    "data_source": {
        "mode": "olist",
        "olist_dir": "data/raw/olist",
        "synthetic_n_orders": 60000,
        "synthetic_seed": 20240101,
    },
    "label": {"primary": "date_based", "sensitivity_check": "timestamp_based"},
    "split": {"train_frac": 0.70, "val_frac": 0.15, "test_frac": 0.15},
    "label_maturity": {"enabled": True},
    "features": {
        "include_payment_features": False,
        "include_distance": False,
        "min_category_frequency": 50,
        "max_category_levels": 60,
    },
    "modeling": {
        "logistic_regression": {"C": [0.1, 1.0], "max_iter": 1000, "class_weight": "balanced"},
        "random_forest": {
            "n_estimators": 400,
            "max_depth": [12, 20],
            "min_samples_leaf": [1, 5],
            "max_features": ["sqrt"],
            "class_weight": "balanced_subsample",
            "n_jobs": -1,
        },
        "dummy": {"strategy": "most_frequent"},
    },
    "threshold_selection": {"objective": "max_f1", "min_precision_floor": 0.30},
    "business_case": {
        # ASSUMPTIONS, not measurements: used only for the "is flagging worth it?"
        # analysis and for the cost-optimal threshold. Replace with real numbers.
        "cost_false_negative": 25.0,   # a delay nobody anticipated (support + churn risk)
        "cost_false_positive": 4.0,    # an unnecessary proactive contact / voucher
    },
    "evaluation": {"bootstrap_n": 500, "subgroups_min_n": 100},
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path | None = None) -> Dict[str, Any]:
    """Load the YAML config, merged on top of :data:`DEFAULTS`.

    Parameters
    ----------
    path:
        Optional override. ``None`` uses ``config/config.yaml``.

    Raises
    ------
    FileNotFoundError
        If an explicit ``path`` was given but does not exist. A missing *default*
        config is tolerated and the built-in defaults are used instead.
    """
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        if path is not None:
            raise FileNotFoundError(f"Config file not found: {cfg_path}")
        return copy.deepcopy(DEFAULTS)

    raw_text = cfg_path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(raw_text) or {}
    except ImportError:  # pragma: no cover - PyYAML is in requirements.txt
        loaded = json.loads(raw_text)

    if not isinstance(loaded, dict):
        raise ValueError(f"Config root must be a mapping, got {type(loaded).__name__}")
    return _deep_merge(DEFAULTS, loaded)


def resolve(cfg: Dict[str, Any], *keys: str, must_exist: bool = False) -> Path:
    """Resolve a dotted config path (e.g. ``paths.reports_dir``) to an absolute path."""
    node: Any = cfg
    for key in keys:
        node = node[key]
    path = Path(node)
    if not path.is_absolute():
        path = ROOT_DIR / path
    if must_exist and not path.exists():
        raise FileNotFoundError(f"Configured path does not exist: {path}")
    return path


def project_path(*parts: str) -> Path:
    """Absolute path relative to the repository root."""
    return ROOT_DIR.joinpath(*parts)
