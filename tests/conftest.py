"""Shared pytest fixtures and path setup.

Deliberately small. Anything expensive is generated once per session and named
after what it costs:

* ``synthetic_tables`` - 3 000 synthetic orders (fast, used by most tests);
* ``big_tables``       - 20 000 synthetic orders (used by the seasonality check,
  which needs enough November/December orders to draw a conclusion).

Everything is built through the *production* code paths, so a test failure means
the pipeline really is broken rather than that the test used its own copy of the
logic. Run the suite from the repository root::

    python -m pytest
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # allow `import src...` without installing
    sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config  # noqa: E402


@pytest.fixture(scope="session")
def cfg() -> Dict[str, Any]:
    """The shipped configuration.

    Tests use the real config rather than a hand-written dict so that a config and
    code drifting apart is a test failure. Tests that must not depend on the
    configured numbers overwrite the specific key they care about.
    """
    return load_config()


@pytest.fixture(scope="session")
def synthetic_tables() -> Dict[str, pd.DataFrame]:
    """A small Olist-shaped dataset, keyed by the *logical* table names.

    3 000 orders is enough for the joins, the label logic and a 3-way chronological
    split to be exercised end to end, and small enough to stay fast.
    """
    from src import synthetic

    return logical_tables(synthetic.generate_tables(n_orders=3000, seed=7))


def logical_tables(tables: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Re-key the generator's file-shaped output to the names the pipeline reads.

    ``generate_tables`` returns ``{"olist_orders_dataset": ...}`` because those keys
    become the CSV file names; ``build_order_dataset`` expects ``{"orders": ...}``.
    ``src.data_prep.TABLE_FILES`` is the single place that maps one to the other, so the
    translation is done through it rather than with a second hard-coded dictionary.
    """
    from src.data_prep import TABLE_FILES

    return {name: tables[Path(filename).stem] for name, filename in TABLE_FILES.items()}


@pytest.fixture(scope="session")
def big_tables() -> Dict[str, pd.DataFrame]:
    """A larger synthetic dataset, for claims that need statistical support."""
    from src import synthetic

    return logical_tables(synthetic.generate_tables(n_orders=20000, seed=11))


@pytest.fixture(scope="session")
def prepared_pair(synthetic_tables, cfg) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """``(prepared_frame, statistics)`` from the production preparation function.

    The frame is round-tripped through CSV first, because that is the only form the
    pipeline ever sees in production: ``prepare_dataset`` writes
    ``orders_prepared.csv`` and every later step reads it back. In-memory pandas 3
    frames keep Arrow-backed string columns containing ``pd.NA``, which scikit-learn
    cannot index ("boolean value of NA is ambiguous"); a test that fitted models on
    such a frame would be testing dtypes production never has.
    """
    from src.data_prep import build_order_dataset

    stats: Dict[str, Any] = {}
    frame = through_csv(build_order_dataset(synthetic_tables, cfg, stats))
    return frame, stats


def through_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Persist and reload a frame in memory, restoring the parsed timestamp columns.

    Mirrors :func:`src.data_prep.load_processed` without touching the disk.
    """
    import io

    import pandas as pd

    reloaded = pd.read_csv(io.StringIO(df.to_csv(index=False)))
    for column in ("order_purchase_timestamp", "order_estimated_delivery_date",
                   "order_delivered_customer_date", "promised_date", "delivered_date"):
        if column in reloaded.columns:
            reloaded[column] = pd.to_datetime(reloaded[column], errors="coerce")
    return reloaded


@pytest.fixture(scope="session")
def as_production_frame():
    """Expose :func:`through_csv` to test modules that build their own frames.

    A fixture rather than a plain import so the helper stays available regardless of
    how pytest is invoked.
    """
    return through_csv


@pytest.fixture(scope="session")
def prepared(prepared_pair) -> pd.DataFrame:
    """The prepared one-row-per-order frame."""
    return prepared_pair[0]


@pytest.fixture
def tmp_config(cfg, tmp_path) -> Dict[str, Any]:
    """A copy of the real config redirected into ``tmp_path`` with cheap models.

    Used by the end-to-end test so that running the suite cannot touch (or depend
    on) ``data/``, ``models/`` or ``reports/``. The model search space is shrunk,
    not the pipeline: the same functions run, just smaller.
    """
    tmp_cfg = copy.deepcopy(cfg)
    root = tmp_path / "project"
    tmp_cfg["paths"].update(
        {
            "raw_dir": str(root / "data" / "raw"),
            "synthetic_raw_dir": str(root / "data" / "raw" / "synthetic"),
            "processed_dir": str(root / "data" / "processed"),
            "models_dir": str(root / "models"),
            "reports_dir": str(root / "reports"),
            "figures_dir": str(root / "reports" / "figures"),
        }
    )
    tmp_cfg["data_source"].update(
        {"mode": "synthetic", "synthetic_n_orders": 1200, "synthetic_seed": 5}
    )
    tmp_cfg["modeling"]["random_forest"].update(
        {"n_estimators": 40, "max_depth": [6], "min_samples_leaf": [2], "n_jobs": 1}
    )
    tmp_cfg["modeling"]["logistic_regression"]["C"] = [1.0]
    # tmp_path is absolute, so every redirected path is absolute too - which is what
    # src.config.resolve() requires before it will trust a path.
    return tmp_cfg
