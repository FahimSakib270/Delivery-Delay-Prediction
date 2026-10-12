"""Small shared helpers: seeding, logging, IO, fingerprints and provenance."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-14s | %(message)s"


def get_logger(name: str = "profast") -> logging.Logger:
    """Return a logger that prints to stdout exactly once per name."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def set_seed(seed: int = 42) -> None:
    """Seed every random source the project uses (Python, NumPy).

    scikit-learn estimators additionally receive ``random_state=seed`` explicitly
    so reproducibility does not depend on global process state.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str | Path) -> Path:
    """Create ``path`` (and parents) if needed and return it as a ``Path``."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Any, path: str | Path) -> Path:
    """Write ``obj`` as pretty, UTF-8 JSON, converting NumPy scalars."""
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(json.dumps(_jsonable(obj), indent=2, ensure_ascii=False), encoding="utf-8")
    return p


def load_json(path: str | Path) -> Any:
    """Read JSON produced by :func:`save_json`."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _jsonable(obj: Any) -> Any:
    """Recursively convert NumPy / pandas types into plain JSON types."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return None if not np.isfinite(value) else value
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, (np.ndarray,)):
        return [_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if isinstance(obj, (Path,)):
        return str(obj)
    return obj


def file_sha256(path: str | Path, chunk_size: int = 1 << 20) -> Optional[str]:
    """SHA-256 of a file, or ``None`` when the file does not exist."""
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    digest = hashlib.sha256()
    with p.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataframe_fingerprint(df: pd.DataFrame, columns: Optional[Iterable[str]] = None) -> str:
    """Order-insensitive, content-based fingerprint of a DataFrame.

    Rows are hashed as text fields, then sorted, so the fingerprint depends on the
    *content* rather than the row order produced by the loader.
    """
    cols = list(columns) if columns is not None else list(df.columns)
    subset = df.loc[:, cols].astype("string").fillna("<NA>")
    row_hashes = pd.util.hash_pandas_object(subset, index=False).to_numpy()
    # ``np.sort`` returns a new array; ``ndarray.sort`` fails on the read-only view
    # that pandas hands back for its hashes.
    row_hashes = np.sort(row_hashes)
    digest = hashlib.sha256()
    digest.update(",".join(cols).encode("utf-8"))
    digest.update(row_hashes.tobytes())
    return digest.hexdigest()


def environment_info() -> Dict[str, str]:
    """Versions of everything that could plausibly change the results."""
    import sklearn  # local import keeps module import cheap

    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
    }
    try:  # optional but recorded because figures depend on them
        import matplotlib

        info["matplotlib"] = matplotlib.__version__
    except ImportError:  # pragma: no cover
        pass
    try:
        import seaborn

        info["seaborn"] = seaborn.__version__
    except ImportError:  # pragma: no cover
        pass
    return info


class Timer:
    """Tiny context manager used to record how long pipeline stages take."""

    def __init__(self) -> None:
        self.seconds: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.seconds = time.perf_counter() - self._start
