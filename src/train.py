"""CLI alias: ``python -m src.train`` -> :func:`src.training.main`.

The project documents its entry points as commands (``src.prepare``, ``src.train``,
``src.evaluate``), so this module exists purely to make ``src.train`` a valid module
name. All logic lives in :mod:`src.training`.
"""

from __future__ import annotations

from .training import main

if __name__ == "__main__":  # pragma: no cover - CLI convenience
    raise SystemExit(main())
