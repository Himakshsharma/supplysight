"""Small shared helpers (logging, seeding, timing, optional imports)."""

from __future__ import annotations

import importlib
import logging
import random
import time
from collections.abc import Iterator
from contextlib import contextmanager
from types import ModuleType

import numpy as np

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def get_logger(name: str) -> logging.Logger:
    """Return a module logger with a consistent format."""
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT)
    return logging.getLogger(name)


def set_seed(seed: int) -> np.random.Generator:
    """Seed Python and NumPy global RNGs and return a fresh NumPy Generator."""
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    return np.random.default_rng(seed)


def optional_import(name: str) -> ModuleType | None:
    """Import a module if it is installed, otherwise return None.

    SupplySight uses this so the core pipeline runs on a minimal install
    (numpy/pandas/scikit-learn) and upgrades itself when LightGBM, PyTorch,
    MLflow, rapidfuzz or sentence-transformers are available.
    """
    try:
        return importlib.import_module(name)
    except Exception:
        return None


@contextmanager
def timer() -> Iterator[dict]:
    """Context manager that records elapsed milliseconds in the yielded dict."""
    out: dict = {}
    start = time.perf_counter()
    try:
        yield out
    finally:
        out["ms"] = (time.perf_counter() - start) * 1000.0
