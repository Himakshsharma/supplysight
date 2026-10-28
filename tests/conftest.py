"""Shared fixtures: a small, seeded dataset loaded into a temporary SQLite database."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(scope="session")
def small_cfg():
    from supplysight.datagen.generator import GenConfig

    return GenConfig(seed=7, start="2025-01-06", end="2026-06-28", material_scale=0.3, n_vendors=25)


@pytest.fixture(scope="session")
def small_tables(small_cfg):
    from supplysight.datagen.generator import generate

    return generate(small_cfg)


@pytest.fixture(scope="session")
def tmp_env(tmp_path_factory):
    """Point settings at a temp data/artifacts dir for the whole test session."""
    base = tmp_path_factory.mktemp("supplysight")
    os.environ["SUPPLYSIGHT_DATA_DIR"] = str(base / "data")
    os.environ["SUPPLYSIGHT_ARTIFACTS_DIR"] = str(base / "artifacts")
    os.environ["DATABASE_URL"] = f"sqlite:///{(base / 'data' / 'test.db').as_posix()}"
    os.environ["JWT_SECRET"] = "test-secret-key-for-unit-tests-only-32b"
    os.environ.pop("GEMINI_API_KEY", None)
    from supplysight.config import reset_settings_cache

    reset_settings_cache()
    yield base


@pytest.fixture(scope="session")
def loaded_db(tmp_env, small_tables):
    from supplysight.db import load_tables

    load_tables(small_tables)
    return small_tables


@pytest.fixture(scope="session")
def trained(loaded_db):
    """Train every model once on the small dataset (session-scoped, ~seconds)."""
    from supplysight.pipeline import train_all

    return train_all(modules=("anomaly", "forecast", "copilot"), quick=True)
