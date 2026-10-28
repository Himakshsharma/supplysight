"""Build the demo state (data, models, reports) on a fresh deployment.

Hosted environments such as Streamlit Community Cloud start from a clean checkout, where the
database and trained models do not exist yet (both are git-ignored). ``ensure_demo_state``
generates the synthetic dataset, trains the models and produces the drift report once, then
returns immediately on later calls.
"""

from __future__ import annotations

import shutil

from supplysight.config import get_settings
from supplysight.data import load_source_tables, read_csv_tables
from supplysight.datagen.generator import DRIFT_TABLES, GenConfig, drift_quarter_config, generate, write_csv
from supplysight.db import load_tables, table_exists
from supplysight.mlops import Registry
from supplysight.pipeline import ANOMALY_MODEL, FORECAST_MODEL, train_all
from supplysight.utils import get_logger

log = get_logger(__name__)


def is_ready() -> bool:
    """True when the database and both production models exist."""
    reg = Registry()
    return (
        table_exists("po_scores")
        and table_exists("demand_forecast")
        and reg.get_version(ANOMALY_MODEL, "Production") is not None
        and reg.get_version(FORECAST_MODEL, "Production") is not None
    )


def ensure_demo_state(seed: int | None = None, quick: bool = True) -> bool:
    """Generate data and train models if needed. Returns True if anything was built.

    ``quick`` uses two backtest origins instead of six, which halves the CPU needed on a cold start
    of a free hosting tier. ``python run.py all`` always runs the full evaluation.
    """
    if is_ready():
        return False
    s = get_settings()
    seed = s.seed if seed is None else seed
    log.info("No trained state found - building the demo dataset and models (seed %d)", seed)
    tables = generate(GenConfig(seed=seed))
    write_csv(tables, s.data_dir / "raw")
    load_tables(tables)
    train_all(("anomaly", "forecast", "copilot"), quick=quick, promote="always")
    _build_drift_report(seed)
    return True


def _build_drift_report(seed: int) -> None:
    """Simulate the Q4 2026 drift quarter and write the drift report (no retraining)."""
    from supplysight.monitoring.retrain import drift_check, drift_dir

    cfg = drift_quarter_config(seed)
    shutil.rmtree(drift_dir(), ignore_errors=True)
    write_csv({k: v for k, v in generate(cfg).items() if k in DRIFT_TABLES}, drift_dir())
    try:
        drift_check(load_source_tables(), read_csv_tables(drift_dir()))
    except Exception as exc:  # the dashboard still works without a drift report
        log.warning("Drift report skipped: %s", exc)
