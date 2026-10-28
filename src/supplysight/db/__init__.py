"""Database access: schema DDL, loaders and read-only connections.

SQLite is the zero-setup default. Set ``DATABASE_URL=postgresql+psycopg2://...`` to use
PostgreSQL (needs SQLAlchemy + psycopg2, both in the ``full`` extra). All table and column
names are lower-case so the same SQL runs on both engines.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd

from supplysight.config import get_settings
from supplysight.utils import get_logger, optional_import

log = get_logger(__name__)

T, F, I = "TEXT", "DOUBLE PRECISION", "INTEGER"  # noqa: E741

# table -> (columns, primary key). Documented in docs/DATA_DICTIONARY.md
SCHEMA: dict[str, tuple[dict[str, str], list[str]]] = {
    "lfa1": ({"lifnr": T, "name1": T, "stras": T, "ort01": T, "regio": T, "pstlz": T, "stcd3": T,
              "bank_ref": T, "created_on": T, "land1": T}, ["lifnr"]),
    "mara": ({"matnr": T, "mtart": T, "matkl": T, "meins": T, "ersda": T}, ["matnr"]),
    "makt": ({"matnr": T, "maktx": T, "spras": T}, ["matnr", "spras"]),
    "marc": ({"matnr": T, "werks": T, "plifz": I, "eisbe": F, "minbe": F, "dispo": T, "dismm": T},
             ["matnr", "werks"]),
    "mard": ({"matnr": T, "werks": T, "lgort": T, "labst": F, "open_po_qty": F}, ["matnr", "werks", "lgort"]),
    "eina": ({"infnr": T, "matnr": T, "lifnr": T, "created_on": T, "is_primary": I}, ["infnr"]),
    "eine": ({"infnr": T, "ekorg": T, "netpr": F, "peinh": I, "waers": T, "aplfz": I, "valid_from": T},
             ["infnr", "ekorg", "valid_from"]),  # one condition record per validity start
    "eban": ({"banfn": T, "bnfpo": T, "matnr": T, "werks": T, "menge": F, "requested_by": T,
              "created_on": T, "ebeln": T, "ebelp": T}, ["banfn", "bnfpo"]),
    "ekko": ({"ebeln": T, "bukrs": T, "bsart": T, "lifnr": T, "bedat": T, "created_at": T, "ekorg": T,
              "ekgrp": T, "created_by": T, "waers": T, "approval_status": T, "status": T}, ["ebeln"]),
    "ekpo": ({"ebeln": T, "ebelp": T, "matnr": T, "werks": T, "menge": F, "meins": T, "netpr": F,
              "peinh": I, "netwr": F, "eindt": T, "banfn": T, "matkl": T}, ["ebeln", "ebelp"]),
    "mkpf": ({"mblnr": T, "mjahr": I, "budat": T, "usnam": T}, ["mblnr", "mjahr"]),
    "mseg": ({"mblnr": T, "zeile": T, "bwart": T, "matnr": T, "werks": T, "menge": F, "meins": T,
              "budat": T, "ebeln": T, "aufnr": T, "lifnr": T}, ["mblnr", "zeile"]),
    "afko": ({"aufnr": T, "werks": T, "plnbez": T, "gamng": F, "gstrp": T, "gltrp": T, "auart": T}, ["aufnr"]),
    "afpo": ({"aufnr": T, "posnr": T, "matnr": T, "psmng": F, "wemng": F, "werks": T}, ["aufnr", "posnr"]),
    "stko": ({"stlnr": T, "matnr": T, "bmeng": F, "bmein": T}, ["stlnr"]),
    "stpo": ({"stlnr": T, "posnr": T, "idnrk": T, "menge": F, "meins": T}, ["stlnr", "posnr"]),
    "plaf": ({"plnum": T, "matnr": T, "plwrk": T, "gsmng": F, "psttr": T, "paart": T}, ["plnum"]),
    "anomaly_labels": ({"ebeln": T, "ebelp": T, "anomaly_type": T}, ["ebeln", "ebelp"]),
    # ---- application tables written by the ML pipeline / API
    "po_scores": ({"ebeln": T, "ebelp": T, "lifnr": T, "matnr": T, "werks": T, "bedat": T, "netwr": F,
                   "score": F, "risk_level": T, "reasons": T, "rule_score": F, "iforest_score": F,
                   "ae_score": F, "model_version": T}, ["ebeln", "ebelp"]),
    "vendor_duplicates": ({"lifnr_a": T, "lifnr_b": T, "name_a": T, "name_b": T, "similarity": F,
                           "match_on": T}, ["lifnr_a", "lifnr_b"]),
    "reorder_recommendations": ({"matnr": T, "werks": T, "maktx": T, "matkl": T, "avg_weekly_forecast": F,
                                 "sigma": F, "lead_time_days": I, "safety_stock": F, "recommended_rop": F,
                                 "current_minbe": F, "current_stock": F, "open_po_qty": F,
                                 "weeks_of_cover": F, "stockout_risk": T, "rop_gap_pct": F}, ["matnr", "werks"]),
    "demand_forecast": ({"matnr": T, "werks": T, "week": T, "yhat": F, "p10": F, "p90": F,
                         "model_version": T}, ["matnr", "werks", "week"]),
    "review_feedback": ({"id": T, "ebeln": T, "ebelp": T, "verdict": T, "comment": T, "reviewer": T,
                         "reviewed_at": T}, ["id"]),
}
APP_TABLES = ("po_scores", "vendor_duplicates", "reorder_recommendations", "demand_forecast", "review_feedback")
SOURCE_TABLES = [t for t in SCHEMA if t not in APP_TABLES]
# tables the copilot SQL tool may read (labels and feedback are deliberately excluded)
COPILOT_TABLES = ["lfa1", "mara", "makt", "marc", "mard", "eina", "eine", "eban", "ekko", "ekpo",
                  "mseg", "mkpf", "afko", "afpo", "stko", "stpo", "plaf", "po_scores", "reorder_recommendations",
                  "vendor_duplicates", "demand_forecast"]


def _sqlite_path(url: str) -> Path:
    return Path(url.replace("sqlite:///", "", 1))


def ddl(table: str) -> str:
    """Return a portable CREATE TABLE statement for ``table``."""
    cols, pk = SCHEMA[table]
    body = ",\n  ".join(f"{c} {t}" for c, t in cols.items())
    return f"CREATE TABLE IF NOT EXISTS {table} (\n  {body},\n  PRIMARY KEY ({', '.join(pk)})\n)"


@contextmanager
def connect(readonly: bool = False, url: str | None = None) -> Iterator[Any]:
    """Yield a DB-API connection (sqlite3) or SQLAlchemy connection (other URLs).

    ``readonly=True`` opens SQLite in ``mode=ro`` and puts PostgreSQL sessions into a
    read-only transaction with a statement timeout — the copilot uses this.
    """
    s = get_settings()
    url = url or s.database_url
    if url.startswith("sqlite"):
        path = _sqlite_path(url)
        path.parent.mkdir(parents=True, exist_ok=True)
        if readonly:
            conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True, check_same_thread=False)
        else:
            conn = sqlite3.connect(path.as_posix(), check_same_thread=False)
        try:
            yield conn
            if not readonly:
                conn.commit()
        finally:
            conn.close()
        return
    sa = optional_import("sqlalchemy")
    if sa is None:  # pragma: no cover
        raise RuntimeError("Non-SQLite DATABASE_URL requires SQLAlchemy (pip install '.[full]')")
    engine = sa.create_engine(url, pool_pre_ping=True)
    with engine.connect() as conn:  # pragma: no cover - exercised in docker-compose
        if readonly:
            conn.exec_driver_sql("SET TRANSACTION READ ONLY")
            conn.exec_driver_sql(f"SET LOCAL statement_timeout = {int(s.sql_timeout_s * 1000)}")
        yield conn
        if not readonly:
            conn.commit()


def init_db(drop: bool = False) -> None:
    """Create all tables (optionally dropping them first)."""
    with connect() as conn:
        for t in SCHEMA:
            if drop:
                _exec(conn, f"DROP TABLE IF EXISTS {t}")
            _exec(conn, ddl(t))


def _exec(conn: Any, sql: str, params: Any = None) -> None:
    if isinstance(conn, sqlite3.Connection):
        conn.execute(sql, params or ())
    else:  # pragma: no cover
        conn.exec_driver_sql(sql, params or ())


def write_table(name: str, df: pd.DataFrame, replace: bool = True) -> None:
    """Write a DataFrame to a schema table (delete+append keeps the DDL and keys)."""
    cols = list(SCHEMA[name][0])
    df = df.reindex(columns=cols)
    with connect() as conn:
        _exec(conn, ddl(name))
        if replace:
            _exec(conn, f"DELETE FROM {name}")
        df.to_sql(name, conn, if_exists="append", index=False, chunksize=5000)


def load_tables(tables: dict[str, pd.DataFrame]) -> None:
    """(Re)create the schema and load the generated source tables."""
    init_db(drop=True)
    for name, df in tables.items():
        if name in SCHEMA:
            write_table(name, df)
    log.info("Loaded %d tables into %s", len(tables), get_settings().database_url.split("@")[-1])


def read_sql(sql: str, params: Any = None, readonly: bool = True) -> pd.DataFrame:
    """Run a query and return a DataFrame."""
    with connect(readonly=readonly) as conn:
        return pd.read_sql_query(sql, conn, params=params)


def read_table(name: str) -> pd.DataFrame:
    """Read a whole table (string keys stay strings)."""
    if name not in SCHEMA:
        raise KeyError(name)
    df = read_sql(f"SELECT * FROM {name}")
    for c, t in SCHEMA[name][0].items():
        if t == T and c in df:
            df[c] = df[c].where(df[c].isna(), df[c].astype(str))
    return df


def table_exists(name: str) -> bool:
    """True if ``name`` exists and has at least one row."""
    try:
        return len(read_sql(f"SELECT 1 FROM {name} LIMIT 1")) > 0
    except Exception:
        return False


def execute(sql: str, params: Any = None) -> None:
    """Execute a write statement."""
    with connect() as conn:
        _exec(conn, sql, params)
