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

TEXT, FLOAT, INT = "TEXT", "DOUBLE PRECISION", "INTEGER"

# table -> (columns, primary key). Documented in docs/DATA_DICTIONARY.md
SCHEMA: dict[str, tuple[dict[str, str], list[str]]] = {
    "lfa1": (
        {
            "lifnr": TEXT,
            "name1": TEXT,
            "stras": TEXT,
            "ort01": TEXT,
            "regio": TEXT,
            "pstlz": TEXT,
            "stcd3": TEXT,
            "bank_ref": TEXT,
            "created_on": TEXT,
            "land1": TEXT,
        },
        ["lifnr"],
    ),
    "mara": ({"matnr": TEXT, "mtart": TEXT, "matkl": TEXT, "meins": TEXT, "ersda": TEXT}, ["matnr"]),
    "makt": ({"matnr": TEXT, "maktx": TEXT, "spras": TEXT}, ["matnr", "spras"]),
    "marc": (
        {"matnr": TEXT, "werks": TEXT, "plifz": INT, "eisbe": FLOAT, "minbe": FLOAT, "dispo": TEXT, "dismm": TEXT},
        ["matnr", "werks"],
    ),
    "mard": (
        {"matnr": TEXT, "werks": TEXT, "lgort": TEXT, "labst": FLOAT, "open_po_qty": FLOAT},
        ["matnr", "werks", "lgort"],
    ),
    "eina": ({"infnr": TEXT, "matnr": TEXT, "lifnr": TEXT, "created_on": TEXT, "is_primary": INT}, ["infnr"]),
    "eine": (
        {"infnr": TEXT, "ekorg": TEXT, "netpr": FLOAT, "peinh": INT, "waers": TEXT, "aplfz": INT, "valid_from": TEXT},
        ["infnr", "ekorg", "valid_from"],
    ),  # one condition record per validity start
    "eban": (
        {
            "banfn": TEXT,
            "bnfpo": TEXT,
            "matnr": TEXT,
            "werks": TEXT,
            "menge": FLOAT,
            "requested_by": TEXT,
            "created_on": TEXT,
            "ebeln": TEXT,
            "ebelp": TEXT,
        },
        ["banfn", "bnfpo"],
    ),
    "ekko": (
        {
            "ebeln": TEXT,
            "bukrs": TEXT,
            "bsart": TEXT,
            "lifnr": TEXT,
            "bedat": TEXT,
            "created_at": TEXT,
            "ekorg": TEXT,
            "ekgrp": TEXT,
            "created_by": TEXT,
            "waers": TEXT,
            "approval_status": TEXT,
            "status": TEXT,
        },
        ["ebeln"],
    ),
    "ekpo": (
        {
            "ebeln": TEXT,
            "ebelp": TEXT,
            "matnr": TEXT,
            "werks": TEXT,
            "menge": FLOAT,
            "meins": TEXT,
            "netpr": FLOAT,
            "peinh": INT,
            "netwr": FLOAT,
            "eindt": TEXT,
            "banfn": TEXT,
            "matkl": TEXT,
        },
        ["ebeln", "ebelp"],
    ),
    "mkpf": ({"mblnr": TEXT, "mjahr": INT, "budat": TEXT, "usnam": TEXT}, ["mblnr", "mjahr"]),
    "mseg": (
        {
            "mblnr": TEXT,
            "zeile": TEXT,
            "bwart": TEXT,
            "matnr": TEXT,
            "werks": TEXT,
            "menge": FLOAT,
            "meins": TEXT,
            "budat": TEXT,
            "ebeln": TEXT,
            "aufnr": TEXT,
            "lifnr": TEXT,
        },
        ["mblnr", "zeile"],
    ),
    "afko": (
        {"aufnr": TEXT, "werks": TEXT, "plnbez": TEXT, "gamng": FLOAT, "gstrp": TEXT, "gltrp": TEXT, "auart": TEXT},
        ["aufnr"],
    ),
    "afpo": (
        {"aufnr": TEXT, "posnr": TEXT, "matnr": TEXT, "psmng": FLOAT, "wemng": FLOAT, "werks": TEXT},
        ["aufnr", "posnr"],
    ),
    "stko": ({"stlnr": TEXT, "matnr": TEXT, "bmeng": FLOAT, "bmein": TEXT}, ["stlnr"]),
    "stpo": ({"stlnr": TEXT, "posnr": TEXT, "idnrk": TEXT, "menge": FLOAT, "meins": TEXT}, ["stlnr", "posnr"]),
    "plaf": ({"plnum": TEXT, "matnr": TEXT, "plwrk": TEXT, "gsmng": FLOAT, "psttr": TEXT, "paart": TEXT}, ["plnum"]),
    "anomaly_labels": ({"ebeln": TEXT, "ebelp": TEXT, "anomaly_type": TEXT}, ["ebeln", "ebelp"]),
    # ---- application tables written by the ML pipeline / API
    "po_scores": (
        {
            "ebeln": TEXT,
            "ebelp": TEXT,
            "lifnr": TEXT,
            "matnr": TEXT,
            "werks": TEXT,
            "bedat": TEXT,
            "netwr": FLOAT,
            "score": FLOAT,
            "risk_level": TEXT,
            "reasons": TEXT,
            "rule_score": FLOAT,
            "iforest_score": FLOAT,
            "ae_score": FLOAT,
            "model_version": TEXT,
        },
        ["ebeln", "ebelp"],
    ),
    "vendor_duplicates": (
        {"lifnr_a": TEXT, "lifnr_b": TEXT, "name_a": TEXT, "name_b": TEXT, "similarity": FLOAT, "match_on": TEXT},
        ["lifnr_a", "lifnr_b"],
    ),
    "reorder_recommendations": (
        {
            "matnr": TEXT,
            "werks": TEXT,
            "maktx": TEXT,
            "matkl": TEXT,
            "avg_weekly_forecast": FLOAT,
            "sigma": FLOAT,
            "lead_time_days": INT,
            "safety_stock": FLOAT,
            "recommended_rop": FLOAT,
            "current_minbe": FLOAT,
            "current_stock": FLOAT,
            "open_po_qty": FLOAT,
            "weeks_of_cover": FLOAT,
            "stockout_risk": TEXT,
            "rop_gap_pct": FLOAT,
        },
        ["matnr", "werks"],
    ),
    "demand_forecast": (
        {"matnr": TEXT, "werks": TEXT, "week": TEXT, "yhat": FLOAT, "p10": FLOAT, "p90": FLOAT, "model_version": TEXT},
        ["matnr", "werks", "week"],
    ),
    "review_feedback": (
        {
            "id": TEXT,
            "ebeln": TEXT,
            "ebelp": TEXT,
            "verdict": TEXT,
            "comment": TEXT,
            "reviewer": TEXT,
            "reviewed_at": TEXT,
        },
        ["id"],
    ),
}
APP_TABLES = ("po_scores", "vendor_duplicates", "reorder_recommendations", "demand_forecast", "review_feedback")
SOURCE_TABLES = [t for t in SCHEMA if t not in APP_TABLES]
# tables the copilot SQL tool may read (labels and feedback are deliberately excluded)
COPILOT_TABLES = [
    "lfa1",
    "mara",
    "makt",
    "marc",
    "mard",
    "eina",
    "eine",
    "eban",
    "ekko",
    "ekpo",
    "mseg",
    "mkpf",
    "afko",
    "afpo",
    "stko",
    "stpo",
    "plaf",
    "po_scores",
    "reorder_recommendations",
    "vendor_duplicates",
    "demand_forecast",
]


_ENGINES: dict[str, Any] = {}


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
        raise RuntimeError("Non-SQLite DATABASE_URL requires SQLAlchemy (pip install '.[serve]')")
    engine = _ENGINES.get(url)
    if engine is None:  # pragma: no cover
        engine = _ENGINES[url] = sa.create_engine(url, pool_pre_ping=True)
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
        if not isinstance(conn, sqlite3.Connection):  # pragma: no cover
            for stmt in PG_HELPERS:
                _exec(conn, stmt)
        for t in SCHEMA:
            if drop:
                _exec(conn, f"DROP TABLE IF EXISTS {t}")
            _exec(conn, ddl(t))


def _pg(sql: str) -> str:
    """Translate SQLite '?' placeholders to the psycopg2 paramstyle."""
    return sql.replace("%", "%%").replace("?", "%s")


def _exec(conn: Any, sql: str, params: Any = None) -> None:
    if isinstance(conn, sqlite3.Connection):
        conn.execute(sql, params or ())
    else:  # pragma: no cover
        conn.exec_driver_sql(_pg(sql) if params else sql, tuple(params) if params else None)


PG_HELPERS = [  # keep SQLite-style SQL portable on PostgreSQL
    "CREATE OR REPLACE FUNCTION round(double precision, integer) RETURNS numeric "
    "AS $$ SELECT round($1::numeric, $2) $$ LANGUAGE SQL IMMUTABLE",
]


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
        if params and not isinstance(conn, sqlite3.Connection):  # pragma: no cover
            return pd.read_sql_query(_pg(sql), conn, params=tuple(params))
        return pd.read_sql_query(sql, conn, params=params)


def read_table(name: str) -> pd.DataFrame:
    """Read a whole table (string keys stay strings)."""
    if name not in SCHEMA:
        raise KeyError(name)
    df = read_sql(f"SELECT * FROM {name}")
    for c, t in SCHEMA[name][0].items():
        if t == TEXT and c in df:
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
