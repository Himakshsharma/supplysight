"""Data-quality checks and convenience loaders shared by training scripts and services."""

from __future__ import annotations

import pandas as pd

from supplysight.db import SOURCE_TABLES, read_table

# (child table, child column, parent table, parent column)
FOREIGN_KEYS = [
    ("ekpo", "matnr", "mara", "matnr"),
    ("ekpo", "ebeln", "ekko", "ebeln"),
    ("ekko", "lifnr", "lfa1", "lifnr"),
    ("eina", "matnr", "mara", "matnr"),
    ("eina", "lifnr", "lfa1", "lifnr"),
    ("eine", "infnr", "eina", "infnr"),
    ("marc", "matnr", "mara", "matnr"),
    ("makt", "matnr", "mara", "matnr"),
    ("eban", "matnr", "mara", "matnr"),
    ("mseg", "matnr", "mara", "matnr"),
    ("mseg", "mblnr", "mkpf", "mblnr"),
    ("afko", "plnbez", "mara", "matnr"),
    ("afpo", "aufnr", "afko", "aufnr"),
    ("stko", "matnr", "mara", "matnr"),
    ("stpo", "idnrk", "mara", "matnr"),
    ("stpo", "stlnr", "stko", "stlnr"),
    ("plaf", "matnr", "mara", "matnr"),
    ("anomaly_labels", "ebeln", "ekko", "ebeln"),
]


def validate(tables: dict[str, pd.DataFrame]) -> list[str]:
    """Return a list of referential-integrity / sanity problems (empty list = OK)."""
    problems: list[str] = []
    for child, ccol, parent, pcol in FOREIGN_KEYS:
        if child not in tables or parent not in tables:
            continue
        vals = tables[child][ccol].dropna().astype(str)
        missing = set(vals) - set(tables[parent][pcol].astype(str))
        if missing:
            problems.append(f"{child}.{ccol} -> {parent}.{pcol}: {len(missing)} orphan keys")
    ekpo = tables.get("ekpo")
    if ekpo is not None:
        if (ekpo.menge <= 0).any():
            problems.append("ekpo has non-positive quantities")
        if (ekpo.netpr <= 0).any():
            problems.append("ekpo has non-positive prices")
        if ekpo.duplicated(["ebeln", "ebelp"]).any():
            problems.append("ekpo has duplicate keys")
    mseg = tables.get("mseg")
    if mseg is not None and not set(mseg.bwart.unique()) <= {"101", "261"}:
        problems.append("unexpected movement types in mseg")
    return problems


def load_source_tables(names: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Read source tables from the configured database."""
    return {n: read_table(n) for n in (names or SOURCE_TABLES)}


def po_items(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Join PO items with header, vendor, material and info-record data (one row per item)."""
    ekpo, ekko = tables["ekpo"], tables["ekko"]
    df = ekpo.merge(ekko[["ebeln", "lifnr", "bedat", "created_at", "created_by", "ekgrp"]], on="ebeln")
    df = df.merge(tables["lfa1"][["lifnr", "name1", "created_on"]].rename(columns={"created_on": "vendor_created_on"}),
                  on="lifnr", how="left")
    df = df.merge(tables["makt"][["matnr", "maktx"]], on="matnr", how="left")
    df = df.merge(tables["eban"][["banfn", "menge"]].rename(columns={"menge": "pr_menge"}), on="banfn", how="left")
    info = tables["eina"].merge(tables["eine"], on="infnr")
    df = df.merge(info[["matnr", "lifnr", "netpr", "aplfz"]].rename(columns={"netpr": "info_netpr"}),
                  on=["matnr", "lifnr"], how="left")
    prim = info[info.is_primary.astype(int) == 1][["matnr", "aplfz", "netpr"]].rename(
        columns={"aplfz": "prim_aplfz", "netpr": "prim_netpr"})
    df = df.merge(prim, on="matnr", how="left")
    df["created_at"] = pd.to_datetime(df.created_at)
    df["bedat"] = pd.to_datetime(df.bedat)
    df["eindt"] = pd.to_datetime(df.eindt)
    df["vendor_created_on"] = pd.to_datetime(df.vendor_created_on)
    return df.sort_values(["created_at", "ebeln"]).reset_index(drop=True)


def weekly_consumption(mseg: pd.DataFrame) -> pd.DataFrame:
    """Weekly goods issues to production (movement 261) per material and plant.

    Returns a dense frame (every week for every series) with columns
    ``matnr, werks, week, qty``; weeks start on Monday.
    """
    x = mseg[mseg.bwart.astype(str) == "261"].copy()
    d = pd.to_datetime(x.budat)
    x["week"] = (d - pd.to_timedelta(d.dt.weekday, unit="D")).dt.normalize()
    g = x.groupby(["matnr", "werks", "week"], as_index=False).menge.sum().rename(columns={"menge": "qty"})
    weeks = pd.date_range(g.week.min(), g.week.max(), freq="W-MON")
    keys = g[["matnr", "werks"]].drop_duplicates()
    full = keys.merge(pd.DataFrame({"week": weeks}), how="cross")
    out = full.merge(g, on=["matnr", "werks", "week"], how="left").fillna({"qty": 0.0})
    return out.sort_values(["matnr", "werks", "week"]).reset_index(drop=True)
