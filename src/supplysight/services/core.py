"""Business services shared by the FastAPI app and the Streamlit dashboard.

Everything reads the serving tables written by the pipeline and the models from the registry
(Production stage). Heavy objects are cached and refreshed when the database or the registry
changes.
"""

from __future__ import annotations

import json
import time
import uuid
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from supplysight.config import get_settings
from supplysight.data import current_info_records, load_source_tables, po_items, weekly_consumption
from supplysight.db import execute, read_sql, table_exists, write_table
from supplysight.features.po_features import build_po_features
from supplysight.mlops import Registry, read_runs
from supplysight.pipeline import ANOMALY_MODEL, FORECAST_MODEL

_CACHE: dict = {}


class NotFound(LookupError):
    """Requested entity does not exist."""


# ------------------------------------------------------------------ cache helpers
def _db_stamp() -> float:
    s = get_settings()
    if s.is_sqlite:
        p = Path(s.database_url.replace("sqlite:///", "", 1))
        return p.stat().st_mtime if p.exists() else 0.0
    return time.time() // 300  # pragma: no cover - refresh every 5 min on Postgres


def _registry_stamp() -> float:
    p = Registry().index_path
    return p.stat().st_mtime if p.exists() else 0.0


def _cached(key: str, stamp: float, build):
    hit = _CACHE.get(key)
    if hit and hit[0] == stamp:
        return hit[1]
    val = build()
    _CACHE[key] = (stamp, val)
    return val


def tables() -> dict[str, pd.DataFrame]:
    """Source tables (cached until the database changes)."""
    return _cached("tables", _db_stamp(), load_source_tables)


def history_items() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(PO items, weekly consumption) used to featurise new POs."""
    def build():
        t = tables()
        return po_items(t), weekly_consumption(t["mseg"])
    return _cached("items", _db_stamp(), build)


def anomaly_model():
    """(model, metadata) of the Production anomaly detector."""
    return _cached("anomaly_model", _registry_stamp(), lambda: Registry().load(ANOMALY_MODEL))


def anchor_date() -> pd.Timestamp:
    """Latest PO date in the data (the synthetic 'today')."""
    d = read_sql("SELECT MAX(bedat) AS d FROM ekko").d.iloc[0]
    return pd.Timestamp(d)


# ------------------------------------------------------------------ anomaly
def score_po(payload: dict) -> dict:
    """Score a single (new) purchase-order item.

    payload keys: lifnr, matnr, werks, menge, netpr, created_by; optional created_at (ISO),
    eindt (date) or lead_time_days, pr_menge, ebeln.
    """
    model, meta = anomaly_model()
    t = tables()
    hist, weekly = history_items()
    mara = t["mara"].set_index("matnr")
    if str(payload["matnr"]) not in mara.index:
        raise NotFound(f"material {payload['matnr']} not found")
    lfa1 = t["lfa1"].set_index("lifnr")
    lifnr = str(payload["lifnr"])
    created = pd.Timestamp(payload.get("created_at") or (anchor_date() + pd.Timedelta(days=1, hours=11)))
    bedat = created.normalize()
    eindt = pd.Timestamp(payload["eindt"]) if payload.get("eindt") else bedat + pd.Timedelta(
        days=int(payload.get("lead_time_days", 14)))
    info = current_info_records(t)
    rec = info[(info.matnr == str(payload["matnr"])) & (info.lifnr == lifnr)]
    prim = info[(info.matnr == str(payload["matnr"])) & (info.is_primary.astype(int) == 1)]
    menge, netpr = float(payload["menge"]), float(payload["netpr"])
    row = {
        "ebeln": str(payload.get("ebeln") or "NEW"), "ebelp": "00010", "matnr": str(payload["matnr"]),
        "werks": str(payload.get("werks", "1000")), "menge": menge, "meins": mara.loc[str(payload["matnr"]), "meins"],
        "netpr": netpr, "peinh": 1, "netwr": menge * netpr, "eindt": eindt, "banfn": None,
        "matkl": mara.loc[str(payload["matnr"]), "matkl"], "lifnr": lifnr, "bedat": bedat, "created_at": created,
        "created_by": payload.get("created_by", "API_USER"), "ekgrp": "P01",
        "name1": lfa1.name1.get(lifnr, "UNKNOWN VENDOR"),
        "vendor_created_on": pd.Timestamp(lfa1.created_on.get(lifnr, created)),
        "maktx": t["makt"].set_index("matnr").maktx.get(str(payload["matnr"]), ""),
        "pr_menge": float(payload.get("pr_menge") or menge),
        "info_netpr": float(rec.netpr.iloc[0]) if len(rec) else np.nan,
        "aplfz": float(rec.aplfz.iloc[0]) if len(rec) else np.nan,
        "prim_aplfz": float(prim.aplfz.iloc[0]) if len(prim) else np.nan,
        "prim_netpr": float(prim.netpr.iloc[0]) if len(prim) else np.nan,
    }
    recent = hist[hist.created_at >= created - pd.Timedelta(days=400)]
    items = pd.concat([recent, pd.DataFrame([row])], ignore_index=True)
    f = build_po_features(items, weekly, getattr(model, "dup_scores", {}), get_settings().approval_threshold_inr)
    s = model.score(f.iloc[[-1]])
    r = s.iloc[0]
    return {"score": round(float(r.score), 4), "risk_level": r.risk_level, "reasons": list(r.reasons),
            "components": {"rules": round(float(r.rule_score), 4), "isolation_forest": round(float(r.iforest_score), 4),
                           "autoencoder": round(float(r.ae_score), 4)},
            "model_version": str(meta["version"])}


def list_flags(risk: str | None = None, date_from: str | None = None, date_to: str | None = None,
               status: str | None = None, limit: int = 200, offset: int = 0) -> list[dict]:
    """Flagged POs (Medium/High by default) with vendor/material names and review status."""
    where = ["1=1"]
    params: list = []
    if risk:
        levels = [x.strip().capitalize() for x in risk.split(",")]
        where.append(f"s.risk_level IN ({','.join('?' * len(levels))})")
        params += levels
    else:
        where.append("s.risk_level IN ('High', 'Medium')")
    if date_from:
        where.append("s.bedat >= ?")
        params.append(date_from)
    if date_to:
        where.append("s.bedat <= ?")
        params.append(date_to)
    fb = ("LEFT JOIN (SELECT ebeln, ebelp, verdict, comment, reviewer, reviewed_at FROM review_feedback) f "
          "ON f.ebeln = s.ebeln AND f.ebelp = s.ebelp") if table_exists("review_feedback") else \
        "LEFT JOIN (SELECT NULL AS ebeln, NULL AS ebelp, NULL AS verdict, NULL AS comment, NULL AS reviewer, NULL AS reviewed_at) f ON 1=0"
    if status == "open":
        where.append("f.verdict IS NULL")
    elif status == "reviewed":
        where.append("f.verdict IS NOT NULL")
    sql = (f"SELECT s.ebeln || '-' || s.ebelp AS id, s.ebeln, s.ebelp, s.bedat, s.lifnr, v.name1 AS vendor, s.matnr, "
           f"t.maktx, s.werks, s.netwr, s.score, s.risk_level, s.reasons, s.rule_score, s.iforest_score, s.ae_score, "
           f"s.model_version, f.verdict, f.comment, f.reviewer, f.reviewed_at "
           f"FROM po_scores s JOIN lfa1 v ON v.lifnr = s.lifnr JOIN makt t ON t.matnr = s.matnr {fb} "
           f"WHERE {' AND '.join(where)} ORDER BY s.score DESC LIMIT {int(limit)} OFFSET {int(offset)}")
    df = read_sql(sql, params=tuple(params))
    df["reasons"] = df.reasons.map(lambda x: json.loads(x) if isinstance(x, str) else [])
    return df.replace({np.nan: None}).to_dict("records")


def review_flag(flag_id: str, verdict: str, comment: str = "", reviewer: str = "") -> dict:
    """Store an analyst verdict ('true_issue' | 'false_positive') for a flag id 'EBELN-EBELP'."""
    if verdict not in ("true_issue", "false_positive"):
        raise ValueError("verdict must be 'true_issue' or 'false_positive'")
    try:
        ebeln, ebelp = flag_id.split("-", 1)
    except ValueError as exc:
        raise NotFound(f"invalid flag id {flag_id}") from exc
    hit = read_sql("SELECT 1 FROM po_scores WHERE ebeln = ? AND ebelp = ?", params=(ebeln, ebelp))
    if hit.empty:
        raise NotFound(f"flag {flag_id} not found")
    rec = {"id": f"{flag_id}", "ebeln": ebeln, "ebelp": ebelp, "verdict": verdict, "comment": comment[:500],
           "reviewer": reviewer, "reviewed_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")}
    if not table_exists("review_feedback"):
        write_table("review_feedback", pd.DataFrame([rec]))
    else:
        execute("DELETE FROM review_feedback WHERE id = ?", (rec["id"],))
        execute("INSERT INTO review_feedback (id, ebeln, ebelp, verdict, comment, reviewer, reviewed_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)", tuple(rec.values()))
    return rec


def duplicate_vendors() -> list[dict]:
    """Suspected duplicate vendor pairs with both master records side by side."""
    sql = ("SELECT d.*, a.ort01 AS city_a, b.ort01 AS city_b, a.stras AS street_a, b.stras AS street_b, "
           "a.stcd3 AS gstin_a, b.stcd3 AS gstin_b, a.bank_ref AS bank_a, b.bank_ref AS bank_b, "
           "a.created_on AS created_a, b.created_on AS created_b, "
           "(SELECT COUNT(*) FROM ekko k WHERE k.lifnr = d.lifnr_b) AS pos_b, "
           "(SELECT ROUND(SUM(p.netwr), 0) FROM ekko k JOIN ekpo p ON p.ebeln = k.ebeln WHERE k.lifnr = d.lifnr_b) AS spend_b "
           "FROM vendor_duplicates d JOIN lfa1 a ON a.lifnr = d.lifnr_a JOIN lfa1 b ON b.lifnr = d.lifnr_b "
           "ORDER BY d.similarity DESC")
    return read_sql(sql).replace({np.nan: None}).to_dict("records")


# ------------------------------------------------------------------ forecast
def get_forecast(matnr: str, plant: str | None = None, horizon: int = 8, history_weeks: int = 26) -> dict:
    """History + P10/P50/P90 forecast + reorder recommendation for one material."""
    fc = read_sql("SELECT * FROM demand_forecast WHERE matnr = ? ORDER BY werks, week", params=(str(matnr),))
    if fc.empty:
        raise NotFound(f"no forecast for material {matnr}")
    plant = plant or fc.werks.iloc[0]
    fc = fc[fc.werks == plant].head(max(1, min(horizon, 12)))
    if fc.empty:
        raise NotFound(f"no forecast for material {matnr} at plant {plant}")
    hist = read_sql("SELECT budat, menge FROM mseg WHERE bwart = '261' AND matnr = ? AND werks = ?",
                    params=(str(matnr), plant))
    h = pd.DataFrame(columns=["week", "qty"])
    if not hist.empty:
        d = pd.to_datetime(hist.budat)
        hist["week"] = (d - pd.to_timedelta(d.dt.weekday, unit="D")).dt.date.astype(str)
        h = hist.groupby("week", as_index=False).menge.sum().rename(columns={"menge": "qty"}).tail(history_weeks)
    reo = read_sql("SELECT * FROM reorder_recommendations WHERE matnr = ? AND werks = ?", params=(str(matnr), plant))
    desc = read_sql("SELECT maktx FROM makt WHERE matnr = ?", params=(str(matnr),))
    return {"matnr": str(matnr), "maktx": desc.maktx.iloc[0] if len(desc) else "", "plant": plant,
            "unit": read_sql("SELECT meins FROM mara WHERE matnr = ?", params=(str(matnr),)).meins.iloc[0],
            "history": h.to_dict("records"),
            "forecast": fc[["week", "yhat", "p10", "p90"]].round(3).to_dict("records"),
            "model_version": fc.model_version.iloc[0],
            "reorder": reo.replace({np.nan: None}).to_dict("records")[0] if len(reo) else None}


def reorder_risk(risk: str | None = None, limit: int = 500) -> list[dict]:
    """Reorder recommendations (optionally filtered by stock-out risk)."""
    if risk:
        levels = [x.strip().capitalize() for x in risk.split(",")]
        df = read_sql(f"SELECT * FROM reorder_recommendations WHERE stockout_risk IN ({','.join('?' * len(levels))}) "
                      f"ORDER BY weeks_of_cover LIMIT {int(limit)}", params=tuple(levels))
    else:
        df = read_sql("SELECT * FROM reorder_recommendations ORDER BY CASE stockout_risk WHEN 'High' THEN 0 "
                      f"WHEN 'Medium' THEN 1 ELSE 2 END, weeks_of_cover LIMIT {int(limit)}")
    return df.replace({np.nan: None}).to_dict("records")


# ------------------------------------------------------------------ overview
def overview() -> dict:
    """KPIs for the dashboard landing page."""
    a = anchor_date()
    d90, d180 = (a - pd.Timedelta(days=90)).date().isoformat(), (a - pd.Timedelta(days=180)).date().isoformat()
    sp = read_sql("SELECT SUM(CASE WHEN k.bedat > ? THEN p.netwr ELSE 0 END) AS cur, "
                  "SUM(CASE WHEN k.bedat <= ? AND k.bedat > ? THEN p.netwr ELSE 0 END) AS prev "
                  "FROM ekpo p JOIN ekko k ON k.ebeln = p.ebeln", params=(d90, d90, d180)).iloc[0]
    op = read_sql("SELECT COUNT(DISTINCT k.ebeln) AS n, SUM(p.netwr) AS v FROM ekko k JOIN ekpo p ON p.ebeln = k.ebeln "
                  "WHERE k.status = 'OPEN'").iloc[0]
    fl = read_sql("SELECT risk_level, COUNT(*) AS n FROM po_scores WHERE bedat > ? GROUP BY risk_level", params=(d90,))
    fl = dict(zip(fl.risk_level, fl.n, strict=True))
    so = read_sql("SELECT stockout_risk, COUNT(*) AS n FROM reorder_recommendations GROUP BY stockout_risk")
    so = dict(zip(so.stockout_risk, so.n, strict=True))
    monthly = read_sql("SELECT SUBSTR(k.bedat, 1, 7) AS month, ROUND(SUM(p.netwr), 0) AS spend, "
                       "COUNT(DISTINCT k.ebeln) AS pos FROM ekpo p JOIN ekko k ON k.ebeln = p.ebeln "
                       "GROUP BY month ORDER BY month")
    groups = read_sql("SELECT p.matkl, ROUND(SUM(p.netwr), 0) AS spend FROM ekpo p JOIN ekko k ON k.ebeln = p.ebeln "
                      "WHERE k.bedat > ? GROUP BY p.matkl ORDER BY spend DESC", params=(d90,))
    flags_m = read_sql("SELECT SUBSTR(bedat, 1, 7) AS month, SUM(CASE WHEN risk_level = 'High' THEN 1 ELSE 0 END) AS high, "
                       "SUM(CASE WHEN risk_level = 'Medium' THEN 1 ELSE 0 END) AS medium FROM po_scores "
                       "GROUP BY month ORDER BY month")
    dups = read_sql("SELECT COUNT(*) AS n FROM vendor_duplicates").n.iloc[0]
    return {"as_of": a.date().isoformat(), "spend_90d": float(sp.cur or 0), "spend_prev_90d": float(sp.prev or 0),
            "open_pos": int(op.n or 0), "open_po_value": float(op.v or 0),
            "high_risk_90d": int(fl.get("High", 0)), "medium_risk_90d": int(fl.get("Medium", 0)),
            "stockout_high": int(so.get("High", 0)), "stockout_medium": int(so.get("Medium", 0)),
            "duplicate_vendor_pairs": int(dups), "monthly": monthly.to_dict("records"),
            "spend_by_group_90d": groups.to_dict("records"), "flags_by_month": flags_m.to_dict("records")}


# ------------------------------------------------------------------ model health
def model_health() -> dict:
    """Registry versions, latest training metrics and the drift summary."""
    rep = get_settings().reports_dir
    drift = json.loads((rep / "drift_summary.json").read_text()) if (rep / "drift_summary.json").exists() else None
    decision = json.loads((rep / "retrain_decision.json").read_text()) if (rep / "retrain_decision.json").exists() else None
    reg = Registry()
    prod = {n: reg.get_version(n, "Production") for n in (ANOMALY_MODEL, FORECAST_MODEL)}
    return {"registry": reg.list(), "production": prod, "drift": drift, "retrain_decision": decision,
            "runs": read_runs()[-30:],
            "drift_report_path": str(rep / "drift_report.html") if (rep / "drift_report.html").exists() else None}


# ------------------------------------------------------------------ request metrics
class RequestMetrics:
    """In-process request counters and latency percentiles for /metrics."""

    def __init__(self) -> None:
        self.started = time.time()
        self.counts: dict[str, int] = defaultdict(int)
        self.errors: dict[str, int] = defaultdict(int)
        self.latency: dict[str, list[float]] = defaultdict(list)

    def record(self, route: str, status: int, ms: float) -> None:
        """Record one request."""
        self.counts[route] += 1
        if status >= 400:
            self.errors[route] += 1
        lat = self.latency[route]
        lat.append(ms)
        if len(lat) > 2000:
            del lat[:1000]

    def snapshot(self) -> dict:
        """Counts, error counts and p50/p95 latency per route."""
        routes = {}
        for r, n in self.counts.items():
            lat = np.array(self.latency[r]) if self.latency[r] else np.array([0.0])
            routes[r] = {"count": n, "errors": self.errors.get(r, 0), "p50_ms": round(float(np.percentile(lat, 50)), 2),
                         "p95_ms": round(float(np.percentile(lat, 95)), 2)}
        return {"uptime_s": round(time.time() - self.started, 1), "routes": routes}


METRICS = RequestMetrics()


def new_request_id() -> str:
    """Short request id for logs."""
    return uuid.uuid4().hex[:10]
