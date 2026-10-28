"""Insight tool: answers from the anomaly and forecast services' outputs (serving tables)."""

from __future__ import annotations

import json
import re

import pandas as pd

from supplysight.copilot.sql_tool import catalog, resolve_materials, resolve_period, run_sql


def answer_insight(question: str) -> dict:
    """Return {'kind', 'title', 'rows' (DataFrame), 'sql'} for an insight question."""
    low = question.lower()
    if re.search(r"duplicate", low):
        sql = ("SELECT lifnr_a AS original_vendor, name_a, lifnr_b AS suspect_vendor, name_b, similarity, match_on\n"
               "FROM vendor_duplicates ORDER BY similarity DESC")
        rows, sql = run_sql(sql)
        return {"kind": "duplicate_vendors", "title": "Suspected duplicate vendors", "rows": rows, "sql": sql}

    if re.search(r"stock ?-?out|run(ning)? (out|short)|shortage|reorder|at risk|forecast|demand", low) and not re.search(
            r"\bpo\b|purchase order|anomal|flag|fraud", low):
        mats = resolve_materials(question, purchased_only=True)
        if len(mats) and re.search(r"forecast|demand", low):
            ids = ", ".join(repr(m) for m in mats.matnr)
            sql = (f"SELECT f.matnr, t.maktx, f.werks, f.week, ROUND(f.yhat, 2) AS forecast, ROUND(f.p10, 2) AS p10,\n"
                   f"       ROUND(f.p90, 2) AS p90\nFROM demand_forecast f JOIN makt t ON t.matnr = f.matnr\n"
                   f"WHERE f.matnr IN ({ids})\nORDER BY f.matnr, f.werks, f.week")
            rows, sql = run_sql(sql)
            return {"kind": "forecast", "title": "Weekly demand forecast (P10–P90)", "rows": rows, "sql": sql}
        risk = "('High')" if "high" in low else "('High', 'Medium')"
        sql = (f"SELECT matnr, maktx, werks, stockout_risk, ROUND(current_stock, 1) AS stock, ROUND(open_po_qty, 1) AS open_po,\n"
               f"       ROUND(avg_weekly_forecast, 1) AS weekly_forecast, lead_time_days, ROUND(weeks_of_cover, 1) AS weeks_cover,\n"
               f"       ROUND(recommended_rop, 1) AS recommended_rop, ROUND(current_minbe, 1) AS current_minbe\n"
               f"FROM reorder_recommendations WHERE stockout_risk IN {risk}\n"
               f"ORDER BY CASE stockout_risk WHEN 'High' THEN 0 ELSE 1 END, weeks_of_cover")
        rows, sql = run_sql(sql)
        return {"kind": "stockout", "title": "Materials at stock-out risk", "rows": rows, "sql": sql}

    # default: risky purchase orders
    period = resolve_period(question)
    if period is None and re.search(r"recent|latest|current", low):
        a = catalog()["anchor"]
        period = ((a - pd.Timedelta(days=30)).date().isoformat(), a.date().isoformat(), "last 30 days")
    levels = "('High')" if "high" in low and "medium" not in low else "('High', 'Medium')"
    pf = f" AND s.bedat BETWEEN '{period[0]}' AND '{period[1]}'" if period else ""
    sql = (f"SELECT s.ebeln, s.bedat, s.lifnr, v.name1 AS vendor, s.matnr, ROUND(s.netwr, 0) AS value_inr,\n"
           f"       s.risk_level, ROUND(s.score, 3) AS score, s.reasons\n"
           f"FROM po_scores s JOIN lfa1 v ON v.lifnr = s.lifnr\nWHERE s.risk_level IN {levels}{pf}\n"
           f"ORDER BY s.score DESC LIMIT 25")
    rows, sql = run_sql(sql)
    if "reasons" in rows:
        rows["reasons"] = rows.reasons.map(lambda r: "; ".join(json.loads(r)) if isinstance(r, str) and r.startswith("[") else r)
    title = "High-risk purchase orders" if levels == "('High')" else "Flagged purchase orders"
    return {"kind": "risky_pos", "title": title + (f" ({period[2]})" if period else ""), "rows": rows, "sql": sql}
