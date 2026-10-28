"""Read-only text-to-SQL tool.

Two generators share one execution path:

* **LLM** (when ``GEMINI_API_KEY`` is set): Gemini writes SQL from the schema description.
* **Template engine** (always available, used offline and as fallback): intent patterns +
  entity resolution (material / vendor / plant / material group / period) -> parameter-free SQL.

Every query - LLM or template - goes through :func:`guardrails.validate_sql` and runs on a
read-only connection with a row limit and timeout.
"""

from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache

import pandas as pd

from supplysight.config import get_settings
from supplysight.copilot.guardrails import validate_sql
from supplysight.db import COPILOT_TABLES, connect, read_sql

SCHEMA_DOC = """Tables (SQLite/PostgreSQL, lower-case SAP names; dates are ISO text 'YYYY-MM-DD'):
lfa1(lifnr vendor id, name1 vendor name, stras street, ort01 city, regio state code, stcd3 GSTIN, bank_ref, created_on)
mara(matnr material, mtart ROH|VERP|FERT, matkl material group e.g. RM-FRAG, meins unit)
makt(matnr, maktx material description)
marc(matnr, werks plant 1000|2000, plifz planned delivery days, eisbe safety stock, minbe reorder point)
mard(matnr, werks, labst unrestricted stock, open_po_qty)
eina(infnr info record, matnr, lifnr, is_primary 1/0)
eine(infnr, netpr contract price INR, aplfz lead time days, valid_from) - several conditions per infnr; latest valid_from is current
eban(banfn requisition, matnr, werks, menge, requested_by, created_on, ebeln)
ekko(ebeln PO, lifnr, bedat PO date, created_at, ekgrp, created_by buyer, approval_status, status OPEN|CLOSED)
ekpo(ebeln, ebelp, matnr, werks, menge qty, meins, netpr unit price, netwr net value INR, eindt delivery date, matkl)
mseg(mblnr, bwart 101 goods receipt|261 issue to production, matnr, werks, menge, budat posting date, ebeln, aufnr, lifnr)
afko(aufnr production order, werks, plnbez finished material, gamng qty, gstrp start date)
stko(stlnr BOM, matnr finished material)  stpo(stlnr, posnr, idnrk component material, menge per 1000 EA)
po_scores(ebeln, ebelp, lifnr, matnr, bedat, netwr, score, risk_level High|Medium|Low, reasons)
reorder_recommendations(matnr, werks, maktx, avg_weekly_forecast, recommended_rop, current_minbe, current_stock, stockout_risk)
vendor_duplicates(lifnr_a, lifnr_b, name_a, name_b, similarity, match_on)"""

GENERIC = set("""ip usp bss pure solution refined deodorised white light liquid soft triple pressed sweet virgin golden
oil oils fragrance extract acid sodium potassium iron oxide lake grade premium organic ecocert imported cosmetic coated nano
pet hdpe ldpe pp acrylic glass bottle jar tube cap case carton label mono front back generic shipper ply clear amber printed
recycled pcr matte black gold flip top disc screw pump dispenser dropper assembly induction seal wad shrink sleeve pvc bopp
alcohol wax butter glycol gum ml mm kg lami compact powder vit""".split())
GROUP_WORDS = {"fragrance": "RM-FRAG", "fragrances": "RM-FRAG", "perfume": "RM-FRAG", "emollient": "RM-EMOLL",
               "emollients": "RM-EMOLL", "butter": "RM-BUTTER", "butters": "RM-BUTTER", "wax": "RM-WAX", "waxes": "RM-WAX",
               "emulsifier": "RM-EMULS", "emulsifiers": "RM-EMULS", "surfactant": "RM-SURF", "surfactants": "RM-SURF",
               "thickener": "RM-THICK", "thickeners": "RM-THICK", "preservative": "RM-PRESV", "preservatives": "RM-PRESV",
               "active": "RM-ACTIVE", "actives": "RM-ACTIVE", "oil": "RM-OIL", "oils": "RM-OIL", "uv": "RM-UVFIL",
               "sunscreen filter": "RM-UVFIL", "pigment": "RM-PIGM", "pigments": "RM-PIGM", "humectant": "RM-HUMECT",
               "humectants": "RM-HUMECT", "bottles": "PK-BOTTLE", "bottle": "PK-BOTTLE", "jars": "PK-JAR", "tubes": "PK-TUBE",
               "caps": "PK-CAP", "closures": "PK-CAP", "cartons": "PK-CARTON", "labels": "PK-LABEL",
               "packaging": "VERP", "raw material": "ROH", "raw materials": "ROH"}
STATES = {"maharashtra": "MH", "gujarat": "GJ", "delhi": "DL", "haryana": "HR", "uttar pradesh": "UP",
          "himachal": "HP", "punjab": "PB", "tamil nadu": "TN", "karnataka": "KA", "telangana": "TS",
          "west bengal": "WB", "rajasthan": "RJ", "madhya pradesh": "MP", "uttarakhand": "UK", "daman": "DD"}


@dataclass
class SQLPlan:
    """A generated query plus a human description of what it answers."""

    sql: str
    intent: str
    description: str
    entities: dict


# ---------------------------------------------------------------- entity resolution
@lru_cache(maxsize=4)
def _catalog(db_url: str) -> dict:
    mats = read_sql("SELECT m.matnr, m.mtart, m.matkl, t.maktx FROM mara m JOIN makt t ON t.matnr = m.matnr")
    vend = read_sql("SELECT lifnr, name1, ort01, regio FROM lfa1")
    anchor = read_sql("SELECT MAX(bedat) AS d FROM ekko").d.iloc[0]
    return {"mats": mats, "vend": vend, "anchor": pd.Timestamp(anchor) if anchor else pd.Timestamp.today()}


def catalog() -> dict:
    """Cached master-data catalogue for entity resolution."""
    return _catalog(get_settings().database_url)


def _words(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


def _near(a: str, b: str) -> bool:
    return a == b or (len(a) >= 5 and len(b) >= 5 and SequenceMatcher(None, a, b).ratio() >= 0.86)


def resolve_materials(q: str, purchased_only: bool = False) -> pd.DataFrame:
    """Materials whose distinctive description words appear in the question (fuzzy)."""
    mats = catalog()["mats"]
    if purchased_only:
        mats = mats[mats.mtart != "FERT"]
    num = re.findall(r"\b([123]\d{5})\b", q)
    if num:
        return mats[mats.matnr.isin(num)]
    qw = [w for w in _words(q) if len(w) >= 3]
    best, hits = 0.0, []
    for r in mats.itertuples():
        dw = [w for w in _words(r.maktx) if w not in GENERIC and not w.isdigit() and len(w) >= 3]
        if not dw:
            continue
        matched = sum(1 for w in dw if any(_near(w, x) for x in qw))
        score = matched / len(dw) + 0.01 * matched
        if matched and score >= best - 1e-9:
            if score > best + 1e-9:
                hits = []
            best = score
            hits.append(r.matnr)
    if best < 0.5:
        return mats.iloc[0:0]
    return mats[mats.matnr.isin(hits)]


def resolve_vendor(q: str) -> pd.DataFrame:
    """Vendors referenced by id or by a distinctive name word."""
    vend = catalog()["vend"]
    ids = re.findall(r"\b(5\d{5})\b", q)
    if ids:
        return vend[vend.lifnr.isin(ids)]
    qw = [w for w in _words(q) if len(w) >= 4]
    generic = {"chemicals", "chemical", "specialty", "industries", "packaging", "polymers", "plastics", "print", "pack",
               "containers", "labels", "corrugators", "closures", "aromatics", "organics", "oleochem", "fine", "chem",
               "extracts", "naturals", "colours", "private", "limited", "enterprises", "trading", "traders", "supplies",
               "vendor", "vendors", "supplier", "branch", "company"}
    scores = []
    for r in vend.itertuples():
        nw = [w for w in _words(r.name1) if w not in generic and len(w) >= 4]
        m = sum(1 for w in nw if any(_near(w, x) for x in qw))
        if m:
            scores.append((m / max(1, len(nw)), r.lifnr))
    if not scores:
        return vend.iloc[0:0]
    top = max(s for s, _ in scores)
    if top < 0.5:
        return vend.iloc[0:0]
    return vend[vend.lifnr.isin([lf for s, lf in scores if s == top])]


def resolve_period(q: str) -> tuple[str, str, str] | None:
    """(from, to, label) relative to the latest PO date in the data (the synthetic 'today')."""
    a = catalog()["anchor"]
    low = q.lower()
    qstart = pd.Timestamp(year=a.year, month=3 * ((a.month - 1) // 3) + 1, day=1)
    if "last quarter" in low or "previous quarter" in low:
        s = qstart - pd.DateOffset(months=3)
        return s.date().isoformat(), (qstart - pd.Timedelta(days=1)).date().isoformat(), "last quarter"
    if "this quarter" in low:
        return qstart.date().isoformat(), a.date().isoformat(), "this quarter"
    if "last month" in low or "previous month" in low:
        s = (a.replace(day=1) - pd.DateOffset(months=1))
        return s.date().isoformat(), (a.replace(day=1) - pd.Timedelta(days=1)).date().isoformat(), "last month"
    if "this month" in low:
        return a.replace(day=1).date().isoformat(), a.date().isoformat(), "this month"
    m = re.search(r"last (\d+) (day|week|month)s?", low)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        delta = pd.Timedelta(days=n) if unit == "day" else (pd.Timedelta(weeks=n) if unit == "week" else pd.DateOffset(months=n))
        return (a - delta).date().isoformat(), a.date().isoformat(), f"last {n} {unit}s"
    if "this year" in low or "ytd" in low:
        return pd.Timestamp(year=a.year, month=1, day=1).date().isoformat(), a.date().isoformat(), "this year"
    if "last year" in low or "past year" in low or "last 12 months" in low:
        return (a - pd.DateOffset(years=1)).date().isoformat(), a.date().isoformat(), "last 12 months"
    y = re.search(r"\b(20\d\d)\b", low)
    if y:
        return f"{y.group(1)}-01-01", f"{y.group(1)}-12-31", y.group(1)
    return None


def _plant(q: str) -> str | None:
    low = q.lower()
    if "baddi" in low or re.search(r"\bplant\s*1000\b|\b1000\b", low):
        return "1000"
    if "vapi" in low or re.search(r"\bplant\s*2000\b|\b2000\b", low):
        return "2000"
    return None


def _in(col: str, vals) -> str:
    return f"{col} IN ({', '.join(repr(str(v)) for v in vals)})"


def _top_n(q: str, default: int = 5) -> int:
    m = re.search(r"\btop\s+(\d+)", q.lower())
    return min(int(m.group(1)), 50) if m else default


# ---------------------------------------------------------------- template engine
def plan_query(q: str) -> SQLPlan | None:
    """Map a data question to SQL with the template engine (None if no intent matches)."""
    low = q.lower()
    period = resolve_period(q)
    plant = _plant(q)
    pf = f" AND k.bedat BETWEEN '{period[0]}' AND '{period[1]}'" if period else ""
    plf = f" AND p.werks = '{plant}'" if plant else ""
    plabel = f" ({period[2]})" if period else ""

    # vendors by spend
    if re.search(r"(top|biggest|largest|highest).*(vendor|supplier)|(vendor|supplier)s? by spend|spend by vendor", low):
        n = _top_n(q)
        sql = (f"SELECT v.lifnr, v.name1, v.ort01, ROUND(SUM(p.netwr), 0) AS spend_inr, COUNT(DISTINCT p.ebeln) AS pos\n"
               f"FROM ekpo p JOIN ekko k ON k.ebeln = p.ebeln JOIN lfa1 v ON v.lifnr = k.lifnr\n"
               f"WHERE 1=1{pf}{plf}\nGROUP BY v.lifnr, v.name1, v.ort01\nORDER BY spend_inr DESC\nLIMIT {n}")
        return SQLPlan(sql, "top_vendors_spend", f"Top {n} vendors by PO value{plabel}", {"period": period})

    # top materials by spend
    if re.search(r"(top|biggest|largest|highest).*(material|item)s?.*(spend|value|cost)|spend by material", low):
        n = _top_n(q)
        sql = (f"SELECT p.matnr, t.maktx, ROUND(SUM(p.netwr), 0) AS spend_inr, ROUND(SUM(p.menge), 1) AS qty\n"
               f"FROM ekpo p JOIN ekko k ON k.ebeln = p.ebeln JOIN makt t ON t.matnr = p.matnr\n"
               f"WHERE 1=1{pf}{plf}\nGROUP BY p.matnr, t.maktx\nORDER BY spend_inr DESC\nLIMIT {n}")
        return SQLPlan(sql, "top_materials_spend", f"Top {n} materials by PO value{plabel}", {"period": period})

    # spend by plant / month / group
    if "spend" in low or "purchase value" in low or "total value" in low:
        if "plant" in low:
            sql = (f"SELECT p.werks AS plant, ROUND(SUM(p.netwr), 0) AS spend_inr, COUNT(DISTINCT p.ebeln) AS pos\n"
                   f"FROM ekpo p JOIN ekko k ON k.ebeln = p.ebeln WHERE 1=1{pf}\nGROUP BY p.werks ORDER BY p.werks")
            return SQLPlan(sql, "spend_by_plant", f"Spend by plant{plabel}", {"period": period})
        if "group" in low or "category" in low:
            sql = (f"SELECT p.matkl AS material_group, ROUND(SUM(p.netwr), 0) AS spend_inr\n"
                   f"FROM ekpo p JOIN ekko k ON k.ebeln = p.ebeln WHERE 1=1{pf}{plf}\n"
                   f"GROUP BY p.matkl ORDER BY spend_inr DESC")
            return SQLPlan(sql, "spend_by_group", f"Spend by material group{plabel}", {"period": period})
        vend = resolve_vendor(q)
        vf = f" AND {_in('k.lifnr', vend.lifnr)}" if len(vend) else ""
        sql = (f"SELECT SUBSTR(k.bedat, 1, 7) AS month, ROUND(SUM(p.netwr), 0) AS spend_inr, COUNT(DISTINCT p.ebeln) AS pos\n"
               f"FROM ekpo p JOIN ekko k ON k.ebeln = p.ebeln WHERE 1=1{pf}{plf}{vf}\nGROUP BY month ORDER BY month")
        return SQLPlan(sql, "spend_by_month", f"Monthly spend{plabel}", {"period": period, "vendors": list(vend.lifnr)})

    mats = resolve_materials(q)
    vend = resolve_vendor(q)

    # BOM usage (before supplier intent: "used in" questions)
    if re.search(r"(used in|components? of|bom|bill of material|ingredients? of|recipe)", low):
        fg = mats[mats.mtart == "FERT"]
        if len(fg):
            sql = (f"SELECT f.maktx AS finished_good, s.idnrk AS component, t.maktx AS component_desc, s.menge AS qty_per_1000, s.meins\n"
                   f"FROM stko k JOIN stpo s ON s.stlnr = k.stlnr JOIN makt t ON t.matnr = s.idnrk JOIN makt f ON f.matnr = k.matnr\n"
                   f"WHERE {_in('k.matnr', fg.matnr)}\nORDER BY f.maktx, s.posnr")
            return SQLPlan(sql, "bom_components", "Bill of materials", {"materials": list(fg.matnr)})
        rm = mats[mats.mtart != "FERT"]
        if len(rm):
            sql = (f"SELECT t.maktx AS component_desc, f.matnr AS finished_good, f.maktx AS finished_good_desc, s.menge AS qty_per_1000\n"
                   f"FROM stpo s JOIN stko k ON k.stlnr = s.stlnr JOIN makt f ON f.matnr = k.matnr JOIN makt t ON t.matnr = s.idnrk\n"
                   f"WHERE {_in('s.idnrk', rm.matnr)}\nORDER BY f.maktx")
            return SQLPlan(sql, "where_used", "Where-used list", {"materials": list(rm.matnr)})

    # who supplies X / lead time of X
    if re.search(r"(who|which|what)\b.*\b(vendor|supplier)s?|suppl(y|ies|ier)|lead ?time|sourced? from|buy .* from", low) and len(mats):
        pm = mats[mats.mtart != "FERT"]
        if len(pm):
            sql = (f"SELECT t.matnr, t.maktx, v.lifnr, v.name1 AS vendor, v.ort01 AS city, e.netpr AS contract_price_inr,\n"
                   f"       e.aplfz AS lead_time_days, CASE WHEN a.is_primary = 1 THEN 'primary' ELSE 'secondary' END AS source\n"
                   f"FROM eina a JOIN eine e ON e.infnr = a.infnr\n"
               f"       AND e.valid_from = (SELECT MAX(e2.valid_from) FROM eine e2 WHERE e2.infnr = a.infnr)\n"
               f"JOIN lfa1 v ON v.lifnr = a.lifnr JOIN makt t ON t.matnr = a.matnr\n"
                   f"WHERE {_in('a.matnr', pm.matnr)}\nORDER BY t.matnr, a.is_primary DESC, e.aplfz")
            return SQLPlan(sql, "vendor_for_material", "Vendors, contract prices and lead times", {"materials": list(pm.matnr)})

    # what does vendor V supply
    if len(vend) and re.search(r"(what|which).*(material|item|suppl)|supplies|catalog", low):
        sql = (f"SELECT v.lifnr, v.name1, a.matnr, t.maktx, e.netpr AS contract_price_inr, e.aplfz AS lead_time_days\n"
               f"FROM eina a JOIN eine e ON e.infnr = a.infnr\n"
               f"       AND e.valid_from = (SELECT MAX(e2.valid_from) FROM eine e2 WHERE e2.infnr = a.infnr)\n"
               f"JOIN lfa1 v ON v.lifnr = a.lifnr JOIN makt t ON t.matnr = a.matnr\n"
               f"WHERE {_in('a.lifnr', vend.lifnr)}\nORDER BY t.maktx")
        return SQLPlan(sql, "materials_of_vendor", "Materials supplied by vendor", {"vendors": list(vend.lifnr)})

    # open POs
    if re.search(r"open (purchase orders|pos?|orders)|pending (purchase orders|pos?|deliver)", low):
        f = [f"k.status = 'OPEN'"]
        if len(vend):
            f.append(_in("k.lifnr", vend.lifnr))
        if len(mats):
            f.append(_in("p.matnr", mats.matnr))
        if plant:
            f.append(f"p.werks = '{plant}'")
        sql = (f"SELECT k.ebeln, k.bedat, v.name1 AS vendor, p.matnr, t.maktx, p.menge, p.meins, p.netwr, p.eindt AS delivery_date\n"
               f"FROM ekko k JOIN ekpo p ON p.ebeln = k.ebeln JOIN lfa1 v ON v.lifnr = k.lifnr JOIN makt t ON t.matnr = p.matnr\n"
               f"WHERE {' AND '.join(f)}\nORDER BY p.eindt")
        return SQLPlan(sql, "open_pos", "Open purchase orders", {})

    # price
    if re.search(r"\bprice|cost per|rate of|how much does", low) and len(mats):
        sql = (f"SELECT p.matnr, t.maktx, SUBSTR(k.bedat, 1, 7) AS month, ROUND(AVG(p.netpr), 2) AS avg_price_inr,\n"
               f"       ROUND(MIN(p.netpr), 2) AS min_price, ROUND(MAX(p.netpr), 2) AS max_price, COUNT(*) AS pos\n"
               f"FROM ekpo p JOIN ekko k ON k.ebeln = p.ebeln JOIN makt t ON t.matnr = p.matnr\n"
               f"WHERE {_in('p.matnr', mats.matnr)}{pf}\nGROUP BY p.matnr, t.maktx, month ORDER BY p.matnr, month DESC")
        return SQLPlan(sql, "price_history", f"Average purchase price by month{plabel}", {"materials": list(mats.matnr)})

    # stock
    if re.search(r"\bstock\b|inventory|on hand", low) and len(mats):
        sql = (f"SELECT d.matnr, t.maktx, d.werks, d.labst AS stock, d.open_po_qty, c.minbe AS reorder_point, c.eisbe AS safety_stock\n"
               f"FROM mard d JOIN makt t ON t.matnr = d.matnr LEFT JOIN marc c ON c.matnr = d.matnr AND c.werks = d.werks\n"
               f"WHERE {_in('d.matnr', mats.matnr)}" + (f" AND d.werks = '{plant}'" if plant else "") + "\nORDER BY d.matnr, d.werks")
        return SQLPlan(sql, "stock", "Current stock and reorder settings", {"materials": list(mats.matnr)})

    # consumption
    if re.search(r"consum|usage|used per|issued", low) and len(mats):
        cf = f" AND m.budat BETWEEN '{period[0]}' AND '{period[1]}'" if period else ""
        sql = (f"SELECT m.matnr, t.maktx, m.werks, SUBSTR(m.budat, 1, 7) AS month, ROUND(SUM(m.menge), 2) AS consumed, m.meins\n"
               f"FROM mseg m JOIN makt t ON t.matnr = m.matnr\nWHERE m.bwart = '261' AND {_in('m.matnr', mats.matnr)}{cf}\n"
               f"GROUP BY m.matnr, t.maktx, m.werks, month, m.meins ORDER BY month DESC, m.werks")
        return SQLPlan(sql, "consumption", f"Monthly consumption{plabel}", {"materials": list(mats.matnr)})

    # vendor master details
    if len(vend) and re.search(r"gstin|gst|bank|address|detail|located|where is|city|contact", low):
        sql = (f"SELECT lifnr, name1, stras, ort01 AS city, regio AS state, stcd3 AS gstin, bank_ref, created_on\n"
               f"FROM lfa1 WHERE {_in('lifnr', vend.lifnr)}")
        return SQLPlan(sql, "vendor_details", "Vendor master data", {"vendors": list(vend.lifnr)})

    # vendors in a state / city
    for name, code in STATES.items():
        if name in low:
            sql = f"SELECT lifnr, name1, ort01 AS city, regio AS state, created_on FROM lfa1 WHERE regio = '{code}' ORDER BY name1"
            return SQLPlan(sql, "vendors_in_state", f"Vendors registered in {name.title()}", {})
    cities = catalog()["vend"].ort01.unique()
    for c in cities:
        if re.search(rf"\b{re.escape(c.lower())}\b", low) and "vendor" in low:
            sql = f"SELECT lifnr, name1, ort01 AS city, regio AS state FROM lfa1 WHERE ort01 = '{c}' ORDER BY name1"
            return SQLPlan(sql, "vendors_in_city", f"Vendors in {c}", {})

    # count of POs
    if re.search(r"how many (purchase orders|pos?|orders)|number of (purchase orders|pos)|count of (purchase orders|pos)", low):
        by = "k.created_by" if re.search(r"buyer|per user|by user|created by", low) else None
        f = "1=1" + pf + plf + (f" AND {_in('k.lifnr', vend.lifnr)}" if len(vend) else "")
        if by:
            sql = (f"SELECT {by} AS buyer, COUNT(DISTINCT k.ebeln) AS pos, ROUND(SUM(p.netwr), 0) AS value_inr\n"
                   f"FROM ekko k JOIN ekpo p ON p.ebeln = k.ebeln WHERE {f}\nGROUP BY {by} ORDER BY pos DESC")
        else:
            sql = (f"SELECT COUNT(DISTINCT k.ebeln) AS pos, ROUND(SUM(p.netwr), 0) AS value_inr\n"
                   f"FROM ekko k JOIN ekpo p ON p.ebeln = k.ebeln WHERE {f}")
        return SQLPlan(sql, "po_count", f"Purchase-order count{plabel}", {"period": period})

    # materials in a group
    for w, g in GROUP_WORDS.items():
        if re.search(rf"\b{w}\b", low) and re.search(r"list|which|show|all|materials", low):
            col = "m.mtart" if g in ("ROH", "VERP") else "m.matkl"
            sql = (f"SELECT m.matnr, t.maktx, m.matkl, m.meins FROM mara m JOIN makt t ON t.matnr = m.matnr\n"
                   f"WHERE {col} = '{g}' ORDER BY t.maktx")
            return SQLPlan(sql, "materials_in_group", f"Materials in {g}", {})

    if len(mats):  # generic: recent POs for the material
        sql = (f"SELECT k.ebeln, k.bedat, v.name1 AS vendor, p.matnr, t.maktx, p.menge, p.netpr, p.netwr\n"
               f"FROM ekpo p JOIN ekko k ON k.ebeln = p.ebeln JOIN lfa1 v ON v.lifnr = k.lifnr JOIN makt t ON t.matnr = p.matnr\n"
               f"WHERE {_in('p.matnr', mats.matnr)}{pf}{plf}\nORDER BY k.bedat DESC LIMIT 20")
        return SQLPlan(sql, "recent_pos", "Recent purchase orders", {"materials": list(mats.matnr)})
    if len(vend):
        sql = (f"SELECT k.ebeln, k.bedat, p.matnr, t.maktx, p.menge, p.netwr, k.status\n"
               f"FROM ekko k JOIN ekpo p ON p.ebeln = k.ebeln JOIN makt t ON t.matnr = p.matnr\n"
               f"WHERE {_in('k.lifnr', vend.lifnr)}{pf}\nORDER BY k.bedat DESC LIMIT 20")
        return SQLPlan(sql, "recent_pos_vendor", "Recent purchase orders for vendor", {"vendors": list(vend.lifnr)})
    return None


# ---------------------------------------------------------------- execution
def run_sql(sql: str) -> tuple[pd.DataFrame, str]:
    """Validate and execute on a read-only connection. Returns (rows, executed_sql)."""
    s = get_settings()
    ok, reason, safe = validate_sql(sql, COPILOT_TABLES, s.sql_row_limit)
    if not ok:
        raise PermissionError(f"SQL blocked by guardrails: {reason}")
    with connect(readonly=True) as conn:
        if isinstance(conn, sqlite3.Connection):
            deadline = time.monotonic() + s.sql_timeout_s
            conn.set_progress_handler(lambda: int(time.monotonic() > deadline), 10_000)
            return pd.read_sql_query(safe, conn), safe
        return pd.read_sql_query(safe, conn), safe  # pragma: no cover
