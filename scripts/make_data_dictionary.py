"""Generate docs/DATA_DICTIONARY.md from the database schema + descriptions below."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from supplysight.db import SCHEMA  # noqa: E402

TABLES = {
    "lfa1": ("LFA1", "Vendor master (general data)"),
    "mara": ("MARA", "Material master (general data)"),
    "makt": ("MAKT", "Material descriptions"),
    "marc": ("MARC", "Plant data for material (MRP view)"),
    "mard": ("MARD", "Storage-location stock (snapshot at data end)"),
    "eina": ("EINA", "Purchasing info record – general data (vendor ↔ material)"),
    "eine": ("EINE", "Purchasing info record – purchasing-org data / price conditions (dated)"),
    "eban": ("EBAN", "Purchase requisitions"),
    "ekko": ("EKKO", "Purchase order header"),
    "ekpo": ("EKPO", "Purchase order item"),
    "mkpf": ("MKPF", "Material document header"),
    "mseg": ("MSEG", "Material document item (goods movements)"),
    "afko": ("AFKO", "Production order header"),
    "afpo": ("AFPO", "Production order item"),
    "stko": ("STKO", "Bill of materials header"),
    "stpo": ("STPO", "Bill of materials item"),
    "plaf": ("PLAF", "Planned orders (frozen 4-week production plan after the data end)"),
    "anomaly_labels": ("—", "Ground truth for injected anomalies. **Evaluation only – never a model input.**"),
    "po_scores": ("—", "Anomaly-engine output per PO item (serving table)"),
    "vendor_duplicates": ("—", "Suspected duplicate vendor pairs (serving table)"),
    "reorder_recommendations": ("—", "Forecast-based reorder points vs MARC-MINBE (serving table)"),
    "demand_forecast": ("—", "12-week P10/P50/P90 weekly demand forecast (serving table)"),
    "review_feedback": ("—", "Analyst verdicts from the review queue (labels for retraining)"),
}
COLS = {
    "lifnr": "Vendor number", "name1": "Vendor name", "stras": "Street / plot", "ort01": "City", "regio": "State code",
    "pstlz": "PIN code", "stcd3": "GSTIN (format-valid, fake; masked for non-admin users)",
    "bank_ref": "Bank IFSC/account (fake; masked for non-admin users)", "created_on": "Creation date",
    "land1": "Country", "matnr": "Material number", "mtart": "Material type: ROH raw, VERP packaging, FERT finished",
    "matkl": "Material group (e.g. RM-FRAG, PK-BOTTLE, FG-CREAM)", "meins": "Base unit of measure",
    "ersda": "Material created on", "maktx": "Material description", "spras": "Language",
    "werks": "Plant (1000 Baddi, 2000 Vapi)", "plifz": "Planned delivery time (days)", "eisbe": "Safety stock",
    "minbe": "Reorder point (deliberately stale – set from the first 6 months)", "dispo": "MRP controller",
    "dismm": "MRP type (VB reorder-point, PD MRP)", "lgort": "Storage location", "labst": "Unrestricted stock",
    "open_po_qty": "Quantity on open POs", "infnr": "Info record number", "is_primary": "1 = primary source",
    "ekorg": "Purchasing organisation", "netpr": "Net price per unit (INR)", "peinh": "Price unit",
    "waers": "Currency", "aplfz": "Planned delivery time of this vendor (days)",
    "valid_from": "Condition valid from (point-in-time contract price)", "banfn": "Purchase requisition number",
    "bnfpo": "Requisition item", "menge": "Quantity", "requested_by": "Requester", "ebeln": "Purchasing document",
    "ebelp": "PO item", "bukrs": "Company code", "bsart": "PO type", "bedat": "PO date",
    "created_at": "Creation timestamp (used for off-hours checks)", "ekgrp": "Purchasing group (P01 RM, P02 PK)",
    "created_by": "Buyer (user id)", "approval_status": "AUTO_RELEASED below ₹5 lakh, APPROVED_MGR above",
    "status": "OPEN / CLOSED (goods received)", "netwr": "Net order value (INR)", "eindt": "Delivery date",
    "mblnr": "Material document", "mjahr": "Document year", "budat": "Posting date", "usnam": "User",
    "zeile": "Document line", "bwart": "Movement type: 101 goods receipt, 261 issue to production order",
    "aufnr": "Production order", "plnbez": "Finished material produced", "gamng": "Planned order quantity",
    "gstrp": "Basic start date", "gltrp": "Basic finish date", "auart": "Order type", "posnr": "Item number",
    "psmng": "Planned quantity", "wemng": "Delivered (actual) quantity", "stlnr": "BOM number",
    "bmeng": "BOM base quantity (1000 EA)", "bmein": "Base unit", "idnrk": "Component material",
    "plnum": "Planned order", "plwrk": "Planning plant", "gsmng": "Planned quantity", "psttr": "Planned start",
    "paart": "Planned-order type", "anomaly_type": "price_spike, quantity_outlier, split_po, offhours_new_vendor, "
    "short_lead_time, duplicate_vendor", "score": "Ensemble anomaly score 0–1", "risk_level": "High / Medium / Low",
    "reasons": "JSON list of up to 3 human-readable reasons", "rule_score": "Rules layer (noisy-OR)",
    "iforest_score": "Isolation Forest ECDF percentile", "ae_score": "Autoencoder ECDF percentile",
    "model_version": "Registry version that produced the row", "lifnr_a": "Existing vendor",
    "lifnr_b": "Suspect (newer) vendor", "name_a": "Existing vendor name", "name_b": "Suspect vendor name",
    "similarity": "Combined similarity 0–1", "match_on": "Evidence (name %, address %, GSTIN/PAN/bank)",
    "avg_weekly_forecast": "Mean weekly forecast over the lead time", "sigma": "Weekly forecast-error std (backtest)",
    "lead_time_days": "Primary vendor lead time", "safety_stock": "z·σ·√L (95 % service)",
    "recommended_rop": "Recommended reorder point", "current_minbe": "Current SAP reorder point",
    "current_stock": "Stock", "weeks_of_cover": "(stock + open POs) / weekly forecast",
    "stockout_risk": "High: cannot cover lead-time demand; Medium: below recommended ROP",
    "rop_gap_pct": "(recommended − MINBE) / MINBE", "week": "Week start (Monday)", "yhat": "P50 forecast",
    "p10": "P10 (conformal-calibrated)", "p90": "P90 (conformal-calibrated)", "id": "Flag id EBELN-EBELP",
    "verdict": "true_issue / false_positive", "comment": "Reviewer comment", "reviewer": "Reviewer user",
    "reviewed_at": "Review timestamp (UTC)",
}


def main() -> None:
    """Write the data dictionary."""
    lines = ["# Data dictionary", "",
             "**All data is synthetic**, produced by `src/supplysight/datagen` (seeded). Table and column names follow "
             "SAP MM/PP (lower-case so the same SQL runs on SQLite and PostgreSQL). Primary keys are in bold.", ""]
    for t, (cols, pk) in SCHEMA.items():
        sap, desc = TABLES.get(t, ("—", ""))
        lines += [f"## `{t}` – {desc}", "", f"SAP table: **{sap}**", "", "| Column | Type | Description |",
                  "|---|---|---|"]
        for c, typ in cols.items():
            name = f"**{c}**" if c in pk else c
            lines.append(f"| {name} | {typ.replace('DOUBLE PRECISION', 'FLOAT')} | {COLS.get(c, '')} |")
        lines.append("")
    out = Path(__file__).resolve().parents[1] / "docs" / "DATA_DICTIONARY.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
