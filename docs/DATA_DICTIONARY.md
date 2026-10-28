# Data dictionary

**All data is synthetic**, produced by `src/supplysight/datagen` (seeded). Table and column names follow SAP MM/PP (lower-case so the same SQL runs on SQLite and PostgreSQL). Primary keys are in bold.

## `lfa1` – Vendor master (general data)

SAP table: **LFA1**

| Column | Type | Description |
|---|---|---|
| **lifnr** | TEXT | Vendor number |
| name1 | TEXT | Vendor name |
| stras | TEXT | Street / plot |
| ort01 | TEXT | City |
| regio | TEXT | State code |
| pstlz | TEXT | PIN code |
| stcd3 | TEXT | GSTIN (format-valid, fake; masked for non-admin users) |
| bank_ref | TEXT | Bank IFSC/account (fake; masked for non-admin users) |
| created_on | TEXT | Creation date |
| land1 | TEXT | Country |

## `mara` – Material master (general data)

SAP table: **MARA**

| Column | Type | Description |
|---|---|---|
| **matnr** | TEXT | Material number |
| mtart | TEXT | Material type: ROH raw, VERP packaging, FERT finished |
| matkl | TEXT | Material group (e.g. RM-FRAG, PK-BOTTLE, FG-CREAM) |
| meins | TEXT | Base unit of measure |
| ersda | TEXT | Material created on |

## `makt` – Material descriptions

SAP table: **MAKT**

| Column | Type | Description |
|---|---|---|
| **matnr** | TEXT | Material number |
| maktx | TEXT | Material description |
| **spras** | TEXT | Language |

## `marc` – Plant data for material (MRP view)

SAP table: **MARC**

| Column | Type | Description |
|---|---|---|
| **matnr** | TEXT | Material number |
| **werks** | TEXT | Plant (1000 Baddi, 2000 Vapi) |
| plifz | INTEGER | Planned delivery time (days) |
| eisbe | FLOAT | Safety stock |
| minbe | FLOAT | Reorder point (deliberately stale – set from the first 6 months) |
| dispo | TEXT | MRP controller |
| dismm | TEXT | MRP type (VB reorder-point, PD MRP) |

## `mard` – Storage-location stock (snapshot at data end)

SAP table: **MARD**

| Column | Type | Description |
|---|---|---|
| **matnr** | TEXT | Material number |
| **werks** | TEXT | Plant (1000 Baddi, 2000 Vapi) |
| **lgort** | TEXT | Storage location |
| labst | FLOAT | Unrestricted stock |
| open_po_qty | FLOAT | Quantity on open POs |

## `eina` – Purchasing info record – general data (vendor ↔ material)

SAP table: **EINA**

| Column | Type | Description |
|---|---|---|
| **infnr** | TEXT | Info record number |
| matnr | TEXT | Material number |
| lifnr | TEXT | Vendor number |
| created_on | TEXT | Creation date |
| is_primary | INTEGER | 1 = primary source |

## `eine` – Purchasing info record – purchasing-org data / price conditions (dated)

SAP table: **EINE**

| Column | Type | Description |
|---|---|---|
| **infnr** | TEXT | Info record number |
| **ekorg** | TEXT | Purchasing organisation |
| netpr | FLOAT | Net price per unit (INR) |
| peinh | INTEGER | Price unit |
| waers | TEXT | Currency |
| aplfz | INTEGER | Planned delivery time of this vendor (days) |
| **valid_from** | TEXT | Condition valid from (point-in-time contract price) |

## `eban` – Purchase requisitions

SAP table: **EBAN**

| Column | Type | Description |
|---|---|---|
| **banfn** | TEXT | Purchase requisition number |
| **bnfpo** | TEXT | Requisition item |
| matnr | TEXT | Material number |
| werks | TEXT | Plant (1000 Baddi, 2000 Vapi) |
| menge | FLOAT | Quantity |
| requested_by | TEXT | Requester |
| created_on | TEXT | Creation date |
| ebeln | TEXT | Purchasing document |
| ebelp | TEXT | PO item |

## `ekko` – Purchase order header

SAP table: **EKKO**

| Column | Type | Description |
|---|---|---|
| **ebeln** | TEXT | Purchasing document |
| bukrs | TEXT | Company code |
| bsart | TEXT | PO type |
| lifnr | TEXT | Vendor number |
| bedat | TEXT | PO date |
| created_at | TEXT | Creation timestamp (used for off-hours checks) |
| ekorg | TEXT | Purchasing organisation |
| ekgrp | TEXT | Purchasing group (P01 RM, P02 PK) |
| created_by | TEXT | Buyer (user id) |
| waers | TEXT | Currency |
| approval_status | TEXT | AUTO_RELEASED below ₹5 lakh, APPROVED_MGR above |
| status | TEXT | OPEN / CLOSED (goods received) |

## `ekpo` – Purchase order item

SAP table: **EKPO**

| Column | Type | Description |
|---|---|---|
| **ebeln** | TEXT | Purchasing document |
| **ebelp** | TEXT | PO item |
| matnr | TEXT | Material number |
| werks | TEXT | Plant (1000 Baddi, 2000 Vapi) |
| menge | FLOAT | Quantity |
| meins | TEXT | Base unit of measure |
| netpr | FLOAT | Net price per unit (INR) |
| peinh | INTEGER | Price unit |
| netwr | FLOAT | Net order value (INR) |
| eindt | TEXT | Delivery date |
| banfn | TEXT | Purchase requisition number |
| matkl | TEXT | Material group (e.g. RM-FRAG, PK-BOTTLE, FG-CREAM) |

## `mkpf` – Material document header

SAP table: **MKPF**

| Column | Type | Description |
|---|---|---|
| **mblnr** | TEXT | Material document |
| **mjahr** | INTEGER | Document year |
| budat | TEXT | Posting date |
| usnam | TEXT | User |

## `mseg` – Material document item (goods movements)

SAP table: **MSEG**

| Column | Type | Description |
|---|---|---|
| **mblnr** | TEXT | Material document |
| **zeile** | TEXT | Document line |
| bwart | TEXT | Movement type: 101 goods receipt, 261 issue to production order |
| matnr | TEXT | Material number |
| werks | TEXT | Plant (1000 Baddi, 2000 Vapi) |
| menge | FLOAT | Quantity |
| meins | TEXT | Base unit of measure |
| budat | TEXT | Posting date |
| ebeln | TEXT | Purchasing document |
| aufnr | TEXT | Production order |
| lifnr | TEXT | Vendor number |

## `afko` – Production order header

SAP table: **AFKO**

| Column | Type | Description |
|---|---|---|
| **aufnr** | TEXT | Production order |
| werks | TEXT | Plant (1000 Baddi, 2000 Vapi) |
| plnbez | TEXT | Finished material produced |
| gamng | FLOAT | Planned order quantity |
| gstrp | TEXT | Basic start date |
| gltrp | TEXT | Basic finish date |
| auart | TEXT | Order type |

## `afpo` – Production order item

SAP table: **AFPO**

| Column | Type | Description |
|---|---|---|
| **aufnr** | TEXT | Production order |
| **posnr** | TEXT | Item number |
| matnr | TEXT | Material number |
| psmng | FLOAT | Planned quantity |
| wemng | FLOAT | Delivered (actual) quantity |
| werks | TEXT | Plant (1000 Baddi, 2000 Vapi) |

## `stko` – Bill of materials header

SAP table: **STKO**

| Column | Type | Description |
|---|---|---|
| **stlnr** | TEXT | BOM number |
| matnr | TEXT | Material number |
| bmeng | FLOAT | BOM base quantity (1000 EA) |
| bmein | TEXT | Base unit |

## `stpo` – Bill of materials item

SAP table: **STPO**

| Column | Type | Description |
|---|---|---|
| **stlnr** | TEXT | BOM number |
| **posnr** | TEXT | Item number |
| idnrk | TEXT | Component material |
| menge | FLOAT | Quantity |
| meins | TEXT | Base unit of measure |

## `plaf` – Planned orders (frozen 4-week production plan after the data end)

SAP table: **PLAF**

| Column | Type | Description |
|---|---|---|
| **plnum** | TEXT | Planned order |
| matnr | TEXT | Material number |
| plwrk | TEXT | Planning plant |
| gsmng | FLOAT | Planned quantity |
| psttr | TEXT | Planned start |
| paart | TEXT | Planned-order type |

## `anomaly_labels` – Ground truth for injected anomalies. **Evaluation only – never a model input.**

SAP table: **—**

| Column | Type | Description |
|---|---|---|
| **ebeln** | TEXT | Purchasing document |
| **ebelp** | TEXT | PO item |
| anomaly_type | TEXT | price_spike, quantity_outlier, split_po, offhours_new_vendor, short_lead_time, duplicate_vendor |

## `po_scores` – Anomaly-engine output per PO item (serving table)

SAP table: **—**

| Column | Type | Description |
|---|---|---|
| **ebeln** | TEXT | Purchasing document |
| **ebelp** | TEXT | PO item |
| lifnr | TEXT | Vendor number |
| matnr | TEXT | Material number |
| werks | TEXT | Plant (1000 Baddi, 2000 Vapi) |
| bedat | TEXT | PO date |
| netwr | FLOAT | Net order value (INR) |
| score | FLOAT | Ensemble anomaly score 0–1 |
| risk_level | TEXT | High / Medium / Low |
| reasons | TEXT | JSON list of up to 3 human-readable reasons |
| rule_score | FLOAT | Rules layer (noisy-OR) |
| iforest_score | FLOAT | Isolation Forest ECDF percentile |
| ae_score | FLOAT | Autoencoder ECDF percentile |
| model_version | TEXT | Registry version that produced the row |

## `vendor_duplicates` – Suspected duplicate vendor pairs (serving table)

SAP table: **—**

| Column | Type | Description |
|---|---|---|
| **lifnr_a** | TEXT | Existing vendor |
| **lifnr_b** | TEXT | Suspect (newer) vendor |
| name_a | TEXT | Existing vendor name |
| name_b | TEXT | Suspect vendor name |
| similarity | FLOAT | Combined similarity 0–1 |
| match_on | TEXT | Evidence (name %, address %, GSTIN/PAN/bank) |

## `reorder_recommendations` – Forecast-based reorder points vs MARC-MINBE (serving table)

SAP table: **—**

| Column | Type | Description |
|---|---|---|
| **matnr** | TEXT | Material number |
| **werks** | TEXT | Plant (1000 Baddi, 2000 Vapi) |
| maktx | TEXT | Material description |
| matkl | TEXT | Material group (e.g. RM-FRAG, PK-BOTTLE, FG-CREAM) |
| avg_weekly_forecast | FLOAT | Mean weekly forecast over the lead time |
| sigma | FLOAT | Weekly forecast-error std (backtest) |
| lead_time_days | INTEGER | Primary vendor lead time |
| safety_stock | FLOAT | z·σ·√L (95 % service) |
| recommended_rop | FLOAT | Recommended reorder point |
| current_minbe | FLOAT | Current SAP reorder point |
| current_stock | FLOAT | Stock |
| open_po_qty | FLOAT | Quantity on open POs |
| weeks_of_cover | FLOAT | (stock + open POs) / weekly forecast |
| stockout_risk | TEXT | High: cannot cover lead-time demand; Medium: below recommended ROP |
| rop_gap_pct | FLOAT | (recommended − MINBE) / MINBE |

## `demand_forecast` – 12-week P10/P50/P90 weekly demand forecast (serving table)

SAP table: **—**

| Column | Type | Description |
|---|---|---|
| **matnr** | TEXT | Material number |
| **werks** | TEXT | Plant (1000 Baddi, 2000 Vapi) |
| **week** | TEXT | Week start (Monday) |
| yhat | FLOAT | P50 forecast |
| p10 | FLOAT | P10 (conformal-calibrated) |
| p90 | FLOAT | P90 (conformal-calibrated) |
| model_version | TEXT | Registry version that produced the row |

## `review_feedback` – Analyst verdicts from the review queue (labels for retraining)

SAP table: **—**

| Column | Type | Description |
|---|---|---|
| **id** | TEXT | Flag id EBELN-EBELP |
| ebeln | TEXT | Purchasing document |
| ebelp | TEXT | PO item |
| verdict | TEXT | true_issue / false_positive |
| comment | TEXT | Reviewer comment |
| reviewer | TEXT | Reviewer user |
| reviewed_at | TEXT | Review timestamp (UTC) |
