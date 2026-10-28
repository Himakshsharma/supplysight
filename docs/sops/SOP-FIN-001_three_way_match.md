# SOP-FIN-001 Three-Way Match and Invoice Verification

*Fictional SOP for the SupplySight demo; synthetic organisation.*

## Purpose
Pay vendors only for goods that were ordered, received and invoiced at the agreed price.

## The three documents
- Purchase order (EKKO/EKPO): ordered quantity and net price.
- Goods receipt (MSEG movement type 101): quantity actually received and accepted.
- Vendor invoice (MIRO): invoiced quantity and price.

## Tolerances
| Check | Tolerance | Action when exceeded |
|---|---|---|
| Price (invoice vs PO) | ±2% | Invoice blocked for payment; buyer must confirm |
| Quantity (invoice vs GR) | +0% / −5% | Invoice blocked; stores to confirm receipt |
| Delivery date | none | informational only |

## Blocked invoices
Blocked invoices are released only by the buyer's manager after the discrepancy is resolved. Repeated price variances from the same vendor are reported to the category manager.

## Goods received but not invoiced
GR/IR balances older than 60 days are reviewed monthly by finance with purchasing.
