# SOP-PUR-001 Purchase Requisition Approval

Applies to: all purchase requisitions (SAP transaction ME51N, table EBAN) raised at plants 1000 (Baddi) and 2000 (Vapi). *Fictional SOP for the SupplySight demo; synthetic organisation.*

## Purpose
Make sure every requirement for raw materials (ROH), packaging (VERP) and services is justified, budgeted and approved before a purchase order is created.

## Who can raise a requisition
- MRP controllers (planners) raise requisitions for production materials, normally generated automatically by the MRP run when stock falls below the reorder point (MARC-MINBE).
- Stores and QC users may raise requisitions for consumables and laboratory items.
- Requisitions must reference the plant, material number, quantity and the required delivery date.

## Approval matrix
| Estimated value (₹) | Approver |
|---|---|
| Up to 1,00,000 | MRP controller / section head (auto-release for MRP-generated PRs) |
| 1,00,001 – 5,00,000 | Plant planning manager |
| Above 5,00,000 | Plant head and finance controller |

Approval is performed with the SAP release strategy. A requisition must never be split to stay under an approval limit (see SOP-PUR-007).

## Conversion to purchase order
Approved requisitions are converted to purchase orders by the purchasing group (EKGRP P01 for raw materials, P02 for packaging). The buyer must use the vendor and price from the valid purchasing info record (EINA/EINE) unless a deviation is approved under SOP-PUR-006.

## Records
The requisition number (BANFN) must be referenced on the PO item. Requisitions older than 90 days without conversion are closed by the planner.
