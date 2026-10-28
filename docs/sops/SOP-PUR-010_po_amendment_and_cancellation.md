# SOP-PUR-010 Purchase Order Amendment and Cancellation

*Fictional SOP for the SupplySight demo; synthetic organisation.*

## Purpose
Control changes to released purchase orders.

## Allowed amendments
- Delivery date changes requested by planning: buyer may change after confirming with the vendor.
- Quantity increases above 10% or any price change: the PO is reset for release and approval follows SOP-PUR-001 and SOP-PUR-006.
- Vendor change on an existing PO is not allowed; cancel and create a new PO.

## Cancellation
POs can be cancelled only if no goods receipt has been posted. Partially received POs are closed with the "delivery completed" indicator.

## Audit trail
All changes are logged in the SAP change documents. SupplySight reviews unusual quantity increases relative to the purchase requisition as part of anomaly scoring.
