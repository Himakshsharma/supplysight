# SOP-MRP-001 Reorder Point and Safety Stock Maintenance

*Fictional SOP for the SupplySight demo; synthetic organisation.*

## Purpose
Keep reorder points (MARC-MINBE) and safety stocks (MARC-EISBE) aligned with current demand so that materials neither stock out nor pile up.

## Formula
- Safety stock = z × σ × √L, where z is the service-level factor (1.65 for 95%), σ is the standard deviation of weekly forecast error and L is the lead time in weeks.
- Reorder point = average weekly demand during the lead time × L + safety stock.

## Review frequency
MRP controllers review reorder points monthly and before the festive season (September–November) and the summer season, when demand for sunscreens and gels rises sharply.

## Using SupplySight recommendations
The *Demand & Reorder* page lists the recommended reorder point next to the current MINBE. Changes larger than 20% must be approved by the planning manager before updating MARC.

## Stock-out risk escalation
Materials where stock plus open purchase orders cannot cover demand during the vendor lead time are flagged "High stock-out risk". The planner must raise an urgent requisition (SOP-PUR-002) within one working day.
