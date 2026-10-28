# SOP-PUR-007 Prohibition of Split Purchase Orders

*Fictional SOP for the SupplySight demo; synthetic organisation.*

## Policy
Splitting a single requirement into several purchase orders or requisitions in order to stay below an approval limit (for example several POs of ₹4.8 lakh instead of one PO of ₹14 lakh) is prohibited and is treated as a control violation.

## How splits are detected
- Several POs for the same material, vendor and plant created by the same buyer on the same or consecutive days, each just below ₹5,00,000.
- SupplySight's rules engine flags such clusters with the reason "Possible split PO" and assigns them a High risk level.

## Legitimate exceptions
Separate deliveries of one requirement are allowed only when created as delivery schedule lines on a single PO or scheduling agreement, not as separate POs.

## Consequences
Confirmed split POs are reported to internal audit. Repeated violations lead to removal of PO creation authorisation.
