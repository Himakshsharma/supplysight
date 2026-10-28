# SOP-PUR-006 Price Change and Price Deviation Approval

*Fictional SOP for the SupplySight demo; synthetic organisation.*

## Purpose
Control changes to purchase prices so that the plant pays contracted prices and every increase is justified.

## Contract price
The valid price for a material and vendor is the purchasing info record (EINA/EINE, field NETPR). Buyers must not type a manual price on a PO without approval.

## Approval levels for price increases
| Increase vs info record or last 90-day median | Approver |
|---|---|
| Up to 5% | Buyer (document the reason) |
| 5% – 15% | Category manager |
| Above 15% | Head of procurement (CPO) and finance controller |

Commodity-linked materials (fragrance oils, butters, vegetable oils, actives) may follow an index-based price formula agreed in the contract; the index evidence must be attached.

## Price spikes
POs with a unit price more than 1.4× the 90-day median for the same material are flagged automatically by SupplySight and must be reviewed before the PO is released.

## Updating the info record
After approval the master-data team updates the info record with a new validity date. Retroactive price changes are not allowed.
