# SOP-VEN-001 New Vendor Onboarding

*Fictional SOP for the SupplySight demo; synthetic organisation.*

## Purpose
Ensure that only genuine, qualified and non-duplicate suppliers are created in the vendor master (LFA1).

## Documents required
- GST registration certificate (GSTIN, 15 characters, validated for format and checksum).
- PAN card of the business entity.
- Cancelled cheque or bank letter confirming the account number and IFSC.
- For cosmetic raw materials: Certificate of Analysis (COA) for a sample lot, MSDS, and BIS/Drugs & Cosmetics Act compliance declarations.

## Duplicate vendor check
Before creation the vendor master team must search for:
1. The same GSTIN or the same PAN (characters 3–12 of the GSTIN). The same PAN in another state is a legitimate branch and must be created only with a documented branch approval.
2. The same bank account number.
3. A similar name or address (e.g. "Shree" vs "Shri", "Pvt Ltd" vs "Private Limited").
Any hit must be resolved before creation. SupplySight runs a fuzzy duplicate-vendor check nightly and lists suspected pairs on the *Vendor Duplicates* page.

## Segregation of duties
The person who requests a vendor cannot create it, and buyers cannot change vendor bank details. Bank detail changes require a call-back verification to the vendor's registered phone number.

## Trial period
New vendors start with a trial order. Purchases from a vendor in its first 90 days are monitored, and the first three goods receipts undergo full QC testing (SOP-QC-001).
