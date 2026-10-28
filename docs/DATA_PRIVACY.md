# Data privacy

## All data is synthetic

Every table in SupplySight is produced by `src/supplysight/datagen/generator.py` from a random seed
(`--seed 42` by default). The project **never uses, requests or simulates access to real company data**. This
includes data from any employer, internship host or SAP system.

| Item | How it is produced |
|---|---|
| Vendor names | Combinations of common Indian business-name words + industry suffix + legal form, e.g. “Sahyadri Chemicals Pvt Ltd”. Any resemblance to a real business is coincidental. |
| GSTINs | Random PAN-like strings with a valid **format and checksum** so validation logic can be demonstrated. They do not belong to any registered taxpayer. |
| Bank references | `IFSC-like/random-digits` strings. Not real accounts. |
| Materials | Generic cosmetic ingredient and packaging names; prices are drawn from group-level ranges. |
| People | Buyer and requester ids are role codes (`RSHARMA`, `PLN_A01`); demo users are role names. |
| Transactions | Simulated from seasonality curves, bills of materials and an inventory model. |

## Controls in the application

- **Masking:** GSTIN and bank fields are masked in copilot answers, API responses and the dashboard for every role
  except `admin` (`copilot/guardrails.py: mask_frame / mask_text`).
- **Least privilege:** the copilot's SQL tool reads through a read-only connection and can only query whitelisted
  tables. `anomaly_labels` (ground truth) and `review_feedback` are not on the list.
- **No data leaves the machine by default:** the copilot runs offline. If you set `GEMINI_API_KEY`, the question,
  retrieved SOP passages and up to 30 result rows (already masked) are sent to the Gemini API.
- **Secrets:** use `.env` (git-ignored) or your platform's secret store; change `JWT_SECRET` before deploying.

## If you adapt this to real data

Run a data-protection review first (India's DPDP Act 2023 and your organisation's policies). Use pseudonymised
extracts, keep vendor bank data out of analytics stores, and restrict the `admin` role.
