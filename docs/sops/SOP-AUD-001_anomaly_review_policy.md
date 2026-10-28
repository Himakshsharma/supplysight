# SOP-AUD-001 AI Anomaly Flag Review (Human in the Loop)

*Fictional SOP for the SupplySight demo; synthetic organisation.*

## Purpose
Define how procurement analysts review purchase orders flagged by the SupplySight anomaly engine.

## Risk levels
- High: review before the PO is released or within one working day if already released.
- Medium: review within five working days.
- Low: no action; sampled monthly by internal audit.

## Review steps
1. Open the flag in the *Anomaly Review Queue* and read the reasons (e.g. price ratio vs 90-day median, split-PO cluster, new vendor).
2. Check supporting documents: requisition, quotation, approval e-mails, vendor master history.
3. Mark the flag as **true issue** or **false positive** and add a comment.

## What happens to the feedback
Reviewer verdicts are stored and used as labelled data when the model is retrained, so frequent false-positive patterns are down-weighted over time. The AI never blocks a PO on its own; a person always takes the decision.

## Escalation
Confirmed fraud indicators (duplicate vendor with the same bank account, split POs) are escalated to internal audit within 24 hours.
