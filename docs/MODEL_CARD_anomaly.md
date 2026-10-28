# Model card – PO anomaly detector (`anomaly-detector`)

## Intended use

The detector prioritises purchase-order items for **human review** by procurement analysts and internal audit at a
cosmetics plant on SAP MM. It produces a score in [0, 1], a risk level (Low / Medium / High) and up to three
human-readable reasons per PO item.

It is **not** meant to block, approve or change POs automatically, to judge individual employees, or to be used
on data unlike the training data (other industries or currencies) without re-validation.

## Model

| Part | What it detects | Notes |
|---|---|---|
| Rules (noisy-OR) | Split PO (≥ 2 POs same day/buyer/vendor/material, each 75–100 % of the ₹5 lakh limit, total above it) · after-hours or weekend PO to a vendor < 90 days old · lead time < 25 % of normal or ≤ 1 day · duplicate vendor (similarity ≥ 0.8) · no info record (weak) | Auditable, deterministic |
| Duplicate-vendor detector | Fuzzy name (+ canonicalisation Shri/Shree, Pvt/Private) and address similarity; GSTIN / PAN / bank matches | rapidfuzz if installed, difflib otherwise |
| Isolation Forest | Multivariate outliers on 15 robust-scaled features | 300 trees, `max_samples=512` |
| Autoencoder | Reconstruction error (16-6-16 bottleneck), trained on the 95 % least-isolated rows | PyTorch if installed, sklearn MLP otherwise |
| Ensemble | `score = 1 − (1 − rules) × (1 − ml_tail)`, `ml_tail = clip((mean ECDF − 0.5)/0.5)³` | High = top 2.5 % of training scores, Medium = next 5 % |

**Features** (all computed from data before the PO's creation time): price vs 90-day material median, price vs
contracted info-record price valid on the PO date, quantity vs typical order size, weeks of consumption covered,
PO vs requisition quantity, vendor tenure, lead time vs vendor norm, off-hours/weekend, buyer's share of recent POs,
value vs approval limit, same-day cluster size, duplicate-vendor similarity, vendor PO history, missing info record.

**Explanations:** rule reasons first, then the features whose robust z-score deviates most in the *risky*
direction, rendered as text, e.g. “Unit price ₹935 is 2.7× the 90-day median (₹349) for material 100037
(Hydroxyethyl Cellulose)”. ML reasons that repeat a rule reason are skipped.

## Training and evaluation data

The data is the synthetic seed-42 dataset: 4,726 PO items over 2 years, 187 (4.0 %) injected anomalies of six types.
The anomaly labels are **never** used for training. They are used only for evaluation and for choosing between
champion and challenger.

Evaluation uses a **time-based holdout**: the model is fitted on the first 70 % of POs and scored on the last 30 %
(1,418 items, 65 anomalies). The production model is then refitted on all history.

| Model | PR-AUC | Precision@50 | Precision@100 | Recall@65 |
|---|---|---|---|---|
| Rules only | 0.589 | 0.78 | 0.40 | 0.62 |
| Isolation Forest | 0.804 | 0.80 | 0.51 | 0.72 |
| Autoencoder | 0.666 | 0.72 | 0.51 | 0.72 |
| **Ensemble** | **0.837** | **0.86** | **0.55** | **0.79** |

Recall of Medium/High flags by anomaly type (holdout): duplicate vendor 11/11, off-hours + new vendor 15/15,
quantity outlier 14/14, split PO 9/9, **price spike 5/9**, **short lead time 3/7**. Flag rate is 7.5 % (High 2.5 %).

**Robustness under drift:** on a simulated Q4 2026 (+18 % price shock, festive demand surge, six new vendors), the
production model's PR-AUC on the new quarter fell to 0.33 and its flag rate rose from 8 % to 19 %. A challenger
trained on a 12-month window reached 0.35 and was promoted. Drift lowers precision substantially, which is why
drift monitoring is part of the system.

## Limitations

- Price spikes on volatile commodity materials (fragrance oils, butters, actives, ±10 % normal noise) and modest
  lead-time reductions overlap with legitimate expedites and market moves.
- Split POs spread over several days are only caught through quantity/value features, not by the same-day rule.
- New legitimate vendors look riskier for their first ~90 days (tenure and history features).
- The holdout has only 65 anomalies, so metrics carry roughly ±0.05 sampling noise.

## Cost of errors and the review policy

| Error | Cost | Mitigation |
|---|---|---|
| False positive | About 5–15 minutes of analyst time; possible delay if a High flag holds a PO | High ≈ 2.5 % of POs, about 20 reviews/month at this volume; reasons speed review |
| False negative | Overpayment, duplicate payment or fraud loss (often lakhs of ₹) | Thresholds favour recall on High-severity rule types; monthly audit sample of Low |

Review policy (SOP-AUD-001): High is reviewed before release or within one working day; Medium within five days;
Low is sampled monthly by audit. Reviewers record **true issue / false positive** with a comment. Verdicts become
labels: they override the evaluation ground truth and, with ≥ 20 verdicts, fit a logistic stacking layer on the
component scores at the next retraining. **The model never blocks a PO on its own.**

## Monitoring

Weekly/quarterly: PSI/KS for all 15 features and for the score distribution, flag rate, share of flags confirmed by
reviewers. Retraining is triggered when ≥ 30 % of features drift or score PSI ≥ 0.2 (`scripts/retrain.py`).
