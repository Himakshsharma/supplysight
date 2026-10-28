# Interview notes – SupplySight

Short, honest answers to the questions this project usually triggers. The numbers are from the seed-42 run.

### 1. Explain the project in 30 seconds.
A cosmetics plant on SAP MM/PP has two problems: risky purchase orders slip through, and reorder points go stale
across festive and summer seasons. I built one platform with three AI parts. An explainable anomaly ensemble scores
every PO and gives reasons (PR-AUC 0.84 vs 0.59 for rules alone). A global gradient-boosted forecaster predicts
weekly material consumption and recommends reorder points (20 % WMAPE vs 23.5 % for the MRP plan explosion). A
copilot answers data, process and insight questions with citations. It ships with FastAPI, Streamlit, a model
registry, drift monitoring and retraining. All data is synthetic, from a seeded generator I wrote.

### 2. Why synthetic data, and how did you keep it realistic?
I can't use real company data. To make the synthetic data useful, it is causal: seasonal demand → weekly production
orders → raw-material consumption = production × BOM → an inventory simulation that raises PRs and POs → goods
receipts after vendor lead times. Then I added realistic noise: legitimate expedites, pre-buys, price moves,
weekend POs, slipped and rush production orders, branch vendors with the same PAN. My first version gave PR-AUC
0.996, which meant the anomalies were too easy, so I hardened the generator rather than report that number.

### 3. How do you evaluate anomaly detection when there are no labels?
In this project, injected anomalies give ground truth. They are stored in a separate table and used only for
evaluation (time-based holdout, PR-AUC, precision@k, recall per type). In real life there are no labels, so the
plan is: (a) proxy metrics such as flag rate, score stability and drift; (b) precision@k from the analysts' review
queue, since every flag gets a true-issue / false-positive verdict; (c) occasional audit samples of Low-risk POs to
estimate recall. The code already treats analyst verdicts as labels: they override the evaluation ground truth and
train a stacking layer once there are enough of them.

### 4. Why PR-AUC and precision@k instead of accuracy or ROC-AUC?
With 4 % positives, a model that flags nothing is 96 % accurate, and ROC-AUC looks good because of the huge negative
class. Analysts review a fixed number of POs per week, so the business question is “how many of the top-k flags are
real?” That is precision@k. PR-AUC summarises the precision/recall trade-off across thresholds.

### 5. What does a false positive cost compared with a false negative, and how did that shape thresholds?
A false positive costs an analyst 5–15 minutes and possibly delays a PO. A false negative can be a duplicate payment
or fraud worth lakhs. So the rules for severe patterns (split PO, after-hours + new vendor, impossible lead time,
duplicate vendor) carry high weights, High is the top ~2.5 % (about 20 reviews a month) and Medium the next 5 %. The
model only prioritises; a person always decides.

### 6. Why an ensemble, and how are the scores combined?
Each part sees different things. Rules are precise on known patterns but blind to new ones. Isolation Forest finds
multivariate outliers. The autoencoder measures “how unlike normal” a PO is, with per-feature errors for explanation.
Their raw scores aren't comparable, so I convert the ML scores to ECDF percentiles on the training data, keep only the
extreme tail, and combine with the rule score as a noisy-OR. The result is 0.837 PR-AUC vs 0.804 for the best
single model.

### 7. How do you make the anomaly flags explainable?
Rules emit their own sentences. For the ML part I compute a robust z-score per feature, keep only deviations in the
risky direction (high price, low tenure, short lead time…) and turn the top ones into text with real values: “Unit
price ₹935 is 2.7× the 90-day median for material 100037”. ML reasons that repeat a rule are skipped, and every flag
shows at most three reasons.

### 8. How did you prevent data leakage?
(a) PO features use time windows with `closed='left'`, so only strictly earlier POs count. (b) Contract prices are
looked up as of the PO date (dated `EINE` conditions). (c) The forecaster uses a rolling-origin backtest, retraining
at each origin. (d) The production plan is only “known” inside the frozen 4-week horizon. (e) Tests poison future
values and assert nothing changes. That test caught a real bug: a feature normalised by a dataset-wide mean, which I
fixed. (f) Labels sit in a separate table the models never read.

### 9. Why did the GBM beat the baselines, and why no LSTM?
The strongest baseline is the MRP explosion (planned production × BOM). Inside the frozen plan it's almost as good
(19.2 % vs 18.5 %), because planned production explains most consumption. Beyond the plan, the GBM wins clearly
(21.5 % vs 27.8 %) because one global model learns seasonality shared across 344 series. An LSTM needs far more
history than 105 weeks per series; I'd only try it with more data, and benchmark it fairly against the GBM.

### 10. How do you compute the reorder point?
Safety stock = z·σ·√L, with z = 1.645 for a 95 % service level, σ = the per-series standard deviation of backtest
forecast errors, and L = vendor lead time in weeks. ROP = mean weekly forecast × L + safety stock. I compare it with
`MARC-MINBE` and flag High stock-out risk when stock plus open POs can't cover lead-time demand. P10–P90 bands are
conformal-calibrated from 66 % to 80 % coverage, so σ and the bands are honest.

### 11. How do you control hallucination in the copilot?
(a) The router sends numbers questions to SQL. The model never “remembers” a number; it comes from a query whose SQL
and row count are shown. (b) Process answers come only from retrieved SOP sections and cite file and section; the LLM
prompt says to answer only from the excerpts or say it can't find the answer. (c) Offline mode returns passages and
query results without generated prose. (d) Evaluation: router accuracy, retrieval hit@3/MRR, SQL execution
accuracy and citation validity on a 42-question gold set, run in CI as quality gates.

### 12. What guardrails does the SQL tool have?
Single statement, SELECT/WITH only, a regex plus (when installed) a `sqlglot` parse, table whitelist (labels and
feedback excluded), no system catalogues, enforced LIMIT, read-only connection (`mode=ro` on SQLite,
`SET TRANSACTION READ ONLY` + statement timeout on PostgreSQL) and a progress-handler timeout. Prompt-injection
patterns and destructive requests are refused before routing. GSTIN and bank details are masked unless the user is
admin. Tests try `DROP`, `UPDATE`, stacked statements, `PRAGMA`, `ATTACH` and CTE-wrapped deletes.

### 13. How do you detect drift and decide to retrain?
For each period I compare reference and current data: PSI and KS for all 15 PO features, PSI of the anomaly score
(prediction drift), scaled-demand and price-index drift, and the **live** WMAPE of the stored forecast against
actuals. On a simulated Q4 (+18 % prices, festive surge, new vendors), 38 % of features drifted, score PSI was 0.94
and live WMAPE was 37.6 % vs 20 % in backtest. Retraining is triggered by thresholds (≥ 30 % features, PSI ≥ 0.2,
WMAPE ≥ 1.25× backtest). A challenger is trained on a 12-month window and must beat the champion **on the new
period** before promotion (0.331 → 0.354 PR-AUC, promoted).

### 14. How would this scale to many plants, or connect to a real SAP system?
Plant is already in every key, and the forecaster is global, so more plants means more series, not more models. For
SAP I'd read via OData services (`API_PURCHASEORDER_PROCESS_SRV`, `API_MATERIAL_DOCUMENT_SRV`), BAPIs
(`BAPI_PO_GETDETAIL1`) or SLT/Datasphere replication into the same schema, and run batch scoring hourly plus
on-demand scoring in the PO release workflow (a BAdI or workflow step calling `/anomaly/score`). Write-back would
only set a release block or a workflow task, never change the PO.

### 15. What would you improve next, and what are the weaknesses?
Weakest anomaly types are price spikes on volatile commodities (56 % recall) and modest lead-time cuts (43 %).
Commodity price indices and expedite reason codes would help. The copilot's gold set is also its dev set, so I'd
build a blind test set. I'd add the LSTM comparison, the ONNX latency benchmark for the PyTorch autoencoder, Alembic
migrations, SSO, and feed analyst feedback into supervised fine-tuning once there are a few hundred verdicts.

---

## The 5 things to be able to explain without notes

1. **The data story:** causal generator (production × BOM → consumption → inventory → POs) and why I made it *harder*
   when results looked too good.
2. **The ensemble:** rules + duplicate matching + Isolation Forest + autoencoder → ECDF → noisy-OR → risk levels and
   three reasons; PR-AUC 0.84 vs 0.59 for rules.
3. **Leakage control:** past-only windows, point-in-time prices, rolling-origin backtest, frozen-plan rule, and the
   test that caught a real leak.
4. **Copilot design:** router + SQL + RAG + insights, guardrails, citations, offline mode, and how it was evaluated.
5. **MLOps loop:** registry stages, drift metrics (PSI/KS, live WMAPE), champion/challenger on the new period, and
   analyst feedback as labels.
