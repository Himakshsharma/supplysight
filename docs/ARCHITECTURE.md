# Architecture

SupplySight is one Python package (`src/supplysight`) with four layers:

1. **Data**: a seeded SAP MM/PP generator and a database layer (SQLite by default, PostgreSQL in Docker).
2. **ML**: shared feature engineering plus three capabilities: anomaly detection, demand forecasting and the copilot.
3. **MLOps**: a training pipeline, model registry with stages, run tracking (with MLflow mirroring), drift
   monitoring and champion/challenger retraining.
4. **Serving**: a service layer used by both the FastAPI app and the Streamlit dashboard.

All data is synthetic.

## Component view

```mermaid
flowchart TB
  subgraph Data
    GEN[datagen.generator<br/>production → BOM → consumption → inventory sim → PR/PO → GR<br/>+ labelled anomalies]
    GEN --> CSV[data/raw/*.csv]
    GEN --> DB[(db: SQLite / PostgreSQL<br/>lower-case SAP tables)]
  end
  subgraph ML
    FE[features.po_features<br/>past-only rolling windows]
    PAN[forecast.panel<br/>weekly MSEG-261 + planned requirement]
    AN[anomaly.detector<br/>rules · duplicates · IForest · AE · ECDF noisy-OR]
    FC[forecast.models<br/>global GBM P10/P50/P90 + baselines]
    RO[forecast.reorder<br/>z·σ·√L safety stock, ROP vs MINBE]
    CP[copilot.agent<br/>router → sql_tool / retriever / insights]
  end
  subgraph MLOps
    PIPE[pipeline<br/>train → evaluate → register → publish]
    REG[(mlops.Registry<br/>None/Staging/Production/Archived)]
    MLF[(MLflow - optional mirror)]
    DRIFT[monitoring.drift + retrain<br/>PSI/KS · live WMAPE · champion/challenger]
  end
  subgraph Serving
    SVC[services.core + services.auth]
    API[api.main FastAPI /api/v1]
    UI[dashboard Streamlit]
  end
  DB --> FE --> AN
  DB --> PAN --> FC --> RO
  AN & FC --> PIPE --> REG
  PIPE -.-> MLF
  PIPE -->|po_scores · vendor_duplicates · demand_forecast · reorder_recommendations| DB
  REG --> SVC
  DB --> SVC
  CP --> SVC
  SVC --> API --> UI
  SVC --> UI
  DB --> DRIFT --> REG
  UI -->|review_feedback| DB
```

## Scoring a new purchase order

```mermaid
sequenceDiagram
  participant B as Buyer (UI / API)
  participant A as FastAPI
  participant S as services.core
  participant R as Registry
  participant M as AnomalyDetector
  B->>A: POST /api/v1/anomaly/score (JWT)
  A->>A: verify token, role has anomaly:score
  A->>S: score_po(payload)
  S->>R: load("anomaly-detector", stage="Production") (cached)
  S->>S: build item row (point-in-time contract price, vendor tenure)
  S->>S: append to last 400 days of history → build_po_features
  S->>M: score(row)
  M-->>S: score, risk level, rule/IF/AE components, top-3 reasons
  S-->>A: JSON
  A-->>B: 200 {score, risk_level, reasons, components, model_version}
```

## Copilot request

```mermaid
sequenceDiagram
  participant U as User
  participant G as Guardrails
  participant RT as Router
  participant SQL as sql_tool
  participant RAG as HybridRetriever
  participant INS as insights
  U->>G: question + role
  G-->>U: refuse (injection / destructive / off-topic)
  G->>RT: clean question
  alt data question
    RT->>SQL: plan_query (templates) or Gemini SQL
    SQL->>SQL: validate_sql (SELECT-only, whitelist, LIMIT) → read-only connection + timeout
  else process question
    RT->>RAG: BM25 + dense → reciprocal rank fusion → top-3 SOP sections
  else insight question
    RT->>INS: po_scores / reorder_recommendations / vendor_duplicates / demand_forecast
  end
  RT-->>U: answer + citations (SOP § or SQL + row count) + GSTIN/bank masked unless admin
```

## Retraining loop

```mermaid
flowchart LR
  NEW[New period<br/>data/drift] --> CHK{drift_check}
  REF[(History)] --> CHK
  CHK -->|PSI ≥ 0.2 · score PSI · live WMAPE ≥ 1.25× backtest| TRAIN[Challenger<br/>12-month window<br/>+ feedback labels]
  CHK -->|no drift| KEEP[Keep champion]
  TRAIN --> CMP{Better on the<br/>new period?}
  CMP -->|yes| PROD[Promote to Production<br/>re-publish serving tables]
  CMP -->|no| STG[Stay in Staging]
  FB[Analyst verdicts] --> TRAIN
```

## Key design choices

| Decision | Why |
|---|---|
| Lower-case SAP table/column names | The same SQL runs on SQLite and PostgreSQL; still recognisable to SAP users (EKPO-NETPR → `ekpo.netpr`). |
| Serving tables written by the pipeline | The API/dashboard answer list queries (flags, forecasts) from indexed tables in milliseconds; only single-PO scoring runs the model online. |
| File registry + MLflow mirror | Serving never depends on a tracking server; MLflow still gets runs and models for its UI. |
| Service layer shared by API and UI | One implementation of business logic; the dashboard can run in-process (demo) or through the API (Docker). |
| Optional backends detected at runtime | Runs on a laptop with only numpy/pandas/sklearn; upgrades to LightGBM, PyTorch, sentence-transformers, rapidfuzz, sqlglot and Evidently when installed. |

## Scaling to more plants / real SAP

- **Ingestion:** replace the generator with extracts via SAP OData services (e.g. `API_PURCHASEORDER_PROCESS_SRV`,
  `API_MATERIAL_DOCUMENT_SRV`), BAPIs (`BAPI_PO_GETDETAIL1`) or replication (SLT / Datasphere) into the same schema.
- **Multi-plant:** `werks` is already part of every series key and feature; the global forecaster scales with
  more series. Partition serving tables by plant and run training per company code if data volume requires.
- **Write-back:** flags stay advisory. A real integration would set a release-strategy block or a workflow item
  in SAP, never change POs automatically.
