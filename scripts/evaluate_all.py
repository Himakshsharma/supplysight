"""Evaluate trained models and write the results tables used in the README and docs.

Usage:
    python scripts/evaluate_all.py                 # all modules
    python scripts/evaluate_all.py --module copilot
    python scripts/evaluate_all.py --min-router 0.9 --min-sql 0.8    # CI quality gates
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from supplysight.config import get_settings  # noqa: E402
from supplysight.copilot.evaluate import evaluate_copilot, write_report  # noqa: E402
from supplysight.copilot.retriever import build_index  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def latency_benchmark(n: int = 300) -> dict:
    """p50/p95 single-row inference latency of the production models (and ONNX if exported)."""
    import time

    import numpy as np

    from supplysight.mlops import Registry
    from supplysight.pipeline import ANOMALY_MODEL

    model, _ = Registry().load(ANOMALY_MODEL)
    x = np.zeros((1, len(model.scaler.center_)))
    out = {}
    for name, fn in {"isolation_forest": lambda: model.iforest.score_samples(x),
                     "autoencoder": lambda: model.ae.score(x)}.items():
        ts = []
        for _ in range(n):
            t = time.perf_counter()
            fn()
            ts.append((time.perf_counter() - t) * 1000)
        out[name] = {"p50_ms": round(float(np.percentile(ts, 50)), 3), "p95_ms": round(float(np.percentile(ts, 95)), 3),
                     "backend": model.ae.backend if name == "autoencoder" else "sklearn"}
    return out


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--module", choices=["copilot", "latency", "all"], default="all")
    ap.add_argument("--min-router", type=float, default=0.0)
    ap.add_argument("--min-sql", type=float, default=0.0)
    ap.add_argument("--min-hit3", type=float, default=0.0)
    args = ap.parse_args()
    rep = get_settings().reports_dir
    rep.mkdir(parents=True, exist_ok=True)
    result: dict = {}
    if args.module in ("copilot", "all"):
        build_index()
        res = evaluate_copilot()
        write_report(res, ROOT / "docs" / "COPILOT_EVAL.md")
        res["details"].to_csv(rep / "copilot_eval_details.csv", index=False)
        (rep / "copilot_eval.json").write_text(json.dumps(res["metrics"], indent=2))
        result["copilot"] = res["metrics"]
        m = res["metrics"]
        gates = [(m["router_accuracy"], args.min_router, "router accuracy"),
                 (m["sql_execution_accuracy"], args.min_sql, "SQL execution accuracy"),
                 (m["retrieval_hit@3"], args.min_hit3, "retrieval hit@3")]
        failed = [f"{n} {v:.2f} < {g}" for v, g, n in gates if v < g]
        if m["guardrail_block_rate"] < 1.0:
            failed.append("a guardrail test question was not blocked")
        if failed:
            print(json.dumps(result, indent=2))
            raise SystemExit("Quality gate failed: " + "; ".join(failed))
    if args.module in ("latency", "all"):
        result["latency"] = latency_benchmark()
        (rep / "latency.json").write_text(json.dumps(result["latency"], indent=2))
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
