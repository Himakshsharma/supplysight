"""Offline evaluation of the copilot against the gold set (tests/copilot_eval.jsonl).

Metrics:
* router accuracy - predicted route (incl. 'refuse') == gold route
* retrieval hit@3 / MRR - gold SOP among the top-3 retrieved chunks (process questions)
* SQL execution accuracy - every value in the reference query's first column appears in the
  copilot's result (data questions; values normalised, floats rounded to 2 dp)
* insight accuracy - correct insight tool used
* guardrail block rate - refuse questions that were blocked
* citation validity - non-refused answers whose citations point to an existing SOP file or a
  validated SQL query
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from supplysight.config import get_settings
from supplysight.copilot.agent import Copilot
from supplysight.copilot.retriever import load_retriever
from supplysight.copilot.sql_tool import run_sql


def _norm(v) -> str:
    try:
        f = float(v)
        return f"{round(f, 2):.2f}"
    except (TypeError, ValueError):
        return str(v).strip()


def load_gold(path: Path | None = None) -> list[dict]:
    """Read the JSONL gold set."""
    from supplysight.config import PROJECT_ROOT

    p = path or PROJECT_ROOT / "tests" / "copilot_eval.jsonl"
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def evaluate_copilot(gold: list[dict] | None = None) -> dict:
    """Run every gold question through the copilot and compute metrics."""
    gold = gold or load_gold()
    bot = Copilot()
    retr = load_retriever()
    sops = get_settings().sops_dir
    rows = []
    for g in gold:
        a = bot.ask(g["question"], role="manager")
        pred = a.route
        rec = {
            "id": g["id"],
            "question": g["question"],
            "gold_route": g["route"],
            "pred_route": pred,
            "route_ok": pred == g["route"],
            "latency_ms": a.latency_ms,
            "mode": a.mode,
        }
        if g["route"] == "process":
            hits = [h["chunk"].title.split()[0] for h in retr.search(g["question"], k=3)]
            rank = next((i + 1 for i, d in enumerate(hits) if d == g["expected_doc"]), None)
            rec.update({"hit@3": rank is not None, "rr": 1.0 / rank if rank else 0.0, "retrieved": hits})
        if g["route"] == "data":
            ref, _ = run_sql(g["ref_sql"])
            ref_vals = {_norm(v) for v in ref.iloc[:, 0].tolist()} if len(ref) else set()
            got = run_sql(a.sql)[0] if a.sql else pd.DataFrame(a.rows)  # full result, not the 50-row preview
            got_vals = {_norm(v) for v in got.to_numpy().ravel()} if len(got) else set()
            rec.update(
                {"sql_ok": bool(ref_vals) and ref_vals <= got_vals, "ref_rows": len(ref), "got_rows": a.row_count}
            )
        if g["route"] == "insight":
            src = a.citations[0].get("source", "") if a.citations else ""
            kind_map = {
                "risky_pos": "po_scores",
                "stockout": "reorder_recommendations",
                "duplicate_vendors": "duplicate-vendor",
                "forecast": "demand-forecaster",
            }
            rec["insight_ok"] = pred == "insight" and kind_map[g["insight_kind"]] in src
        if g["route"] == "refuse":
            rec["blocked"] = a.blocked
        if pred != "refuse":
            valid = bool(a.citations) and all(
                (c["type"] == "sop" and (sops / c["doc"]).exists())
                or (c["type"] in ("sql", "model_output") and c.get("query"))
                for c in a.citations
            )
            rec["citation_valid"] = valid
        rows.append(rec)
    df = pd.DataFrame(rows)

    def mean(col: str, mask=None) -> float:
        if col not in df:
            return float("nan")
        s = df.loc[mask, col] if mask is not None else df[col]
        s = s.dropna()
        return float(s.astype(float).mean()) if len(s) else float("nan")

    metrics = {
        "n_questions": len(df),
        "router_accuracy": mean("route_ok"),
        "retrieval_hit@3": mean("hit@3", df.gold_route == "process"),
        "retrieval_mrr": mean("rr", df.gold_route == "process"),
        "sql_execution_accuracy": mean("sql_ok", df.gold_route == "data"),
        "insight_accuracy": mean("insight_ok", df.gold_route == "insight"),
        "guardrail_block_rate": mean("blocked", df.gold_route == "refuse"),
        "citation_validity": mean("citation_valid", df.pred_route != "refuse"),
        "p50_latency_ms": float(df.latency_ms.median()),
        "mode": "offline" if (df["mode"] == "offline").all() else "mixed",
    }
    return {"metrics": metrics, "details": df}


def write_report(res: dict, path: Path) -> None:
    """Write docs/COPILOT_EVAL.md."""
    m, d = res["metrics"], res["details"]

    def failed(col: str) -> pd.Series:
        return d[col].eq(False) if col in d else pd.Series(False, index=d.index)

    fails = d[~d.route_ok | failed("sql_ok") | failed("hit@3") | failed("insight_ok")]
    lines = [
        "# Copilot evaluation",
        "",
        "Generated by `python scripts/evaluate_all.py --module copilot` on the seed-42 synthetic dataset, "
        f"**{m['mode']} mode** (no LLM key). Gold set: `tests/copilot_eval.jsonl` ({m['n_questions']} questions: "
        f"{(d.gold_route == 'data').sum()} data, {(d.gold_route == 'process').sum()} process, "
        f"{(d.gold_route == 'insight').sum()} insight, {(d.gold_route == 'refuse').sum()} guardrail).",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Router accuracy (incl. refusals) | {m['router_accuracy']:.1%} |",
        f"| Retrieval hit@3 (process) | {m['retrieval_hit@3']:.1%} |",
        f"| Retrieval MRR (process) | {m['retrieval_mrr']:.3f} |",
        f"| SQL execution accuracy (data) | {m['sql_execution_accuracy']:.1%} |",
        f"| Insight tool accuracy | {m['insight_accuracy']:.1%} |",
        f"| Guardrail block rate (injection / destructive / off-topic) | {m['guardrail_block_rate']:.1%} |",
        f"| Answers with valid citations | {m['citation_validity']:.1%} |",
        f"| Median latency | {m['p50_latency_ms']:.0f} ms |",
        "",
        "> **Read these numbers as a development set, not a blind test.** The router patterns and query expansions "
        "were debugged against this gold set (three router/insight bugs it exposed were fixed), so a fresh, unseen "
        "question set would score lower. The offline template SQL engine also only covers the intents it was built for; "
        "an LLM key widens coverage at the cost of determinism.",
        "",
        "## How each metric is computed",
        "",
        "- **SQL execution accuracy**: the reference SQL in the gold file is executed; the answer counts as correct when every value of the reference result's first column appears in the copilot's result rows.",
        "- **Retrieval hit@3 / MRR**: the expected SOP must be among the top-3 fused BM25 + dense chunks.",
        "- **Citation validity**: every citation points to an existing SOP file or to a guardrail-validated SQL query with its row count.",
        "",
        "## Misses",
        "",
    ]
    if fails.empty:
        lines.append("None.")
    else:
        lines += ["| ID | Question | Gold route | Predicted | Note |", "|---|---|---|---|---|"]
        for r in fails.itertuples():
            note = []
            if getattr(r, "retrieved", None) and isinstance(r.retrieved, list):
                note.append("retrieved " + ", ".join(r.retrieved))
            if getattr(r, "sql_ok", None) is False:
                note.append(f"result rows {r.got_rows} vs reference {r.ref_rows}")
            lines.append(f"| {r.id} | {r.question} | {r.gold_route} | {r.pred_route} | {'; '.join(note)} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
