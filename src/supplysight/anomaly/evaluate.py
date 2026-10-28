"""Evaluation of anomaly scores against the injected ground-truth labels."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def precision_at_k(y: np.ndarray, s: np.ndarray, k: int) -> float:
    """Share of true anomalies among the k highest scores."""
    k = min(k, len(s))
    top = np.argsort(-s, kind="stable")[:k]
    return float(np.asarray(y)[top].mean()) if k else 0.0


def recall_at_k(y: np.ndarray, s: np.ndarray, k: int) -> float:
    """Share of all anomalies found within the top-k scores."""
    y = np.asarray(y)
    if y.sum() == 0:
        return 0.0
    top = np.argsort(-s, kind="stable")[: min(k, len(s))]
    return float(y[top].sum() / y.sum())


def evaluate_scores(
    y: np.ndarray, scores: dict[str, np.ndarray], flagged: np.ndarray | None = None, types: pd.Series | None = None
) -> dict:
    """Compare several scorers on the same labelled rows.

    Returns ``{"table": DataFrame, "per_type_recall": DataFrame | None}``.
    """
    y = np.asarray(y).astype(int)
    rows = []
    n_pos = int(y.sum())
    for name, s in scores.items():
        s = np.asarray(s, float)
        rows.append(
            {
                "model": name,
                "pr_auc": average_precision_score(y, s) if n_pos else np.nan,
                "precision@50": precision_at_k(y, s, 50),
                "precision@100": precision_at_k(y, s, 100),
                f"recall@{n_pos}": recall_at_k(y, s, n_pos),
                "recall@2x": recall_at_k(y, s, 2 * n_pos),
            }
        )
    table = pd.DataFrame(rows)
    per_type = None
    if types is not None and flagged is not None:
        t = pd.DataFrame({"type": types.fillna("normal"), "flagged": np.asarray(flagged).astype(int)})
        per_type = (
            t[t.type != "normal"]
            .groupby("type")
            .flagged.agg(["mean", "size"])
            .rename(columns={"mean": "recall_flagged", "size": "n"})
            .reset_index()
        )
    return {"table": table, "per_type_recall": per_type, "n_pos": n_pos, "n": len(y)}
