"""Deterministic, auditable procurement rules. Each rule returns a weight in [0, 1] and a
human-readable reason; rule scores combine with a noisy-OR."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RuleConfig:
    """Thresholds for the rules layer (documented in MODEL_CARD_anomaly.md)."""

    approval_threshold: float = 500_000.0
    split_band_low: float = 0.75
    new_vendor_days: int = 90
    min_lt_ratio: float = 0.25
    dup_vendor_min: float = 0.80


def _inr(x: float) -> str:
    """Format rupees Indian-style (₹5,00,000)."""
    s = f"{int(round(x))}"
    if len(s) <= 3:
        return "₹" + s
    head, tail = s[:-3], s[-3:]
    parts = []
    while len(head) > 2:
        parts.insert(0, head[-2:])
        head = head[:-2]
    if head:
        parts.insert(0, head)
    return "₹" + ",".join(parts) + "," + tail


def apply_rules(f: pd.DataFrame, cfg: RuleConfig) -> pd.DataFrame:
    """Evaluate all rules. Returns columns rule_score and rule_reasons (list[str])."""
    thr = cfg.approval_threshold
    n = len(f)
    hits: list[list[tuple[float, str]]] = [[] for _ in range(n)]

    split = (
        (f.same_day_cluster >= 2) & (f.netwr >= cfg.split_band_low * thr) & (f.netwr < thr) & (f.cluster_value >= thr)
    ).to_numpy()
    new_vendor = (f.tenure_days < cfg.new_vendor_days).to_numpy()
    odd_time = ((f.is_weekend == 1) | (f.is_offhours == 1)).to_numpy()
    short_lt = ((f.lt_ratio < cfg.min_lt_ratio) | (f.lead_time_days <= 1)).to_numpy()
    dup = (f.vendor_dup_score >= cfg.dup_vendor_min).to_numpy()
    no_info = (f.no_info_record == 1).to_numpy()

    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    for i, r in enumerate(f.itertuples()):
        if split[i]:
            hits[i].append(
                (
                    0.9,
                    f"Possible split PO: {int(r.same_day_cluster)} POs for material {r.matnr} to vendor "
                    f"{r.lifnr} by {r.created_by} on the same day, each just below the {_inr(thr)} "
                    f"approval limit (total {_inr(r.cluster_value)})",
                )
            )
        if odd_time[i] and new_vendor[i]:
            hits[i].append(
                (
                    0.85,
                    f"Created {day_names[r.dow]} {r.hour:02d}:00 (outside business hours) for a vendor "
                    f"created only {int(r.tenure_days)} days earlier",
                )
            )
        elif odd_time[i]:
            hits[i].append((0.15, f"Created outside business hours ({day_names[r.dow]} {r.hour:02d}:00)"))
        if short_lt[i]:
            hits[i].append(
                (
                    0.85,
                    f"Delivery date {int(r.lead_time_days)} day(s) after PO date vs vendor's normal "
                    f"lead time of {int(r.normal_lt_days)} days",
                )
            )
        if dup[i]:
            hits[i].append((0.8, f"Vendor {r.lifnr} looks like a duplicate of vendor {r.dup_of} ({r.dup_evidence})"))
        if no_info[i] and not dup[i]:
            hits[i].append(
                (0.35, f"No purchasing info record (contract price) for vendor {r.lifnr} and material {r.matnr}")
            )

    score = np.array([1 - np.prod([1 - w for w, _ in h]) if h else 0.0 for h in hits])
    reasons = [[t for _, t in sorted(h, key=lambda x: -x[0])] for h in hits]
    return pd.DataFrame({"rule_score": score, "rule_reasons": reasons}, index=f.index)
