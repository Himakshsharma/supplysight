"""Feature engineering for purchase-order items (shared by rules, ML models and the API).

Every statistic is computed from data *before* the PO's creation time (``closed='left'``
rolling windows), so features never peek at the future.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ML_FEATURES = [
    "log_price_ratio",
    "log_price_vs_info",
    "log_qty_ratio",
    "log_weeks_cover",
    "log_pr_ratio",
    "log_tenure_days",
    "log_lt_ratio",
    "is_offhours",
    "is_weekend",
    "requester_freq",
    "value_to_threshold",
    "same_day_cluster",
    "vendor_dup_score",
    "log_vendor_prior_pos",
    "no_info_record",
]


def _rolling_past(df: pd.DataFrame, by: list[str], col: str, window: str, how: str) -> pd.Series:
    """Rolling statistic over strictly earlier rows in the same group (time-based window)."""
    tmp = df[by + ["created_at", col]].copy()
    tmp["_i"] = np.arange(len(tmp))
    tmp = tmp.sort_values(by + ["created_at", "_i"])
    roll = tmp.groupby(by, sort=False).rolling(window, on="created_at", closed="left")[col]
    vals = getattr(roll, how)().reset_index(level=list(range(len(by))), drop=True)
    tmp["_v"] = vals.values
    return tmp.sort_values("_i")["_v"].reset_index(drop=True)


def build_po_features(
    items: pd.DataFrame, weekly: pd.DataFrame | None, dup_scores: dict, approval_threshold: float
) -> pd.DataFrame:
    """Compute raw and model-ready features for each PO item.

    Args:
        items: output of :func:`supplysight.data.po_items` (sorted by created_at).
        weekly: dense weekly consumption (matnr, werks, week, qty) or None.
        dup_scores: suspect vendor -> (similarity, original vendor, evidence).
        approval_threshold: PO value needing manager release (₹).
    """
    df = items.copy().reset_index(drop=True)
    df["created_at"] = pd.to_datetime(df.created_at)
    df["bedat"] = pd.to_datetime(df.bedat)
    df["eindt"] = pd.to_datetime(df.eindt)

    # --- price vs recent history of the same material
    med90 = _rolling_past(df, ["matnr"], "netpr", "90D", "median")
    med365 = _rolling_past(df, ["matnr"], "netpr", "365D", "median")
    ref = med90.fillna(med365).fillna(df.prim_netpr).fillna(df.netpr)
    df["ref_price"] = ref.values
    df["price_ratio"] = df.netpr / df.ref_price
    contract = df.info_netpr.fillna(df.prim_netpr)
    df["price_vs_info"] = df.netpr / contract

    # --- quantity vs typical order size and vs consumption need
    qmed = _rolling_past(df, ["matnr", "werks"], "menge", "365D", "median")
    df["ref_qty"] = qmed.fillna(df.pr_menge).fillna(df.menge).values
    df["qty_ratio"] = df.menge / df.ref_qty
    if weekly is not None and len(weekly):
        w = weekly.sort_values("week").copy()
        w["avg8"] = w.groupby(["matnr", "werks"]).qty.transform(lambda s: s.shift(1).rolling(8, min_periods=2).mean())
        w = w.dropna(subset=["avg8"])[["matnr", "werks", "week", "avg8"]]
        key = df[["matnr", "werks", "created_at"]].copy()
        key["_i"] = np.arange(len(key))
        key = key.sort_values("created_at")
        m = pd.merge_asof(
            key,
            w.sort_values("week"),
            left_on="created_at",
            right_on="week",
            by=["matnr", "werks"],
            direction="backward",
        )
        df["avg_weekly_use"] = m.sort_values("_i").avg8.values
    else:
        df["avg_weekly_use"] = np.nan
    df["weeks_cover"] = (df.menge / df.avg_weekly_use.replace(0, np.nan)).fillna(6.0)
    df["pr_ratio"] = (df.menge / df.pr_menge).fillna(1.0)

    # --- vendor, timing, lead time
    df["tenure_days"] = (df.created_at - pd.to_datetime(df.vendor_created_on)).dt.days.clip(lower=0).fillna(3650)
    df["lead_time_days"] = (df.eindt - df.bedat).dt.days
    normal_lt = df.aplfz.fillna(df.prim_aplfz).fillna(14)
    df["normal_lt_days"] = normal_lt
    df["lt_ratio"] = (df.lead_time_days.clip(lower=0) + 0.5) / (normal_lt + 0.5)
    df["dow"] = df.created_at.dt.weekday
    df["hour"] = df.created_at.dt.hour
    df["is_weekend"] = (df.dow >= 5).astype(int)
    df["is_offhours"] = ((df.hour < 8) | (df.hour >= 20)).astype(int)

    # --- requester behaviour
    df["_one"], df["_all"] = 1.0, "all"
    freq = _rolling_past(df, ["created_by"], "_one", "30D", "sum").fillna(0)
    total = _rolling_past(df, ["_all"], "_one", "30D", "sum").fillna(0)
    # share of all POs in the previous 30 days created by this buyer (past-only, no global stats)
    df["requester_freq"] = (freq / total.clip(lower=1.0)).values
    df["value_to_threshold"] = df.netwr / approval_threshold

    # --- split-PO cluster: same day, buyer, vendor and material
    day = df.created_at.dt.date
    grp = df.groupby([day, df.created_by, df.lifnr, df.matnr])
    df["same_day_cluster"] = grp.netwr.transform("size").astype(float)
    df["cluster_value"] = grp.netwr.transform("sum")

    # --- vendor master signals
    df["vendor_dup_score"] = df.lifnr.map(lambda v: dup_scores.get(v, (0.0, "", ""))[0]).astype(float)
    df["dup_of"] = df.lifnr.map(lambda v: dup_scores.get(v, (0.0, "", ""))[1])
    df["dup_evidence"] = df.lifnr.map(lambda v: dup_scores.get(v, (0.0, "", ""))[2])
    df["vendor_prior_pos"] = df.groupby("lifnr").cumcount()
    df["no_info_record"] = df.info_netpr.isna().astype(int)

    # --- model-ready transforms
    df["log_price_ratio"] = np.log(df.price_ratio.clip(0.05, 20))
    df["log_price_vs_info"] = np.log(df.price_vs_info.fillna(1).clip(0.05, 20))
    df["log_qty_ratio"] = np.log(df.qty_ratio.clip(0.01, 100))
    df["log_weeks_cover"] = np.log1p(df.weeks_cover.clip(0, 500))
    df["log_pr_ratio"] = np.log(df.pr_ratio.clip(0.01, 100))
    df["log_tenure_days"] = np.log1p(df.tenure_days)
    df["log_lt_ratio"] = np.log(df.lt_ratio.clip(0.01, 20))
    df["log_vendor_prior_pos"] = np.log1p(df.vendor_prior_pos)
    df[ML_FEATURES] = df[ML_FEATURES].astype(float).fillna(0.0)
    return df.drop(columns=["_one", "_all"])
