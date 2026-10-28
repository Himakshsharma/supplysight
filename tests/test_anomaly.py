"""Phase 2: features, rules, duplicate detection, metrics and the trained ensemble."""

from __future__ import annotations

import numpy as np
import pandas as pd

from supplysight.anomaly.duplicates import find_duplicate_vendors, normalise_name, text_similarity
from supplysight.anomaly.evaluate import evaluate_scores, precision_at_k, recall_at_k
from supplysight.anomaly.rules import RuleConfig, _inr, apply_rules
from supplysight.data import po_items, weekly_consumption
from supplysight.features.po_features import ML_FEATURES, build_po_features


def _features(tables):
    return build_po_features(po_items(tables), weekly_consumption(tables["mseg"]), {}, 500_000)


def test_features_are_finite_and_past_only(small_tables):
    f = _features(small_tables)
    assert np.isfinite(f[ML_FEATURES].to_numpy()).all()
    # appending a future PO must not change features of earlier POs (no look-ahead)
    items = po_items(small_tables)
    cut = items.created_at.quantile(0.5)
    early = items[items.created_at <= cut]
    f_early = build_po_features(early, weekly_consumption(small_tables["mseg"]), {}, 500_000)
    f_all = _features(small_tables).set_index(["ebeln", "ebelp"])
    cols = ["ref_price", "ref_qty", "requester_freq", "vendor_prior_pos"]
    a = f_early.set_index(["ebeln", "ebelp"])[cols]
    b = f_all.loc[a.index, cols]
    pd.testing.assert_frame_equal(a, b, check_dtype=False)


def test_split_po_rule_fires():
    base = dict(
        matnr="100001",
        lifnr="500001",
        created_by="X",
        werks="1000",
        dow=1,
        hour=11,
        is_weekend=0,
        is_offhours=0,
        tenure_days=900,
        lead_time_days=14,
        normal_lt_days=14,
        lt_ratio=1.0,
        vendor_dup_score=0.0,
        dup_of="",
        dup_evidence="",
        no_info_record=0,
    )
    f = pd.DataFrame(
        [
            {**base, "netwr": 480_000, "same_day_cluster": 3, "cluster_value": 1_440_000},
            {**base, "netwr": 120_000, "same_day_cluster": 1, "cluster_value": 120_000},
        ]
    )
    r = apply_rules(f, RuleConfig())
    assert r.rule_score.iloc[0] >= 0.9 and "split PO" in r.rule_reasons.iloc[0][0]
    assert r.rule_score.iloc[1] == 0


def test_offhours_new_vendor_and_short_lead_time_rules():
    base = dict(
        matnr="1",
        lifnr="2",
        created_by="X",
        werks="1000",
        netwr=1e4,
        same_day_cluster=1,
        cluster_value=1e4,
        vendor_dup_score=0.0,
        dup_of="",
        dup_evidence="",
        no_info_record=0,
        normal_lt_days=20,
    )
    f = pd.DataFrame(
        [
            {
                **base,
                "dow": 6,
                "hour": 23,
                "is_weekend": 1,
                "is_offhours": 1,
                "tenure_days": 10,
                "lead_time_days": 20,
                "lt_ratio": 1.0,
            },
            {
                **base,
                "dow": 2,
                "hour": 11,
                "is_weekend": 0,
                "is_offhours": 0,
                "tenure_days": 999,
                "lead_time_days": 1,
                "lt_ratio": 0.07,
            },
        ]
    )
    r = apply_rules(f, RuleConfig())
    assert r.rule_score.iloc[0] >= 0.85 and "outside business hours" in r.rule_reasons.iloc[0][0]
    assert r.rule_score.iloc[1] >= 0.85 and "lead time" in r.rule_reasons.iloc[1][0]


def test_duplicate_vendor_detection():
    lfa1 = pd.DataFrame(
        [
            {
                "lifnr": "1",
                "name1": "Shree Ganesh Chemicals Pvt Ltd",
                "stras": "Plot 4, MIDC",
                "ort01": "Pune",
                "stcd3": "27ABCCD1234E1Z5",
                "bank_ref": "HDFC0001234/50200012345678",
                "created_on": "2018-01-01",
            },
            {
                "lifnr": "2",
                "name1": "Shri Ganesh Chemical Pvt. Ltd.",
                "stras": "Plot No. 4, MIDC",
                "ort01": "Pune",
                "stcd3": "27ZZZCD9999E1Z5",
                "bank_ref": "HDFC0001234/50200012345678",
                "created_on": "2026-03-01",
            },
            {
                "lifnr": "3",
                "name1": "Kaveri Packaging LLP",
                "stras": "Ring Road",
                "ort01": "Surat",
                "stcd3": "24QWECX1111A1Z1",
                "bank_ref": "SBIN0000001/11111111111",
                "created_on": "2019-01-01",
            },
        ]
    )
    pairs = find_duplicate_vendors(lfa1)
    assert len(pairs) == 1
    assert (pairs.lifnr_a.iloc[0], pairs.lifnr_b.iloc[0]) == ("1", "2")
    assert "same bank account" in pairs.match_on.iloc[0]
    assert normalise_name("Shri Ganesh Chemical Pvt. Ltd.") == normalise_name("Shree Ganesh Chemicals Private Limited")
    assert text_similarity("a b c", "c b a") == 1.0


def test_metrics():
    y = np.array([1, 0, 1, 0, 0])
    s = np.array([0.9, 0.8, 0.7, 0.1, 0.0])
    assert precision_at_k(y, s, 2) == 0.5
    assert recall_at_k(y, s, 3) == 1.0
    out = evaluate_scores(y, {"m": s})
    assert 0 < out["table"].pr_auc.iloc[0] <= 1


def test_inr_format():
    assert _inr(500000) == "₹5,00,000"
    assert _inr(18_57_035) == "₹18,57,035"
    assert _inr(999) == "₹999"


def test_trained_ensemble_beats_rules_and_explains(trained):
    m = trained["anomaly"]["metrics"]
    hold = trained["anomaly"]["holdout"]["table"].set_index("model")
    assert hold.loc["ensemble", "pr_auc"] >= hold.loc["rules_only", "pr_auc"]
    assert 0 < m["flag_rate"] < 0.2
    from supplysight.db import read_table

    scores = read_table("po_scores")
    flagged = scores[scores.risk_level != "Low"]
    assert len(flagged) > 0
    assert flagged.reasons.map(lambda r: len(r) > 2).all()  # non-empty JSON list
