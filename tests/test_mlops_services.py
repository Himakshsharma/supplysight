"""Phase 5–6: registry, drift metrics, retraining loop, services and auth."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from supplysight.monitoring.drift import drift_table, psi


def test_psi_behaviour():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000)
    assert psi(a, rng.normal(0, 1, 5000)) < 0.05
    assert psi(a, rng.normal(1.0, 1, 5000)) > 0.2
    assert psi(np.r_[np.zeros(900), np.ones(100)], np.r_[np.zeros(500), np.ones(500)]) > 0.2  # binary feature
    t = drift_table(pd.DataFrame({"x": a}), pd.DataFrame({"x": a + 2}), ["x"])
    assert bool(t.drifted.iloc[0])


def test_registry_stages(tmp_env):
    from supplysight.mlops import Registry

    reg = Registry(tmp_env / "reg_test")
    v1 = reg.register("m", {"w": 1}, {"score": 0.5}, stage="Production")
    v2 = reg.register("m", {"w": 2}, {"score": 0.6})
    assert reg.get_version("m", "Production")["version"] == v1
    reg.transition("m", v2, "Production")
    assert reg.get_version("m", "Production")["version"] == v2
    assert reg._index()["m"]["versions"][str(v1)]["stage"] == "Archived"
    model, meta = reg.load("m")
    assert model == {"w": 2} and meta["metrics"]["score"] == 0.6
    with pytest.raises(LookupError):
        reg.load("unknown")


def test_services_and_feedback_loop(trained):
    from supplysight.services import core

    ov = core.overview()
    assert ov["spend_90d"] > 0 and ov["monthly"]
    flags = core.list_flags(risk="High,Medium", limit=50)
    assert flags and all(f["risk_level"] in ("High", "Medium") for f in flags)
    rec = core.review_flag(flags[0]["id"], "false_positive", "test", "tester")
    assert rec["verdict"] == "false_positive"
    assert any(f["id"] == flags[0]["id"] for f in core.list_flags(status="reviewed"))
    with pytest.raises(ValueError):
        core.review_flag(flags[0]["id"], "maybe")
    with pytest.raises(core.NotFound):
        core.review_flag("4599999999-00010", "true_issue")

    t = core.tables()
    info = t["eina"].iloc[0]
    normal = core.score_po({"lifnr": info.lifnr, "matnr": info.matnr, "werks": "1000", "menge": 50, "netpr": 1.0,
                            "lead_time_days": 14, "created_by": "RSHARMA"})
    risky = core.score_po({"lifnr": info.lifnr, "matnr": info.matnr, "werks": "1000", "menge": 50, "netpr": 1.0,
                           "lead_time_days": 0, "created_by": "RSHARMA", "created_at": "2026-06-28T23:30:00"})
    assert risky["score"] >= normal["score"]
    assert risky["reasons"]
    with pytest.raises(core.NotFound):
        core.score_po({"lifnr": "1", "matnr": "999999", "menge": 1, "netpr": 1})

    rr = core.reorder_risk()
    fc = core.get_forecast(rr[0]["matnr"], rr[0]["werks"], 6)
    assert len(fc["forecast"]) == 6 and all(p["p10"] <= p["yhat"] <= p["p90"] for p in fc["forecast"])
    assert core.model_health()["production"]["anomaly-detector"]["version"]


def test_auth():
    from supplysight.services.auth import allowed, authenticate, create_token, decode_token

    assert authenticate("buyer", "buyer123")["role"] == "buyer"
    assert authenticate("buyer", "wrong") is None
    tok = create_token("planner", "planner")
    assert decode_token(tok)["role"] == "planner"
    assert allowed("manager", "anomaly:review") and not allowed("planner", "anomaly:review")


def test_drift_and_retrain_loop(trained, small_cfg):
    """Simulated drift quarter -> drift detected -> challenger trained and evaluated."""
    from dataclasses import replace

    from supplysight.datagen.generator import generate, write_csv
    from supplysight.monitoring.retrain import drift_dir, run_retrain

    cfg = replace(small_cfg, start="2026-07-01", end="2026-09-30", trend_anchor=small_cfg.start, id_offset=100_000,
                  vendor_offset=100, drift={"price_shock": 1.25, "demand_surge": {"FG-LIP": 1.6, "FG-CREAM": 1.4},
                                            "new_vendor_share": 0.15})
    tabs = generate(cfg)
    keep = ["lfa1", "eina", "eine", "eban", "ekko", "ekpo", "mkpf", "mseg", "afko", "afpo", "mard", "plaf",
            "anomaly_labels"]
    write_csv({k: v for k, v in tabs.items() if k in keep}, drift_dir())
    out = run_retrain(force=True)
    d = out["drift"]
    assert d["retrain_recommended"] and d["anomaly_score_psi"] > 0
    assert out["retrained"] and "promoted" in out["decision"]["anomaly"]
