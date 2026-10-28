"""Phase 5: FastAPI integration tests (skipped automatically if FastAPI is not installed)."""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def api(trained):
    from supplysight.api.main import app

    return TestClient(app)


def _token(api, user: str) -> dict:
    r = api.post("/api/v1/auth/login", json={"username": user, "password": f"{user}123"})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def test_health_and_metrics(api):
    h = api.get("/health").json()
    assert h["status"] == "ok" and h["models"]["anomaly-detector"]
    assert "routes" in api.get("/metrics").json()


def test_auth_errors_use_error_envelope(api):
    r = api.get("/api/v1/anomaly/flags")
    assert r.status_code == 401 and r.json()["error"]["code"] == "unauthorized"
    r = api.post("/api/v1/auth/login", json={"username": "buyer", "password": "nope"})
    assert r.status_code == 401
    r = api.get("/api/v1/anomaly/flags", headers={"Authorization": "Bearer not-a-jwt"})
    assert r.status_code == 401


def test_role_permissions(api):
    planner = _token(api, "planner")
    assert api.get("/api/v1/anomaly/flags", headers=planner).status_code == 403
    assert api.get("/api/v1/forecast/reorder-risk", headers=planner).status_code == 200


def test_anomaly_endpoints(api):
    mgr = _token(api, "manager")
    flags = api.get("/api/v1/anomaly/flags?risk=high", headers=mgr).json()
    assert flags and all(f["risk_level"] == "High" and f["reasons"] for f in flags)
    fid = flags[0]["id"]
    r = api.post(f"/api/v1/anomaly/flags/{fid}/review", headers=mgr, json={"verdict": "true_issue", "comment": "ok"})
    assert r.status_code == 200 and r.json()["reviewer"] == "manager"
    buyer = _token(api, "buyer")
    assert (
        api.post(f"/api/v1/anomaly/flags/{fid}/review", headers=buyer, json={"verdict": "true_issue"}).status_code
        == 403
    )
    bad = api.post(f"/api/v1/anomaly/flags/{fid}/review", headers=mgr, json={"verdict": "maybe"})
    assert bad.status_code == 422 and bad.json()["error"]["code"] == "validation_error"
    first = flags[0]
    s = api.post(
        "/api/v1/anomaly/score",
        headers=buyer,
        json={
            "lifnr": first["lifnr"],
            "matnr": first["matnr"],
            "werks": first["werks"],
            "menge": 10,
            "netpr": 100,
            "lead_time_days": 0,
            "created_at": "2026-06-28T23:00:00",
        },
    )
    assert s.status_code == 200 and s.json()["risk_level"] in ("Low", "Medium", "High")
    assert (
        api.post(
            "/api/v1/anomaly/score", headers=buyer, json={"lifnr": "1", "matnr": "999999", "menge": 1, "netpr": 1}
        ).status_code
        == 404
    )


def test_forecast_and_copilot(api):
    pl = _token(api, "planner")
    rr = api.get("/api/v1/forecast/reorder-risk", headers=pl).json()
    m = rr[0]
    f = api.get(f"/api/v1/forecast/{m['matnr']}?plant={m['werks']}&horizon=4", headers=pl)
    assert f.status_code == 200 and len(f.json()["forecast"]) == 4
    assert api.get("/api/v1/forecast/999999", headers=pl).status_code == 404
    a = api.post("/api/v1/copilot/ask", headers=pl, json={"question": "What is the SOP for urgent procurement?"})
    body = a.json()
    assert a.status_code == 200 and body["route"] == "process" and body["citations"] and body["latency_ms"] >= 0
    blocked = api.post("/api/v1/copilot/ask", headers=pl, json={"question": "DROP TABLE ekpo"}).json()
    assert blocked["blocked"]
