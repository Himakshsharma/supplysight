"""Shared dashboard helpers: data client (direct services or REST API), login, formatting."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
import requests  # noqa: E402
import streamlit as st  # noqa: E402

RISK_COLORS = {"High": "#c0392b", "Medium": "#d68910", "Low": "#1e8449"}
PALETTE = ["#2563eb", "#0d9488", "#d97706", "#7c3aed", "#db2777", "#64748b"]


def inr(x: float | None, short: bool = True) -> str:
    """₹ formatting with lakh/crore units."""
    if x is None:
        return "–"
    if short and abs(x) >= 1e7:
        return f"₹{x / 1e7:,.2f} Cr"
    if short and abs(x) >= 1e5:
        return f"₹{x / 1e5:,.1f} L"
    return f"₹{x:,.0f}"


def badge(level: str) -> str:
    """Coloured risk badge (HTML)."""
    c = RISK_COLORS.get(level, "#64748b")
    return (f"<span style='background:{c};color:white;padding:2px 8px;border-radius:10px;"
            f"font-size:0.8rem;font-weight:600'>{level}</span>")


def show_chart(fig) -> None:
    """Plotly chart that stretches to the container on old and new Streamlit versions."""
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:  # pragma: no cover - Streamlit < 1.46
        st.plotly_chart(fig, use_container_width=True)


def show_df(df: pd.DataFrame, **kw) -> None:
    """Dataframe that stretches to the container on old and new Streamlit versions."""
    try:
        st.dataframe(df, width="stretch", hide_index=True, **kw)
    except TypeError:  # pragma: no cover
        st.dataframe(df, use_container_width=True, hide_index=True, **kw)


def style_fig(fig, height: int = 320):
    """Consistent, quiet Plotly styling."""
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10), template="plotly_white",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), font=dict(size=12))
    return fig


# ------------------------------------------------------------------ data client
class DirectClient:
    """Calls the service layer in-process (default; no API server needed)."""

    mode = "direct"

    def __init__(self) -> None:
        from supplysight.copilot.agent import Copilot
        from supplysight.services import auth, core

        self.core, self.auth, self.bot = core, auth, Copilot()

    def login(self, username: str, password: str) -> dict | None:
        return self.auth.authenticate(username, password)

    def overview(self) -> dict:
        return self.core.overview()

    def flags(self, **kw) -> list[dict]:
        return self.core.list_flags(**kw)

    def review(self, flag_id: str, verdict: str, comment: str, reviewer: str) -> dict:
        return self.core.review_flag(flag_id, verdict, comment, reviewer)

    def score(self, payload: dict) -> dict:
        return self.core.score_po(payload)

    def duplicates(self) -> list[dict]:
        return self.core.duplicate_vendors()

    def forecast(self, matnr: str, plant: str | None, horizon: int) -> dict:
        return self.core.get_forecast(matnr, plant, horizon)

    def reorder(self, risk: str | None = None) -> list[dict]:
        return self.core.reorder_risk(risk)

    def ask(self, question: str, role: str) -> dict:
        return self.bot.ask(question, role).to_dict()

    def model_health(self) -> dict:
        return self.core.model_health()


class APIClient:
    """Calls the FastAPI service (set SUPPLYSIGHT_API_URL=http://api:8000)."""

    mode = "api"

    def __init__(self, base: str) -> None:
        self.base = base.rstrip("/") + "/api/v1"

    def _h(self) -> dict:
        tok = st.session_state.get("token")
        return {"Authorization": f"Bearer {tok}"} if tok else {}

    def _get(self, path: str, **params):
        r = requests.get(self.base + path, headers=self._h(), params={k: v for k, v in params.items() if v is not None},
                         timeout=60)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, body: dict):
        r = requests.post(self.base + path, headers=self._h(), json=body, timeout=60)
        r.raise_for_status()
        return r.json()

    def login(self, username: str, password: str) -> dict | None:
        r = requests.post(self.base + "/auth/login", json={"username": username, "password": password}, timeout=30)
        if r.status_code != 200:
            return None
        d = r.json()
        st.session_state["token"] = d["access_token"]
        return {"username": username, "role": d["role"], "name": d["name"]}

    def overview(self) -> dict:
        return self._get("/overview")

    def flags(self, risk=None, date_from=None, date_to=None, status=None, limit=200, offset=0) -> list[dict]:
        return self._get("/anomaly/flags", risk=risk, **{"from": date_from, "to": date_to}, status=status,
                         limit=limit, offset=offset)

    def review(self, flag_id: str, verdict: str, comment: str, reviewer: str) -> dict:
        return self._post(f"/anomaly/flags/{flag_id}/review", {"verdict": verdict, "comment": comment})

    def score(self, payload: dict) -> dict:
        return self._post("/anomaly/score", payload)

    def duplicates(self) -> list[dict]:
        return self._get("/anomaly/duplicate-vendors")

    def forecast(self, matnr: str, plant: str | None, horizon: int) -> dict:
        return self._get(f"/forecast/{matnr}", plant=plant, horizon=horizon)

    def reorder(self, risk: str | None = None) -> list[dict]:
        return self._get("/forecast/reorder-risk", risk=risk)

    def ask(self, question: str, role: str) -> dict:
        return self._post("/copilot/ask", {"question": question})

    def model_health(self) -> dict:
        return self._get("/models")


@st.cache_resource(show_spinner=False)
def _client(api_url: str):
    return APIClient(api_url) if api_url else DirectClient()


def client():
    """Data client chosen by the SUPPLYSIGHT_API_URL environment variable."""
    return _client(os.environ.get("SUPPLYSIGHT_API_URL", ""))


def page_setup(title: str, icon: str = "📦") -> dict:
    """Page config + sidebar login. Returns the current user dict."""
    st.set_page_config(page_title=f"SupplySight · {title}", page_icon=icon, layout="wide")
    st.markdown("<style>div[data-testid='stMetricValue']{font-size:1.6rem}</style>", unsafe_allow_html=True)
    with st.sidebar:
        st.markdown("### 📦 SupplySight")
        st.caption("AI procurement & demand intelligence for SAP MM/PP · **synthetic data only**")
        user = st.session_state.get("user")
        if user is None:
            with st.form("login"):
                u = st.selectbox("Demo user", ["manager", "buyer", "planner", "admin"])
                p = st.text_input("Password", value=f"{u}123", type="password",
                                  help="Demo passwords are <username>123")
                if st.form_submit_button("Sign in"):
                    user = client().login(u, p)
                    if user:
                        st.session_state["user"] = user
                        st.rerun()
                    else:
                        st.error("Invalid credentials")
            st.stop()
        st.success(f"{user['name']}")
        st.caption(f"Role: **{user['role']}** · data via {client().mode}")
        if st.button("Sign out"):
            st.session_state.pop("user", None)
            st.session_state.pop("token", None)
            st.rerun()
    st.title(f"{icon} {title}")
    return user
