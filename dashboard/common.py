"""Shared dashboard helpers: data client (direct services or REST API), login, formatting."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import requests
import streamlit as st

# ------------------------------------------------------------------ design tokens
# Single source of truth for colour. Mirrored in .streamlit/config.toml and DESIGN.md – change all three together.
INK = "#0f172a"  # primary text
MUTED = "#475569"  # secondary text, axis labels (7.5:1 on white)
LINE = "#e2e8f0"  # gridlines, dividers
SURFACE_2 = "#f1f5f9"  # sidebar / secondary panels
ACCENT = "#2563eb"  # primary action, current selection, the forecast line
NEUTRAL_SERIES = "#94a3b8"  # de-emphasised marks (non-drifted features, history)
# Risk colours carry meaning only – never decoration. Each passes 4.5:1 with white text.
RISK_COLORS = {"High": "#b91c1c", "Medium": "#b45309", "Low": "#15803d"}
SERIES = [ACCENT, "#0d9488", "#7c3aed", "#64748b"]  # categorical chart series, in order
PALETTE = SERIES  # backwards-compatible alias
FONT = '"Source Sans Pro", "Source Sans 3", system-ui, sans-serif'  # Streamlit's UI face, reused in charts

# Human labels for SAP field names shown in tables. Technical names stay in the data layer.
LABELS = {
    "id": "Flag",
    "ebeln": "PO",
    "ebelp": "Item",
    "bedat": "PO date",
    "vendor": "Vendor",
    "lifnr": "Vendor no.",
    "maktx": "Material",
    "matnr": "Material no.",
    "werks": "Plant",
    "matkl": "Material group",
    "netwr": "Value (₹)",
    "score": "Risk score",
    "risk_level": "Risk",
    "top reason": "Top reason",
    "verdict": "Verdict",
    "stockout_risk": "Stock-out risk",
    "weeks_of_cover": "Weeks of cover",
    "lead_time_days": "Lead time (days)",
    "recommended_rop": "Recommended ROP",
    "current_minbe": "SAP MINBE",
    "avg_weekly_forecast": "Avg weekly forecast",
    "safety_stock": "Safety stock",
    "rop_gap_pct": "ROP gap %",
    "current_stock": "Stock on hand",
    "open_po_qty": "Open PO qty",
}

_CSS = f"""
<style>
[data-testid="stMetricValue"] {{ font-size: 1.6rem; font-variant-numeric: tabular-nums; letter-spacing: -0.01em; }}
[data-testid="stMetricLabel"] p {{ color: {MUTED}; }}
h1 {{ letter-spacing: -0.02em; }}
::selection {{ background: rgba(37, 99, 235, 0.18); color: {INK}; }}
a:focus-visible, button:focus-visible, [role="tab"]:focus-visible, summary:focus-visible {{
  outline: 2px solid {ACCENT}; outline-offset: 2px; border-radius: 4px; }}
.ss-wordmark {{ font-size: 1.15rem; font-weight: 700; letter-spacing: -0.01em; color: {INK}; margin: 0; }}
.ss-badge {{ display: inline-block; color: #fff; padding: 1px 8px; border-radius: 999px; font-size: 0.8rem;
  font-weight: 600; line-height: 1.5; vertical-align: 0.1em; }}
</style>
"""


def inr(x: float | None, short: bool = True) -> str:
    """₹ formatting with lakh/crore units."""
    if x is None:
        return "–"
    if short and abs(x) >= 1e7:
        return f"₹{x / 1e7:,.2f} Cr"
    if short and abs(x) >= 1e5:
        return f"₹{x / 1e5:,.1f} L"
    return f"₹{x:,.0f}"


def num(x, spec: str) -> str:
    """Format a metric; missing or NaN values show an en dash instead of 'nan'."""
    try:
        if x is None or x != x:  # NaN check
            return "–"
        return format(x, spec)
    except (TypeError, ValueError):
        return "–"


def badge(level: str) -> str:
    """Risk badge (HTML). The level is always written out, so colour is never the only signal."""
    c = RISK_COLORS.get(level, MUTED)
    return f"<span class='ss-badge' style='background:{c}'>{level}</span>"


def show_chart(fig) -> None:
    """Plotly chart that stretches to the container on old and new Streamlit versions."""
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:  # pragma: no cover - Streamlit < 1.46
        st.plotly_chart(fig, use_container_width=True)


def show_df(df: pd.DataFrame, **kw) -> None:
    """Dataframe with human column labels that stretches to the container on old and new Streamlit versions.

    column_config keys must use the labelled names (e.g. "Risk score", not "score")."""
    df = df.rename(columns={k: v for k, v in LABELS.items() if k in df.columns})
    try:
        st.dataframe(df, width="stretch", hide_index=True, **kw)
    except TypeError:  # pragma: no cover
        st.dataframe(df, use_container_width=True, hide_index=True, **kw)


def submit(label: str, icon: str | None = None, **kw) -> bool:
    """Form submit button with a Material icon, falling back to no icon on older Streamlit."""
    try:
        return st.form_submit_button(label, icon=icon, **kw)
    except TypeError:  # pragma: no cover
        return st.form_submit_button(label, **kw)


def style_fig(fig, height: int = 320):
    """Consistent, quiet Plotly styling that matches the Streamlit theme."""
    axis = dict(
        gridcolor=LINE,
        zerolinecolor=LINE,
        linecolor=LINE,
        tickfont=dict(color=MUTED),
        title=dict(font=dict(color=MUTED)),
    )
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=44, b=10),
        template="plotly_white",
        colorway=SERIES,
        font=dict(family=FONT, size=12, color=INK),
        title=dict(x=0, xanchor="left", font=dict(size=14, color=INK)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(color=MUTED)),
        hoverlabel=dict(font=dict(family=FONT), bgcolor="white", bordercolor=LINE),
        separators=".,",
    )
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
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
        r = requests.get(
            self.base + path, headers=self._h(), params={k: v for k, v in params.items() if v is not None}, timeout=60
        )
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
        return self._get(
            "/anomaly/flags", risk=risk, **{"from": date_from, "to": date_to}, status=status, limit=limit, offset=offset
        )

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


@st.cache_resource(show_spinner=False)
def _ensure_demo_state() -> bool:
    """On a fresh deployment, build the synthetic data and train the models once per process."""
    from supplysight.bootstrap import ensure_demo_state, is_ready

    if is_ready():
        return False
    with st.spinner("First start: generating the synthetic SAP dataset and training the models (about a minute)…"):
        return ensure_demo_state()


def _login_form() -> dict:
    """Sign-in form in the main area (visible on phones, where the sidebar starts collapsed)."""
    st.title("SupplySight")
    st.markdown("AI-powered procurement and demand intelligence for SAP MM/PP. **All data is synthetic.**")
    left, _ = st.columns([2, 3])
    with left, st.form("login"):
        u = st.selectbox(
            "Demo user",
            ["manager", "buyer", "planner", "admin"],
            help="manager can review flags · buyer scores POs · planner sees forecasts · admin sees GSTINs",
        )
        p = st.text_input("Password", value=f"{u}123", type="password", help="Demo passwords are <username>123")
        if submit("Sign in", icon=":material/login:", type="primary"):
            user = client().login(u, p)
            if user:
                st.session_state["user"] = user
                st.rerun()
            st.error("That password doesn't match. Demo passwords are the username followed by 123.")
    st.stop()


def page_setup(title: str, icon: str = ":material/inventory_2:") -> dict:
    """Page config, first-start bootstrap and sign-in. Returns the current user dict. `icon` is a Material Symbols shortcode."""
    st.set_page_config(page_title=f"SupplySight · {title}", page_icon=icon, layout="wide")
    st.markdown(_CSS, unsafe_allow_html=True)
    if client().mode == "direct":
        _ensure_demo_state()
    with st.sidebar:
        st.markdown("<p class='ss-wordmark'>SupplySight</p>", unsafe_allow_html=True)
        st.caption("AI procurement & demand intelligence for SAP MM/PP · **synthetic data only**")
    user = st.session_state.get("user") or _login_form()
    with st.sidebar:
        st.markdown(f"**{user['name']}**")
        st.caption(f"Role: **{user['role']}** · data via {client().mode}")
        if st.button("Sign out", icon=":material/logout:"):
            st.session_state.pop("user", None)
            st.session_state.pop("token", None)
            st.rerun()
    st.title(title)
    return user
