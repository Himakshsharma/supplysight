"""Minimal stand-ins for streamlit / plotly so dashboard pages can be smoke-tested where
those packages are not installed. Widgets return their defaults; layout helpers return
context managers. (When streamlit *is* installed, tests use streamlit.testing.AppTest.)"""

from __future__ import annotations

import sys
import types
from datetime import date


class StopPage(Exception):
    """Raised by st.stop()."""


class Anything:
    """Accepts any attribute access / call / context use."""

    def __init__(self, *a, **k):
        pass

    def __getattr__(self, name):
        return Anything()

    def __call__(self, *a, **k):
        return Anything()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __iter__(self):
        return iter([])

    def __bool__(self):
        return False


class SessionState(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as exc:
            raise AttributeError(k) from exc

    def __setattr__(self, k, v):
        self[k] = v


def make_streamlit(session: dict) -> types.ModuleType:
    st = types.ModuleType("streamlit")
    st.session_state = SessionState(session)
    calls: list[str] = []
    st._calls = calls

    def rec(name):
        def f(*a, **k):
            calls.append(name)
            return Anything()
        return f

    for n in ("set_page_config", "markdown", "caption", "title", "subheader", "metric", "plotly_chart", "dataframe",
              "info", "success", "warning", "error", "divider", "code", "image", "write", "download_button",
              "page_link", "rerun", "json"):
        setattr(st, n, rec(n))
    st.stop = lambda: (_ for _ in ()).throw(StopPage())
    st.columns = lambda spec, **k: [Anything() for _ in range(spec if isinstance(spec, int) else len(spec))]
    st.tabs = lambda names: [Anything() for _ in names]
    st.expander = lambda *a, **k: Anything()
    st.container = lambda *a, **k: Anything()
    st.form = lambda *a, **k: Anything()
    st.spinner = lambda *a, **k: Anything()
    st.chat_message = lambda *a, **k: Anything()
    st.sidebar = Anything()
    st.button = lambda *a, **k: False
    st.form_submit_button = lambda *a, **k: False
    st.toggle = lambda *a, **k: False
    st.chat_input = lambda *a, **k: "Which vendor supplies glycerin and what is the lead time?"
    st.selectbox = lambda label, options, index=0, **k: list(options)[index] if len(list(options)) else None
    st.multiselect = lambda label, options, default=None, **k: default or []
    st.text_input = lambda label, value="", **k: value
    st.number_input = lambda label, min_value=None, max_value=None, value=None, *a, **k: value
    st.date_input = lambda label, value=None, **k: value or date(2026, 1, 1)
    st.slider = lambda label, min_value=None, max_value=None, value=None, *a, **k: value
    st.cache_resource = lambda *a, **k: (a[0] if a and callable(a[0]) else (lambda f: f))
    st.cache_data = st.cache_resource
    cc = types.SimpleNamespace(ProgressColumn=lambda *a, **k: None, NumberColumn=lambda *a, **k: None)
    st.column_config = cc
    comps = types.ModuleType("streamlit.components")
    v1 = types.ModuleType("streamlit.components.v1")
    v1.html = rec("html")
    comps.v1 = v1
    st.components = comps
    sys.modules["streamlit.components"] = comps
    sys.modules["streamlit.components.v1"] = v1
    return st


def make_plotly() -> None:
    plotly = types.ModuleType("plotly")
    go = types.ModuleType("plotly.graph_objects")
    go.Figure = Anything
    go.Bar = Anything
    go.Scatter = Anything
    plotly.graph_objects = go
    sys.modules.setdefault("plotly", plotly)
    sys.modules.setdefault("plotly.graph_objects", go)
