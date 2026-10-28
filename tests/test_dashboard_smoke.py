"""Every dashboard page renders against the trained test database without raising."""

from __future__ import annotations

import importlib.util
import runpy
import sys
from pathlib import Path

import pytest

DASH = Path(__file__).resolve().parents[1] / "dashboard"
PAGES = [DASH / "app.py"] + sorted((DASH / "pages").glob("*.py"))
USER = {"username": "manager", "role": "manager", "name": "Test Manager"}


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.stem)
def test_page_renders(page, trained, monkeypatch):
    if importlib.util.find_spec("streamlit") is not None:  # pragma: no cover - real Streamlit available
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(str(page), default_timeout=60)
        at.session_state["user"] = USER
        at.run()
        assert not at.exception, at.exception
        return
    from tests import fake_ui

    st = fake_ui.make_streamlit({"user": USER})
    monkeypatch.setitem(sys.modules, "streamlit", st)
    fake_ui.make_plotly()
    monkeypatch.syspath_prepend(str(DASH))
    sys.modules.pop("common", None)
    try:
        runpy.run_path(str(page), run_name="__main__")
    except fake_ui.StopPage:
        pass
    assert "set_page_config" in st._calls
    assert any(c in st._calls for c in ("dataframe", "metric", "plotly_chart", "markdown"))
