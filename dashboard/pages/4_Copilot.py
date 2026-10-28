"""Procurement copilot chat with citations and the SQL used."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from common import client, page_setup, show_df

user = page_setup("Procurement Copilot", ":material/forum:")
st.caption(
    "Routes each question to **SQL** (purchasing data), **SOP retrieval** (process) or **insights** "
    "(anomaly / forecast outputs). Read-only, with citations. Offline mode works without an LLM key; "
    "set `GEMINI_API_KEY` for generated prose."
)

EXAMPLES = [
    "Which vendor supplies glycerin and what is the lead time?",
    "Top 5 vendors by spend last quarter",
    "What is the approval process for urgent POs?",
    "Show high-risk POs this month",
    "Which materials may stock out?",
    "What documents are required to onboard a new vendor?",
]
st.session_state.setdefault("chat", [])
if not st.session_state["chat"]:
    st.markdown("**Try a question**")
    cols = st.columns(3)
    for i, q in enumerate(EXAMPLES):
        if cols[i % 3].button(q, key=f"ex{i}"):
            st.session_state["pending"] = q


def render(ans: dict) -> None:
    """Render one copilot answer with its citations."""
    st.markdown(ans["answer"].split("\n\n|")[0] if ans["rows"] else ans["answer"])
    if ans["rows"]:
        show_df(pd.DataFrame(ans["rows"], columns=ans["columns"]))
    meta = f"route **{ans['route']}** · {ans['mode']} · {ans['latency_ms']:.0f} ms"
    if ans["blocked"]:
        meta += " · :material/shield: blocked by guardrails"
    st.caption(meta)
    if ans["citations"]:
        with st.expander("Citations" + (" & SQL" if ans.get("sql") else ""), icon=":material/format_quote:"):
            for c in ans["citations"]:
                if c["type"] == "sop":
                    st.markdown(f":material/description: **{c['sop']}** › {c['section']} (`docs/sops/{c['doc']}`)")
                else:
                    st.markdown(f":material/database: {c.get('source', 'SQL query')} · {c['row_count']} row(s)")
                    st.code(c["query"], language="sql")


for turn in st.session_state["chat"]:
    with st.chat_message("user"):
        st.markdown(turn["q"])
    with st.chat_message("assistant"):
        render(turn["a"])

q = st.chat_input("Ask about vendors, POs, prices, stock, SOPs…") or st.session_state.pop("pending", None)
if q:
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"), st.spinner("Looking that up…"):
        ans = client().ask(q, user["role"])
        render(ans)
    st.session_state["chat"].append({"q": q, "a": ans})
    if len(st.session_state["chat"]) == 1:
        st.rerun()  # hide the starter questions once the conversation has begun
