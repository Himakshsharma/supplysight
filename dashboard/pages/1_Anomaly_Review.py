"""Anomaly review queue: filter flags, read reasons, mark true issue / false positive, score a new PO."""

from __future__ import annotations

from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from common import RISK_COLORS, badge, client, inr, page_setup, show_chart, show_df, style_fig

user = page_setup("Anomaly Review Queue", "🚨")
can_review = user["role"] in ("manager", "admin")

with st.expander("ℹ️ How flags are produced", expanded=False):
    st.markdown(
        "Each PO item is scored by an **ensemble**: deterministic rules (split POs, after-hours + new vendor, impossible "
        "lead time, duplicate vendor), an **Isolation Forest** and an **autoencoder** on 15 engineered features. "
        "High ≈ top 2.5 % of scores, Medium ≈ next 5 %. Every flag lists up to three human-readable reasons. "
        "Your verdicts are stored and used as labels at the next retraining (SOP-AUD-001).")

f1, f2, f3, f4 = st.columns([1.2, 1.4, 1, 1.4])
risk = f1.multiselect("Risk level", ["High", "Medium"], default=["High", "Medium"])
d_from = f2.date_input("PO date from", value=date(2026, 4, 1))
status = f3.selectbox("Status", ["open", "reviewed", "all"], index=0)
search = f4.text_input("Search vendor / material / PO")

rows = client().flags(risk=",".join(risk) or "High", date_from=str(d_from), status=None if status == "all" else status,
                      limit=500)
df = pd.DataFrame(rows)
if search and len(df):
    s = search.lower()
    df = df[df.vendor.str.lower().str.contains(s) | df.maktx.str.lower().str.contains(s) | df.ebeln.str.contains(s)]
st.caption(f"{len(df)} flagged PO items")
if df.empty:
    st.info("No flags match the filters.")
    st.stop()

df["top reason"] = df.reasons.map(lambda r: r[0] if r else "")
show_df(df[["id", "bedat", "risk_level", "score", "vendor", "maktx", "werks", "netwr", "top reason", "verdict"]].rename(
    columns={"maktx": "material", "netwr": "value ₹", "risk_level": "risk"}),
    column_config={"score": st.column_config.ProgressColumn("score", min_value=0.0, max_value=1.0, format="%.2f")},
    height=320)

st.divider()
sel = st.selectbox("Open a flag", df.id.tolist(),
                   format_func=lambda i: f"{i} · {df.set_index('id').loc[i, 'vendor']} · {df.set_index('id').loc[i, 'risk_level']}")
r = df.set_index("id").loc[sel]
c1, c2 = st.columns([3, 2])
with c1:
    st.markdown(f"#### PO {r.ebeln} / item {r.ebelp} &nbsp; {badge(r.risk_level)}", unsafe_allow_html=True)
    st.markdown(f"**Vendor:** {r.vendor} ({r.lifnr}) &nbsp;·&nbsp; **Material:** {r.maktx} ({r.matnr}) &nbsp;·&nbsp; "
                f"**Plant:** {r.werks} &nbsp;·&nbsp; **Value:** {inr(r.netwr, short=False)} &nbsp;·&nbsp; **PO date:** {r.bedat}")
    st.markdown("**Why it was flagged**")
    for k, reason in enumerate(r.reasons, 1):
        st.markdown(f"{k}. {reason}")
    if r.verdict:
        st.info(f"Reviewed: **{r.verdict.replace('_', ' ')}** by {r.reviewer} at {r.reviewed_at} — {r.comment or ''}")
with c2:
    comp = {"Rules": r.rule_score, "Isolation Forest (percentile)": r.iforest_score, "Autoencoder (percentile)": r.ae_score}
    fig = go.Figure(go.Bar(x=list(comp.values()), y=list(comp.keys()), orientation="h",
                           marker_color=["#7c3aed", "#2563eb", "#0d9488"], text=[f"{v:.2f}" for v in comp.values()]))
    fig.update_layout(title=f"Component scores · ensemble {r.score:.3f}", xaxis=dict(range=[0, 1]))
    show_chart(style_fig(fig, 220))

with st.form("review"):
    comment = st.text_input("Comment", placeholder="e.g. confirmed with buyer – emergency order for production order 1000123")
    b1, b2, _ = st.columns([1, 1, 3])
    t = b1.form_submit_button("✅ True issue", disabled=not can_review)
    fp = b2.form_submit_button("❎ False positive", disabled=not can_review)
    if t or fp:
        res = client().review(sel, "true_issue" if t else "false_positive", comment, user["username"])
        st.success(f"Saved verdict for {res['id']}: {res['verdict']}")
        st.rerun()
if not can_review:
    st.caption("Only the **manager** and **admin** roles can record verdicts.")

st.divider()
with st.expander("🧪 Score a new PO before release"):
    with st.form("score"):
        a, b, c = st.columns(3)
        lifnr = a.text_input("Vendor (LIFNR)", "500014")
        matnr = b.text_input("Material (MATNR)", "100001")
        werks = c.selectbox("Plant", ["1000", "2000"])
        menge = a.number_input("Quantity", 1.0, 1e7, 120.0)
        netpr = b.number_input("Unit price ₹", 0.01, 1e7, 125.0)
        lt = c.number_input("Lead time (days)", 0, 365, 30)
        created = a.text_input("Created at", "2026-10-01T11:00:00")
        buyer = b.text_input("Buyer", "RSHARMA")
        if st.form_submit_button("Score"):
            try:
                out = client().score({"lifnr": lifnr, "matnr": matnr, "werks": werks, "menge": menge, "netpr": netpr,
                                      "lead_time_days": int(lt), "created_at": created, "created_by": buyer})
                st.markdown(f"Risk {badge(out['risk_level'])} &nbsp; score **{out['score']:.3f}** "
                            f"(model v{out['model_version']})", unsafe_allow_html=True)
                for reason in out["reasons"] or ["No significant risk signals."]:
                    st.markdown(f"- {reason}")
            except Exception as exc:
                st.error(f"Could not score: {exc}")

_ = RISK_COLORS  # keep import used for linters
