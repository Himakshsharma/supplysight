"""SupplySight dashboard – Overview page. Run: streamlit run dashboard/app.py"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from common import PALETTE, client, inr, page_setup, show_chart, show_df, style_fig

user = page_setup("Procurement Overview", "📊")
ov = client().overview()
st.caption(f"Data as of **{ov['as_of']}** · one company code, plants 1000 (Baddi) and 2000 (Vapi) · all data synthetic")

c1, c2, c3, c4, c5 = st.columns(5)
delta = (ov["spend_90d"] - ov["spend_prev_90d"]) / ov["spend_prev_90d"] if ov["spend_prev_90d"] else 0
c1.metric("Spend, last 90 days", inr(ov["spend_90d"]), f"{delta:+.1%} vs previous 90 days")
c2.metric("Open POs", f"{ov['open_pos']:,}", inr(ov["open_po_value"]) + " open value", delta_color="off")
c3.metric("High-risk POs (90 d)", ov["high_risk_90d"], f"{ov['medium_risk_90d']} medium", delta_color="off")
c4.metric("Stock-out risk", ov["stockout_high"], f"{ov['stockout_medium']} medium", delta_color="off")
c5.metric("Suspected duplicate vendors", ov["duplicate_vendor_pairs"])

left, right = st.columns([3, 2])
with left:
    m = pd.DataFrame(ov["monthly"])
    fig = go.Figure()
    fig.add_bar(x=m.month, y=m.spend / 1e5, name="Spend (₹ lakh)", marker_color=PALETTE[0])
    fl = pd.DataFrame(ov["flags_by_month"])
    if len(fl):
        fig.add_scatter(x=fl.month, y=fl.high, name="High-risk POs", yaxis="y2", mode="lines+markers",
                        line=dict(color="#c0392b", width=2))
    fig.update_layout(title="Monthly PO spend and high-risk flags",
                      yaxis=dict(title="₹ lakh"), yaxis2=dict(title="High-risk POs", overlaying="y", side="right",
                                                              showgrid=False, rangemode="tozero"))
    show_chart(style_fig(fig, 360))
with right:
    g = pd.DataFrame(ov["spend_by_group_90d"]).head(12)
    fig = go.Figure(go.Bar(x=g.spend[::-1] / 1e5, y=g.matkl[::-1], orientation="h", marker_color=PALETTE[1]))
    fig.update_layout(title="Spend by material group, last 90 days (₹ lakh)")
    show_chart(style_fig(fig, 360))

a, b = st.columns(2)
with a:
    st.subheader("Top flagged purchase orders")
    flags = pd.DataFrame(client().flags(risk="High", limit=8))
    if len(flags):
        flags["top reason"] = flags.reasons.map(lambda r: r[0] if r else "")
        show_df(flags[["ebeln", "bedat", "vendor", "maktx", "netwr", "score", "top reason"]].rename(
            columns={"netwr": "value ₹", "maktx": "material"}))
    if hasattr(st, "page_link"):
        st.page_link("pages/1_Anomaly_Review.py", label="Open the review queue →")
with b:
    st.subheader("Materials at stock-out risk")
    rr = pd.DataFrame(client().reorder("High,Medium")).head(8)
    if len(rr):
        show_df(rr[["matnr", "maktx", "werks", "stockout_risk", "weeks_of_cover", "lead_time_days",
                    "recommended_rop", "current_minbe"]].round(1))
    if hasattr(st, "page_link"):
        st.page_link("pages/3_Demand_and_Reorder.py", label="Plan reorders →")
