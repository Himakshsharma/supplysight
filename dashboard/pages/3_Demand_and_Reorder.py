"""Demand forecasts with P10–P90 bands and reorder-point recommendations."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import ACCENT, MUTED, client, page_setup, show_chart, show_df, style_fig

user = page_setup("Demand & Reorder", ":material/trending_up:")
rr = pd.DataFrame(client().reorder())
if rr.empty:
    st.warning(
        "No forecasts yet. Train the models with `python run.py train`, then reload this page.",
        icon=":material/model_training:",
    )
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("High stock-out risk", int((rr.stockout_risk == "High").sum()))
c2.metric("Medium stock-out risk", int((rr.stockout_risk == "Medium").sum()))
gap = rr.rop_gap_pct.dropna()
c3.metric("MINBE >20 % below recommended", int((gap > 20).sum()), help="SAP reorder point (MARC-MINBE) is stale")

rr["label"] = rr.matnr + " · " + rr.maktx + " · plant " + rr.werks
choice = st.selectbox("Material / plant", rr.label.tolist(), index=0)
row = rr[rr.label == choice].iloc[0]
horizon = st.slider("Horizon (weeks)", 4, 12, 8)
fc = client().forecast(row.matnr, row.werks, horizon)

h = pd.DataFrame(fc["history"])
f = pd.DataFrame(fc["forecast"])
fig = go.Figure()
if len(h):
    fig.add_scatter(
        x=h.week,
        y=h.qty,
        name="Actual consumption (MSEG 261)",
        mode="lines+markers",
        line=dict(color=MUTED, width=1.5),
        marker=dict(size=5),
    )
fig.add_scatter(
    x=list(f.week) + list(f.week[::-1]),
    y=list(f.p90) + list(f.p10[::-1]),
    fill="toself",
    fillcolor="rgba(37,99,235,0.15)",
    line=dict(width=0),
    name="P10–P90 band",
    hoverinfo="skip",
)
fig.add_scatter(x=f.week, y=f.yhat, name="Forecast (P50)", mode="lines+markers", line=dict(color=ACCENT, width=3))
fig.update_layout(
    title=f"{fc['maktx']} ({fc['matnr']}) · plant {fc['plant']} · weekly demand in {fc['unit']} "
    f"· model v{fc['model_version']}",
    yaxis_title=fc["unit"],
)
show_chart(style_fig(fig, 380))

rec = fc["reorder"]
if rec:
    a, b, c, d = st.columns(4)
    a.metric(
        "Recommended reorder point",
        f"{rec['recommended_rop']:,.1f}",
        f"{rec['rop_gap_pct']:+.0f}% vs MINBE" if rec["rop_gap_pct"] is not None else None,
        delta_color="off",
    )
    b.metric("Current MINBE (SAP)", f"{rec['current_minbe']:,.1f}")
    c.metric("Safety stock (95 %)", f"{rec['safety_stock']:,.1f}", f"σ {rec['sigma']:,.1f}/wk", delta_color="off")
    d.metric(
        "Stock + open POs",
        f"{rec['current_stock'] + rec['open_po_qty']:,.1f}",
        f"{rec['weeks_of_cover']:.1f} weeks cover · LT {rec['lead_time_days']} d",
        delta_color="off",
    )
    st.caption("Safety stock = z·σ·√L, reorder point = mean weekly forecast × L + safety stock (SOP-MRP-001).")

st.subheader("All materials")
risk = st.multiselect("Stock-out risk", ["High", "Medium", "Low"], default=["High", "Medium"])
view = rr[rr.stockout_risk.isin(risk)] if risk else rr
cols = [
    "matnr",
    "maktx",
    "werks",
    "matkl",
    "stockout_risk",
    "avg_weekly_forecast",
    "lead_time_days",
    "safety_stock",
    "recommended_rop",
    "current_minbe",
    "rop_gap_pct",
    "current_stock",
    "open_po_qty",
    "weeks_of_cover",
]
if view.empty:
    st.info("No materials at the selected risk levels.", icon=":material/verified:")
else:
    show_df(view[cols].round(2), height=380)
csv = view.drop(columns="label").to_csv(index=False).encode()
try:
    st.download_button(
        "Download recommendations (CSV)",
        csv,
        "reorder_recommendations.csv",
        "text/csv",
        icon=":material/download:",
        disabled=view.empty,
    )
except TypeError:  # pragma: no cover - Streamlit without button icons
    st.download_button(
        "Download recommendations (CSV)", csv, "reorder_recommendations.csv", "text/csv", disabled=view.empty
    )
