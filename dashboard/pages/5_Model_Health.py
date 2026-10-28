"""Model registry, evaluation metrics, drift summary and the latest drift report."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from common import NEUTRAL_SERIES, RISK_COLORS, client, num, page_setup, show_chart, show_df, style_fig
from supplysight.config import get_settings

user = page_setup("Model Health", ":material/monitor_heart:")
mh = client().model_health()
prod = mh["production"]

a, b = st.columns(2)
an, fc = prod.get("anomaly-detector") or {}, prod.get("demand-forecaster") or {}
with a:
    st.subheader(f"Anomaly detector · v{an.get('version', '–')}")
    m = an.get("metrics", {})
    x1, x2, x3 = st.columns(3)
    x1.metric(
        "Holdout PR-AUC",
        num(m.get("holdout_pr_auc"), ".3f"),
        f"rules only {num(m.get('holdout_rules_pr_auc'), '.3f')}",
        delta_color="off",
    )
    x2.metric("Precision@50", num(m.get("holdout_precision_at_50"), ".2f"))
    x3.metric("Flag rate", num(m.get("flag_rate"), ".1%"), help="Share of PO items flagged Medium or High")
with b:
    st.subheader(f"Demand forecaster · v{fc.get('version', '–')}")
    m = fc.get("metrics", {})
    y1, y2, y3 = st.columns(3)
    y1.metric(
        "Backtest WMAPE",
        num(m.get("gbm_wmape"), ".1%"),
        f"best baseline {num(m.get('best_baseline_wmape'), '.1%')}",
        delta_color="off",
        help="Weighted mean absolute percentage error – lower is better",
    )
    y2.metric(
        "P10–P90 coverage",
        num(m.get("interval_coverage_calibrated"), ".0%"),
        help="Share of actuals inside the band; 80 % is the target",
    )
    y3.metric(
        "Bias", num(m.get("gbm_bias"), "+.1%"), help="Positive = the model over-forecasts on average; near 0 is best"
    )

rep = get_settings().reports_dir
t1, t2, t3, t4 = st.tabs(["Drift", "Anomaly evaluation", "Forecast backtest", "Registry & runs"])
with t1:
    d = mh.get("drift")
    if not d:
        st.info(
            "No drift report yet. Run `python scripts/simulate_drift.py` then `python scripts/retrain.py --check`.",
            icon=":material/query_stats:",
        )
    else:
        st.markdown(f"Reference **{d['reference_period']}** vs current **{d['current_period']}**")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Features drifted", num(d["share_drifted_features"], ".0%"))
        c2.metric("Anomaly-score PSI", num(d["anomaly_score_psi"], ".2f"), help="Above 0.2 indicates a material shift")
        c3.metric(
            "Flag rate", num(d["flag_rate_cur"], ".1%"), f"ref {num(d['flag_rate_ref'], '.1%')}", delta_color="off"
        )
        c4.metric(
            "Live forecast WMAPE",
            num(d["forecast_live_wmape"], ".1%"),
            f"backtest {num(d['forecast_backtest_wmape'], '.1%')}",
            delta_color="off",
        )
        if d["retrain_recommended"]:
            st.error("Retraining recommended: " + "; ".join(d["reasons"]), icon=":material/warning:")
        else:
            st.success("No retraining needed.", icon=":material/check_circle:")
        dec = mh.get("retrain_decision")
        if dec:
            st.markdown("**Last champion / challenger decision**")
            show_df(pd.DataFrame([{"model": k, **v} for k, v in dec.items() if isinstance(v, dict)]))
        feats = rep / "drift_features.csv"
        if feats.exists():
            ft = pd.read_csv(feats)
            fig = go.Figure(
                go.Bar(
                    x=ft.psi,
                    y=ft.feature,
                    orientation="h",
                    marker_color=[RISK_COLORS["High"] if x else NEUTRAL_SERIES for x in ft.drifted],
                )
            )
            fig.add_vline(
                x=0.2,
                line_dash="dash",
                line_color=RISK_COLORS["High"],
                annotation_text="PSI 0.2 threshold",
                annotation_position="top",
            )
            fig.update_layout(title="PSI per PO feature (red = drifted)", yaxis=dict(autorange="reversed"))
            show_chart(style_fig(fig, 420))
        html = rep / "drift_report.html"
        if html.exists() and st.toggle("Show full drift report"):
            components.html(html.read_text(encoding="utf-8"), height=900, scrolling=True)
with t2:
    shown = False
    for name, title in (
        ("anomaly_holdout.csv", "Holdout (last 30 % of POs) – rules vs ML vs ensemble"),
        ("anomaly_per_type_holdout.csv", "Recall of flagged (Medium/High) POs by anomaly type"),
    ):
        p = rep / name
        if p.exists():
            shown = True
            st.markdown(f"**{title}**")
            show_df(pd.read_csv(p).round(3))
    if not shown:
        st.info("No anomaly evaluation yet. Run `python run.py eval`.", icon=":material/query_stats:")
with t3:
    shown = False
    for name, title in (
        ("forecast_backtest_overall.csv", "Rolling-origin backtest (6 origins × 8 weeks)"),
        ("forecast_backtest_by_zone.csv", "Inside vs beyond the frozen production plan"),
    ):
        p = rep / name
        if p.exists():
            shown = True
            st.markdown(f"**{title}**")
            show_df(pd.read_csv(p).round(3))
    img = rep / "forecast_vs_actual.png"
    if img.exists():
        shown = True
        st.image(str(img), caption="Forecast vs actual for three sample materials")
    if not shown:
        st.info("No forecast backtest yet. Run `python run.py eval`.", icon=":material/query_stats:")
with t4:
    show_df(
        pd.DataFrame(
            [
                {
                    **{k: v for k, v in r.items() if k != "metrics"},
                    **{k: round(v, 3) for k, v in r["metrics"].items() if isinstance(v, float)},
                }
                for r in mh["registry"]
            ]
        )
    )
    runs = pd.DataFrame(mh["runs"])
    if len(runs):
        runs["metric"] = runs.metrics.map(lambda m: m.get("holdout_pr_auc", m.get("gbm_wmape")))
        fig = go.Figure()
        for exp, g in runs.groupby("experiment"):
            fig.add_scatter(x=g.ended_at, y=g.metric, mode="lines+markers", name=exp)
        fig.update_layout(title="Key metric per training run (anomaly PR-AUC / forecast WMAPE)")
        show_chart(style_fig(fig, 300))
