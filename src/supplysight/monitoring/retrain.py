"""Drift check + champion/challenger retraining.

1. Compare the reference period with the newest period (``data/drift``):
   PO-feature drift, anomaly-score (prediction) drift, demand drift, and *live* forecast
   accuracy of the production forecaster against the new actuals.
2. Retrain when drift crosses thresholds (or ``force``): the challenger is trained on
   reference + new data (+ analyst feedback) and registered in *Staging*.
3. Champion and challenger are compared on the newest period; the challenger is promoted
   to *Production* only if it is better.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from supplysight.config import get_settings
from supplysight.data import load_source_tables, merge_periods, read_csv_tables, weekly_consumption
from supplysight.db import load_tables, read_table, table_exists, write_table
from supplysight.features.po_features import ML_FEATURES
from supplysight.mlops import Registry
from supplysight.monitoring.drift import PSI_ALERT, drift_table, evidently_report, psi, save_summary, write_html_report
from supplysight.pipeline import (
    ANOMALY_MODEL,
    FORECAST_MODEL,
    _labels,
    _load_feedback,
    anomaly_features,
    publish_anomaly,
    publish_forecast,
    train_anomaly,
    train_forecast,
)
from supplysight.utils import get_logger

log = get_logger(__name__)
DEFAULT_THRESHOLDS = {"share_drifted": 0.3, "score_psi": PSI_ALERT, "forecast_degradation": 1.25}


def drift_dir() -> Path:
    """Folder holding the newest period produced by scripts/simulate_drift.py."""
    return get_settings().data_dir / "drift"


def _live_forecast_wmape(cur: dict[str, pd.DataFrame]) -> float:
    """Accuracy of the stored production forecast against the new period's actual consumption."""
    if not table_exists("demand_forecast"):
        return float("nan")
    fc = read_table("demand_forecast")
    fc["week"] = pd.to_datetime(fc.week)
    act = weekly_consumption(cur["mseg"])
    m = fc.merge(act, on=["matnr", "werks", "week"], how="inner")
    return float(np.abs(m.qty - m.yhat).sum() / max(m.qty.abs().sum(), 1e-9)) if len(m) else float("nan")


def drift_check(ref: dict[str, pd.DataFrame], cur: dict[str, pd.DataFrame], thresholds: dict | None = None) -> dict:
    """Compute drift metrics, write reports, and decide whether to retrain."""
    th = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    rep = get_settings().reports_dir
    rep.mkdir(parents=True, exist_ok=True)
    merged = merge_periods(ref, cur)
    f, _, _ = anomaly_features(merged)
    f["log_netwr"] = np.log1p(f.netwr)
    is_cur = f.ebeln.isin(set(cur["ekpo"].ebeln)).to_numpy()
    ref_start = f.loc[~is_cur, "created_at"].max() - pd.Timedelta(days=365)
    f_ref = f[~is_cur & (f.created_at >= ref_start).to_numpy()]
    f_cur = f[is_cur]
    feats = ML_FEATURES + ["log_netwr"]
    po_tab = drift_table(f_ref, f_cur, feats)

    reg = Registry()
    model, meta = reg.load(ANOMALY_MODEL)
    s_ref, s_cur = model.score(f_ref), model.score(f_cur)
    score_psi = psi(s_ref.score.to_numpy(), s_cur.score.to_numpy())
    pred_tab = drift_table(
        pd.DataFrame({"anomaly_score": s_ref.score}), pd.DataFrame({"anomaly_score": s_cur.score}), ["anomaly_score"]
    )

    # demand drift: weekly consumption scaled by each series' reference mean
    w_ref, w_cur = weekly_consumption(ref["mseg"]), weekly_consumption(cur["mseg"])
    scale = w_ref.groupby(["matnr", "werks"]).qty.mean().rename("scale")
    last13 = w_ref[w_ref.week > w_ref.week.max() - pd.Timedelta(weeks=26)]
    a = last13.join(scale, on=["matnr", "werks"])
    b = w_cur.join(scale, on=["matnr", "werks"]).dropna(subset=["scale"])
    dem_tab = drift_table(
        pd.DataFrame({"scaled_weekly_consumption": a.qty / a.scale.clip(lower=1e-9)}),
        pd.DataFrame({"scaled_weekly_consumption": b.qty / b.scale.clip(lower=1e-9)}),
        ["scaled_weekly_consumption"],
    )
    # unit price drift per material (ratio to the material's reference median)
    pm = f_ref.groupby("matnr").netpr.median().rename("ref_med")
    pr = pd.DataFrame({"price_index": (f_cur.netpr / f_cur.matnr.map(pm)).dropna()})
    pr_ref = pd.DataFrame({"price_index": (f_ref.netpr / f_ref.matnr.map(pm)).dropna()})
    price_tab = drift_table(pr_ref, pr, ["price_index"])

    fmeta = reg.get_version(FORECAST_MODEL, "Production") or {"metrics": {}}
    live = _live_forecast_wmape(cur)
    backtest_wmape = float(fmeta["metrics"].get("gbm_wmape", np.nan))
    degradation = live / backtest_wmape if backtest_wmape and np.isfinite(live) else float("nan")
    share = float(po_tab.drifted.mean())
    reasons = []
    if share >= th["share_drifted"]:
        reasons.append(f"{share:.0%} of PO features drifted (≥ {th['share_drifted']:.0%})")
    if score_psi >= th["score_psi"]:
        reasons.append(f"anomaly-score PSI {score_psi:.2f} (≥ {th['score_psi']})")
    if np.isfinite(degradation) and degradation >= th["forecast_degradation"]:
        reasons.append(f"live forecast WMAPE {live:.1%} is {degradation:.2f}× the backtest ({backtest_wmape:.1%})")
    if bool(dem_tab.drifted.iloc[0]):
        reasons.append(f"demand distribution drifted (PSI {dem_tab.psi.iloc[0]:.2f})")
    if bool(price_tab.drifted.iloc[0]):
        reasons.append(f"price level drifted (mean price index {price_tab.cur_mean.iloc[0]:.2f}×)")
    summary = {
        "reference_period": f"{f_ref.created_at.min():%Y-%m-%d} – {f_ref.created_at.max():%Y-%m-%d}",
        "current_period": f"{f_cur.created_at.min():%Y-%m-%d} – {f_cur.created_at.max():%Y-%m-%d}",
        "n_ref_pos": len(f_ref),
        "n_cur_pos": len(f_cur),
        "share_drifted_features": share,
        "drifted_features": ", ".join(po_tab[po_tab.drifted].feature),
        "anomaly_score_psi": score_psi,
        "flag_rate_ref": float((s_ref.risk_level != "Low").mean()),
        "flag_rate_cur": float((s_cur.risk_level != "Low").mean()),
        "demand_psi": float(dem_tab.psi.iloc[0]),
        "price_index_cur": float(price_tab.cur_mean.iloc[0]),
        "forecast_backtest_wmape": backtest_wmape,
        "forecast_live_wmape": live,
        "forecast_degradation": degradation,
        "anomaly_model_version": meta["version"],
        "forecast_model_version": fmeta.get("version"),
        "retrain_recommended": bool(reasons),
        "reasons": reasons,
    }
    write_html_report(
        {
            "PO features (anomaly model inputs)": (po_tab, f_ref, f_cur),
            "Prediction drift (anomaly score)": (
                pred_tab,
                pd.DataFrame({"anomaly_score": s_ref.score}),
                pd.DataFrame({"anomaly_score": s_cur.score}),
            ),
            "Demand (scaled weekly consumption)": (
                dem_tab,
                pd.DataFrame({"scaled_weekly_consumption": a.qty / a.scale.clip(lower=1e-9)}),
                pd.DataFrame({"scaled_weekly_consumption": b.qty / b.scale.clip(lower=1e-9)}),
            ),
            "Unit price index": (price_tab, pr_ref, pr),
        },
        summary,
        rep / "drift_report.html",
    )
    summary["evidently_report"] = evidently_report(f_ref[feats], f_cur[feats], rep / "drift_report_evidently.html")
    po_tab.to_csv(rep / "drift_features.csv", index=False)
    save_summary(summary, rep / "drift_summary.json")
    log.info("drift: share=%.2f score_psi=%.3f live_wmape=%.3f -> retrain=%s", share, score_psi, live, bool(reasons))
    return summary


def _pr_auc(model, f: pd.DataFrame, y: np.ndarray) -> float:
    return float(average_precision_score(y, model.score(f).score)) if y.sum() else float("nan")


def run_retrain(force: bool = False, thresholds: dict | None = None) -> dict:
    """Drift check -> (maybe) retrain -> champion/challenger comparison -> promotion."""
    cur = read_csv_tables(drift_dir())
    if not cur:
        raise FileNotFoundError(f"No new-period data in {drift_dir()} - run scripts/simulate_drift.py first")
    ref = load_source_tables()
    summary = drift_check(ref, cur, thresholds)
    result = {"drift": summary, "retrained": False}
    if not (summary["retrain_recommended"] or force):
        log.info("No retraining needed.")
        return result

    reg = Registry()
    champion_anom, champ_meta = reg.load(ANOMALY_MODEL)
    champion_fc, champ_fc_meta = reg.load(FORECAST_MODEL)
    live_wmape = summary["forecast_live_wmape"]

    merged = merge_periods(ref, cur)
    load_tables(merged)  # the database now holds history + new quarter
    feedback = _load_feedback()
    f, pairs, _ = anomaly_features(merged)
    is_cur = f.ebeln.isin(set(cur["ekpo"].ebeln)).to_numpy()
    y = _labels(f, merged, feedback).notna().astype(int).to_numpy()

    # ---- anomaly challenger
    res_a = train_anomaly(merged, promote="never", feedback=feedback, window_days=365)
    challenger, _ = reg.load(ANOMALY_MODEL, version=res_a["version"])
    champ_auc = _pr_auc(champion_anom, f[is_cur], y[is_cur])
    chall_auc = _pr_auc(challenger, f[is_cur], y[is_cur])
    promote_a = bool(np.nan_to_num(chall_auc) > np.nan_to_num(champ_auc))
    if promote_a:
        reg.transition(ANOMALY_MODEL, res_a["version"], "Production")
        publish_anomaly(challenger, f, pairs, res_a["version"])
    else:
        publish_anomaly(champion_anom, f, pairs, champ_meta["version"])

    # ---- forecast challenger (backtest on new history vs champion's live accuracy)
    res_f = train_forecast(merged, quick=True, promote="never")
    chall_wmape = res_f["metrics"]["gbm_wmape"]
    promote_f = bool(np.isfinite(live_wmape) and chall_wmape < live_wmape) or not np.isfinite(live_wmape)
    from supplysight.forecast.panel import build_panel

    panel = build_panel(merged)
    if promote_f:
        reg.transition(FORECAST_MODEL, res_f["version"], "Production")
        challenger_fc, _ = reg.load(FORECAST_MODEL, version=res_f["version"])
        publish_forecast(challenger_fc, panel, merged, res_f["version"])
    else:
        publish_forecast(champion_fc, panel, merged, champ_fc_meta["version"])

    decision = {
        "anomaly": {
            "champion_version": champ_meta["version"],
            "challenger_version": res_a["version"],
            "champion_pr_auc_new_period": champ_auc,
            "challenger_pr_auc_new_period": chall_auc,
            "promoted": promote_a,
        },
        "forecast": {
            "champion_version": champ_fc_meta["version"],
            "challenger_version": res_f["version"],
            "champion_live_wmape": live_wmape,
            "challenger_backtest_wmape": chall_wmape,
            "promoted": promote_f,
        },
        "feedback_labels_used": 0 if feedback is None else len(feedback),
    }
    save_summary(decision, get_settings().reports_dir / "retrain_decision.json")
    write_table("review_feedback", feedback) if feedback is not None else None
    result.update({"retrained": True, "decision": decision})
    log.info("retrain decision: %s", decision)
    return result
