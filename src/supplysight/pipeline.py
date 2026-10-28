"""End-to-end training pipeline used by ``scripts/train_all.py``, ``scripts/retrain.py`` and tests.

Each ``train_*`` function: builds features from the database, evaluates honestly
(time-based holdout / rolling-origin backtest), refits on all data, registers the model,
writes serving tables (po_scores, demand_forecast, reorder_recommendations, ...) and
saves reports under ``artifacts/reports``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from supplysight.anomaly.detector import AnomalyDetector
from supplysight.anomaly.duplicates import find_duplicate_vendors, vendor_dup_scores
from supplysight.anomaly.evaluate import evaluate_scores
from supplysight.anomaly.rules import RuleConfig
from supplysight.config import get_settings
from supplysight.data import load_source_tables, po_items, weekly_consumption
from supplysight.db import read_table, table_exists, write_table
from supplysight.features.po_features import ML_FEATURES, build_po_features
from supplysight.forecast.backtest import backtest, interval_coverage, summarise
from supplysight.forecast.models import GBMForecaster
from supplysight.forecast.panel import build_panel
from supplysight.forecast.reorder import recommend_reorder_points
from supplysight.mlops import Registry, track_run
from supplysight.utils import get_logger, timer

log = get_logger(__name__)
ANOMALY_MODEL = "anomaly-detector"
FORECAST_MODEL = "demand-forecaster"
FORECAST_HORIZON = 12


def _reports() -> Path:
    p = get_settings().reports_dir
    p.mkdir(parents=True, exist_ok=True)
    return p


def _json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


# ============================================================================ anomaly
def anomaly_features(tables: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return (features, duplicate pairs, dup score map) for all PO items."""
    s = get_settings()
    pairs = find_duplicate_vendors(tables["lfa1"])
    dup = vendor_dup_scores(pairs)
    f = build_po_features(po_items(tables), weekly_consumption(tables["mseg"]), dup, s.approval_threshold_inr)
    return f, pairs, dup


def _labels(f: pd.DataFrame, tables: dict[str, pd.DataFrame], feedback: pd.DataFrame | None = None) -> pd.Series:
    """Evaluation labels = injected ground truth, overridden by analyst feedback when present.

    In a real deployment there is no injected truth, so analyst verdicts from the review queue
    become the only labels - the same code path handles both.
    """
    lab = tables.get("anomaly_labels")
    out = pd.Series([None] * len(f), index=f.index, dtype=object)
    if lab is not None and not lab.empty:
        m = f[["ebeln", "ebelp"]].merge(lab, on=["ebeln", "ebelp"], how="left")
        out = pd.Series(m.anomaly_type.values, index=f.index, dtype=object)
    if feedback is not None and not feedback.empty:
        fb = f[["ebeln", "ebelp"]].merge(feedback[["ebeln", "ebelp", "verdict"]].drop_duplicates(["ebeln", "ebelp"], keep="last"),
                                         on=["ebeln", "ebelp"], how="left")
        v = pd.Series(fb.verdict.values, index=f.index)
        out = out.where(~(v == "false_positive"), None)
        out = out.where(~((v == "true_issue") & out.isna()), "analyst_confirmed")
    return out


def _load_feedback() -> pd.DataFrame | None:
    return read_table("review_feedback") if table_exists("review_feedback") else None


def publish_anomaly(model: AnomalyDetector, f: pd.DataFrame, pairs: pd.DataFrame, version: int) -> pd.DataFrame:
    """Score every PO with ``model`` and write po_scores + vendor_duplicates."""
    full = model.score(f)
    out = f[["ebeln", "ebelp", "lifnr", "matnr", "werks", "netwr"]].copy()
    out["bedat"] = f.bedat.dt.date.astype(str)
    out = out.join(full[["score", "risk_level", "rule_score", "iforest_score", "ae_score"]])
    out["reasons"] = full.reasons.map(json.dumps)
    out["model_version"] = str(version)
    write_table("po_scores", out)
    write_table("vendor_duplicates", pairs)
    return full


def _feedback_stacker(det: AnomalyDetector, f: pd.DataFrame, feedback: pd.DataFrame | None):
    """Learn ensemble weights from analyst feedback (true_issue / false_positive)."""
    if feedback is None or len(feedback) < 20 or feedback.verdict.nunique() < 2:
        return None
    from sklearn.linear_model import LogisticRegression

    fb = f.reset_index().merge(feedback[["ebeln", "ebelp", "verdict"]], on=["ebeln", "ebelp"])
    if fb.empty or fb.verdict.nunique() < 2:
        return None
    sub = f.loc[fb["index"]]
    comp = det._combine(sub, det._x(sub))
    z = np.column_stack([comp.rule_score, comp.iforest_score, comp.ae_score, comp.ml_tail])
    yy = (fb.verdict.values == "true_issue").astype(int)
    return LogisticRegression(C=1.0, class_weight="balanced", max_iter=500).fit(z, yy)


def train_anomaly(tables: dict[str, pd.DataFrame] | None = None, quick: bool = False,
                  promote: str = "if_better", feedback: pd.DataFrame | None = None,
                  window_days: int | None = None) -> dict:
    """Train, evaluate, register and deploy the PO anomaly ensemble.

    ``window_days`` restricts training to the most recent period (sliding window), which
    lets a retrained model adapt to a new price/volume regime after drift.
    """
    s = get_settings()
    tables = tables or load_source_tables()
    with timer() as t_feat:
        f_all, pairs, dup = anomaly_features(tables)
    f = f_all
    if window_days:
        f = f_all[f_all.created_at >= f_all.created_at.max() - pd.Timedelta(days=window_days)].copy()
    if feedback is None:
        feedback = _load_feedback()
    types = _labels(f, tables, feedback)
    y = types.notna().astype(int).to_numpy()
    cut = f.created_at.quantile(0.7)
    tr, te = (f.created_at < cut).to_numpy(), (f.created_at >= cut).to_numpy()
    rule_cfg = RuleConfig(approval_threshold=s.approval_threshold_inr)

    with track_run("anomaly", "ensemble") as run:
        det = AnomalyDetector(rule_cfg=rule_cfg, seed=s.seed, nn_backend=s.nn_backend).fit(f[tr])
        sc = det.score(f[te])
        hold = evaluate_scores(y[te], {"rules_only": sc.rule_score, "isolation_forest": sc.iforest_score,
                                       "autoencoder": sc.ae_score, "ensemble": sc.score},
                               flagged=(sc.risk_level != "Low").to_numpy(), types=types[te])
        # production model: refit on the full history
        final = AnomalyDetector(rule_cfg=rule_cfg, seed=s.seed, nn_backend=s.nn_backend).fit(f)
        final.stacker = _feedback_stacker(final, f, feedback)
        final.dup_scores = dup
        full = final.score(f)
        full_eval = evaluate_scores(y, {"rules_only": full.rule_score, "isolation_forest": full.iforest_score,
                                        "autoencoder": full.ae_score, "ensemble": full.score},
                                    flagged=(full.risk_level != "Low").to_numpy(), types=types)
        ens = hold["table"].set_index("model").loc["ensemble"]
        metrics = {"holdout_pr_auc": ens.pr_auc, "holdout_precision_at_50": ens["precision@50"],
                   "holdout_precision_at_100": ens["precision@100"],
                   "holdout_rules_pr_auc": hold["table"].set_index("model").loc["rules_only", "pr_auc"],
                   "full_pr_auc": full_eval["table"].set_index("model").loc["ensemble", "pr_auc"],
                   "flag_rate": float((full.risk_level != "Low").mean()),
                   "high_rate": float((full.risk_level == "High").mean()),
                   "n_items": len(f), "n_labelled": int(y.sum()),
                   "feedback_labels": 0 if feedback is None else len(feedback)}
        run["params"] = {"features": ",".join(ML_FEATURES), "nn_backend": final.ae.backend, "train_cut": str(cut),
                         "window_days": window_days,
                         "approval_threshold": s.approval_threshold_inr, "stacker": final.stacker is not None}
        run["metrics"] = metrics
        rep = _reports()
        hold["table"].to_csv(rep / "anomaly_holdout.csv", index=False)
        full_eval["table"].to_csv(rep / "anomaly_full.csv", index=False)
        if hold["per_type_recall"] is not None:
            hold["per_type_recall"].to_csv(rep / "anomaly_per_type_holdout.csv", index=False)
        if full_eval["per_type_recall"] is not None:
            full_eval["per_type_recall"].to_csv(rep / "anomaly_per_type_full.csv", index=False)
        run["artifacts"] = [str(rep / "anomaly_holdout.csv"), str(rep / "anomaly_full.csv")]
        run["model"], run["model_name"] = final, ANOMALY_MODEL

    reg = Registry()
    version = reg.register(ANOMALY_MODEL, final, metrics, run["params"], run["run_id"])
    promoted = _maybe_promote(reg, ANOMALY_MODEL, version, metrics["holdout_pr_auc"], promote, higher_is_better=True)

    if promoted:
        publish_anomaly(final, f_all, pairs, version)
    _json(_reports() / "anomaly_metrics.json", {**metrics, "version": version, "promoted": promoted,
                                                "feature_secs": t_feat["ms"] / 1000})
    log.info("anomaly: holdout PR-AUC %.3f (rules %.3f), v%d promoted=%s", metrics["holdout_pr_auc"],
             metrics["holdout_rules_pr_auc"], version, promoted)
    return {"metrics": metrics, "version": version, "promoted": promoted, "holdout": hold, "full": full_eval}


def _maybe_promote(reg: Registry, name: str, version: int, value: float, policy: str,
                   higher_is_better: bool) -> bool:
    """Promote to Production when policy allows (always / if_better / never)."""
    if policy == "never":
        reg.transition(name, version, "Staging")
        return False
    prod = reg.get_version(name, "Production")
    key = "holdout_pr_auc" if name == ANOMALY_MODEL else "gbm_wmape"
    better = prod is None or policy == "always" or (
        value > prod["metrics"].get(key, -np.inf) if higher_is_better else value < prod["metrics"].get(key, np.inf))
    reg.transition(name, version, "Production" if better else "Staging")
    return bool(better)


# ============================================================================ forecast
def _conformal_scale(bt: pd.DataFrame, target: float = 0.8) -> float:
    g = bt[bt.model == "gbm"].dropna(subset=["p10", "p90"])
    for k in np.arange(1.0, 4.01, 0.05):
        lo = g.yhat - k * (g.yhat - g.p10)
        hi = g.yhat + k * (g.p90 - g.yhat)
        if ((g.y >= lo) & (g.y <= hi)).mean() >= target:
            return float(round(k, 2))
    return 4.0


def _forecast_chart(bt: pd.DataFrame, panel, path: Path, n: int = 3) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = bt[bt.model == "gbm"]
    vol = pd.Series(np.nansum(panel.Y[:, : panel.n_hist], 1)).rank(pct=True)
    picks = [int((vol - q).abs().idxmin()) for q in (0.95, 0.7, 0.4)][:n]
    fig, axes = plt.subplots(n, 1, figsize=(10, 3.1 * n), sharex=False)
    for ax, i in zip(np.atleast_1d(axes), picks, strict=False):
        hist = panel.Y[i, : panel.n_hist]
        ax.plot(panel.weeks[: panel.n_hist], hist, color="#6b7280", lw=1.2, label="actual")
        gi = g[(g.series == i)].sort_values(["origin", "h"])
        for k, (o, gg) in enumerate(gi.groupby("origin")):
            wk = [o + pd.Timedelta(weeks=int(h - 1)) for h in gg.h]
            ax.fill_between(wk, gg.p10, gg.p90, color="#2563eb", alpha=0.12, lw=0)
            ax.plot(wk, gg.yhat, color="#2563eb", lw=1.6, label="GBM forecast (8-wk, rolling origins)" if k == 0 else None)
        r = panel.series.iloc[i]
        ax.set_title(f"{r.matnr} · plant {r.werks} · {r.matkl}", fontsize=10, loc="left")
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def train_forecast(tables: dict[str, pd.DataFrame] | None = None, quick: bool = False,
                   promote: str = "if_better") -> dict:
    """Backtest, train, register and publish forecasts + reorder recommendations."""
    s = get_settings()
    tables = tables or load_source_tables()
    panel = build_panel(tables)
    n_origins = 2 if quick else 6
    with track_run("forecast", "gbm_global") as run:
        bt = backtest(panel, horizon=8, n_origins=n_origins, backend=s.gbm_backend, seed=s.seed)
        bt["zone"] = np.where(bt.h <= 4, "h1-4 (plan known)", "h5-8 (beyond frozen plan)")
        overall = summarise(bt)
        by_zone = summarise(bt, by=["zone"])
        by_group = summarise(bt[bt.model.isin(["gbm", "mrp_plan", "moving_avg_4", "seasonal_naive"])], by=["matkl"])
        k = _conformal_scale(bt)
        raw_cov = interval_coverage(bt)
        g = bt[bt.model == "gbm"].copy()
        g["p10"], g["p90"] = g.yhat - k * (g.yhat - g.p10), g.yhat + k * (g.p90 - g.yhat)
        cal_cov = interval_coverage(g)
        resid = g.assign(e=g.y - g.yhat).groupby("series").e.std().reindex(range(len(panel.series)))
        sigma = resid.fillna(resid.median()).fillna(0).to_numpy()

        model = GBMForecaster(backend=s.gbm_backend, seed=s.seed).fit(panel)
        model.interval_scale, model.sigma = k, sigma
        model.series = panel.series.copy()

        ov = overall.set_index("model")
        best_base = ov.drop(index="gbm").wmape.min()
        metrics = {"gbm_wmape": ov.loc["gbm", "wmape"], "best_baseline_wmape": best_base,
                   "best_baseline": ov.drop(index="gbm").wmape.idxmin(),
                   "mrp_plan_wmape": ov.loc["mrp_plan", "wmape"], "seasonal_naive_wmape": ov.loc["seasonal_naive", "wmape"],
                   "gbm_bias": ov.loc["gbm", "bias"], "interval_coverage_raw": raw_cov,
                   "interval_coverage_calibrated": cal_cov, "interval_scale": k,
                   "n_series": len(panel.series), "n_origins": n_origins,
                   "gbm_wmape_beyond_plan": by_zone[(by_zone.model == "gbm") & by_zone.zone.str.startswith("h5")].wmape.iloc[0]}
        run["params"] = {"horizon": 8, "n_origins": n_origins, "backend": s.gbm_backend,
                         "lags": "1,2,3,4,8,13,52", "plan_dropout": model.plan_dropout}
        run["metrics"] = metrics
        rep = _reports()
        overall.to_csv(rep / "forecast_backtest_overall.csv", index=False)
        by_zone.to_csv(rep / "forecast_backtest_by_zone.csv", index=False)
        by_group.to_csv(rep / "forecast_backtest_by_group.csv", index=False)
        bt.to_parquet(rep / "forecast_backtest.parquet") if _has_parquet() else bt.to_csv(rep / "forecast_backtest.csv.gz", index=False)
        _forecast_chart(bt, panel, rep / "forecast_vs_actual.png")
        run["artifacts"] = [str(rep / "forecast_backtest_overall.csv"), str(rep / "forecast_vs_actual.png")]
        run["model"], run["model_name"] = model, FORECAST_MODEL

    reg = Registry()
    version = reg.register(FORECAST_MODEL, model, metrics, run["params"], run["run_id"])
    promoted = _maybe_promote(reg, FORECAST_MODEL, version, metrics["gbm_wmape"], promote, higher_is_better=False)
    if promoted:
        reorder = publish_forecast(model, panel, tables, version)
        metrics["stockout_high"] = int((reorder.stockout_risk == "High").sum())
        metrics["stockout_medium"] = int((reorder.stockout_risk == "Medium").sum())
    _json(_reports() / "forecast_metrics.json", {**metrics, "version": version, "promoted": promoted})
    log.info("forecast: GBM WMAPE %.3f vs best baseline %.3f (%s), coverage %.2f->%.2f, v%d promoted=%s",
             metrics["gbm_wmape"], best_base, metrics["best_baseline"], raw_cov, cal_cov, version, promoted)
    return {"metrics": metrics, "version": version, "promoted": promoted, "overall": overall, "by_zone": by_zone,
            "by_group": by_group}


def publish_forecast(model: GBMForecaster, panel, tables: dict[str, pd.DataFrame], version: int) -> pd.DataFrame:
    """Forecast 12 weeks from the data end, write demand_forecast and reorder_recommendations."""
    k = getattr(model, "interval_scale", 1.0)
    fc = model.predict(panel, panel.n_hist, FORECAST_HORIZON)
    p10 = np.clip(fc["p50"] - k * (fc["p50"] - fc["p10"]), 0, None)
    p90 = fc["p50"] + k * (fc["p90"] - fc["p50"])
    sigma = _align_sigma(model, panel)
    weeks = pd.date_range(panel.weeks[panel.n_hist - 1] + pd.Timedelta(days=7), periods=FORECAST_HORIZON, freq="W-MON")
    rows = []
    for i, r in enumerate(panel.series.itertuples()):
        for h, w in enumerate(weeks):
            rows.append((r.matnr, r.werks, w.date().isoformat(), float(fc["p50"][i, h]), float(p10[i, h]),
                         float(p90[i, h]), str(version)))
    write_table("demand_forecast", pd.DataFrame(rows, columns=["matnr", "werks", "week", "yhat", "p10", "p90",
                                                               "model_version"]))
    reorder = recommend_reorder_points(panel.series, fc["p50"], sigma, tables)
    write_table("reorder_recommendations", reorder)
    return reorder


def _align_sigma(model: GBMForecaster, panel) -> np.ndarray:
    """Map the model's per-series error sigma onto the (possibly different) panel series."""
    ref = getattr(model, "series", None)
    sig = getattr(model, "sigma", None)
    if ref is None or sig is None:
        return np.zeros(len(panel.series))
    m = dict(zip(zip(ref.matnr, ref.werks, strict=True), sig, strict=True))
    med = float(np.median(sig)) if len(sig) else 0.0
    return np.array([m.get((a, b), med) for a, b in zip(panel.series.matnr, panel.series.werks, strict=True)])


def _has_parquet() -> bool:
    try:
        import pyarrow  # noqa: F401

        return True
    except Exception:
        return False


# ============================================================================ all
def train_all(modules: tuple[str, ...] = ("anomaly", "forecast", "copilot"), quick: bool = False,
              promote: str = "if_better") -> dict:
    """Train every requested module on the current database."""
    tables = load_source_tables()
    out: dict = {}
    if "anomaly" in modules:
        out["anomaly"] = train_anomaly(tables, quick=quick, promote=promote)
    if "forecast" in modules:
        out["forecast"] = train_forecast(tables, quick=quick, promote=promote)
    if "copilot" in modules:
        from supplysight.copilot.retriever import build_index

        out["copilot"] = {"chunks": build_index()}
    return out
