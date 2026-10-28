"""Phase 3: panel, no-leakage guarantees, backtest metrics, reorder formula."""

from __future__ import annotations

import numpy as np
import pandas as pd

from supplysight.forecast.backtest import bias, rolling_origins, wmape
from supplysight.forecast.models import GBMForecaster, baseline_forecasts, calendar_flags, column_features, series_scale
from supplysight.forecast.panel import build_panel
from supplysight.forecast.reorder import recommend_reorder_points


def test_panel_shapes(small_tables):
    p = build_panel(small_tables)
    assert p.Y.shape == p.P.shape
    assert np.isnan(p.Y[:, p.n_hist :]).all()  # future weeks have no actuals
    assert np.isfinite(p.Y[:, : p.n_hist]).all()
    assert (p.P[:, : p.n_hist] >= 0).all()


def test_features_never_use_target_week_or_future(small_tables):
    p = build_panel(small_tables)
    t = p.n_hist - 5
    static = np.zeros((p.Y.shape[0], 3))
    scale = series_scale(p.Y, t)
    x1 = column_features(p.Y, p.P, t, p.weeks, scale, static)
    y2 = p.Y.copy()
    y2[:, t:] = 1e9  # poison the target week and everything after it
    x2 = column_features(y2, p.P, t, p.weeks, scale, static)
    np.testing.assert_array_equal(np.nan_to_num(x1), np.nan_to_num(x2))


def test_training_rows_respect_cutoff(small_tables):
    p = build_panel(small_tables)
    t_end = p.n_hist - 8
    m = GBMForecaster()
    X1, y1 = m.training_rows(p, t_end)
    y_poisoned = p.Y.copy()
    y_poisoned[:, t_end:] = 1e9
    p2 = type(p)(Y=y_poisoned, P=p.P, weeks=p.weeks, n_hist=p.n_hist, series=p.series)
    X2, y2 = m.training_rows(p2, t_end)
    np.testing.assert_array_equal(np.nan_to_num(X1), np.nan_to_num(X2))
    np.testing.assert_array_equal(y1, y2)


def test_predict_ignores_actuals_after_origin(small_tables):
    p = build_panel(small_tables)
    origin = p.n_hist - 8
    m = GBMForecaster().fit(p, t_end=origin, quantiles=False)
    a = m.predict(p, origin, 4)["p50"]
    y2 = p.Y.copy()
    y2[:, origin:] *= 5
    p2 = type(p)(Y=y2, P=p.P, weeks=p.weeks, n_hist=p.n_hist, series=p.series)
    b = m.predict(p2, origin, 4)["p50"]
    np.testing.assert_allclose(a, b)
    assert (a >= 0).all()


def test_rolling_origins_are_ordered_and_inside_history():
    o = rolling_origins(100, n_origins=6, step=4, horizon=8)
    assert o == sorted(o) and o[-1] == 92 and o[0] == 72


def test_metrics_and_baselines():
    y = np.array([10.0, 10.0, 0.0])
    assert wmape(y, np.array([9.0, 11.0, 0.0])) == 0.1
    assert bias(y, np.array([11.0, 11.0, 0.0])) == 0.1
    Y = np.arange(60, dtype=float).reshape(1, -1)
    b = baseline_forecasts(Y, 56, 2)
    assert b["naive"][0, 0] == 55 and b["moving_avg_4"][0, 0] == np.mean([52, 53, 54, 55])
    assert b["seasonal_naive"][0, 0] == 4


def test_calendar_flags():
    f = calendar_flags(pd.Timestamp("2026-09-21"))  # sales month October
    assert f["festive"] == 1.0 and f["summer"] == 0.0


def test_reorder_point_formula():
    series = pd.DataFrame({"matnr": ["100001"], "werks": ["1000"], "matkl": ["RM-HUMECT"]})
    tables = {
        "eina": pd.DataFrame({"infnr": ["1"], "matnr": ["100001"], "lifnr": ["5"], "is_primary": [1]}),
        "eine": pd.DataFrame({"infnr": ["1"], "aplfz": [14], "netpr": [100.0], "valid_from": ["2024-01-01"]}),
        "marc": pd.DataFrame({"matnr": ["100001"], "werks": ["1000"], "minbe": [100.0]}),
        "mard": pd.DataFrame({"matnr": ["100001"], "werks": ["1000"], "labst": [50.0], "open_po_qty": [0.0]}),
        "makt": pd.DataFrame({"matnr": ["100001"], "maktx": ["Glycerin"]}),
    }
    out = recommend_reorder_points(series, np.array([[100.0, 100.0, 100.0]]), np.array([20.0]), tables).iloc[0]
    expected_ss = 1.6449 * 20 * np.sqrt(2)
    assert abs(out.safety_stock - round(expected_ss, 3)) < 1e-3
    assert abs(out.recommended_rop - round(200 + expected_ss, 3)) < 1e-3
    assert out.stockout_risk == "High"  # 50 in stock < 200 needed over the lead time
    assert out.rop_gap_pct > 100


def test_trained_forecaster_beats_naive(trained):
    m = trained["forecast"]["metrics"]
    ov = trained["forecast"]["overall"].set_index("model")
    assert ov.loc["gbm", "wmape"] < ov.loc["naive", "wmape"]
    assert 0.5 <= m["interval_coverage_calibrated"] <= 1.0
