"""Rolling-origin backtesting (expanding window, no shuffling, no peeking)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from supplysight.forecast.models import FROZEN_WEEKS, GBMForecaster, baseline_forecasts
from supplysight.forecast.panel import Panel


def wmape(y: np.ndarray, yhat: np.ndarray) -> float:
    """Weighted MAPE = sum|y - yhat| / sum|y|."""
    denom = np.nansum(np.abs(y))
    return float(np.nansum(np.abs(y - yhat)) / denom) if denom else float("nan")


def bias(y: np.ndarray, yhat: np.ndarray) -> float:
    """Relative bias = sum(yhat - y) / sum(y) (positive = over-forecast)."""
    denom = np.nansum(np.abs(y))
    return float(np.nansum(yhat - y) / denom) if denom else float("nan")


def rolling_origins(n_hist: int, n_origins: int = 6, step: int = 4, horizon: int = 8) -> list[int]:
    """Origins ending exactly at the last full horizon of history."""
    last = n_hist - horizon
    return sorted(last - step * i for i in range(n_origins))


def backtest(
    panel: Panel, horizon: int = 8, n_origins: int = 6, step: int = 4, backend: str = "auto", seed: int = 42
) -> pd.DataFrame:
    """Return long-format predictions for every model, origin, series and horizon step.

    The GBM is retrained at each origin using only data before that origin.
    """
    rows = []
    for origin in rolling_origins(panel.n_hist, n_origins, step, horizon):
        model = GBMForecaster(backend=backend, seed=seed).fit(panel, t_end=origin, quantiles=True)
        preds = model.predict(panel, origin, horizon)
        preds_all = {"gbm": preds["p50"], **baseline_forecasts(panel.Y, origin, horizon)}
        actual = panel.Y[:, origin : origin + horizon]
        plan = np.nan_to_num(panel.P[:, origin : origin + horizon])
        scrap = np.nansum(panel.Y[:, :origin], 1) / np.maximum(np.nansum(panel.P[:, :origin], 1), 1e-9)
        mrp = plan * np.clip(scrap, 0.8, 1.2)[:, None]
        if horizon > FROZEN_WEEKS:  # MRP has no plan beyond the frozen horizon -> moving average
            mrp[:, FROZEN_WEEKS:] = preds_all["moving_avg_4"][:, FROZEN_WEEKS:]
        preds_all["mrp_plan"] = mrp
        for name, p in preds_all.items():
            for h in range(horizon):
                rows.append(
                    pd.DataFrame(
                        {
                            "model": name,
                            "origin": panel.weeks[origin],
                            "h": h + 1,
                            "series": np.arange(panel.Y.shape[0]),
                            "y": actual[:, h],
                            "yhat": p[:, h],
                            "p10": preds["p10"][:, h] if name == "gbm" else np.nan,
                            "p90": preds["p90"][:, h] if name == "gbm" else np.nan,
                        }
                    )
                )
    df = pd.concat(rows, ignore_index=True)
    df = df.join(panel.series[["matnr", "werks", "matkl", "mtart"]], on="series")
    return df


def summarise(bt: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    """WMAPE, scaled MAE and bias per model (and optional grouping)."""
    keys = ["model"] + (by or [])
    scale = bt.groupby("series").y.transform(lambda s: max(np.abs(s).mean(), 1e-9))
    bt = bt.assign(abs_err_scaled=np.abs(bt.y - bt.yhat) / scale)

    def agg(g: pd.DataFrame) -> pd.Series:
        return pd.Series(
            {
                "wmape": wmape(g.y.values, g.yhat.values),
                "mae_scaled": g.abs_err_scaled.mean(),
                "bias": bias(g.y.values, g.yhat.values),
                "n": len(g),
            }
        )

    out = bt.groupby(keys).apply(agg, include_groups=False).reset_index()
    return out.sort_values(keys[1:] + ["wmape"] if by else ["wmape"]).reset_index(drop=True)


def interval_coverage(bt: pd.DataFrame) -> float:
    """Share of actuals inside the GBM P10–P90 band (target ≈ 0.8)."""
    g = bt[bt.model == "gbm"].dropna(subset=["p10", "p90"])
    return float(((g.y >= g.p10) & (g.y <= g.p90)).mean()) if len(g) else float("nan")
