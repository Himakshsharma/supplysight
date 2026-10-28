"""Forecasting models: naive baselines and a global gradient-boosted model.

The GBM is *global* (one model for all ~350 material/plant series). Targets and lag
features are divided by a per-series scale (mean of the training window) so kilograms of
glycerin and units of bottles share one model. Multi-step forecasts are recursive.
LightGBM is used when installed; scikit-learn's HistGradientBoosting otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from supplysight.forecast.panel import Panel
from supplysight.utils import optional_import

LAGS = (1, 2, 3, 4, 8, 13, 52)
FEATURES = [f"lag_{k}" for k in LAGS] + [
    "roll_mean_4",
    "roll_mean_8",
    "roll_mean_13",
    "roll_std_8",
    "plan",
    "plan_lag_1",
    "plan_known",
    "week_of_year",
    "month",
    "festive",
    "wedding",
    "summer",
    "winter",
    "matkl_code",
    "plant_code",
    "is_pack",
]
FROZEN_WEEKS = 4  # planned production is known this many weeks ahead


def calendar_flags(week: pd.Timestamp) -> dict[str, float]:
    """Calendar and Indian seasonal-event flags for a week (Monday)."""
    m = (week + pd.Timedelta(days=21)).month  # production runs ~3 weeks ahead of sales
    return {
        "week_of_year": float(week.isocalendar().week),
        "month": float(week.month),
        "festive": float(m in (9, 10, 11)),
        "wedding": float(m in (11, 12, 1, 2, 4, 5)),
        "summer": float(m in (3, 4, 5, 6)),
        "winter": float(m in (11, 12, 1, 2)),
    }


def make_regressor(backend: str = "auto", quantile: float | None = None, seed: int = 42):
    """Return a LightGBM or sklearn gradient-boosting regressor."""
    lgb = optional_import("lightgbm") if backend in ("auto", "lightgbm") else None
    if lgb is not None:  # pragma: no cover - not installed in the minimal image
        kw = dict(
            n_estimators=500,
            learning_rate=0.04,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            random_state=seed,
            verbose=-1,
        )
        if quantile is not None:
            return lgb.LGBMRegressor(objective="quantile", alpha=quantile, **kw)
        return lgb.LGBMRegressor(objective="l1", **kw)
    from sklearn.ensemble import HistGradientBoostingRegressor

    kw = dict(
        max_iter=400,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=0.1,
        random_state=seed,
    )
    if quantile is not None:
        return HistGradientBoostingRegressor(loss="quantile", quantile=quantile, **kw)
    return HistGradientBoostingRegressor(loss="absolute_error", **kw)


def series_scale(Y: np.ndarray, t_end: int, window: int = 26) -> np.ndarray:
    """Per-series scale = mean of the last ``window`` training weeks (floored)."""
    lo = max(0, t_end - window)
    s = np.nanmean(Y[:, lo:t_end], axis=1)
    fallback = np.nanmean(Y[:, :t_end], axis=1)
    s = np.where(np.isfinite(s) & (s > 0), s, fallback)
    return np.where(np.isfinite(s) & (s > 0), s, 1.0)


def column_features(
    Y: np.ndarray,
    P: np.ndarray,
    t: int,
    weeks: pd.DatetimeIndex,
    scale: np.ndarray,
    static: np.ndarray,
    plan_known: np.ndarray | bool = True,
) -> np.ndarray:
    """Feature matrix [S, F] for target week ``t`` using only weeks < t of ``Y``."""
    S = Y.shape[0]
    cols = []
    for k in LAGS:
        cols.append(Y[:, t - k] / scale if t - k >= 0 else np.full(S, np.nan))

    def win(w):
        lo = max(0, t - w)
        return Y[:, lo:t] / scale[:, None]

    with np.errstate(all="ignore"):
        cols += [np.nanmean(win(4), 1), np.nanmean(win(8), 1), np.nanmean(win(13), 1), np.nanstd(win(8), 1)]
    known = np.broadcast_to(np.asarray(plan_known, bool), (S,))
    plan = np.where(known, P[:, t] / scale, np.nan)
    plan_l1 = P[:, t - 1] / scale if t >= 1 else np.full(S, np.nan)
    cols += [plan, plan_l1, known.astype(float)]
    cal = calendar_flags(weeks[t])
    cols += [np.full(S, cal[c]) for c in ("week_of_year", "month", "festive", "wedding", "summer", "winter")]
    X = np.column_stack(cols + [static])
    return X


@dataclass
class GBMForecaster:
    """Global gradient-boosted forecaster with P10/P50/P90 outputs."""

    backend: str = "auto"
    seed: int = 42
    plan_dropout: float = 0.3
    min_t: int = 13
    models: dict = field(default_factory=dict)
    matkl_codes: dict = field(default_factory=dict)
    version: str = "0"

    def _static(self, panel: Panel) -> np.ndarray:
        s = panel.series
        if not self.matkl_codes:
            self.matkl_codes = {g: i for i, g in enumerate(sorted(s.matkl.unique()))}
        return np.column_stack(
            [
                s.matkl.map(self.matkl_codes).fillna(-1).to_numpy(float),
                (s.werks == "2000").to_numpy(float),
                (s.mtart == "VERP").to_numpy(float),
            ]
        )

    def training_rows(self, panel: Panel, t_end: int) -> tuple[np.ndarray, np.ndarray]:
        """Stack (X, y) for all target weeks min_t <= t < t_end (no future leakage)."""
        rng = np.random.default_rng(self.seed)
        scale = series_scale(panel.Y, t_end)
        static = self._static(panel)
        Xs, ys = [], []
        for t in range(self.min_t, t_end):
            known = rng.random(panel.Y.shape[0]) >= self.plan_dropout
            Xs.append(column_features(panel.Y, panel.P, t, panel.weeks, scale, static, known))
            ys.append(panel.Y[:, t] / scale)
        X, y = np.vstack(Xs), np.concatenate(ys)
        ok = np.isfinite(y)
        return X[ok], y[ok]

    def fit(self, panel: Panel, t_end: int | None = None, quantiles: bool = True) -> GBMForecaster:
        """Train on weeks before ``t_end`` (default: all history)."""
        t_end = panel.n_hist if t_end is None else t_end
        X, y = self.training_rows(panel, t_end)
        self.models = {"p50": make_regressor(self.backend, None, self.seed).fit(X, y)}
        if quantiles:
            self.models["p10"] = make_regressor(self.backend, 0.1, self.seed).fit(X, y)
            self.models["p90"] = make_regressor(self.backend, 0.9, self.seed).fit(X, y)
        self.t_fit = t_end
        return self

    def predict(self, panel: Panel, origin: int, horizon: int) -> dict[str, np.ndarray]:
        """Recursive forecast for weeks origin..origin+horizon-1 (all series).

        Uses actuals strictly before ``origin``; planned production is only treated as
        known inside the frozen horizon (FROZEN_WEEKS) and where it exists.
        """
        Y = panel.Y.copy()
        S = Y.shape[0]
        T = Y.shape[1]
        if origin + horizon > T:  # extend arrays for weeks beyond the panel
            extra = origin + horizon - T
            Y = np.hstack([Y, np.full((S, extra), np.nan)])
            P = np.hstack([panel.P, np.full((S, extra), np.nan)])
            weeks = panel.weeks.append(
                pd.date_range(panel.weeks[-1] + pd.Timedelta(days=7), periods=extra, freq="W-MON")
            )
        else:
            P, weeks = panel.P, panel.weeks
        Y[:, origin:] = np.nan
        scale = series_scale(Y, origin)
        static = self._static(panel)
        out = {k: np.zeros((S, horizon)) for k in self.models}
        for h in range(horizon):
            t = origin + h
            known = (h < FROZEN_WEEKS) & np.isfinite(P[:, t])
            X = column_features(Y, P, t, weeks, scale, static, known)
            for k, m in self.models.items():
                out[k][:, h] = np.clip(m.predict(X), 0, None) * scale
            Y[:, t] = out["p50"][:, h]
        if "p10" in out:
            lo = np.minimum(out["p10"], out["p50"])
            hi = np.maximum(out["p90"], out["p50"])
            out["p10"], out["p90"] = lo, hi
        return out


def baseline_forecasts(Y: np.ndarray, origin: int, horizon: int) -> dict[str, np.ndarray]:
    """Naive, seasonal-naive and 4-week moving-average forecasts from ``origin``."""
    S = Y.shape[0]
    last = Y[:, origin - 1]
    ma4 = np.nanmean(Y[:, origin - 4 : origin], axis=1)
    out = {"naive": np.repeat(last[:, None], horizon, 1), "moving_avg_4": np.repeat(ma4[:, None], horizon, 1)}
    sn = np.zeros((S, horizon))
    for h in range(horizon):
        j = origin + h - 52
        sn[:, h] = Y[:, j] if j >= 0 else last
    out["seasonal_naive"] = sn
    return out
