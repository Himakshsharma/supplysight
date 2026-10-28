"""Turn SAP tables into a dense weekly panel (series x weeks) for forecasting.

* ``Y``  – actual weekly consumption (MSEG 261) per material/plant.
* ``P``  – planned requirement = planned production (AFKO.GAMNG, or PLAF for the frozen
  horizon after the data end) x BOM quantity. Known in advance, so a legitimate feature.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from supplysight.data import weekly_consumption


@dataclass
class Panel:
    """Dense arrays for all series. Future weeks (no actuals yet) are NaN in ``Y``."""

    Y: np.ndarray  # [S, T + H_future]
    P: np.ndarray  # [S, T + H_future] planned requirement (NaN where unknown)
    weeks: pd.DatetimeIndex
    n_hist: int  # number of weeks with actuals
    series: pd.DataFrame  # matnr, werks, matkl, mtart, scale

    def index_of(self, matnr: str, werks: str) -> int:
        """Row index for one series."""
        hit = np.flatnonzero((self.series.matnr.values == str(matnr)) & (self.series.werks.values == str(werks)))
        if not len(hit):
            raise KeyError(f"{matnr}/{werks}")
        return int(hit[0])


def _monday(d: pd.Series) -> pd.Series:
    d = pd.to_datetime(d)
    return (d - pd.to_timedelta(d.dt.weekday, unit="D")).dt.normalize()


def planned_requirements(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Weekly planned component requirement from production orders and planned orders."""
    bom = tables["stpo"].merge(tables["stko"][["stlnr", "matnr", "bmeng"]], on="stlnr")
    bom["per_unit"] = bom.menge.astype(float) / bom.bmeng.astype(float)
    hist = tables["afko"][["plnbez", "werks", "gamng", "gstrp"]].rename(
        columns={"plnbez": "fg", "gamng": "qty", "gstrp": "date"}
    )
    plaf = tables.get("plaf")
    if plaf is not None and len(plaf):
        fut = plaf[["matnr", "plwrk", "gsmng", "psttr"]].rename(
            columns={"matnr": "fg", "plwrk": "werks", "gsmng": "qty", "psttr": "date"}
        )
        hist = pd.concat([hist, fut], ignore_index=True)
    x = hist.merge(bom[["matnr", "idnrk", "per_unit"]], left_on="fg", right_on="matnr")
    x["req"] = x.qty.astype(float) * x.per_unit
    x["week"] = _monday(x.date)
    return x.groupby(["idnrk", "werks", "week"], as_index=False).req.sum().rename(columns={"idnrk": "matnr"})


def build_panel(tables: dict[str, pd.DataFrame], min_nonzero: int = 8) -> Panel:
    """Build the forecasting panel from source tables."""
    wc = weekly_consumption(tables["mseg"])
    plan = planned_requirements(tables)
    hist_weeks = pd.date_range(wc.week.min(), wc.week.max(), freq="W-MON")
    last_plan = plan.week.max() if len(plan) else hist_weeks[-1]
    all_weeks = pd.date_range(hist_weeks[0], max(hist_weeks[-1], last_plan), freq="W-MON")
    nz = wc.groupby(["matnr", "werks"]).qty.apply(lambda s: int((s > 0).sum()))
    keys = nz[nz >= min_nonzero].reset_index()[["matnr", "werks"]]
    keys = keys.sort_values(["matnr", "werks"]).reset_index(drop=True)
    S, T = len(keys), len(all_weeks)
    pos = {k: i for i, k in enumerate(zip(keys.matnr, keys.werks, strict=True))}
    wpos = {w: j for j, w in enumerate(all_weeks)}
    Y = np.full((S, T), np.nan)
    Y[:, : len(hist_weeks)] = 0.0
    for r in wc.itertuples():
        i = pos.get((r.matnr, r.werks))
        if i is not None:
            Y[i, wpos[r.week]] = r.qty
    P = np.full((S, T), np.nan)
    P[:, : len(hist_weeks)] = 0.0
    for r in plan.itertuples():
        i = pos.get((r.matnr, r.werks))
        j = wpos.get(r.week)
        if i is not None and j is not None:
            P[i, j] = (0.0 if np.isnan(P[i, j]) else P[i, j]) + r.req
    # future weeks covered by the frozen plan: absent plan => 0 planned, beyond => unknown
    if last_plan > hist_weeks[-1]:
        P[:, len(hist_weeks) : wpos[last_plan] + 1] = np.nan_to_num(P[:, len(hist_weeks) : wpos[last_plan] + 1])
    mara = tables["mara"].set_index("matnr")
    keys["matkl"] = keys.matnr.map(mara.matkl)
    keys["mtart"] = keys.matnr.map(mara.mtart)
    return Panel(Y=Y, P=P, weeks=all_weeks, n_hist=len(hist_weeks), series=keys)
