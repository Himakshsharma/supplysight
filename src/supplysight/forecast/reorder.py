"""Reorder-point recommendations from the forecast, its error and vendor lead time.

    safety_stock = z * sigma_weekly * sqrt(lead_time_weeks)
    reorder_point = mean_weekly_forecast * lead_time_weeks + safety_stock
"""

from __future__ import annotations

import numpy as np
import pandas as pd

Z_SERVICE = {0.90: 1.2816, 0.95: 1.6449, 0.98: 2.0537, 0.99: 2.3263}


def recommend_reorder_points(series: pd.DataFrame, forecast: np.ndarray, sigma: np.ndarray,
                             tables: dict[str, pd.DataFrame], service_level: float = 0.95) -> pd.DataFrame:
    """Compare recommended ROPs with SAP MARC.MINBE and flag stock-out risk.

    Args:
        series: panel series (matnr, werks, matkl, ...), one row per forecast row.
        forecast: [S, H] weekly P50 forecast from the data end date.
        sigma: [S] weekly forecast-error standard deviation (from the backtest).
        tables: needs eina/eine (primary lead time), marc (MINBE), mard (stock), makt.
    """
    z = Z_SERVICE.get(service_level, 1.6449)
    from supplysight.data import current_info_records

    info = current_info_records(tables)
    lt = info[info.is_primary.astype(int) == 1].groupby("matnr").aplfz.first().astype(float)
    marc = tables["marc"].set_index(["matnr", "werks"])
    mard = tables["mard"].set_index(["matnr", "werks"])
    makt = tables["makt"].set_index("matnr").maktx
    rows = []
    for i, r in enumerate(series.itertuples()):
        lt_days = float(lt.get(r.matnr, 14.0))
        lt_w = max(lt_days / 7.0, 0.5)
        n = int(np.ceil(lt_w))
        d = float(np.mean(forecast[i, : max(1, min(n, forecast.shape[1]))]))
        ss = z * float(sigma[i]) * np.sqrt(lt_w)
        rop = d * lt_w + ss
        minbe = float(marc.loc[(r.matnr, r.werks), "minbe"]) if (r.matnr, r.werks) in marc.index else np.nan
        stock = float(mard.loc[(r.matnr, r.werks), "labst"]) if (r.matnr, r.werks) in mard.index else 0.0
        open_q = float(mard.loc[(r.matnr, r.werks), "open_po_qty"]) if (r.matnr, r.werks) in mard.index else 0.0
        need_lt = d * lt_w
        if stock + open_q < need_lt:
            risk = "High"
        elif stock + open_q < rop:
            risk = "Medium"
        else:
            risk = "Low"
        rows.append({"matnr": r.matnr, "werks": r.werks, "maktx": makt.get(r.matnr, ""), "matkl": r.matkl,
                     "avg_weekly_forecast": round(d, 3), "sigma": round(float(sigma[i]), 3),
                     "lead_time_days": int(lt_days), "safety_stock": round(ss, 3), "recommended_rop": round(rop, 3),
                     "current_minbe": round(minbe, 3), "current_stock": round(stock, 3),
                     "open_po_qty": round(open_q, 3),
                     "weeks_of_cover": round((stock + open_q) / d, 2) if d > 0 else 999.0,
                     "stockout_risk": risk,
                     "rop_gap_pct": round(100 * (rop - minbe) / minbe, 1) if minbe and minbe > 0 else np.nan})
    return pd.DataFrame(rows)
