"""Data and prediction drift: PSI + Kolmogorov–Smirnov per feature, plus an HTML report.

If the ``evidently`` package is installed an Evidently DataDriftPreset report is written as
well; the built-in PSI/KS report needs only numpy/scipy/matplotlib.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from supplysight.utils import get_logger, optional_import

log = get_logger(__name__)
PSI_WARN, PSI_ALERT = 0.1, 0.2


def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index with quantile bins of the reference distribution."""
    ref, cur = np.asarray(ref, float), np.asarray(cur, float)
    ref, cur = ref[np.isfinite(ref)], cur[np.isfinite(cur)]
    if len(ref) == 0 or len(cur) == 0:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # (near-)constant or binary feature: compare category shares
        vals = np.union1d(np.unique(ref), np.unique(cur))
        p = np.array([(ref == v).mean() for v in vals])
        q = np.array([(cur == v).mean() for v in vals])
    else:
        edges[0], edges[-1] = -np.inf, np.inf
        p = np.histogram(ref, edges)[0] / len(ref)
        q = np.histogram(cur, edges)[0] / len(cur)
    p, q = np.clip(p, 1e-4, None), np.clip(q, 1e-4, None)
    return float(np.sum((q - p) * np.log(q / p)))


def drift_table(ref: pd.DataFrame, cur: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Per-feature PSI, KS statistic/p-value and a drift flag."""
    rows = []
    for c in features:
        a, b = ref[c].to_numpy(float), cur[c].to_numpy(float)
        ks = ks_2samp(a[np.isfinite(a)], b[np.isfinite(b)]) if len(a) and len(b) else None
        val = psi(a, b)
        rows.append({"feature": c, "psi": round(val, 4), "ks_stat": round(float(ks.statistic), 4) if ks else np.nan,
                     "ks_pvalue": float(ks.pvalue) if ks else np.nan,
                     "ref_mean": float(np.nanmean(a)), "cur_mean": float(np.nanmean(b)),
                     "drifted": bool(val >= PSI_ALERT or (ks is not None and ks.pvalue < 0.01 and ks.statistic > 0.1))})
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def _hist_png(ref: np.ndarray, cur: np.ndarray, title: str) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(3.6, 2.2))
    lo, hi = np.nanpercentile(np.r_[ref, cur], [1, 99])
    if not np.isfinite(lo) or lo == hi:
        lo, hi = np.nanmin(np.r_[ref, cur]) - 0.5, np.nanmax(np.r_[ref, cur]) + 0.5
    b = np.linspace(lo, hi, 30)
    ax.hist(ref, b, alpha=0.55, density=True, label="reference", color="#64748b")
    ax.hist(cur, b, alpha=0.55, density=True, label="current", color="#2563eb")
    ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=6, frameon=False)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def write_html_report(sections: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]], summary: dict,
                      path: Path) -> None:
    """Self-contained HTML drift report. sections: name -> (table, ref_df, cur_df)."""
    css = ("body{font-family:system-ui,Segoe UI,sans-serif;margin:24px;color:#0f172a;background:#fff}"
           "table{border-collapse:collapse;font-size:13px;margin:8px 0 20px}td,th{border-bottom:1px solid #e2e8f0;"
           "padding:4px 10px;text-align:right}th{background:#f8fafc}td:first-child,th:first-child{text-align:left}"
           ".bad{color:#b91c1c;font-weight:600}.grid{display:flex;flex-wrap:wrap;gap:8px}.card{border:1px solid "
           "#e2e8f0;border-radius:8px;padding:4px}h1{font-size:22px}h2{font-size:17px;margin-top:28px}")
    html = [f"<!doctype html><html><head><meta charset='utf-8'><title>SupplySight drift report</title><style>{css}"
            "</style></head><body><h1>SupplySight – data &amp; prediction drift</h1>",
            "<p>Synthetic data only. PSI ≥ 0.2 or KS p &lt; 0.01 with statistic &gt; 0.1 marks a feature as drifted.</p>",
            "<table><tr><th>Check</th><th>Value</th></tr>"]
    for k, v in summary.items():
        html.append(f"<tr><td>{k}</td><td>{v if not isinstance(v, float) else round(v, 4)}</td></tr>")
    html.append("</table>")
    for name, (tab, ref, cur) in sections.items():
        html.append(f"<h2>{name}</h2><table><tr><th>feature</th><th>PSI</th><th>KS</th><th>KS p</th>"
                    "<th>ref mean</th><th>cur mean</th><th>drifted</th></tr>")
        for r in tab.itertuples():
            cls = " class='bad'" if r.drifted else ""
            html.append(f"<tr><td>{r.feature}</td><td{cls}>{r.psi:.3f}</td><td>{r.ks_stat:.3f}</td><td>{r.ks_pvalue:.2g}</td>"
                        f"<td>{r.ref_mean:.3f}</td><td>{r.cur_mean:.3f}</td><td{cls}>{'yes' if r.drifted else 'no'}</td></tr>")
        html.append("</table><div class='grid'>")
        for r in tab.head(6).itertuples():
            img = _hist_png(ref[r.feature].to_numpy(float), cur[r.feature].to_numpy(float), f"{r.feature} (PSI {r.psi:.2f})")
            html.append(f"<div class='card'><img src='data:image/png;base64,{img}' alt='{r.feature}'></div>")
        html.append("</div>")
    html.append("</body></html>")
    path.write_text("\n".join(html), encoding="utf-8")


def evidently_report(ref: pd.DataFrame, cur: pd.DataFrame, path: Path) -> bool:
    """Write an Evidently data-drift report if evidently is installed (0.4.x API)."""
    ev = optional_import("evidently.report")
    presets = optional_import("evidently.metric_preset")
    if ev is None or presets is None:
        return False
    try:  # pragma: no cover - optional dependency
        rep = ev.Report(metrics=[presets.DataDriftPreset()])
        rep.run(reference_data=ref, current_data=cur)
        rep.save_html(str(path))
        return True
    except Exception as exc:  # pragma: no cover
        log.warning("Evidently report skipped: %s", exc)
        return False


def save_summary(summary: dict, path: Path) -> None:
    """Persist the drift summary for the dashboard / API."""
    path.write_text(json.dumps(summary, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
