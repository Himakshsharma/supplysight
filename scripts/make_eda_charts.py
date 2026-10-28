"""EDA charts for the README / notebook: seasonality and price distributions.

Usage: python scripts/make_eda_charts.py   (writes docs/images/*.png)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from supplysight.data import load_source_tables  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "images"
COLORS = ["#2563eb", "#0d9488", "#d97706", "#7c3aed", "#db2777", "#64748b"]


def main() -> None:
    """Write seasonality, price and anomaly-type charts."""
    OUT.mkdir(parents=True, exist_ok=True)
    t = load_source_tables(["afko", "mara", "ekpo", "ekko", "anomaly_labels", "mseg"])
    af = t["afko"].merge(t["mara"][["matnr", "matkl"]], left_on="plnbez", right_on="matnr")
    af["month"] = pd.to_datetime(af.gstrp).dt.to_period("M").dt.to_timestamp()
    g = af.groupby(["month", "matkl"]).gamng.sum().unstack().fillna(0)
    wk = af.groupby("month").gstrp.apply(lambda d: pd.to_datetime(d).dt.isocalendar().week.nunique())
    g = g[wk.reindex(g.index) >= 3]  # drop partial months at the edges
    groups = ["FG-SUN", "FG-LIP", "FG-CREAM", "FG-HAIR"]
    fig, ax = plt.subplots(figsize=(10, 3.6))
    for c, col in zip(groups, COLORS, strict=False):
        ax.plot(g.index, g[c] / g[c].mean(), label=c, color=col, lw=2)
    ax.axhline(1, color="#94a3b8", lw=0.8)
    ax.set_title("Planned production by finished-goods group (index, mean = 1): summer sunscreen peak, festive lipsticks/creams",
                 fontsize=10, loc="left")
    ax.legend(frameon=False, ncol=4, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "eda_seasonality.png", dpi=120)
    plt.close(fig)

    po = t["ekpo"].merge(t["ekko"][["ebeln", "bedat"]], on="ebeln").merge(t["anomaly_labels"], on=["ebeln", "ebelp"],
                                                                           how="left")
    med = po[po.anomaly_type.isna()].groupby("matnr").netpr.median()
    po["price_ratio"] = po.netpr / po.matnr.map(med)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    bins = np.linspace(0.5, 3.2, 60)
    axes[0].hist(po[po.anomaly_type.isna()].price_ratio.clip(0.5, 3.2), bins, color="#94a3b8", label="normal", density=True)
    axes[0].hist(po[po.anomaly_type == "price_spike"].price_ratio.clip(0.5, 3.2), bins, color="#c0392b", alpha=0.7,
                 label="injected price spike", density=True)
    axes[0].set_title("Unit price / material median", fontsize=10, loc="left")
    axes[0].legend(frameon=False, fontsize=8)
    vc = po.anomaly_type.value_counts()
    axes[1].barh(vc.index[::-1], vc.values[::-1], color=COLORS[0])
    axes[1].set_title(f"Injected anomalies ({len(po[po.anomaly_type.notna()])} of {len(po)} PO items)", fontsize=10,
                      loc="left")
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "eda_prices_anomalies.png", dpi=120)
    plt.close(fig)
    rep = Path(__file__).resolve().parents[1] / "artifacts" / "reports"
    if (rep / "forecast_vs_actual.png").exists():
        import shutil

        shutil.copy(rep / "forecast_vs_actual.png", OUT / "forecast_vs_actual.png")
    if (rep / "drift_features.csv").exists():
        ft = pd.read_csv(rep / "drift_features.csv").sort_values("psi")
        fig, ax = plt.subplots(figsize=(8, 4.2))
        ax.barh(ft.feature, ft.psi, color=["#c0392b" if d else "#94a3b8" for d in ft.drifted])
        ax.axvline(0.2, ls="--", color="#c0392b", lw=1)
        ax.set_title("Drift quarter vs reference: PSI per PO feature (red = drifted)", fontsize=10, loc="left")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(OUT / "drift_psi.png", dpi=120)
        plt.close(fig)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
