# Model card – material demand forecaster (`demand-forecaster`)

## Intended use

The forecaster predicts **weekly consumption (SAP movement type 261, issues to production orders)** per purchased
material and plant for the next 12 weeks, with P10/P50/P90 bands. MRP controllers use it to review reorder points
(`MARC-MINBE`) and safety stock, and to spot materials that will stock out before the vendor can deliver. It
supports the planner (SOP-MRP-001) and does not replace MRP runs.

## Model

- **Global gradient-boosted regressor** (LightGBM if installed, otherwise scikit-learn HistGradientBoosting). One
  model covers all 344 material/plant series.
- **Target:** weekly consumption divided by the series' mean over the last 26 training weeks (scale-free).
- **Features:** lags 1, 2, 3, 4, 8, 13 and 52; rolling means (4/8/13) and std (8); **planned requirement** =
  planned production orders (AFKO / PLAF) × BOM quantity for the target week and the week before; calendar and
  Indian seasonal flags (festive Sep–Nov, wedding, summer, winter, shifted 3 weeks for production lead); material
  group, plant and packaging flags.
- **Frozen-plan realism:** the production plan is treated as known only for the first 4 forecast weeks. During
  training it is randomly hidden 30 % of the time, so the model also learns to forecast without it.
- **Multi-step:** recursive (each prediction feeds the next week's lags).
- **Uncertainty:** quantile models (α = 0.1/0.9), widened by a conformal factor (×1.85) chosen on the backtest so the
  P10–P90 band covers 80 % of actuals (raw coverage 66 %).

## Evaluation: rolling-origin backtest

Six origins, 4 weeks apart. At each origin the model is retrained on data strictly before it and forecasts 8 weeks
ahead (16,512 forecast points). There is no shuffling. Tests poison future values to prove that features, training
rows and predictions never use them.

| Model | WMAPE | Scaled MAE | Bias |
|---|---|---|---|
| **GBM (global)** | **20.0 %** | **0.225** | −3.0 % |
| MRP plan explosion (planned production × BOM × historical scrap; moving average beyond the plan) | 23.5 % | 0.266 | −1.1 % |
| Moving average (4 wk) | 26.4 % | 0.300 | −0.9 % |
| Seasonal naive (52 wk) | 29.7 % | 0.343 | −7.0 % |
| Naive (last week) | 34.5 % | 0.380 | −0.9 % |

| Horizon | GBM | MRP plan | Moving average |
|---|---|---|---|
| Weeks 1–4 (plan known) | 18.5 % | 19.2 % | 25.0 % |
| Weeks 5–8 (beyond the frozen plan) | **21.5 %** | 27.8 % | 27.8 % |

**Where it wins and why:** the GBM beats every baseline in all 21 material groups. Inside the frozen horizon the
plan already explains most variance, so the GBM's gain over the MRP explosion is small (it learns scrap, slipped
orders and rush batches). Beyond the plan it wins clearly because it learns shared seasonality across hundreds of
series, which naive methods cannot. Hardest groups: UV filters (26 %), jars (25 %) and pigments (24 %), which are
small-volume, lumpy series. Easiest: thickeners (17 %).

**Why no LSTM in this build:** 105 weekly points per series is little for a sequence model, and the GBM already
captures cross-series patterns and gives quantiles cheaply. An LSTM comparison is listed as future work.

## Reorder recommendation

`safety stock = z · σ · √L` (z = 1.645 for 95 % service; σ = per-series std of backtest errors; L = primary vendor
lead time in weeks) and `ROP = mean weekly forecast × L + safety stock`. Stock-out risk is **High** when stock + open
POs < forecast demand over the lead time, and **Medium** when below the recommended ROP. In the base dataset, 3
material/plant combinations are High and 14 are Medium. Many `MINBE` values are deliberately stale (set from the
first 6 months), which the recommendation exposes.

## Limitations

- Lead times are taken from the primary info record; actual delivery variability is not yet in σ.
- Recursive errors compound; P50 bias is slightly negative (−3 %).
- Under a demand surge (simulated Q4 2026) live WMAPE rose to 37.6 % until retraining; drift monitoring compares live
  WMAPE with the backtest and triggers retraining at 1.25×.
- Forecasts consumption, not sales; finished-goods demand sensing is out of scope.
