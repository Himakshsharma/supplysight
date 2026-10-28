"""Explainable PO anomaly ensemble: rules + Isolation Forest + autoencoder.

Score combination (noisy-OR, every part in [0, 1]):

    ml      = mean(ECDF_iforest(x), ECDF_autoencoder(x))         # calibrated on training data
    ml_tail = clip((ml - 0.5) / 0.5, 0, 1) ** 3                  # only the extreme tail matters
    score   = 1 - (1 - rule_score) * (1 - ml_tail)

Risk levels come from training-score quantiles (High ≈ top 2.5 %, Medium ≈ next 5 %), and
every PO gets up to three human-readable reasons (rules first, then the features that
deviate most in the "risky" direction).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

from supplysight.anomaly.autoencoder import Autoencoder
from supplysight.anomaly.rules import RuleConfig, _inr, apply_rules
from supplysight.features.po_features import ML_FEATURES

# +1: high values are risky, -1: low values are risky, 0: not used for explanations
DIRECTION = {
    "log_price_ratio": 1, "log_price_vs_info": 1, "log_qty_ratio": 1, "log_weeks_cover": 1,
    "log_pr_ratio": 1, "log_tenure_days": -1, "log_lt_ratio": -1, "is_offhours": 1, "is_weekend": 1,
    "requester_freq": 0, "value_to_threshold": 0, "same_day_cluster": 1, "vendor_dup_score": 1,
    "log_vendor_prior_pos": -1, "no_info_record": 1,
}


def _reason(feat: str, r: pd.Series) -> str | None:
    desc = f"{r.matnr} ({r.maktx})" if isinstance(r.get("maktx"), str) else str(r.matnr)
    if feat == "log_price_ratio":
        return (f"Unit price {_inr(r.netpr)} is {r.price_ratio:.1f}× the 90-day median "
                f"({_inr(r.ref_price)}) for material {desc}")
    if feat == "log_price_vs_info":
        return f"Unit price is {r.price_vs_info:.1f}× the contracted info-record price"
    if feat == "log_qty_ratio":
        return f"Quantity {r.menge:,.0f} {r.meins} is {r.qty_ratio:.1f}× the typical order size at plant {r.werks}"
    if feat == "log_weeks_cover":
        return f"Order covers ~{r.weeks_cover:.0f} weeks of consumption (typical 3–8 weeks)"
    if feat == "log_pr_ratio":
        return f"PO quantity is {r.pr_ratio:.1f}× the purchase-requisition quantity"
    if feat == "log_tenure_days":
        return f"Vendor {r.lifnr} was created only {int(r.tenure_days)} days before this PO"
    if feat == "log_lt_ratio":
        return f"Requested delivery in {int(r.lead_time_days)} days vs normal {int(r.normal_lt_days)} days"
    if feat in ("is_offhours", "is_weekend"):
        return f"Created at an unusual time ({r.created_at:%a %d-%b %H:%M})"
    if feat == "same_day_cluster":
        return f"{int(r.same_day_cluster)} near-identical POs created the same day by {r.created_by}"
    if feat == "vendor_dup_score":
        return f"Vendor resembles existing vendor {r.dup_of} ({r.dup_evidence})"
    if feat == "log_vendor_prior_pos":
        return f"Vendor {r.lifnr} has little purchasing history ({int(r.vendor_prior_pos)} earlier POs)"
    if feat == "no_info_record":
        return "No info record (contract) exists for this vendor–material pair"
    return None


@dataclass
class AnomalyDetector:
    """Fit/score wrapper persisted as one joblib artifact."""

    rule_cfg: RuleConfig = field(default_factory=RuleConfig)
    seed: int = 42
    nn_backend: str = "auto"
    weights: dict = field(default_factory=lambda: {"iforest": 0.5, "ae": 0.5})
    tail_start: float = 0.5
    scaler: RobustScaler | None = None
    iforest: IsolationForest | None = None
    ae: Autoencoder | None = None
    ecdf_if: np.ndarray | None = None
    ecdf_ae: np.ndarray | None = None
    thresholds: dict = field(default_factory=dict)
    explain_center: np.ndarray | None = None
    explain_scale: np.ndarray | None = None
    stacker: object | None = None  # optional logistic model learned from analyst feedback
    version: str = "0"

    # ------------------------------------------------------------------ helpers
    def _x(self, f: pd.DataFrame) -> np.ndarray:
        return np.clip(self.scaler.transform(f[ML_FEATURES].to_numpy(float)), -10, 10)

    @staticmethod
    def _ecdf(ref: np.ndarray, v: np.ndarray) -> np.ndarray:
        return np.searchsorted(ref, v, side="right") / len(ref)

    # ------------------------------------------------------------------ fit
    def fit(self, f: pd.DataFrame) -> AnomalyDetector:
        """Fit scaler, Isolation Forest and autoencoder on training PO features."""
        self.scaler = RobustScaler(quantile_range=(10, 90)).fit(f[ML_FEATURES].to_numpy(float))
        # binary / near-constant columns: avoid division blow-up
        self.scaler.scale_ = np.where(self.scaler.scale_ < 1e-6, 1.0, self.scaler.scale_)
        x = self._x(f)
        self.iforest = IsolationForest(n_estimators=300, max_samples=min(512, len(x)),
                                       random_state=self.seed).fit(x)
        if_raw = -self.iforest.score_samples(x)
        keep = if_raw <= np.quantile(if_raw, 0.95)  # trimmed training: AE learns "normal"
        self.ae = Autoencoder(self.nn_backend, self.seed).fit(x[keep])
        ae_raw = self.ae.score(x)
        self.ecdf_if, self.ecdf_ae = np.sort(if_raw), np.sort(ae_raw)
        raw = f[ML_FEATURES].to_numpy(float)
        self.explain_center = np.median(raw, axis=0)
        mad = 1.4826 * np.median(np.abs(raw - self.explain_center), axis=0)
        self.explain_scale = np.maximum.reduce([mad, 0.5 * raw.std(axis=0), np.full(raw.shape[1], 1e-3)])
        scores = self._combine(f, x)["score"].to_numpy()
        self.thresholds = {"high": float(max(np.quantile(scores, 0.975), 0.6)),
                           "medium": float(max(np.quantile(scores, 0.925), 0.3))}
        return self

    # ------------------------------------------------------------------ score
    def _combine(self, f: pd.DataFrame, x: np.ndarray) -> pd.DataFrame:
        rules = apply_rules(f, self.rule_cfg)
        if_raw = -self.iforest.score_samples(x)
        ae_raw = self.ae.score(x)
        p_if, p_ae = self._ecdf(self.ecdf_if, if_raw), self._ecdf(self.ecdf_ae, ae_raw)
        ml = self.weights["iforest"] * p_if + self.weights["ae"] * p_ae
        ml_tail = np.clip((ml - self.tail_start) / (1 - self.tail_start), 0, 1) ** 3
        score = 1 - (1 - rules.rule_score.to_numpy()) * (1 - ml_tail)
        if self.stacker is not None:  # learned from analyst feedback (closing the loop)
            z = np.column_stack([rules.rule_score, p_if, p_ae, ml_tail])
            score = 0.5 * score + 0.5 * self.stacker.predict_proba(z)[:, 1]
        return pd.DataFrame({"score": score, "rule_score": rules.rule_score.to_numpy(), "iforest_score": p_if,
                             "ae_score": p_ae, "ml_tail": ml_tail, "rule_reasons": rules.rule_reasons.to_numpy()},
                            index=f.index)

    def score(self, f: pd.DataFrame, top_k: int = 3) -> pd.DataFrame:
        """Score feature rows; returns score, risk_level, reasons and component scores."""
        x = self._x(f)
        out = self._combine(f, x)
        z = (f[ML_FEATURES].to_numpy(float) - self.explain_center) / self.explain_scale
        sign = np.array([DIRECTION[c] for c in ML_FEATURES])
        contrib = np.clip(z * sign, 0, None)
        reasons = []
        for i, (idx, r) in enumerate(f.iterrows()):
            rs = list(out.at[idx, "rule_reasons"])
            if out.at[idx, "ml_tail"] > 0.05 or (rs and out.at[idx, "rule_score"] >= 0.5):
                for j in np.argsort(-contrib[i]):
                    if contrib[i, j] < 2.5 or len(rs) >= top_k:
                        break
                    feat = ML_FEATURES[j]
                    if feat in ("is_offhours", "is_weekend", "same_day_cluster", "vendor_dup_score",
                                "no_info_record") and rs:
                        continue  # already explained by a rule
                    txt = _reason(feat, r)
                    if txt and txt not in rs:
                        rs.append(txt)
            reasons.append(rs[:top_k])
        out["reasons"] = reasons
        out["risk_level"] = np.select([out.score >= self.thresholds["high"], out.score >= self.thresholds["medium"]],
                                      ["High", "Medium"], "Low")
        return out.drop(columns=["rule_reasons"])
