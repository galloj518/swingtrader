"""Hard eligibility gates for the primary long-swing candidate pool.

Gates are applied in order; the first rejection marks the symbol ineligible.
Eligible symbols flow into setup-bucket assignment; ineligible symbols are
excluded from the primary long list and appear only in rejected artifacts.

Gate philosophy
---------------
Hard gates answer: "Is this name even a candidate for a long swing trade today?"
They are intentionally strict because:
  - Surfacing weak / downtrend names destroys signal quality
  - Screening should happen BEFORE scoring, not after
  - A smaller, higher-quality list beats a larger, noisy list

Every gate stores a rejection reason so the trader can see WHY a name
was excluded. Rejection reasons are persisted in eligibility_results.json.

Gate catalogue
--------------
  NON_EQUITY          — cash / non-equity instruments (SPAXX, etc.)
  INVALID_STATE       — state not in scoreable set (NONE, FAILED, ERROR, …)
  BROKEN_TREND        — price significantly below 200 SMA (clear downtrend)
  POOR_RS             — relative strength vs SPY well below zero over 63 days
  WEAK_REGIME_POOR_RS — market in downtrend AND stock also underperforming
  HIGH_FAILURE_RISK   — failure_risk model output exceeds hard ceiling
  LOW_SCORE           — composite_score below hard floor (structural weakness)
  THIN_BASE           — base_length too short to be a real base structure

Warning conditions (not rejected, but flagged):
  BELOW_SMA50         — price below 50-day SMA (weak short-term trend)
  NEUTRAL_REGIME      — SPY in neutral / sideways regime
  AGING_BASE          — BASE state but days_in_state > threshold
  ELEVATED_FAILURE    — failure_risk elevated but not at hard ceiling
"""
from __future__ import annotations

import math

import pandas as pd

from swingtrader.dashboard.freshness import SCORED_STATES

# ── Gate reason constants ─────────────────────────────────────────────────────

GATE_NON_EQUITY         = "non_equity"
GATE_INVALID_STATE      = "invalid_state"
GATE_BROKEN_TREND       = "broken_trend"
GATE_POOR_RS            = "poor_rs"
GATE_WEAK_REGIME_POOR_RS = "weak_regime_poor_rs"
GATE_HIGH_FAILURE_RISK  = "high_failure_risk"
GATE_LOW_SCORE          = "low_score"
GATE_THIN_BASE          = "thin_base"

WARN_BELOW_SMA50        = "below_sma50"
WARN_NEUTRAL_REGIME     = "neutral_regime"
WARN_AGING_BASE         = "aging_base"
WARN_ELEVATED_FAILURE   = "elevated_failure_risk"

# ── Thresholds ─────────────────────────────────────────────────────────────────

# Price below 200 SMA by more than this fraction → broken trend gate.
# close_vs_sma200 is typically (close − sma200) / sma200.
BROKEN_TREND_SMA200: float = -0.08   # 8% below 200 SMA

# Relative strength 63-day (vs SPY) below this → poor RS gate.
POOR_RS_THRESHOLD: float = -0.10     # underperforming SPY by 10%+ over 63 days

# In a market downtrend (regime_spy_trend < 0), the RS threshold is tighter:
# any negative RS + downtrend → excluded.
DOWNTREND_RS_THRESHOLD: float = 0.0  # must outperform SPY in a down market

# Hard ceiling on failure_risk (model probability of setup failing).
FAILURE_RISK_HARD: float = 0.65

# Hard floor on composite_score (below this = structurally weak regardless of state).
SCORE_FLOOR: float = 0.18

# Minimum base_length for ARMED / BASE states to be considered a real base.
MIN_BASE_LENGTH: int = 5

# Warning: price below 50 SMA by more than this fraction.
BELOW_SMA50_WARN: float = -0.02

# Warning: BASE aging out after this many days.
AGING_BASE_DAYS: int = 45

# Warning: failure_risk in this range triggers a soft warning (not hard rejection).
ELEVATED_FAILURE_WARN: float = 0.45


def _f(row: pd.Series, key: str, default: float = math.nan) -> float:
    try:
        v = float(row.get(key, default))
        return v if math.isfinite(v) else math.nan
    except (TypeError, ValueError):
        return math.nan


def assess_eligibility(row: pd.Series) -> dict:
    """Assess eligibility of a single snapshot row for the primary long pool.

    Parameters
    ----------
    row : one row from the scored snapshot DataFrame.

    Returns
    -------
    dict with keys:
        eligible          : bool — True if the symbol passes all hard gates
        rejection_reasons : list[str] — gate names that fired (empty if eligible)
        warnings          : list[str] — soft conditions present (even if eligible)
    """
    rejection_reasons: list[str] = []
    warnings: list[str] = []

    is_non_equity = bool(row.get("is_non_equity", False))
    state = str(row.get("state", "NONE"))
    composite = _f(row, "composite_score")
    failure = _f(row, "failure_risk")
    cvs200 = _f(row, "close_vs_sma200")
    cvs50 = _f(row, "close_vs_sma50")
    rs63 = _f(row, "daily_rs_63")
    regime_trend = _f(row, "regime_spy_trend")
    base_length = int(row.get("base_length", 0) or 0)
    days = int(row.get("days_in_state", 0) or 0)

    # ── Gate 1: Non-equity ────────────────────────────────────────────────────
    if is_non_equity:
        rejection_reasons.append(GATE_NON_EQUITY)
        return {"eligible": False, "rejection_reasons": rejection_reasons, "warnings": warnings}

    # ── Gate 2: Invalid / unscorable state ────────────────────────────────────
    if state not in SCORED_STATES:
        rejection_reasons.append(GATE_INVALID_STATE)
        return {"eligible": False, "rejection_reasons": rejection_reasons, "warnings": warnings}

    # ── Gate 3: Broken trend — price well below 200 SMA ──────────────────────
    # Only apply when close_vs_sma200 is available and significant.
    if math.isfinite(cvs200) and cvs200 < BROKEN_TREND_SMA200:
        rejection_reasons.append(GATE_BROKEN_TREND)

    # ── Gate 4: Poor relative strength ───────────────────────────────────────
    if math.isfinite(rs63) and rs63 < POOR_RS_THRESHOLD:
        rejection_reasons.append(GATE_POOR_RS)

    # ── Gate 5: Weak market regime + stock also underperforming ───────────────
    # In a broad downtrend, only names outperforming SPY qualify.
    if (
        math.isfinite(regime_trend)
        and regime_trend < 0
        and math.isfinite(rs63)
        and rs63 < DOWNTREND_RS_THRESHOLD
    ):
        rejection_reasons.append(GATE_WEAK_REGIME_POOR_RS)

    # ── Gate 6: High failure risk ─────────────────────────────────────────────
    if math.isfinite(failure) and failure > FAILURE_RISK_HARD:
        rejection_reasons.append(GATE_HIGH_FAILURE_RISK)

    # ── Gate 7: Low composite score ───────────────────────────────────────────
    if math.isfinite(composite) and composite < SCORE_FLOOR:
        rejection_reasons.append(GATE_LOW_SCORE)

    # ── Gate 8: Thin base (too short to be a real base) ───────────────────────
    if state in {"BASE", "ARMED"} and base_length > 0 and base_length < MIN_BASE_LENGTH:
        rejection_reasons.append(GATE_THIN_BASE)

    # ── Warnings (not disqualifying) ─────────────────────────────────────────
    if math.isfinite(cvs50) and cvs50 < BELOW_SMA50_WARN:
        warnings.append(WARN_BELOW_SMA50)

    if math.isfinite(regime_trend) and regime_trend == 0:
        warnings.append(WARN_NEUTRAL_REGIME)

    if state == "BASE" and days > AGING_BASE_DAYS:
        warnings.append(WARN_AGING_BASE)

    if math.isfinite(failure) and ELEVATED_FAILURE_WARN <= failure <= FAILURE_RISK_HARD:
        warnings.append(WARN_ELEVATED_FAILURE)

    eligible = len(rejection_reasons) == 0

    return {
        "eligible": eligible,
        "rejection_reasons": rejection_reasons,
        "warnings": warnings,
    }


def add_eligibility_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add eligibility assessment columns to snapshot DataFrame.

    Parameters
    ----------
    df : snapshot DataFrame with scored + freshness + action columns.

    Returns
    -------
    Copy of df with added columns:
        eligible           : bool
        rejection_reasons  : str  (comma-separated gate names, or "" if eligible)
        eligibility_warnings : str (comma-separated warning names, or "")
    """
    if df.empty:
        return df.copy()

    result = df.copy()
    eligible_vals: list[bool] = []
    rejection_vals: list[str] = []
    warning_vals: list[str] = []

    for _, row in df.iterrows():
        assessment = assess_eligibility(row)
        eligible_vals.append(assessment["eligible"])
        rejection_vals.append(", ".join(assessment["rejection_reasons"]))
        warning_vals.append(", ".join(assessment["warnings"]))

    result["eligible"] = eligible_vals
    result["rejection_reasons"] = rejection_vals
    result["eligibility_warnings"] = warning_vals
    return result


# ── Human-readable descriptions ───────────────────────────────────────────────

GATE_DESCRIPTIONS: dict[str, str] = {
    GATE_NON_EQUITY:          "Non-equity / cash instrument — not a swing trade candidate",
    GATE_INVALID_STATE:       "State not in scoreable set (NONE, FAILED, ERROR, etc.)",
    GATE_BROKEN_TREND:        "Price significantly below 200-day SMA — broken trend structure",
    GATE_POOR_RS:             "Relative strength vs SPY well below zero over 63 days",
    GATE_WEAK_REGIME_POOR_RS: "Market in downtrend and stock underperforming SPY",
    GATE_HIGH_FAILURE_RISK:   "Model failure-risk estimate exceeds hard ceiling",
    GATE_LOW_SCORE:           "Composite score below structural minimum floor",
    GATE_THIN_BASE:           "Base structure too short — likely a bounce, not a real base",
}

WARN_DESCRIPTIONS: dict[str, str] = {
    WARN_BELOW_SMA50:      "Price below 50-day SMA — weak short-term trend",
    WARN_NEUTRAL_REGIME:   "Broad market in neutral / sideways regime",
    WARN_AGING_BASE:       "BASE state aging out — may need to refresh base structure",
    WARN_ELEVATED_FAILURE: "Failure risk elevated (above warning threshold)",
}


__all__ = [  # noqa: RUF022
    # Gate constants
    "GATE_NON_EQUITY",
    "GATE_INVALID_STATE",
    "GATE_BROKEN_TREND",
    "GATE_POOR_RS",
    "GATE_WEAK_REGIME_POOR_RS",
    "GATE_HIGH_FAILURE_RISK",
    "GATE_LOW_SCORE",
    "GATE_THIN_BASE",
    # Warning constants
    "WARN_BELOW_SMA50",
    "WARN_NEUTRAL_REGIME",
    "WARN_AGING_BASE",
    "WARN_ELEVATED_FAILURE",
    # Thresholds
    "BROKEN_TREND_SMA200",
    "POOR_RS_THRESHOLD",
    "DOWNTREND_RS_THRESHOLD",
    "FAILURE_RISK_HARD",
    "SCORE_FLOOR",
    "MIN_BASE_LENGTH",
    # Functions
    "assess_eligibility",
    "add_eligibility_columns",
    # Descriptions
    "GATE_DESCRIPTIONS",
    "WARN_DESCRIPTIONS",
]
