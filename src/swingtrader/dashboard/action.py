"""Rule-based action label assignment.

Each setup receives exactly one of five action labels.
Rules are evaluated in priority order; the first matching rule wins.

Labels
------
  ACTION_NOW       = "Actionable now"
    State is TRIGGERED or ACCEPTED AND fresh (days_in_state within window)
    AND composite_score meets minimum threshold.

  ACTION_BREAKOUT  = "Actionable on breakout"
    State is ARMED or BASE AND close is within ARM_DIST_ATR of the pivot
    AND score meets minimum threshold.
    Interpretation: setup is ready; wait for the pivot breach.

  ACTION_PULLBACK  = "Actionable on pullback"
    State is BASE or ARMED but too far below the pivot to expect an
    imminent breakout. Or state is ACCEPTED/TRIGGERED with close well
    above the entry zone — better to wait for a re-test.
    Interpretation: the setup exists; wait for a lower-risk entry.

  ACTION_EXTENDED  = "Extended, wait"
    Symbol has already moved more than EXT_ATR units above the pivot,
    is in LATE/EXHAUSTED state, or is a stale CONFIRMED name.
    Interpretation: the move is mature; do not chase.

  ACTION_AVOID     = "Avoid / low quality"
    Score below MIN_SCORE threshold, or failure_risk above MAX_FAILURE_RISK,
    or state not in scored states (FAILED, NONE, SKIPPED, etc.).
    Interpretation: model assigns low probability or high failure risk.

Thresholds are module-level constants, documented below.
"""
from __future__ import annotations

import math

import pandas as pd

from swingtrader.dashboard.freshness import EXT_ATR, SCORED_STATES

# ── Label constants ───────────────────────────────────────────────────────────

ACTION_NOW      = "Actionable now"
ACTION_BREAKOUT = "Actionable on breakout"
ACTION_PULLBACK = "Actionable on pullback"
ACTION_EXTENDED = "Extended, wait"
ACTION_AVOID    = "Avoid / low quality"
ACTION_PORTFOLIO = "Portfolio hold"   # for is_portfolio symbols in action routing

# Ordered list (determines priority in the CSS / table display)
ALL_LABELS = [ACTION_NOW, ACTION_BREAKOUT, ACTION_PULLBACK, ACTION_EXTENDED, ACTION_AVOID]

# ── Thresholds ────────────────────────────────────────────────────────────────

# Minimum composite_score to be considered non-avoid (below this → AVOID).
MIN_SCORE: float = 0.20

# Maximum failure_risk to be considered non-avoid (above this → AVOID if score also low).
MAX_FAILURE_RISK: float = 0.70

# Close must be within this many ATR of the pivot to be labeled ACTION_BREAKOUT.
ARM_DIST_ATR: float = 1.5

# Minimum composite_score to qualify for ACTION_NOW (higher bar than generic min).
NOW_MIN_SCORE: float = 0.30

# Fresh trigger window (days_in_state ≤ this → fresh for TRIGGERED/ACCEPTED).
FRESH_TRIGGER_DAYS: int = 8

# In a confirmed market downtrend (regime_spy_trend < 0), the score threshold
# for non-AVOID is raised: weaker setups are suppressed.
DOWNTREND_SCORE_PENALTY: float = 0.08   # effective MIN_SCORE += this in downtrend
DOWNTREND_NOW_PENALTY: float = 0.05     # effective NOW_MIN_SCORE += this in downtrend


def _f(row: pd.Series, key: str) -> float:
    try:
        v = float(row.get(key, math.nan))
        return v if math.isfinite(v) else math.nan
    except (TypeError, ValueError):
        return math.nan


def _regime_adjusted_thresholds(row: pd.Series) -> tuple[float, float]:
    """Return (min_score, now_min_score) adjusted for market regime.

    In a confirmed market downtrend, score thresholds are raised so that
    weaker setups that would normally squeak through are suppressed.
    """
    regime_trend = _f(row, "regime_spy_trend")
    min_score = MIN_SCORE
    now_min_score = NOW_MIN_SCORE
    if math.isfinite(regime_trend) and regime_trend < 0:
        min_score += DOWNTREND_SCORE_PENALTY
        now_min_score += DOWNTREND_NOW_PENALTY
    return min_score, now_min_score


def assign_action(row: pd.Series) -> str:
    """Return one action label for a single snapshot row.

    Parameters
    ----------
    row : Series from the scored snapshot DataFrame.
          Expected fields: state, composite_score, failure_risk, dist_to_pivot_atr,
          days_in_state, is_extended (optional), is_fresh (optional),
          regime_spy_trend (optional), is_portfolio (optional).

    Returns
    -------
    One of the ACTION_* constants.
    """
    state = str(row.get("state", "NONE"))
    score = _f(row, "composite_score")
    failure = _f(row, "failure_risk")
    dist = _f(row, "dist_to_pivot_atr")
    days = int(row.get("days_in_state", 0) or 0)
    is_extended = bool(row.get("is_extended", dist > EXT_ATR if math.isfinite(dist) else False))
    is_non_equity = bool(row.get("is_non_equity", False))

    # ── 0. Non-equity: always portfolio/informational ─────────────────────────
    if is_non_equity:
        return ACTION_PORTFOLIO

    # ── 1. Extended states (before avoid so LATE → EXTENDED not AVOID) ───────
    if state in {"LATE", "EXHAUSTED"}:
        return ACTION_EXTENDED

    # ── 2. Avoid ──────────────────────────────────────────────────────────────
    if state not in SCORED_STATES:
        return ACTION_AVOID

    # Regime-adjusted thresholds: stricter scoring in downtrend markets
    min_score, now_min_score = _regime_adjusted_thresholds(row)

    if math.isfinite(score) and score < min_score:
        return ACTION_AVOID
    # High failure risk + weak score
    if math.isfinite(failure) and math.isfinite(score) and failure > MAX_FAILURE_RISK and score < 0.35:
        return ACTION_AVOID

    # ── 3. Extended (scored state but price has run) ──────────────────────────
    if is_extended:
        return ACTION_EXTENDED

    # ── 4. Portfolio holdings: route to portfolio action, not entry action ────
    # Portfolio labels are handled separately; action_label is informational.
    # Still use entry labels here so the dashboard can show them in portfolio
    # section, but the bucket system ensures they don't enter fresh entry lists.

    # ── 5. Actionable now (TRIGGERED / ACCEPTED, fresh) ───────────────────────
    if state in {"TRIGGERED", "ACCEPTED"}:
        fresh = days <= FRESH_TRIGGER_DAYS
        score_ok = not math.isfinite(score) or score >= now_min_score
        if fresh and score_ok:
            return ACTION_NOW
        # Triggered but not fresh, or score too low: pullback wait
        return ACTION_PULLBACK

    # ── 6. Actionable on breakout (ARMED / BASE near pivot) ───────────────────
    if state in {"ARMED", "BASE"}:
        near_pivot = math.isfinite(dist) and abs(dist) <= ARM_DIST_ATR
        if near_pivot:
            return ACTION_BREAKOUT
        return ACTION_PULLBACK

    return ACTION_AVOID


def add_action_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'action_label' column to a snapshot DataFrame.

    Parameters
    ----------
    df : snapshot DataFrame; must include freshness columns if available.

    Returns
    -------
    Copy of df with 'action_label' column added.
    """
    if df.empty:
        return df.copy()
    result = df.copy()
    result["action_label"] = df.apply(assign_action, axis=1)
    return result


# ── Setup classification ──────────────────────────────────────────────────────

def classify_setup(row: pd.Series) -> str:
    """Return a human-readable setup classification label.

    Priority order: Extended first, then Failed, then state-based logic.

    Parameters
    ----------
    row : Series from the scored snapshot DataFrame.

    Returns
    -------
    One of the human-readable classification strings.
    """
    state = str(row.get("state", "NONE"))
    days = int(row.get("days_in_state", 0) or 0)
    dist = row.get("dist_to_pivot_atr", math.nan)
    try:
        dist = float(dist)
        if not math.isfinite(dist):
            dist = 0.0
    except (TypeError, ValueError):
        dist = 0.0

    is_extended = bool(row.get("is_extended", False))
    is_stale_confirmed = bool(row.get("is_stale_confirmed", False))
    action_label = str(row.get("action_label", ""))
    skip_reason = row.get("skip_reason", None)

    # ── Priority 1: Extended ──────────────────────────────────────────────────
    if is_extended or state in {"LATE", "EXHAUSTED"}:
        return "Extended / chase risk"

    # ── Priority 2: Failed ────────────────────────────────────────────────────
    if state == "FAILED" or action_label == ACTION_AVOID:
        return "Failed / avoid"

    # ── Priority 3: State-based logic ─────────────────────────────────────────
    if state in {"TRIGGERED", "ACCEPTED"}:
        # Pulled back below pivot
        if dist < 0:
            return "Pullback entry"
        # Active breakout: fresh and near pivot
        if days <= 5 and 0 <= dist <= 2.5:
            return "Active breakout"
        # Early breakout: days 6–15, not extended (already excluded above)
        if 6 <= days <= 15:
            return "Early breakout"
        # Default for triggered states beyond the above
        return "Early breakout"

    if state == "CONFIRMED":
        if is_stale_confirmed:
            return "Mature trend"
        return "Confirmed uptrend"

    if state in {"ARMED", "BASE"}:
        if abs(dist) <= 0.75:
            return "Near breakout / poised"
        if -1.5 <= dist <= -0.75:
            return "Approaching pivot"
        if dist < -1.5:
            return "Building base"
        # dist > 0.75 but not extended (already excluded): near-poised region
        return "Near breakout / poised"

    # ── Watching ──────────────────────────────────────────────────────────────
    if state in {"NONE"} or skip_reason:
        return "Watching"

    return "Watching"


def add_setup_classification_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'setup_classification' column to a snapshot DataFrame.

    Parameters
    ----------
    df : snapshot DataFrame.

    Returns
    -------
    Copy of df with 'setup_classification' column added.
    """
    if df.empty:
        return df.copy()
    result = df.copy()
    result["setup_classification"] = df.apply(classify_setup, axis=1)
    return result


# ── Portfolio guidance ────────────────────────────────────────────────────────

def portfolio_guidance(row: pd.Series) -> str:
    """Return actionable guidance for a portfolio holding.

    Parameters
    ----------
    row : Series from the scored snapshot DataFrame.

    Returns
    -------
    Actionable guidance string (first-match rule set).
    """
    is_non_equity = bool(row.get("is_non_equity", False))
    state = str(row.get("state", "NONE"))
    days = int(row.get("days_in_state", 0) or 0)
    dist = _f(row, "dist_to_pivot_atr")
    pivot = _f(row, "pivot")
    atr = _f(row, "atr14")
    failure_risk = _f(row, "failure_risk")
    is_extended = bool(row.get("is_extended", False))
    skip_reason = row.get("skip_reason", None)

    stop = (pivot - 1.0 * atr) if (math.isfinite(pivot) and math.isfinite(atr)) else math.nan

    # ── Non-equity / cash ─────────────────────────────────────────────────────
    if is_non_equity:
        return "Non-equity / cash — informational only"

    # ── Failed ────────────────────────────────────────────────────────────────
    if state == "FAILED":
        return "Exit / review — state machine flagged failure"

    # ── Not evaluated ─────────────────────────────────────────────────────────
    if state in {"NONE", "SKIPPED"} or skip_reason:
        return "Not currently evaluated"

    # ── TRIGGERED / ACCEPTED ─────────────────────────────────────────────────
    if state in {"TRIGGERED", "ACCEPTED"}:
        if math.isfinite(failure_risk) and failure_risk > 0.65:
            return "Tighten stop — elevated failure risk. Hold but do not add."
        if is_extended:
            return "Hold — extended from entry. Consider trimming into resistance."
        if days <= 10:
            return "Hold — active breakout. Add only on pullback to pivot zone."
        return "Hold — in trade. No adds at current extension."

    # ── CONFIRMED ─────────────────────────────────────────────────────────────
    if state == "CONFIRMED":
        if days <= 20:
            return "Hold — recently confirmed. No adds; protect gains."
        return "Hold — mature position. Extended from base; monitor for rotation."

    # ── ARMED ─────────────────────────────────────────────────────────────────
    if state == "ARMED":
        if math.isfinite(failure_risk) and failure_risk > 0.65:
            return "Defend — elevated failure risk on base. Watch for breakdown."
        if math.isfinite(dist) and abs(dist) <= 1.0:
            return "Watch for breakout. Not yet actionable; hold existing position."
        if math.isfinite(stop):
            return f"Hold — building base. Defend on close below {stop:.2f}."
        return "Hold — building base. Monitor structure."

    # ── BASE ──────────────────────────────────────────────────────────────────
    if state == "BASE":
        if math.isfinite(failure_risk) and failure_risk > 0.65:
            return "Defend — elevated failure risk on base. Tighten stop."
        if math.isfinite(stop):
            return f"Hold — building base. Defend on close below {stop:.2f}."
        return "Hold — building base. Monitor structure."

    # ── LATE / EXHAUSTED ─────────────────────────────────────────────────────
    if state in {"LATE", "EXHAUSTED"}:
        return "Trim / de-risk — setup is late or exhausted. Protect profits."

    # ── Fallback ──────────────────────────────────────────────────────────────
    if math.isfinite(failure_risk) and failure_risk > 0.65:
        return "Defend — elevated failure risk. Tighten stop."

    return "Hold — monitor."


def add_portfolio_guidance_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'portfolio_guidance' column to a snapshot DataFrame.

    Parameters
    ----------
    df : snapshot DataFrame.

    Returns
    -------
    Copy of df with 'portfolio_guidance' column added.
    """
    if df.empty:
        return df.copy()
    result = df.copy()
    result["portfolio_guidance"] = df.apply(portfolio_guidance, axis=1)
    return result


__all__ = [
    "ACTION_AVOID",
    "ACTION_BREAKOUT",
    "ACTION_EXTENDED",
    "ACTION_NOW",
    "ACTION_PORTFOLIO",
    "ACTION_PULLBACK",
    "ALL_LABELS",
    "ARM_DIST_ATR",
    "DOWNTREND_NOW_PENALTY",
    "DOWNTREND_SCORE_PENALTY",
    "FRESH_TRIGGER_DAYS",
    "MAX_FAILURE_RISK",
    "MIN_SCORE",
    "NOW_MIN_SCORE",
    "add_action_column",
    "add_portfolio_guidance_column",
    "add_setup_classification_column",
    "assign_action",
    "classify_setup",
    "portfolio_guidance",
]
