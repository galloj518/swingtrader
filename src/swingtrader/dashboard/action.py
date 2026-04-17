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


def _f(row: pd.Series, key: str) -> float:
    try:
        v = float(row.get(key, math.nan))
        return v if math.isfinite(v) else math.nan
    except (TypeError, ValueError):
        return math.nan


def assign_action(row: pd.Series) -> str:
    """Return one action label for a single snapshot row.

    Parameters
    ----------
    row : Series from the scored snapshot DataFrame.
          Expected fields: state, composite_score, failure_risk, dist_to_pivot_atr,
          days_in_state, is_extended (optional), is_fresh (optional).

    Returns
    -------
    One of the five ACTION_* constants.
    """
    state = str(row.get("state", "NONE"))
    score = _f(row, "composite_score")
    failure = _f(row, "failure_risk")
    dist = _f(row, "dist_to_pivot_atr")
    days = int(row.get("days_in_state", 0) or 0)
    is_extended = bool(row.get("is_extended", dist > EXT_ATR if math.isfinite(dist) else False))

    # ── 1. Extended states (before avoid so LATE → EXTENDED not AVOID) ───────
    if state in {"LATE", "EXHAUSTED"}:
        return ACTION_EXTENDED

    # ── 2. Avoid ──────────────────────────────────────────────────────────────
    if state not in SCORED_STATES:
        return ACTION_AVOID
    if math.isfinite(score) and score < MIN_SCORE:
        return ACTION_AVOID
    # High failure risk + weak score
    if math.isfinite(failure) and math.isfinite(score) and failure > MAX_FAILURE_RISK and score < 0.35:
        return ACTION_AVOID

    # ── 3. Extended (scored state but price has run) ──────────────────────────
    if is_extended:
        return ACTION_EXTENDED

    # ── 3. Actionable now (TRIGGERED / ACCEPTED, fresh) ───────────────────────
    if state in {"TRIGGERED", "ACCEPTED"}:
        fresh = days <= FRESH_TRIGGER_DAYS
        score_ok = not math.isfinite(score) or score >= NOW_MIN_SCORE
        if fresh and score_ok:
            return ACTION_NOW
        # Triggered but not fresh, or not re-test-able: pullback wait
        return ACTION_PULLBACK

    # ── 4. Actionable on breakout (ARMED / BASE near pivot) ───────────────────
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
