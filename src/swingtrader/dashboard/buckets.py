"""Setup bucket assignment — separates symbols into distinct setup categories.

Buckets are mutually exclusive. A symbol belongs to exactly one bucket.
Assignment happens AFTER eligibility gating and AFTER freshness classification.

Bucket definitions
------------------

BREAKOUT_LONG ("breakout_long")
    Eligible, non-portfolio symbols with a fresh setup at or near a pivot.
    These are the primary fresh-entry candidates.
    Entry condition: ARMED/BASE near pivot OR fresh TRIGGERED/ACCEPTED.
    Only names that pass ALL eligibility gates enter this bucket.

PULLBACK_LONG ("pullback_long")
    Eligible, non-portfolio symbols that are constructive pullbacks within
    a confirmed uptrend but too far from the pivot for a clean breakout entry.
    These are secondary candidates: add-on setups, pullback re-entries.
    State: BASE/ARMED far from pivot, or ACCEPTED/CONFIRMED with a pullback
    below the old pivot.

EXTENDED_LEADER ("extended_leader")
    Eligible names that are too extended from the base for attractive new entries.
    State: LATE, EXHAUSTED, or dist_to_pivot_atr > EXT_ATR.
    Useful for monitoring, NOT for top actionable fresh-entry ranking.

PORTFOLIO_HOLD ("portfolio_hold")
    All portfolio holdings (is_portfolio=True), regardless of state.
    These are analyzed separately via portfolio_guidance logic.
    Do NOT mix into fresh-entry buckets.

NON_EQUITY ("non_equity")
    Cash instruments, non-equity symbols (SPAXX, etc.).
    Always informational-only.

EXCLUDED ("excluded")
    Failed one or more hard eligibility gates.
    Shown in rejected artifacts, not in actionable lists.

REVERSAL_SPEC ("reversal_speculative")
    Weak RS or broken structure names with some positive catalyst.
    Explicitly NOT mixed into the primary long list.
    Optional monitoring bucket only.

Ranking within buckets
----------------------
Each bucket is ranked independently:
  - BREAKOUT_LONG: ranked by setup_score (BASE/ARMED) or early trade_score
    (TRIGGERED days 1-5), adjusted down by failure_risk
  - PULLBACK_LONG: ranked by composite_score within BASE/ARMED states
  - EXTENDED_LEADER: ranked by composite_score (informational only)
  - PORTFOLIO_HOLD: ranked by position health (failure_risk inverse + trend strength)
"""
from __future__ import annotations

import math

import pandas as pd

from swingtrader.dashboard.freshness import SCORED_STATES

# ── Bucket constants ──────────────────────────────────────────────────────────

BUCKET_BREAKOUT    = "breakout_long"
BUCKET_PULLBACK    = "pullback_long"
BUCKET_EXTENDED    = "extended_leader"
BUCKET_PORTFOLIO   = "portfolio_hold"
BUCKET_NON_EQUITY  = "non_equity"
BUCKET_EXCLUDED    = "excluded"
BUCKET_REVERSAL    = "reversal_speculative"

# Ordered for display (primary list order)
BUCKET_DISPLAY_ORDER = [
    BUCKET_BREAKOUT,
    BUCKET_PULLBACK,
    BUCKET_EXTENDED,
    BUCKET_PORTFOLIO,
    BUCKET_REVERSAL,
    BUCKET_NON_EQUITY,
    BUCKET_EXCLUDED,
]

# Distance threshold: ARMED/BASE within this ATR of pivot → BREAKOUT bucket
# (vs farther away → PULLBACK bucket)
BREAKOUT_DIST_ATR: float = 1.5

# Fresh TRIGGERED/ACCEPTED: within this many days of trigger → BREAKOUT bucket
BREAKOUT_TRIGGER_DAYS: int = 7


def _f(row: pd.Series, key: str) -> float:
    try:
        v = float(row.get(key, math.nan))
        return v if math.isfinite(v) else math.nan
    except (TypeError, ValueError):
        return math.nan


def assign_bucket(row: pd.Series) -> str:
    """Return the bucket for a single snapshot row.

    Parameters
    ----------
    row : one row from the scored snapshot DataFrame.
          Must have freshness, action, and eligibility columns added.

    Returns
    -------
    One of the BUCKET_* constants.
    """
    is_non_equity = bool(row.get("is_non_equity", False))
    is_portfolio = bool(row.get("is_portfolio", False))
    eligible = bool(row.get("eligible", False))
    state = str(row.get("state", "NONE"))
    dist = _f(row, "dist_to_pivot_atr")
    days = int(row.get("days_in_state", 0) or 0)
    is_extended = bool(row.get("is_extended", False))
    is_fresh = bool(row.get("is_fresh", False))

    # ── Priority 1: Non-equity ────────────────────────────────────────────────
    if is_non_equity:
        return BUCKET_NON_EQUITY

    # ── Priority 2: Portfolio holdings — always separate ─────────────────────
    # Portfolio holdings go to PORTFOLIO_HOLD regardless of state/score.
    # They are analyzed via portfolio_guidance, not fresh-entry logic.
    if is_portfolio:
        return BUCKET_PORTFOLIO

    # ── Priority 3: Excluded (failed eligibility) ─────────────────────────────
    if not eligible:
        return BUCKET_EXCLUDED

    # ── Priority 4: Extended leader ───────────────────────────────────────────
    # Extended names are interesting to watch but not entry candidates.
    if is_extended or state in {"LATE", "EXHAUSTED"}:
        return BUCKET_EXTENDED

    # ── Priority 5: State not in scored set ───────────────────────────────────
    # Eligible check above requires state in SCORED_STATES, but double-check.
    if state not in SCORED_STATES:
        return BUCKET_EXCLUDED

    # ── Priority 6: BREAKOUT_LONG ─────────────────────────────────────────────
    # Two sub-cases:
    #   a) BASE/ARMED near pivot — ready for breakout entry
    #   b) Fresh TRIGGERED/ACCEPTED — active breakout, early days
    if state in {"ARMED", "BASE"}:
        near_pivot = math.isfinite(dist) and abs(dist) <= BREAKOUT_DIST_ATR
        if near_pivot and is_fresh:
            return BUCKET_BREAKOUT
        # Far from pivot or stale → pullback candidate
        return BUCKET_PULLBACK

    if state in {"TRIGGERED", "ACCEPTED"}:
        if is_fresh and days <= BREAKOUT_TRIGGER_DAYS:
            return BUCKET_BREAKOUT
        # Past the breakout window but still in trade → pullback re-entry context
        return BUCKET_PULLBACK

    # ── Default ───────────────────────────────────────────────────────────────
    return BUCKET_EXCLUDED


def add_bucket_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'bucket' column to snapshot DataFrame.

    Parameters
    ----------
    df : snapshot DataFrame with eligibility, freshness, and action columns added.

    Returns
    -------
    Copy of df with 'bucket' column added.
    """
    if df.empty:
        return df.copy()
    result = df.copy()
    result["bucket"] = df.apply(assign_bucket, axis=1)
    return result


def bucket_counts(df: pd.DataFrame) -> dict[str, int]:
    """Return count of symbols per bucket.

    Useful for summary logging and artifact metadata.
    """
    if df.empty or "bucket" not in df.columns:
        return dict.fromkeys(BUCKET_DISPLAY_ORDER, 0)
    counts = df["bucket"].value_counts().to_dict()
    return {b: int(counts.get(b, 0)) for b in BUCKET_DISPLAY_ORDER}


# ── Bucket display labels (human-readable) ────────────────────────────────────

BUCKET_LABELS: dict[str, str] = {
    BUCKET_BREAKOUT:   "Breakout Candidates",
    BUCKET_PULLBACK:   "Pullback / Re-entry Candidates",
    BUCKET_EXTENDED:   "Extended Leaders",
    BUCKET_PORTFOLIO:  "Portfolio Holdings",
    BUCKET_REVERSAL:   "Speculative / Reversal Watch",
    BUCKET_NON_EQUITY: "Non-Equity / Informational",
    BUCKET_EXCLUDED:   "Excluded (failed eligibility)",
}


__all__ = [  # noqa: RUF022
    # Constants
    "BUCKET_BREAKOUT",
    "BUCKET_PULLBACK",
    "BUCKET_EXTENDED",
    "BUCKET_PORTFOLIO",
    "BUCKET_NON_EQUITY",
    "BUCKET_EXCLUDED",
    "BUCKET_REVERSAL",
    "BUCKET_DISPLAY_ORDER",
    # Thresholds
    "BREAKOUT_DIST_ATR",
    "BREAKOUT_TRIGGER_DAYS",
    # Functions
    "assign_bucket",
    "add_bucket_column",
    "bucket_counts",
    # Labels
    "BUCKET_LABELS",
]
