"""Top setup selection logic — bucket-aware.

Implements three separate selection functions:

  select_breakout_candidates(df, n)
      Primary long candidates.  Eligible, fresh, non-portfolio symbols in the
      BREAKOUT_LONG bucket, ranked by composite_score descending.
      This is the main "best swing setups today" list.

  select_pullback_candidates(df, n)
      Secondary candidates.  Eligible, non-portfolio symbols in the
      PULLBACK_LONG bucket, ranked by composite_score descending.
      These are add-on / pullback re-entry setups.

  select_portfolio_holdings(df)
      All portfolio holdings (PORTFOLIO_HOLD bucket).
      Not ranked by entry quality — used for position guidance.

  select_top_setups(df, n)
      Combined shortlist for backward compatibility and the main dashboard
      card section.  Selects BREAKOUT_LONG first, then fills remaining
      slots from PULLBACK_LONG if needed.
      Only eligible, fresh, non-portfolio names are included.

Design notes
------------
- All selection functions require eligibility and bucket columns (from
  add_eligibility_columns and add_bucket_column).
- Diversity cap: at most MAX_PER_GROUP symbols from the same sector group
  in the combined shortlist to avoid sector clustering.
- Freshness preference: if the breakout bucket has < MIN_FRESH_BREAKOUT
  fresh names, the combined list will be short rather than filled with
  stale/pullback names.  Quality over quantity.
"""
from __future__ import annotations

import math

import pandas as pd

from swingtrader.dashboard.action import (
    ACTION_AVOID,
    ACTION_BREAKOUT,
    ACTION_EXTENDED,
    ACTION_NOW,
    ACTION_PULLBACK,
)
from swingtrader.dashboard.buckets import BUCKET_BREAKOUT, BUCKET_PORTFOLIO, BUCKET_PULLBACK

# ── Configuration ─────────────────────────────────────────────────────────────

# Maximum total setups in the combined top list (breakout + pullback fill)
TOP_N: int = 7

# Maximum breakout candidates shown in their own section
TOP_N_BREAKOUT: int = 7

# Maximum pullback candidates shown in their own section
TOP_N_PULLBACK: int = 5

# Minimum number of breakout-bucket names required before pulling from pullback
# (prevents the top list from being dominated by inferior setups)
MIN_FRESH_BREAKOUT: int = 2

# Max symbols from the same sector/group tag (diversity cap)
MAX_PER_GROUP: int = 3

# Tier mapping within breakout bucket (lower = higher priority)
_BREAKOUT_TIER: dict[str, int] = {
    ACTION_NOW: 1,
    ACTION_BREAKOUT: 2,
    ACTION_PULLBACK: 3,
    ACTION_EXTENDED: 4,
    ACTION_AVOID: 5,
}


def _safe_score(row: pd.Series) -> float:
    v = row.get("composite_score", math.nan)
    try:
        f = float(v)
        return f if math.isfinite(f) else -1.0
    except (TypeError, ValueError):
        return -1.0


def _primary_group(row: pd.Series) -> str:
    """Return the first group tag for diversity checking."""
    g = str(row.get("groups", ""))
    return g.split(",")[0].strip() if g else "other"


def _sort_and_cap(
    candidates: pd.DataFrame,
    n: int,
    diversity: bool = True,
) -> pd.DataFrame:
    """Sort by (tier, score desc), apply group diversity cap, return top-n."""
    if candidates.empty:
        return candidates.head(0)

    df = candidates.copy()
    df["_tier"] = df["action_label"].map(lambda x: _BREAKOUT_TIER.get(x, 9)) \
        if "action_label" in df.columns else 9
    df["_score"] = df.apply(_safe_score, axis=1)
    df["_group"] = df.apply(_primary_group, axis=1)
    df = df.sort_values(["_tier", "_score"], ascending=[True, False])

    if not diversity:
        result = df.head(n)
        return result.drop(columns=["_tier", "_score", "_group"], errors="ignore")

    selected: list[pd.Series] = []
    group_counts: dict[str, int] = {}

    for _, row in df.iterrows():
        if len(selected) >= n:
            break
        g = row["_group"]
        if group_counts.get(g, 0) >= MAX_PER_GROUP:
            continue
        selected.append(row)
        group_counts[g] = group_counts.get(g, 0) + 1

    if not selected:
        return df.head(0).drop(columns=["_tier", "_score", "_group"], errors="ignore")

    result = pd.DataFrame(selected)
    return result.drop(columns=["_tier", "_score", "_group"], errors="ignore").head(n)


def _base_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Apply common pre-filters: remove non-equity, portfolio, and avoid."""
    if df.empty:
        return df

    mask = pd.Series(True, index=df.index)

    # Remove non-equity (always)
    if "is_non_equity" in df.columns:
        mask &= ~df["is_non_equity"].astype(bool)

    # Remove ACTION_AVOID
    if "action_label" in df.columns:
        mask &= df["action_label"] != ACTION_AVOID

    return df[mask].copy()


# ── Primary selection functions ───────────────────────────────────────────────

def select_breakout_candidates(df: pd.DataFrame, n: int = TOP_N_BREAKOUT) -> pd.DataFrame:
    """Select top breakout-bucket candidates for the primary long list.

    Returns only eligible, fresh, non-portfolio symbols in BREAKOUT_LONG.
    Ranked by action tier then composite_score. Diversity-capped.

    Parameters
    ----------
    df : snapshot DataFrame with eligibility, bucket, freshness, and action columns.
    n  : max symbols to return.

    Returns
    -------
    DataFrame of up to n rows, ordered by (tier, composite_score desc).
    """
    if df.empty or "bucket" not in df.columns:
        return df.head(0)

    # Eligible breakout-bucket names only (eligibility already encoded in bucket)
    candidates = df[df["bucket"] == BUCKET_BREAKOUT].copy()

    # Require is_fresh: breakout candidates must be actionably fresh
    if "is_fresh" in candidates.columns:
        candidates = candidates[candidates["is_fresh"].astype(bool)]

    # Remove portfolio (double-check even though bucket should handle this)
    if "is_portfolio" in candidates.columns:
        candidates = candidates[~candidates["is_portfolio"].astype(bool)]

    candidates = _base_filter(candidates)

    return _sort_and_cap(candidates, n, diversity=True)


def select_pullback_candidates(df: pd.DataFrame, n: int = TOP_N_PULLBACK) -> pd.DataFrame:
    """Select top pullback-bucket candidates for the secondary list.

    Returns eligible, non-portfolio symbols in PULLBACK_LONG.
    Ranked by composite_score. Diversity-capped.

    Parameters
    ----------
    df : snapshot DataFrame with eligibility, bucket, freshness, and action columns.
    n  : max symbols to return.

    Returns
    -------
    DataFrame of up to n rows.
    """
    if df.empty or "bucket" not in df.columns:
        return df.head(0)

    candidates = df[df["bucket"] == BUCKET_PULLBACK].copy()

    # Require is_fresh for pullback candidates too (no stale names)
    if "is_fresh" in candidates.columns:
        candidates = candidates[candidates["is_fresh"].astype(bool)]

    if "is_portfolio" in candidates.columns:
        candidates = candidates[~candidates["is_portfolio"].astype(bool)]

    candidates = _base_filter(candidates)

    return _sort_and_cap(candidates, n, diversity=True)


def select_portfolio_holdings(df: pd.DataFrame) -> pd.DataFrame:
    """Return all portfolio holdings for position-guidance review.

    Does NOT apply fresh/score filters — portfolio holdings are always shown
    regardless of state or score.  Sorted by state priority (active trades
    first) then composite_score.

    Parameters
    ----------
    df : snapshot DataFrame with bucket column.

    Returns
    -------
    DataFrame of portfolio holdings, sorted by (state_priority, score desc).
    """
    if df.empty:
        return df.head(0)

    # Try bucket column first; fall back to is_portfolio flag
    if "bucket" in df.columns:
        portfolio = df[df["bucket"] == BUCKET_PORTFOLIO].copy()
    elif "is_portfolio" in df.columns:
        portfolio = df[df["is_portfolio"].astype(bool)].copy()
    else:
        return df.head(0)

    if portfolio.empty:
        return portfolio

    # Sort: active trades first (TRIGGERED, ACCEPTED), then by score
    state_priority = {"TRIGGERED": 1, "ACCEPTED": 2, "CONFIRMED": 3, "ARMED": 4, "BASE": 5}
    portfolio["_state_pri"] = portfolio["state"].map(lambda s: state_priority.get(s, 9))
    portfolio["_score"] = portfolio.apply(_safe_score, axis=1)
    portfolio = portfolio.sort_values(["_state_pri", "_score"], ascending=[True, False])
    return portfolio.drop(columns=["_state_pri", "_score"], errors="ignore")


def select_top_setups(df: pd.DataFrame, n: int = TOP_N) -> pd.DataFrame:
    """Select the combined shortlist for the main dashboard cards section.

    Selects BREAKOUT_LONG first.  If fewer than MIN_FRESH_BREAKOUT breakout
    names are available, still shows what exists rather than filling with
    lower-quality names.  If budget remains after breakout names, fills with
    the best PULLBACK_LONG names (up to total n).

    This function is the primary interface for backward-compatible callers.
    New callers should use select_breakout_candidates() and
    select_pullback_candidates() directly.

    Parameters
    ----------
    df : snapshot DataFrame with eligibility, bucket, freshness, action columns.
    n  : maximum total setups to return.

    Returns
    -------
    DataFrame of up to n rows: breakout names first, then pullback fill.
    """
    if df.empty:
        return df.head(0)

    # If bucket column missing, fall back to legacy behaviour
    if "bucket" not in df.columns:
        return _legacy_select(df, n)

    breakout = select_breakout_candidates(df, n=n)
    n_remaining = n - len(breakout)

    if n_remaining > 0:
        # Avoid duplicating any symbol already in breakout list
        already_selected = set()
        sym_col = "user_symbol" if "user_symbol" in df.columns else "symbol"
        if sym_col in breakout.columns:
            already_selected = set(breakout[sym_col].tolist())

        pullback_candidates = df[df["bucket"] == BUCKET_PULLBACK].copy()
        if sym_col in pullback_candidates.columns and already_selected:
            pullback_candidates = pullback_candidates[
                ~pullback_candidates[sym_col].isin(already_selected)
            ]
        if "is_fresh" in pullback_candidates.columns:
            pullback_candidates = pullback_candidates[pullback_candidates["is_fresh"].astype(bool)]
        if "is_portfolio" in pullback_candidates.columns:
            pullback_candidates = pullback_candidates[~pullback_candidates["is_portfolio"].astype(bool)]
        pullback_candidates = _base_filter(pullback_candidates)
        pullback_fill = _sort_and_cap(pullback_candidates, n_remaining, diversity=True)

        if not pullback_fill.empty:
            result = pd.concat([breakout, pullback_fill], ignore_index=True)
            return result.head(n)

    return breakout.head(n)


# ── Legacy fallback ───────────────────────────────────────────────────────────

def _legacy_select(df: pd.DataFrame, n: int = TOP_N) -> pd.DataFrame:
    """Legacy selection without bucket column (backward compatibility)."""
    if df.empty or "action_label" not in df.columns:
        return df.head(0)

    candidates = df[df["action_label"] != ACTION_AVOID].copy()
    if "is_non_equity" in candidates.columns:
        candidates = candidates[~candidates["is_non_equity"].astype(bool)]

    if candidates.empty:
        return candidates.head(0)

    _tier_map: dict[str, int] = {
        ACTION_NOW: 1,
        ACTION_BREAKOUT: 2,
        ACTION_PULLBACK: 3,
        ACTION_EXTENDED: 4,
        ACTION_AVOID: 5,
    }

    candidates["_tier"] = candidates["action_label"].map(lambda x: _tier_map.get(x, 9))
    candidates["_score"] = candidates.apply(_safe_score, axis=1)
    candidates["_group"] = candidates.apply(_primary_group, axis=1)
    candidates = candidates.sort_values(["_tier", "_score"], ascending=[True, False])

    selected: list[pd.Series] = []
    group_counts: dict[str, int] = {}

    for _, row in candidates.iterrows():
        if len(selected) >= n:
            break
        g = row["_group"]
        if group_counts.get(g, 0) >= MAX_PER_GROUP:
            continue
        selected.append(row)
        group_counts[g] = group_counts.get(g, 0) + 1

    if not selected:
        return candidates.head(0)

    result = pd.DataFrame(selected)
    return result.drop(columns=["_tier", "_score", "_group"], errors="ignore").head(n)
