"""Freshness and actionability classification.

Classifies each symbol in the scored snapshot as fresh, stale, or extended.
This prevents stale CONFIRMED names and over-extended setups from polluting the
top actionable list.

Rules (evaluated per row of the snapshot DataFrame):

  fresh
    State is in SCORED_STATES (BASE/ARMED/TRIGGERED/ACCEPTED) AND
    not extended AND
    days_in_state <= FRESH_MAX_DAYS[state]

  stale_confirmed
    State == CONFIRMED and days_in_state > STALE_CONFIRMED_DAYS.
    These names have already hit the target; they are position-monitoring,
    not new entry candidates.

  extended
    dist_to_pivot_atr > EXT_ATR (close is more than EXT_ATR units above the
    pivot). At this distance from the base, the risk/reward for new entries
    is poor. Symbols in LATE/EXHAUSTED states are always extended.

  is_actionable
    fresh AND state in {TRIGGERED, ACCEPTED, ARMED, BASE} — used by the
    selector to build the top actionable list.

All thresholds are module-level constants so they can be adjusted without
touching multiple files.
"""
from __future__ import annotations

import math

import pandas as pd

# ── Thresholds ────────────────────────────────────────────────────────────────

# Distance-from-pivot above which the symbol is classified as extended (in ATR units).
EXT_ATR: float = 3.0

# Price more than this percent above SMA50 → extended regardless of pivot distance.
# Catches names that are "near their own pivot" but have already run far from MA support.
EXT_SMA50_PCT: float = 0.12   # 12 % above 50-day MA

# Price more than this many ATR above YTD AVWAP → extended vs. year-open cost basis.
EXT_YTD_ATR: float = 5.0

# Maximum days_in_state before a setup is considered stale, by state.
# TRIGGERED is expected to resolve quickly; BASE can sit longer.
FRESH_MAX_DAYS: dict[str, int] = {
    "TRIGGERED": 10,
    "ACCEPTED": 15,
    "ARMED": 30,
    "BASE": 60,
}

# A CONFIRMED trade older than this is stale for entry purposes.
STALE_CONFIRMED_DAYS: int = 20

# States that receive actionable scoring (can appear in top setup list).
SCORED_STATES: frozenset[str] = frozenset({"BASE", "ARMED", "TRIGGERED", "ACCEPTED"})


# ── Per-row classification ────────────────────────────────────────────────────

def _safe_float(v) -> float:
    try:
        f = float(v)
        return f if math.isfinite(f) else math.nan
    except (TypeError, ValueError):
        return math.nan


def classify_row(row: pd.Series) -> dict[str, bool | str]:
    """Return freshness classification for a single snapshot row.

    Parameters
    ----------
    row : one row from the scored snapshot DataFrame.

    Returns
    -------
    dict with keys: is_extended, is_stale_confirmed, is_fresh, is_actionable,
                    freshness_label, extension_reasons (human-readable string).

    Extension is now multi-signal:
      - dist_to_pivot_atr > EXT_ATR (3.0)   — too far above the base pivot
      - close_vs_sma50 > EXT_SMA50_PCT       — 12% above 50-day MA (run from MA support)
      - ytd_dist_atr > EXT_YTD_ATR           — 5 ATR above YTD AVWAP (extended vs cost basis)
      - state in {LATE, EXHAUSTED}            — state machine says late/exhausted

    A name can be near its own pivot but still be extended vs. MA or cost basis,
    which represents poor risk/reward for a fresh entry.
    """
    state = str(row.get("state", "NONE"))
    dist = _safe_float(row.get("dist_to_pivot_atr", math.nan))
    days = int(row.get("days_in_state", 0) or 0)
    close_vs_sma50 = _safe_float(row.get("close_vs_sma50", math.nan))
    ytd_dist_atr = _safe_float(row.get("ytd_dist_atr", math.nan))

    # ── Extension reasons (each tracked independently) ────────────────────────
    ext_late_state = state in {"LATE", "EXHAUSTED"}
    ext_pivot = math.isfinite(dist) and dist > EXT_ATR
    ext_sma50 = math.isfinite(close_vs_sma50) and close_vs_sma50 > EXT_SMA50_PCT
    ext_ytd = math.isfinite(ytd_dist_atr) and ytd_dist_atr > EXT_YTD_ATR

    is_extended = ext_late_state or ext_pivot or ext_sma50 or ext_ytd

    # Build a human-readable reason string for transparency in cards/artifacts
    reasons: list[str] = []
    if ext_late_state:
        reasons.append(f"state={state}")
    if ext_pivot:
        reasons.append(f"pivot+{dist:.1f}ATR>{EXT_ATR}")
    if ext_sma50:
        reasons.append(f"SMA50+{close_vs_sma50 * 100:.0f}%>{EXT_SMA50_PCT * 100:.0f}%")
    if ext_ytd:
        reasons.append(f"YTD+{ytd_dist_atr:.1f}ATR>{EXT_YTD_ATR}")

    # Stale confirmed: hit target already, position-monitoring only
    is_stale_confirmed = state == "CONFIRMED" and days > STALE_CONFIRMED_DAYS

    # Fresh: in scored state, not extended, not aged out
    max_days = FRESH_MAX_DAYS.get(state, 0)
    in_scored_state = state in SCORED_STATES
    is_fresh = in_scored_state and not is_extended and (max_days == 0 or days <= max_days)

    # Actionable: fresh and in one of the four scored states
    is_actionable = is_fresh and in_scored_state

    # Human label
    if not in_scored_state:
        label = "not-scored"
    elif is_stale_confirmed:
        label = "stale-confirmed"
    elif is_extended:
        label = "extended"
    elif not is_fresh:
        label = "stale"
    else:
        label = "fresh"

    return {
        "is_extended": is_extended,
        "is_stale_confirmed": is_stale_confirmed,
        "is_fresh": is_fresh,
        "is_actionable": is_actionable,
        "freshness_label": label,
        "extension_reasons": ", ".join(reasons) if reasons else "",
    }


def add_freshness_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add freshness columns to a snapshot DataFrame in place.

    Parameters
    ----------
    df : snapshot DataFrame with columns state, dist_to_pivot_atr, days_in_state.

    Returns
    -------
    Copy of df with added columns: is_extended, is_stale_confirmed, is_fresh,
    is_actionable, freshness_label.
    """
    if df.empty:
        return df.copy()
    records = df.apply(classify_row, axis=1)
    fresh_df = pd.DataFrame(list(records), index=df.index)
    return pd.concat([df, fresh_df], axis=1)
