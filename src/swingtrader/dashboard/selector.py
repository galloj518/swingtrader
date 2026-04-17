"""Top setup selection logic.

Selects the top N actionable setups from the scored snapshot.

Selection criteria (applied in order):
  1. Exclude non-equity, skipped, and AVOID-labeled symbols.
  2. Prefer fresh setups (is_actionable == True).
  3. If fewer than MIN_TOP fresh setups exist, fill from non-extended scored states.
  4. Sort by priority tier first, then by composite_score descending.

Priority tiers (lower number = higher priority):
  Tier 1: ACTION_NOW (TRIGGERED/ACCEPTED, fresh)
  Tier 2: ACTION_BREAKOUT (ARMED near pivot)
  Tier 3: ACTION_PULLBACK (BASE/ARMED, further from pivot)
  Tier 4: everything else (safety net, rarely shown)

Diversity: limit to MAX_PER_GROUP symbols from the same group tag to avoid
  the top list being dominated by one sector ETF cluster.
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

# ── Configuration ─────────────────────────────────────────────────────────────

TOP_N: int = 7           # maximum setups in the top list
MIN_TOP: int = 3         # minimum to show even if quality is low
MAX_PER_GROUP: int = 3   # max symbols from the same group tag

_TIER: dict[str, int] = {
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


def select_top_setups(df: pd.DataFrame) -> pd.DataFrame:
    """Select the top N actionable setups.

    Parameters
    ----------
    df : snapshot DataFrame with freshness and action_label columns added.

    Returns
    -------
    DataFrame of up to TOP_N rows, sorted by (tier, composite_score desc).
    Empty DataFrame if no actionable setups exist.
    """
    if df.empty:
        return df.head(0)

    # Exclude unscorable symbols
    if "action_label" not in df.columns:
        return df.head(0)

    candidates = df[df["action_label"] != ACTION_AVOID].copy()
    candidates = candidates[candidates.get("is_non_equity", pd.Series(False, index=candidates.index)) != True]  # noqa: E712

    if candidates.empty:
        return candidates.head(0)

    # Add sort key
    candidates["_tier"] = candidates["action_label"].map(lambda x: _TIER.get(x, 9))
    candidates["_score"] = candidates.apply(_safe_score, axis=1)
    candidates["_group"] = candidates.apply(_primary_group, axis=1)

    candidates = candidates.sort_values(["_tier", "_score"], ascending=[True, False])

    # Diversity filter: cap per group
    selected: list[pd.Series] = []
    group_counts: dict[str, int] = {}

    for _, row in candidates.iterrows():
        if len(selected) >= TOP_N:
            break
        g = row["_group"]
        if group_counts.get(g, 0) >= MAX_PER_GROUP:
            continue
        selected.append(row)
        group_counts[g] = group_counts.get(g, 0) + 1

    if not selected:
        return candidates.head(0)

    result = pd.DataFrame(selected).drop(columns=["_tier", "_score", "_group"], errors="ignore")
    return result.head(TOP_N)
