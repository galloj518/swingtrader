"""Composite score ranking.

Rank method: percentile rank within each state group.

Rationale: a symbol in ARMED state should be ranked against other ARMED symbols,
not against CONFIRMED symbols (which are tracking open positions, not entry setups).
Cross-state comparisons are meaningless because the models predict different things
for different states.

Percentile rank is robust to non-normal score distributions and avoids any
implicit calibration assumption about the absolute scale of composite_score.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from swingtrader.utils.logging import get_logger

log = get_logger(__name__)

# States that receive a composite score and meaningful rank
_RANKED_STATES = {"BASE", "ARMED", "TRIGGERED", "ACCEPTED"}


def rank_within_state(scores_df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'percentile_rank' column: 0–100 within each state group.

    Parameters
    ----------
    scores_df : DataFrame with columns ['state', 'composite_score', ...].
                Index is symbol.

    Returns
    -------
    Copy of scores_df with added 'percentile_rank' column.
    NaN for states not in _RANKED_STATES or where composite_score is NaN.
    """
    df = scores_df.copy()
    df["percentile_rank"] = np.nan

    for state, group in df.groupby("state", sort=False):
        if state not in _RANKED_STATES:
            continue
        valid = group["composite_score"].dropna()
        if valid.empty:
            continue
        # Percentile rank: fraction of group with score ≤ this score, × 100
        ranks = valid.rank(method="average", pct=True) * 100
        df.loc[valid.index, "percentile_rank"] = ranks

    return df


def top_n_per_state(
    ranked_df: pd.DataFrame,
    n: int = 20,
    *,
    states: list[str] | None = None,
) -> pd.DataFrame:
    """Return the top-N symbols by percentile_rank within each state.

    Parameters
    ----------
    ranked_df : output of rank_within_state()
    n : max symbols per state
    states : state names to include; defaults to _RANKED_STATES

    Returns
    -------
    DataFrame sorted by (state, percentile_rank desc).
    """
    target_states = set(states) if states else _RANKED_STATES
    frames: list[pd.DataFrame] = []
    for state in sorted(target_states):
        group = ranked_df[ranked_df["state"] == state].copy()
        group = group.dropna(subset=["percentile_rank"])
        group = group.sort_values("percentile_rank", ascending=False).head(n)
        frames.append(group)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames)


def build_ranked_snapshot(
    snapshot_df: pd.DataFrame,
    scores_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge pipeline snapshot with scores, compute ranks, return unified table.

    snapshot_df : from DailyRunner._build_snapshot() — indexed by integer, has 'provider_symbol'
    scores_df   : from score_all_symbols() — indexed by symbol (provider_symbol)

    Returns a DataFrame with one row per symbol, all snapshot + score columns,
    plus 'percentile_rank'.
    """
    # scores_df index is provider_symbol
    merged = snapshot_df.copy()
    if "provider_symbol" in merged.columns:
        key = "provider_symbol"
    elif "symbol" in merged.columns:
        key = "symbol"
    else:
        log.warning("build_ranked_snapshot: no symbol key found — returning snapshot unchanged")
        return merged

    scores_df = scores_df.rename_axis("_score_sym").reset_index()
    scores_df = scores_df.rename(columns={"_score_sym": key})

    merged = merged.merge(
        scores_df[[key, "setup_score", "trade_score", "failure_risk", "composite_score"]],
        on=key,
        how="left",
    )

    ranked = rank_within_state(
        merged.set_index(key).assign(state=merged["state"].values)
        if "state" in merged.columns
        else merged.set_index(key)
    )
    ranked = ranked.reset_index().rename(columns={"index": key})
    return ranked


def summary_by_state(ranked_df: pd.DataFrame) -> pd.DataFrame:
    """One-row-per-state summary: count, mean/median composite_score."""
    agg = (
        ranked_df.groupby("state")["composite_score"]
        .agg(count="count", mean="mean", median="median")
        .reset_index()
    )
    return agg.sort_values("state")
