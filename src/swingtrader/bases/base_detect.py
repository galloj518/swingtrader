"""Base detection and pivot geometry.

At each bar t, detect the longest flat consolidation (base) ending at t:
  A base is a contiguous window [t-L+1, t] where:
    (max_high - min_low) / max_high ≤ max_depth_pct
  with L ∈ [min_days, max_days].

Algorithm: for each bar t, expand the window backward from L=1 to L=max_days,
  tracking running max_high and min_low. Record the longest L that stays within
  max_depth_pct. O(n × max_days) — acceptable for daily bars.

The pivot is max_high of the base window — the resistance level a breakout must
clear. This is defined by geometry (highest resistance in the period), not lore.

Output columns in the returned DataFrame (indexed by date):
  pivot           — resistance / pivot level (NaN if no valid base)
  base_length     — number of bars in the detected base (0 if none)
  base_depth_pct  — (pivot - base_low) / pivot
  base_low        — lowest low in the base window

Assumption:
  If the highest high occurred on the very last bar (today), price may be breaking
  out rather than basing. This is handled downstream by the state machine — base_detect
  does not try to exclude it. The state machine will transition to TRIGGERED if close
  sufficiently exceeds the pivot.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from swingtrader.utils.config import load_config
from swingtrader.utils.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class BaseInfo:
    """Single-bar snapshot of the detected base."""

    pivot: float          # max high of the base window
    base_low: float       # min low of the base window
    base_length: int      # number of bars in window
    base_depth_pct: float # (pivot - base_low) / pivot


def detect_bases(
    df: pd.DataFrame,
    *,
    min_days: int | None = None,
    max_days: int | None = None,
    max_depth_pct: float | None = None,
    cfg=None,
) -> pd.DataFrame:
    """Compute base metrics for every bar in df.

    Parameters fall back to config/scoring.yaml → state_machine.base when None.

    Returns a DataFrame indexed like df with columns:
        pivot, base_length, base_depth_pct, base_low
    All NaN when no valid base exists at that bar.
    """
    cfg = cfg or load_config("scoring")
    sm_base = cfg.get("state_machine", {}).get("base", {})
    min_days = min_days if min_days is not None else int(sm_base.get("min_days", 15))
    max_days = max_days if max_days is not None else int(sm_base.get("max_days", 120))
    max_depth_pct = max_depth_pct if max_depth_pct is not None else float(sm_base.get("max_depth_pct", 0.25))

    n = len(df)
    if n < min_days:
        return pd.DataFrame(
            {"pivot": np.nan, "base_length": 0, "base_depth_pct": np.nan, "base_low": np.nan},
            index=df.index,
        )

    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)

    pivot_arr = np.full(n, np.nan)
    base_low_arr = np.full(n, np.nan)
    base_len_arr = np.zeros(n, dtype=int)
    base_depth_arr = np.full(n, np.nan)

    for t in range(n):
        best_len = 0
        run_hi = highs[t]
        run_lo = lows[t]
        lim = min(max_days, t + 1)

        for bar_len in range(1, lim + 1):
            j = t - bar_len + 1
            run_hi = max(run_hi, highs[j])
            run_lo = min(run_lo, lows[j])
            if run_hi <= 0:
                continue
            depth = (run_hi - run_lo) / run_hi
            if depth <= max_depth_pct and min_days <= bar_len:
                best_len = bar_len

        if best_len >= min_days:
            # Recompute final window stats for best_len
            w_hi = np.max(highs[t - best_len + 1: t + 1])
            w_lo = np.min(lows[t - best_len + 1: t + 1])
            depth = (w_hi - w_lo) / w_hi if w_hi > 0 else np.nan
            pivot_arr[t] = w_hi
            base_low_arr[t] = w_lo
            base_len_arr[t] = best_len
            base_depth_arr[t] = depth

    return pd.DataFrame(
        {
            "pivot": pivot_arr,
            "base_length": base_len_arr,
            "base_depth_pct": base_depth_arr,
            "base_low": base_low_arr,
        },
        index=df.index,
    )


def resistance_touches(
    df: pd.DataFrame,
    pivot: float,
    *,
    lookback_days: int = 60,
    tolerance_atr_mult: float = 0.25,
    atr: pd.Series | None = None,
) -> int:
    """Count bars where the high came within ``tolerance_atr_mult × ATR`` of ``pivot``.

    More touches → more defined resistance, flatter top → stronger pivot signal.
    """
    if atr is None:
        from swingtrader.features.primitives import atr_wilder
        atr = atr_wilder(df, 14)
    window = df.tail(lookback_days)
    atr_window = atr.tail(lookback_days)
    tol = float(atr_window.mean()) * tolerance_atr_mult
    return int(((window["high"] - pivot).abs() <= tol).sum())
