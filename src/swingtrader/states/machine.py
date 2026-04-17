"""Breakout state machine.

States (string literals):
  NONE       — no valid base at this bar
  BASE       — valid base detected, no trigger
  ARMED      — in BASE with compression criteria met near pivot
  TRIGGERED  — close broke above pivot with required confirmation
  ACCEPTED   — N consecutive closes above trigger_pivot; holding
  CONFIRMED  — price reached pivot + M×ATR before hitting stop
  FAILED     — close back below pivot after trigger, or stop hit
  LATE       — price has already extended far beyond pivot (entry risk too high)
  EXHAUSTED  — post-CONFIRMED with very extended price (not used in v1; reserved)

Transitions are deterministic given the bar data and previous state.
All thresholds come from config/scoring.yaml → state_machine.

Persistence:
  compute_states() returns a DataFrame with one row per bar. The daily pipeline
  writes it to data/states/{SYMBOL}.parquet and appends new bars incrementally.

Design notes:
  - The state machine is a single forward pass (no look-back into the future).
  - Once TRIGGERED, the stored trigger_pivot and trigger_atr are used for
    CONFIRMED/FAILED thresholds — not the live pivot from base_detect.
  - FAILED resets to BASE if a new valid base appears in the next bar.
  - LATE is a soft warning state; the symbol can still TRIGGER from LATE.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

from swingtrader.features.primitives import atr_wilder, ema
from swingtrader.utils.config import load_config
from swingtrader.utils.logging import get_logger

log = get_logger(__name__)

# Canonical state labels
NONE: Final = "NONE"
BASE: Final = "BASE"
ARMED: Final = "ARMED"
TRIGGERED: Final = "TRIGGERED"
ACCEPTED: Final = "ACCEPTED"
CONFIRMED: Final = "CONFIRMED"
FAILED: Final = "FAILED"
LATE: Final = "LATE"
EXHAUSTED: Final = "EXHAUSTED"

_LIVE_STATES = frozenset({TRIGGERED, ACCEPTED, CONFIRMED})


@dataclass
class _MachineState:
    """Mutable carry-through state for the iterative loop."""

    state: str = NONE
    trigger_pivot: float = np.nan
    trigger_atr: float = np.nan
    trigger_date: object = None
    consecutive_above: int = 0
    days_in_state: int = 0
    prev_state: str = NONE


def compute_states(
    df: pd.DataFrame,
    bases: pd.DataFrame,
    *,
    cfg=None,
) -> pd.DataFrame:
    """Run the state machine over the full history of ``df``.

    Parameters
    ----------
    df : daily OHLCV DataFrame indexed by date.
    bases : output of bases.base_detect.detect_bases() aligned to df.index.
    cfg : optional pre-loaded scoring config.

    Returns
    -------
    DataFrame indexed by date with columns:
        state, pivot, trigger_pivot, trigger_atr, trigger_date,
        consecutive_above, days_in_state, state_changed
    """
    cfg = cfg or load_config("scoring")
    sm = cfg.get("state_machine", {})
    arm_cfg = sm.get("armed", {})
    trig_cfg = sm.get("triggered", {})
    acc_cfg = sm.get("accepted", {})
    conf_cfg = sm.get("confirmed", {})
    late_cfg = sm.get("late", {})

    breach_atr: float = float(trig_cfg.get("pivot_breach_atr", 0.10))
    consec: int = int(acc_cfg.get("consecutive_closes", 3))
    max_viol: float = float(acc_cfg.get("max_violation_atr", 0.5))
    conf_target: float = float(conf_cfg.get("atr_gain_target", 2.0))
    conf_stop: float = float(conf_cfg.get("atr_stop", 1.0))
    late_ext: float = float(late_cfg.get("max_extension_from_ema20_atr", 4.0))
    arm_dist: float = float(arm_cfg.get("max_dist_to_pivot_atr", 2.0))
    arm_atr_pct: float = float(arm_cfg.get("atr_contraction_pct", 70))
    arm_vol_pct: float = float(arm_cfg.get("volume_contraction_pct", 80))
    arm_atr_lb: int = int(arm_cfg.get("atr_contraction_lookback", 50))
    arm_vol_lb: int = int(arm_cfg.get("volume_contraction_lookback", 50))

    # Pre-compute indicator series
    atr14 = atr_wilder(df, 14).to_numpy(dtype=float)
    ema20 = ema(df["close"], 20).to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    pivot_arr = bases["pivot"].to_numpy(dtype=float)

    # ATR and volume rolling percentiles for ARMED detection
    atr_series = pd.Series(atr14, index=df.index)
    vol_series = df["volume"]
    atr_pct_rank = atr_series.rolling(arm_atr_lb, min_periods=10).apply(
        lambda x: float(np.sum(x[:-1] <= x[-1]) / max(len(x) - 1, 1) * 100), raw=True
    ).to_numpy(dtype=float)
    vol_10m = vol_series.rolling(10, min_periods=3).mean()
    vol_50pct = vol_series.rolling(arm_vol_lb, min_periods=10).apply(
        lambda x: float(np.percentile(x, arm_vol_pct)), raw=True
    )

    n = len(df)
    ms = _MachineState()

    # Output arrays
    states = [""] * n
    pivots = np.full(n, np.nan)
    trig_pivots = np.full(n, np.nan)
    trig_atrs = np.full(n, np.nan)
    trig_dates = [None] * n
    consec_above = np.zeros(n, dtype=int)
    days_in = np.zeros(n, dtype=int)
    changed = np.zeros(n, dtype=bool)

    for t in range(n):
        c = close[t]
        a = atr14[t]
        p = pivot_arr[t]          # current bar's base-detect pivot (includes today's high)
        # Use the PRIOR bar's pivot for trigger comparison — the resistance level was
        # established before today. Using today's pivot would make close > pivot impossible
        # since base_detect always includes today's high, keeping pivot ≥ high ≥ close.
        prior_p = pivot_arr[t - 1] if t > 0 else p
        e20 = ema20[t]
        has_base = not np.isnan(p)

        ms.prev_state = ms.state
        new_state = ms.state  # tentative

        if ms.state in _LIVE_STATES:
            # ── Active lifecycle: use stored trigger_pivot ──────────────────
            tp = ms.trigger_pivot
            ta = ms.trigger_atr
            if np.isnan(tp) or np.isnan(ta):
                new_state = FAILED
            elif c >= tp + conf_target * ta:
                new_state = CONFIRMED
            elif c <= tp - conf_stop * ta:
                new_state = FAILED
            elif ms.state in (TRIGGERED, ACCEPTED):
                if c > tp:
                    ms.consecutive_above += 1
                else:
                    ms.consecutive_above = 0
                if c < tp - max_viol * ta:
                    new_state = FAILED
                elif ms.consecutive_above >= consec:
                    new_state = ACCEPTED
                else:
                    new_state = TRIGGERED
            # CONFIRMED stays CONFIRMED

        elif ms.state == FAILED:
            # ── Recovery: wait for new valid base ───────────────────────────
            new_state = BASE if has_base else NONE

        elif ms.state == NONE:
            new_state = BASE if has_base else NONE

        else:
            # ── BASE, ARMED, LATE ───────────────────────────────────────────
            if not has_base:
                new_state = NONE
                ms.consecutive_above = 0
            else:
                # LATE check: too extended from EMA20
                late_condition = (not np.isnan(e20)) and (not np.isnan(a)) and (
                    c > e20 + late_ext * a
                )
                # TRIGGER check (overrides LATE — late triggers are still logged).
                # Uses prior_p (yesterday's pivot) so today's breakout can clear the
                # resistance defined by the prior base.
                trigger_level = prior_p if not np.isnan(prior_p) else p
                if not np.isnan(a) and not np.isnan(trigger_level) and c > trigger_level + breach_atr * a:
                    new_state = TRIGGERED
                    ms.trigger_pivot = trigger_level
                    ms.trigger_atr = a
                    ms.trigger_date = df.index[t]
                    ms.consecutive_above = 1 if c > p else 0
                elif late_condition:
                    new_state = LATE
                else:
                    # ARMED check
                    atr_contracted = (not np.isnan(atr_pct_rank[t])) and (atr_pct_rank[t] <= arm_atr_pct)
                    vol_contracted = (not np.isnan(vol_10m.iloc[t])) and (not np.isnan(vol_50pct.iloc[t])) and (
                        vol_10m.iloc[t] <= vol_50pct.iloc[t]
                    )
                    near_pivot = (not np.isnan(a)) and (abs(c - p) <= arm_dist * a)
                    if near_pivot and atr_contracted and vol_contracted:
                        new_state = ARMED
                    else:
                        new_state = BASE

        # ── Reset trigger state when entering FAILED / NONE ────────────────
        if new_state in (FAILED, NONE) and ms.state not in (FAILED, NONE):
            ms.trigger_pivot = np.nan
            ms.trigger_atr = np.nan
            ms.trigger_date = None
            ms.consecutive_above = 0

        # ── Track days_in_state ─────────────────────────────────────────────
        if new_state == ms.state:
            ms.days_in_state += 1
        else:
            ms.days_in_state = 1

        ms.state = new_state

        states[t] = ms.state
        pivots[t] = p if has_base else np.nan
        trig_pivots[t] = ms.trigger_pivot
        trig_atrs[t] = ms.trigger_atr
        trig_dates[t] = ms.trigger_date
        consec_above[t] = ms.consecutive_above
        days_in[t] = ms.days_in_state
        changed[t] = ms.state != ms.prev_state

    return pd.DataFrame(
        {
            "state": states,
            "pivot": pivots,
            "trigger_pivot": trig_pivots,
            "trigger_atr": trig_atrs,
            "trigger_date": trig_dates,
            "consecutive_above": consec_above,
            "days_in_state": days_in,
            "state_changed": changed,
        },
        index=df.index,
    )
