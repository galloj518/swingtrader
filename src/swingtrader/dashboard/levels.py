"""Trade level computation.

Computes actionable trade levels from the pivot price and ATR.

All levels are derived from observables (pivot, atr14, close) plus
ATR multiples from config/scoring.yaml. No levels are fabricated.

Method
------
Pivot
  The base's resistance high, from base_detect output. Already in the snapshot.

Entry zone
  Breakout entry (BASE/ARMED state, waiting for trigger):
    lo = pivot
    hi = pivot + BREACH_ATR * atr    (= labels.yaml triggered_breakout.pivot_breach_atr = 0.10)
  Already triggered (TRIGGERED/ACCEPTED):
    lo = trigger_pivot  (use pivot as proxy if trigger_pivot unavailable)
    hi = trigger_pivot + BREACH_ATR * atr

Stop / invalidation
  stop = pivot - STOP_ATR * atr     (= scoring.yaml confirmed.atr_stop = 1.0)
  Rationale: a close below pivot minus one ATR invalidates the base.

Targets
  T1 = pivot + CONF_ATR * atr       (= scoring.yaml confirmed.atr_gain_target = 2.0)
  T2 = pivot + T2_ATR * atr         (3.5× ATR — midpoint between T1 and T3)
  T3 = pivot + T3_ATR * atr         (5.0× ATR — full-extension target)

Support ladder (S1 / S2 / S3)
  BASE/ARMED (entry not yet triggered):
    S1 = close - 0.5 * atr          (nearest support)
    S2 = stop                        (hard stop / invalidation)
    S3 = pivot - 2.0 * atr          (deep base support)
  TRIGGERED/ACCEPTED (in trade):
    S1 = pivot                       (pivot is now support)
    S2 = stop
    S3 = pivot - 1.5 * atr

Resistance ladder (R1 / R2 / R3)
  BASE/ARMED:
    R1 = pivot                       (immediate resistance = breakout level)
    R2 = T1
    R3 = T2
  TRIGGERED/ACCEPTED:
    R1 = T1
    R2 = T2
    R3 = T3

When pivot or atr14 is NaN, all levels are returned as NaN.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

# ── ATR multipliers (mirrors config/scoring.yaml and config/labels.yaml) ──────

BREACH_ATR: float = 0.10    # labels.yaml triggered_breakout.pivot_breach_atr
STOP_ATR: float = 1.0       # scoring.yaml confirmed.atr_stop
CONF_ATR: float = 2.0       # scoring.yaml confirmed.atr_gain_target
T2_ATR: float = 3.5
T3_ATR: float = 5.0

_PRE_TRIGGER = frozenset({"NONE", "BASE", "ARMED", "LATE", "EXHAUSTED"})
_IN_TRADE = frozenset({"TRIGGERED", "ACCEPTED", "CONFIRMED"})


@dataclass
class TradeLevels:
    """All computed trade levels for one symbol."""
    pivot: float = math.nan
    entry_lo: float = math.nan
    entry_hi: float = math.nan
    stop: float = math.nan
    t1: float = math.nan
    t2: float = math.nan
    t3: float = math.nan
    s1: float = math.nan
    s2: float = math.nan
    s3: float = math.nan
    r1: float = math.nan
    r2: float = math.nan
    r3: float = math.nan
    # Risk/reward from entry midpoint to T1
    risk_reward_t1: float = math.nan
    # ATR multiple from entry to stop (should be ~1.0 by construction)
    entry_stop_atr: float = math.nan

    def to_dict(self) -> dict[str, float]:
        return {
            "pivot": self.pivot,
            "entry_lo": self.entry_lo,
            "entry_hi": self.entry_hi,
            "stop": self.stop,
            "t1": self.t1,
            "t2": self.t2,
            "t3": self.t3,
            "s1": self.s1,
            "s2": self.s2,
            "s3": self.s3,
            "r1": self.r1,
            "r2": self.r2,
            "r3": self.r3,
            "risk_reward_t1": self.risk_reward_t1,
            "entry_stop_atr": self.entry_stop_atr,
        }


def compute_levels(row: pd.Series) -> TradeLevels:
    """Compute trade levels for a single snapshot row.

    Parameters
    ----------
    row : Series with fields: state, pivot, atr14, close.

    Returns
    -------
    TradeLevels dataclass. All NaN if pivot or atr14 is missing.
    """
    def _f(k: str) -> float:
        try:
            v = float(row.get(k, math.nan))
            return v if math.isfinite(v) else math.nan
        except (TypeError, ValueError):
            return math.nan

    pivot = _f("pivot")
    atr = _f("atr14")
    close = _f("close")
    state = str(row.get("state", "NONE"))

    if not (math.isfinite(pivot) and math.isfinite(atr) and atr > 0):
        return TradeLevels(pivot=pivot)

    # Shared levels
    entry_lo = pivot
    entry_hi = pivot + BREACH_ATR * atr
    stop = pivot - STOP_ATR * atr
    t1 = pivot + CONF_ATR * atr
    t2 = pivot + T2_ATR * atr
    t3 = pivot + T3_ATR * atr

    if state in _IN_TRADE:
        # Already triggered: pivot is now support
        s1 = pivot
        s2 = stop
        s3 = pivot - 1.5 * atr
        r1 = t1
        r2 = t2
        r3 = t3
    else:
        # Pre-trigger: pivot is resistance
        nearby_support = (close - 0.5 * atr) if math.isfinite(close) else (pivot - 0.5 * atr)
        s1 = max(nearby_support, stop + 0.01)  # S1 always above stop
        s2 = stop
        s3 = pivot - 2.0 * atr
        r1 = pivot
        r2 = t1
        r3 = t2

    # Risk/reward from entry midpoint to T1
    entry_mid = (entry_lo + entry_hi) / 2
    risk = entry_mid - stop
    reward = t1 - entry_mid
    rr = (reward / risk) if risk > 1e-6 else math.nan
    entry_stop_atr_dist = risk / atr if atr > 1e-6 else math.nan

    return TradeLevels(
        pivot=round(pivot, 2),
        entry_lo=round(entry_lo, 2),
        entry_hi=round(entry_hi, 2),
        stop=round(stop, 2),
        t1=round(t1, 2),
        t2=round(t2, 2),
        t3=round(t3, 2),
        s1=round(s1, 2),
        s2=round(s2, 2),
        s3=round(s3, 2),
        r1=round(r1, 2),
        r2=round(r2, 2),
        r3=round(r3, 2),
        risk_reward_t1=round(rr, 2),
        entry_stop_atr=round(entry_stop_atr_dist, 2),
    )
