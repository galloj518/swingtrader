"""Rule-based trade narrative generation.

Produces a short, standardised trade narrative for each top setup.
All text is generated deterministically from feature values — no LLM calls.

Output format (dict):
  setup    : one-sentence setup description
  why      : why this name is relevant today
  entry    : entry idea
  risk     : invalidation / stop description
  targets  : T1 / T2 / T3 summary
  ma_context : moving-average slope context
  verdict  : action verdict (mirrors action_label)

The narrative is designed to be: concise, trader-readable, and honest about
uncertainty. It does not make hard forecasts; it describes the current setup
state and what would need to happen for the trade to work.
"""
from __future__ import annotations

import math
from typing import Any

import pandas as pd

from swingtrader.dashboard.action import (
    ACTION_BREAKOUT,
    ACTION_EXTENDED,
    ACTION_NOW,
    ACTION_PULLBACK,
)
from swingtrader.dashboard.levels import TradeLevels

_NAN = math.nan


def _f(v: Any, decimals: int = 2) -> str:
    """Format a float as a string; '—' if NaN."""
    try:
        fv = float(v)
        if not math.isfinite(fv):
            return "—"
        return f"{fv:.{decimals}f}"
    except (TypeError, ValueError):
        return "—"


def _rs_context(daily_rs_63: float) -> str:
    """Describe relative-strength context."""
    if not math.isfinite(daily_rs_63):
        return "RS unavailable"
    if daily_rs_63 > 0.05:
        return f"outperforming SPY by {daily_rs_63 * 100:.0f}% over 63 days"
    if daily_rs_63 < -0.05:
        return f"underperforming SPY by {abs(daily_rs_63) * 100:.0f}% over 63 days"
    return "tracking SPY closely over 63 days"


def _ma_context(close_vs_sma50: float, close: float, atr: float) -> str:
    """Describe MA position and slope context.

    close_vs_sma50 is (close - SMA50) / close (fractional distance).
    We estimate next-bar SMA50 direction: if today's close > SMA50, adding
    another bar at this price will nudge the 50-bar average higher.
    """
    if not (math.isfinite(close_vs_sma50) and math.isfinite(close)):
        return "MA data unavailable"

    sma50_est = close / (1 + close_vs_sma50) if abs(close_vs_sma50) < 1 else _NAN
    if not math.isfinite(sma50_est):
        return "MA data unavailable"

    dist_atr = (close - sma50_est) / atr if math.isfinite(atr) and atr > 0 else _NAN
    pos = "above" if close_vs_sma50 >= 0 else "below"
    slope = "rising" if close_vs_sma50 >= 0 else "at risk of falling"

    parts = [f"Close is {pos} the 50-day MA"]
    if math.isfinite(dist_atr):
        parts.append(f"({_f(abs(dist_atr), 1)} ATR {pos})")
    parts.append(f"— MA trend is {slope}.")

    # Threshold close to keep SMA50 rising (close must be above SMA50)
    if close_vs_sma50 < 0 and math.isfinite(sma50_est):
        needed = sma50_est
        parts.append(f"Close above {_f(needed)} needed to stop MA from declining.")

    return " ".join(parts)


def _avwap_context(ytd_dist_atr: float) -> str:
    """Describe YTD AVWAP context."""
    if not math.isfinite(ytd_dist_atr):
        return ""
    if ytd_dist_atr > 1.5:
        return f"Price is {_f(ytd_dist_atr, 1)} ATR above YTD AVWAP — extended vs year-open cost basis."
    if ytd_dist_atr < -1.5:
        return f"Price is {_f(abs(ytd_dist_atr), 1)} ATR below YTD AVWAP — below year-open cost basis."
    return f"Price is near YTD AVWAP ({_f(ytd_dist_atr, 1)} ATR) — neutral vs year-open cost basis."


def _compression_context(atr_compression_pct: float) -> str:
    """Describe ATR compression."""
    if not math.isfinite(atr_compression_pct):
        return ""
    if atr_compression_pct <= 30:
        return f"ATR is in the {atr_compression_pct:.0f}th percentile of its own 50-bar history — tight compression."
    if atr_compression_pct <= 60:
        return f"ATR compression is moderate ({atr_compression_pct:.0f}th pct)."
    return f"ATR is elevated ({atr_compression_pct:.0f}th pct) — not compressed; wait for tightening."


def build_narrative(
    row: pd.Series,
    levels: TradeLevels,
    action_label: str,
) -> dict[str, str]:
    """Build the full narrative dict for one symbol.

    Parameters
    ----------
    row          : one row from the scored snapshot DataFrame.
    levels       : computed TradeLevels for this symbol.
    action_label : one of the ACTION_* constants.

    Returns
    -------
    dict with keys: setup, why, entry, risk, targets, ma_context, avwap_context, verdict.
    """
    def _frow(k: str) -> float:
        try:
            v = float(row.get(k, math.nan))
            return v if math.isfinite(v) else math.nan
        except (TypeError, ValueError):
            return math.nan

    sym = str(row.get("user_symbol", row.get("symbol", "?")))
    state = str(row.get("state", "NONE"))
    base_len = int(row.get("base_length", 0) or 0)
    days = int(row.get("days_in_state", 0) or 0)
    score = _frow("composite_score")
    failure = _frow("failure_risk")
    daily_rs = _frow("daily_rs_63")
    close_vs_sma50 = _frow("close_vs_sma50")
    close = _frow("close")
    atr = _frow("atr14")
    atr_comp = _frow("atr_compression_pct")
    ytd_dist = _frow("ytd_dist_atr")
    vol_dry = _frow("volume_dryup")

    # ── Setup sentence ────────────────────────────────────────────────────────
    base_desc = f"{base_len}-bar base" if base_len > 0 else "base"
    pivot_str = _f(levels.pivot)

    if state in {"TRIGGERED", "ACCEPTED"}:
        setup = (
            f"{sym} has triggered above a {base_desc}. "
            f"Trigger pivot: {pivot_str}. "
            f"Now {days} bar(s) post-trigger."
        )
    elif state in {"ARMED"}:
        setup = (
            f"{sym} is armed near the pivot of a {base_desc}. "
            f"Pivot: {pivot_str}. "
            f"ATR and volume are contracting; setup is coiled."
        )
    else:
        vol_str = (" Volume drying up." if math.isfinite(vol_dry) and vol_dry > 0.5 else "")
        comp_str = (_compression_context(atr_comp)) if atr_comp < 50 else ""
        setup = (
            f"{sym} is building a {base_desc}. "
            f"Pivot resistance: {pivot_str}."
            + (f" {comp_str}" if comp_str else "")
            + (f"{vol_str}" if vol_str else "")
        )

    # ── Why it matters now ────────────────────────────────────────────────────
    rs_str = _rs_context(daily_rs)
    score_str = f"Score: {_f(score, 2)}" if math.isfinite(score) else "Score: N/A (models not yet fitted)"
    failure_str = (f", failure risk: {_f(failure, 2)}" if math.isfinite(failure) else "")
    why = f"{rs_str.capitalize()}. {score_str}{failure_str}."

    # ── Entry idea ────────────────────────────────────────────────────────────
    if state in {"TRIGGERED", "ACCEPTED"}:
        if math.isfinite(levels.entry_lo) and math.isfinite(levels.entry_hi):
            entry = (
                f"In trade. Original breakout zone: {_f(levels.entry_lo)}-{_f(levels.entry_hi)}. "
                f"Pullback to pivot ({pivot_str}) would be a second-chance entry."
            )
        else:
            entry = "In trade. Use breakout zone for position sizing reference."
    elif action_label == ACTION_BREAKOUT:
        if math.isfinite(levels.entry_lo):
            entry = (
                f"Buy breakout above {_f(levels.entry_hi)} on volume expansion. "
                f"Entry zone: {_f(levels.entry_lo)}-{_f(levels.entry_hi)}."
            )
        else:
            entry = f"Buy breakout above pivot ({pivot_str}) on volume."
    elif action_label == ACTION_PULLBACK:
        if math.isfinite(levels.entry_lo):
            entry = (
                f"Wait for pullback to {_f(levels.entry_lo)}-{_f(levels.entry_hi)} range "
                f"before entering. Avoid chasing."
            )
        else:
            entry = f"Wait for a controlled pullback to the pivot area ({pivot_str})."
    else:
        entry = "No entry recommended at current levels."

    # ── Risk / invalidation ───────────────────────────────────────────────────
    if math.isfinite(levels.stop) and math.isfinite(atr):
        risk = (
            f"Invalidation: close below {_f(levels.stop)} "
            f"(pivot minus {_f(atr, 2)} ATR). "
            f"This violates the base structure."
        )
    elif math.isfinite(levels.stop):
        risk = f"Invalidation: close below {_f(levels.stop)}."
    else:
        risk = "Stop level unavailable (pivot or ATR missing)."

    # ── Targets ───────────────────────────────────────────────────────────────
    if math.isfinite(levels.t1):
        rr_str = (f" (R/R to T1: {_f(levels.risk_reward_t1, 1)}x)" if math.isfinite(levels.risk_reward_t1) else "")
        targets = (
            f"T1: {_f(levels.t1)}{rr_str}. "
            f"T2: {_f(levels.t2)}. "
            f"T3: {_f(levels.t3)}."
        )
    else:
        targets = "Targets unavailable (pivot or ATR missing)."

    # ── MA context ────────────────────────────────────────────────────────────
    ma_ctx = _ma_context(close_vs_sma50, close, atr)

    # ── AVWAP context ─────────────────────────────────────────────────────────
    avwap_ctx = _avwap_context(ytd_dist)

    # ── Verdict ───────────────────────────────────────────────────────────────
    if action_label == ACTION_NOW:
        verdict = f"Active breakout - consider entry within {_f(levels.entry_lo)}-{_f(levels.entry_hi)} range."
    elif action_label == ACTION_BREAKOUT:
        verdict = f"Ready to trigger. Buy above {_f(levels.entry_hi)} on volume."
    elif action_label == ACTION_PULLBACK:
        verdict = "Setup exists but extended from optimal entry. Wait for pullback."
    elif action_label == ACTION_EXTENDED:
        verdict = "Do not chase. Let the move consolidate before reassessing."
    else:
        verdict = "Avoid — score or failure risk does not meet minimum threshold."

    return {
        "setup": setup,
        "why": why,
        "entry": entry,
        "risk": risk,
        "targets": targets,
        "ma_context": ma_ctx,
        "avwap_context": avwap_ctx,
        "verdict": verdict,
    }
