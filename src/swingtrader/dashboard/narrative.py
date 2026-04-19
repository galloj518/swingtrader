"""Rule-based trade narrative generation.

Produces a short, standardised trade narrative for each top setup.
All text is generated deterministically from feature values — no LLM calls.

Output format (dict):
  setup       : one-sentence setup description
  why         : why this name is relevant today
  entry       : entry idea
  risk        : invalidation / stop description
  targets     : T1 / T2 / T3 summary
  ma_context  : moving-average slope context
  avwap_context : AVWAP position context
  verdict     : action verdict (mirrors action_label)
  trade_plan  : structured trade plan with all key levels

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


def _weekly_context(row: pd.Series) -> str:
    """Describe weekly trend context from WMA10 distance and RS.

    Uses weekly_dist_wma10 (distance from 10-week WMA in ATR units) and
    weekly_rs_26 (26-week relative strength vs SPY, fractional).
    """
    weekly_dist = row.get("weekly_dist_wma10", _NAN)
    weekly_rs = row.get("weekly_rs_26", _NAN)

    try:
        weekly_dist = float(weekly_dist)
        if not math.isfinite(weekly_dist):
            weekly_dist = _NAN
    except (TypeError, ValueError):
        weekly_dist = _NAN

    try:
        weekly_rs = float(weekly_rs)
        if not math.isfinite(weekly_rs):
            weekly_rs = _NAN
    except (TypeError, ValueError):
        weekly_rs = _NAN

    if not math.isfinite(weekly_dist) and not math.isfinite(weekly_rs):
        return "Weekly: data unavailable."

    parts = []

    if math.isfinite(weekly_dist):
        direction = "above" if weekly_dist >= 0 else "below"
        parts.append(f"Weekly: price {direction} 10-week WMA ({_f(weekly_dist, 1)} ATR).")
    else:
        parts.append("Weekly: WMA data unavailable.")

    if math.isfinite(weekly_rs):
        if weekly_rs > 0.03:
            parts.append(f"Weekly RS: outperforming ({weekly_rs * 100:.1f}% vs SPY).")
        elif weekly_rs < -0.03:
            parts.append(f"Weekly RS: underperforming ({weekly_rs * 100:.1f}% vs SPY).")
        else:
            parts.append("Weekly RS: in line with SPY.")
    else:
        parts.append("Weekly RS: unavailable.")

    return " ".join(parts)


def _regime_context(row: pd.Series) -> str:
    """Describe current market regime context.

    Uses regime_spy_trend, regime_vix_level, regime_spy_above_200sma.
    """
    spy_trend = row.get("regime_spy_trend", None)
    vix_level = row.get("regime_vix_level", _NAN)
    spy_above_200 = row.get("regime_spy_above_200sma", None)

    try:
        vix_level = float(vix_level)
        if not math.isfinite(vix_level):
            vix_level = _NAN
    except (TypeError, ValueError):
        vix_level = _NAN

    parts = ["Market regime:"]

    # SPY trend
    if spy_trend is not None:
        trend_str = str(spy_trend).lower()
        if "up" in trend_str:
            parts.append("SPY uptrend.")
        elif "down" in trend_str:
            parts.append("SPY downtrend.")
        else:
            parts.append(f"SPY trend: {spy_trend}.")
    elif spy_above_200 is not None:
        above = bool(spy_above_200)
        parts.append("SPY above 200-day MA." if above else "SPY below 200-day MA.")
    else:
        parts.append("SPY trend: unavailable.")

    # VIX
    if math.isfinite(vix_level):
        if vix_level < 15:
            vix_desc = "low (complacent)"
            env = "Breakout environment: favorable."
        elif vix_level < 20:
            vix_desc = "neutral"
            env = "Breakout environment: selective."
        elif vix_level < 30:
            vix_desc = "elevated"
            env = "Breakout environment: cautious — size smaller."
        else:
            vix_desc = "high (risk-off)"
            env = "Breakout environment: unfavorable — wait for stabilization."
        parts.append(f"VIX {vix_level:.1f} ({vix_desc}).")
        parts.append(env)
    else:
        parts.append("VIX: unavailable.")

    return " ".join(parts)


def build_narrative(
    row: pd.Series,
    levels: TradeLevels,
    action_label: str,
    bucket: str = "",
) -> dict[str, str]:
    """Build the full narrative dict for one symbol.

    Routes to a bucket-specific narrative template when the bucket is known,
    giving each setup type a tailored narrative that matches its trade logic.
    Falls back to the generic narrative for unrecognised buckets.

    Parameters
    ----------
    row          : one row from the scored snapshot DataFrame.
    levels       : computed TradeLevels for this symbol.
    action_label : one of the ACTION_* constants.
    bucket       : optional bucket string (BUCKET_* constant) for routing.

    Returns
    -------
    dict with keys: setup, why, entry, risk, targets, ma_context, avwap_context,
    verdict, trade_plan.
    """
    if bucket == "breakout_long":
        return _build_breakout_narrative(row, levels, action_label)
    if bucket == "pullback_long":
        return _build_pullback_narrative(row, levels, action_label)
    if bucket == "extended_leader":
        return _build_extended_leader_narrative(row, levels)
    if bucket == "reversal_speculative":
        return _build_reversal_narrative(row, levels, action_label)
    return _build_generic_narrative(row, levels, action_label)


def _build_generic_narrative(
    row: pd.Series,
    levels: TradeLevels,
    action_label: str,
) -> dict[str, str]:
    """Generic narrative — original logic, used as fallback."""
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
    base_desc = f"{base_len}-bar" if base_len > 0 else ""
    pivot_str = _f(levels.pivot)

    if state in {"TRIGGERED", "ACCEPTED"}:
        comp_str = _compression_context(atr_comp) if math.isfinite(atr_comp) and atr_comp < 50 else ""
        setup = (
            f"{sym} has triggered above a {base_desc + ' ' if base_desc else ''}base "
            f"(Day {days} in {state} state). "
            f"Trigger pivot: ${pivot_str}."
            + (f" {comp_str}" if comp_str else "")
        )
    elif state == "ARMED":
        comp_str = _compression_context(atr_comp) if math.isfinite(atr_comp) and atr_comp < 50 else ""
        setup = (
            f"{sym} is armed near the pivot of a {base_desc + ' ' if base_desc else ''}base "
            f"(Day {days} in ARMED state). "
            f"Pivot resistance: ${pivot_str}."
            + (f" {comp_str}" if comp_str else " ATR and volume are contracting; setup is coiled.")
        )
    else:
        vol_str = " Volume drying up." if math.isfinite(vol_dry) and vol_dry > 0.5 else ""
        comp_str = _compression_context(atr_comp) if math.isfinite(atr_comp) and atr_comp < 50 else ""
        base_type = f"{base_desc} flat base" if base_desc else "base"
        setup = (
            f"{sym} is building a {base_type} "
            f"(Day {days} in {state} state). "
            f"Pivot resistance: ${pivot_str}."
            + (f" {comp_str}" if comp_str else "")
            + (f"{vol_str}" if vol_str else "")
        )

    # ── Why it matters now ────────────────────────────────────────────────────
    rs_str = _rs_context(daily_rs)

    if math.isfinite(score):
        score_str = f"Score: {_f(score, 2)}"
    else:
        score_str = "Score: N/A (models not yet fitted)"

    failure_str = (f", failure risk {_f(failure, 2)}" if math.isfinite(failure) else "")

    weekly_str = _weekly_context(row)
    regime_str = _regime_context(row)

    why = (
        f"{sym} is {rs_str}. "
        f"{score_str}{failure_str}. "
        f"{weekly_str} "
        f"{regime_str}"
    )

    # ── Entry idea ────────────────────────────────────────────────────────────
    atr_tenth = atr * 0.10 if math.isfinite(atr) else math.nan
    trigger_level = levels.pivot + atr_tenth if (math.isfinite(levels.pivot) and math.isfinite(atr_tenth)) else math.nan
    pullback_lo = levels.pivot - atr_tenth if (math.isfinite(levels.pivot) and math.isfinite(atr_tenth)) else math.nan

    if state in {"TRIGGERED", "ACCEPTED"}:
        if math.isfinite(levels.entry_lo) and math.isfinite(levels.entry_hi):
            entry = (
                f"In trade. Original breakout zone: ${_f(levels.entry_lo)}-${_f(levels.entry_hi)}. "
                f"Pullback entry: ${_f(pullback_lo)}-${pivot_str} zone. "
                f"Avoid adding above ${_f(levels.entry_hi)}."
            )
        else:
            entry = (
                f"In trade. Use breakout zone for position sizing reference. "
                f"Pullback to pivot (${pivot_str}) is the second-chance entry."
            )
    elif action_label == ACTION_BREAKOUT:
        trigger_str = _f(trigger_level) if math.isfinite(trigger_level) else _f(levels.entry_hi)
        if math.isfinite(levels.entry_lo):
            entry = (
                f"Buy break above ${trigger_str} (pivot + 0.10 ATR) on volume expansion. "
                f"Aggressive entry: ${pivot_str} pivot break on close. "
                f"Pullback entry: ${_f(pullback_lo)}-${pivot_str} zone."
            )
        else:
            entry = f"Buy breakout above pivot (${pivot_str}) on volume."
    elif action_label == ACTION_PULLBACK:
        if math.isfinite(levels.entry_lo):
            entry = (
                f"Wait for pullback to ${_f(levels.entry_lo)}-${_f(levels.entry_hi)} range "
                f"before entering. Avoid chasing current levels."
            )
        else:
            entry = f"Wait for a controlled pullback to the pivot area (${pivot_str})."
    else:
        entry = "No entry recommended at current levels."

    # ── Risk / invalidation ───────────────────────────────────────────────────
    if math.isfinite(levels.stop) and math.isfinite(atr):
        daily_stop = levels.stop - atr * 0.5 if math.isfinite(atr) else math.nan
        risk = (
            f"Invalidation: close below ${_f(levels.stop)} "
            f"(pivot minus {_f(atr, 2)} ATR). "
            f"Violates base structure."
            + (f" Daily stop loss: ${_f(daily_stop)}." if math.isfinite(daily_stop) else "")
        )
    elif math.isfinite(levels.stop):
        risk = f"Invalidation: close below ${_f(levels.stop)}."
    else:
        risk = "Stop level unavailable (pivot or ATR missing)."

    # ── Targets ───────────────────────────────────────────────────────────────
    if math.isfinite(levels.t1) and math.isfinite(close) and close > 0:
        t1_pct = (levels.t1 - close) / close * 100
        t2_pct = (levels.t2 - close) / close * 100 if math.isfinite(levels.t2) else math.nan
        t3_pct = (levels.t3 - close) / close * 100 if math.isfinite(levels.t3) else math.nan
        rr_str = (f" (R/R {_f(levels.risk_reward_t1, 1)}x from entry mid)" if math.isfinite(levels.risk_reward_t1) else "")

        t1_str = f"T1: ${_f(levels.t1)} (+{t1_pct:.1f}%){rr_str}."
        t2_str = f" T2: ${_f(levels.t2)} (+{t2_pct:.1f}%)." if math.isfinite(t2_pct) else f" T2: ${_f(levels.t2)}."
        t3_str = f" T3: ${_f(levels.t3)} (+{t3_pct:.1f}%)." if math.isfinite(t3_pct) else f" T3: ${_f(levels.t3)}."

        targets = t1_str + t2_str + t3_str + " Partial at T1 recommended."
    elif math.isfinite(levels.t1):
        rr_str = (f" (R/R to T1: {_f(levels.risk_reward_t1, 1)}x)" if math.isfinite(levels.risk_reward_t1) else "")
        targets = (
            f"T1: ${_f(levels.t1)}{rr_str}. "
            f"T2: ${_f(levels.t2)}. "
            f"T3: ${_f(levels.t3)}. "
            "Partial at T1 recommended."
        )
    else:
        targets = "Targets unavailable (pivot or ATR missing)."

    # ── MA context ────────────────────────────────────────────────────────────
    ma_ctx = _ma_context(close_vs_sma50, close, atr)

    # ── AVWAP context ─────────────────────────────────────────────────────────
    avwap_ctx = _avwap_context(ytd_dist)

    # ── Verdict ───────────────────────────────────────────────────────────────
    # Extension always wins over entry labels — prevents contradictions between
    # action label and narrative.
    is_extended_flag = bool(row.get("is_extended", False))
    ext_reasons_str = str(row.get("extension_reasons", ""))
    regime_brief = _regime_context(row)

    if is_extended_flag or action_label == ACTION_EXTENDED:
        ext_note = f" ({ext_reasons_str})" if ext_reasons_str else ""
        verdict = (
            f"Extended — do not chase{ext_note}. "
            f"Wait for a pullback toward ${_f(levels.t1) if math.isfinite(levels.t1) else _f(levels.pivot)} "
            f"or a new base to form before reassessing."
        )
    elif action_label == ACTION_NOW:
        verdict = (
            f"Actionable now — break above ${_f(trigger_level) if math.isfinite(trigger_level) else _f(levels.entry_hi)} "
            f"on volume. Set alert. Stop: ${_f(levels.stop)}. "
            f"{regime_brief}"
        )
    elif action_label == ACTION_BREAKOUT:
        verdict = (
            f"Actionable on breakout above ${_f(trigger_level) if math.isfinite(trigger_level) else _f(levels.entry_hi)} on volume. "
            f"Stop: ${_f(levels.stop)}. "
            f"{regime_brief}"
        )
    elif action_label == ACTION_PULLBACK:
        verdict = (
            f"Actionable on pullback — wait for entry zone ${_f(levels.entry_lo)}-${_f(levels.entry_hi)}. "
            f"Avoid chasing; setup exists but not at optimal price. "
            f"Stop: ${_f(levels.stop)}."
        )
    else:
        verdict = "Avoid — score or failure risk does not meet minimum threshold."

    # ── Trade plan ────────────────────────────────────────────────────────────
    trade_plan = (
        f"Entry trigger: close/break above ${_f(levels.entry_hi)} on volume. "
        f"Pullback entry: ${_f(levels.entry_lo)}-${_f(levels.entry_hi)}. "
        f"Stop/invalidation: ${_f(levels.stop)}. "
        f"T1: ${_f(levels.t1)} | T2: ${_f(levels.t2)} | T3: ${_f(levels.t3)}. "
        f"R/R to T1: {_f(levels.risk_reward_t1, 1)}x from entry mid."
    )

    return {
        "setup": setup,
        "why": why,
        "entry": entry,
        "risk": risk,
        "targets": targets,
        "ma_context": ma_ctx,
        "avwap_context": avwap_ctx,
        "verdict": verdict,
        "trade_plan": trade_plan,
    }


# ── Bucket-specific narrative templates ───────────────────────────────────────

def _frow_safe(row: pd.Series, k: str) -> float:
    try:
        v = float(row.get(k, math.nan))
        return v if math.isfinite(v) else math.nan
    except (TypeError, ValueError):
        return math.nan


def _build_breakout_narrative(
    row: pd.Series,
    levels: TradeLevels,
    action_label: str,
) -> dict[str, str]:
    """Breakout-specific narrative: pivot clarity, compression, entry trigger, timing."""
    sym = str(row.get("user_symbol", row.get("symbol", "?")))
    state = str(row.get("state", "NONE"))
    base_len = int(row.get("base_length", 0) or 0)
    days = int(row.get("days_in_state", 0) or 0)
    close = _frow_safe(row, "close")
    atr = _frow_safe(row, "atr14")
    atr_comp = _frow_safe(row, "atr_compression_pct")
    daily_rs = _frow_safe(row, "daily_rs_63")
    failure = _frow_safe(row, "failure_risk")
    setup_score = _frow_safe(row, "setup_score")
    weekly_dist = _frow_safe(row, "weekly_dist_wma10")
    vol_dry = _frow_safe(row, "volume_dryup")
    is_extended = bool(row.get("is_extended", False))
    ext_reasons = str(row.get("extension_reasons", ""))

    base_desc = f"{base_len}-bar " if base_len > 0 else ""
    pivot_str = _f(levels.pivot)
    atr_tenth = atr * 0.10 if math.isfinite(atr) else math.nan
    trigger_level = levels.pivot + atr_tenth if (math.isfinite(levels.pivot) and math.isfinite(atr_tenth)) else math.nan

    # Setup — emphasise pivot, base structure, compression
    comp_desc = _compression_context(atr_comp) if math.isfinite(atr_comp) else ""
    vol_desc = "Volume is drying up — institutional supply exhaustion signal." if (math.isfinite(vol_dry) and vol_dry < 0.7) else ""

    if state in {"TRIGGERED", "ACCEPTED"}:
        setup = (
            f"{sym} triggered above a {base_desc}base (Day {days}). "
            f"Pivot: ${pivot_str}. "
            + (f"{comp_desc} " if comp_desc else "")
            + (f"{vol_desc}" if vol_desc else "")
        )
    else:
        setup = (
            f"{sym} is coiled at pivot in a {base_desc}base (Day {days} in {state}). "
            f"Pivot resistance: ${pivot_str}. "
            + (f"{comp_desc} " if comp_desc else "")
            + (f"{vol_desc}" if vol_desc else "Watching for volume contraction to confirm base quality.")
        )

    # Why now — RS + setup score + weekly context + regime
    rs_str = _rs_context(daily_rs)
    ss_str = f"Setup score: {_f(setup_score, 2)}." if math.isfinite(setup_score) else ""
    fr_str = f" Failure risk: {_f(failure, 2)}." if math.isfinite(failure) else ""
    wk_str = ""
    if math.isfinite(weekly_dist):
        wk_str = (f"Weekly: {_f(weekly_dist * 100, 1)}% above 10-wk WMA — strong weekly structure." if weekly_dist > 0
                  else "Weekly: near 10-wk WMA — watch for weekly support hold.")

    why = f"{sym} is {rs_str}. {ss_str}{fr_str} {wk_str} {_regime_context(row)}"

    # Entry — specific breakout trigger
    if math.isfinite(trigger_level) and math.isfinite(levels.entry_lo):
        entry = (
            f"Breakout trigger: break above ${_f(trigger_level)} on volume >= 1.5x average. "
            f"Aggressive pivot buy: ${pivot_str}. "
            f"Pullback entry zone: ${_f(levels.entry_lo)}-${pivot_str}. "
            f"Do not chase more than 0.5 ATR above trigger."
        )
    else:
        entry = f"Buy break above pivot (${pivot_str}) on volume expansion."

    # Risk — invalidation
    if math.isfinite(levels.stop):
        risk = (
            f"Invalidation: close below ${_f(levels.stop)} (pivot - 1 ATR). "
            f"Base structure violated — full exit. "
            + (f"Daily stop: ${levels.stop - atr * 0.3:.2f}." if math.isfinite(atr) else "")
        )
    else:
        risk = "Stop level unavailable (pivot or ATR missing)."

    # Targets
    targets = _format_targets(levels, close)

    # MA and AVWAP context
    ma_ctx = _ma_context(_frow_safe(row, "close_vs_sma50"), close, atr)
    avwap_ctx = _avwap_context(_frow_safe(row, "ytd_dist_atr"))

    # Verdict — extension always wins
    if is_extended:
        ext_note = f" ({ext_reasons})" if ext_reasons else ""
        verdict = (
            f"Extended — do not chase{ext_note}. "
            f"Wait for pullback to ${_f(levels.pivot)} or new base."
        )
    elif action_label in (ACTION_NOW, ACTION_BREAKOUT):
        verdict = (
            f"Decision-ready breakout. Trigger: ${_f(trigger_level) if math.isfinite(trigger_level) else pivot_str}. "
            f"Stop: ${_f(levels.stop)}. T1: ${_f(levels.t1)} (R/R {_f(levels.risk_reward_t1, 1)}x)."
        )
    else:
        verdict = f"Actionable on pullback to ${_f(levels.entry_lo)}-${pivot_str}. Stop: ${_f(levels.stop)}."

    trade_plan = (
        f"Trigger: break above ${_f(trigger_level) if math.isfinite(trigger_level) else _f(levels.entry_hi)} on volume. "
        f"Stop: ${_f(levels.stop)}. "
        f"T1: ${_f(levels.t1)} | T2: ${_f(levels.t2)} | T3: ${_f(levels.t3)}. "
        f"R/R to T1: {_f(levels.risk_reward_t1, 1)}x."
    )

    return {
        "setup": setup, "why": why, "entry": entry, "risk": risk,
        "targets": targets, "ma_context": ma_ctx, "avwap_context": avwap_ctx,
        "verdict": verdict, "trade_plan": trade_plan,
    }


def _build_pullback_narrative(
    row: pd.Series,
    levels: TradeLevels,
    action_label: str,
) -> dict[str, str]:
    """Pullback-specific narrative: support context, MA stack, resumption potential."""
    sym = str(row.get("user_symbol", row.get("symbol", "?")))
    state = str(row.get("state", "NONE"))
    days = int(row.get("days_in_state", 0) or 0)
    close = _frow_safe(row, "close")
    atr = _frow_safe(row, "atr14")
    daily_rs = _frow_safe(row, "daily_rs_63")
    failure = _frow_safe(row, "failure_risk")
    cvs50 = _frow_safe(row, "close_vs_sma50")
    ytd_dist = _frow_safe(row, "ytd_dist_atr")
    swing_low_dist = _frow_safe(row, "swing_low_dist_atr")
    vol_dry = _frow_safe(row, "volume_dryup")
    is_extended = bool(row.get("is_extended", False))
    ext_reasons = str(row.get("extension_reasons", ""))

    pivot_str = _f(levels.pivot)

    # Support context: describe what the pullback is landing on
    support_lines = []
    if math.isfinite(cvs50):
        if -0.05 <= cvs50 <= 0.02:
            support_lines.append(f"testing SMA50 ({_f(cvs50 * 100, 1)}% from it)")
        elif cvs50 > 0.02:
            support_lines.append(f"above SMA50 (+{_f(cvs50 * 100, 1)}%)")
    if math.isfinite(ytd_dist) and abs(ytd_dist) < 1.5:
        support_lines.append(f"near YTD AVWAP ({_f(ytd_dist, 1)} ATR)")
    if math.isfinite(swing_low_dist) and 0 <= swing_low_dist <= 2.0:
        support_lines.append(f"near swing-low AVWAP ({_f(swing_low_dist, 1)} ATR above)")
    support_str = " | ".join(support_lines) if support_lines else "near key support zone"

    vol_desc = "Volume contracting — healthy pullback characteristic." if (math.isfinite(vol_dry) and vol_dry < 0.7) else ""

    setup = (
        f"{sym} is pulling back (Day {days} in {state}). "
        f"Pivot: ${pivot_str}. "
        f"Pullback landing: {support_str}. "
        + (f"{vol_desc}" if vol_desc else "")
    )

    # Why now
    rs_str = _rs_context(daily_rs)
    fr_str = f"Failure risk: {_f(failure, 2)}." if math.isfinite(failure) else ""
    wk_str = _weekly_context(row)

    why = (
        f"{sym} is {rs_str}. {fr_str} "
        f"Structure: {support_str}. "
        f"{wk_str} {_regime_context(row)}"
    )

    # Entry — pullback zone with AVWAP / SMA context
    entry_lo = _f(levels.entry_lo)
    entry_hi = _f(levels.entry_hi)
    if math.isfinite(levels.entry_lo) and math.isfinite(levels.entry_hi):
        entry = (
            f"Pullback entry zone: ${entry_lo}-${entry_hi}. "
            f"Ideal entry: price stabilises with volume dry-up at support ({support_str}). "
            f"Avoid adding if price drops through ${_f(levels.stop)}."
        )
    else:
        entry = f"Wait for pullback into ${pivot_str} support zone with volume dry-up."

    # Risk
    if math.isfinite(levels.stop):
        risk = (
            f"Invalidation: close below ${_f(levels.stop)}. "
            f"Support stack breaks — do not hold through."
        )
    else:
        risk = "Stop level unavailable (pivot or ATR missing)."

    # Targets — for pullback re-entry, T1 is prior high or resistance
    targets = _format_targets(levels, close, pullback_context=True)

    ma_ctx = _ma_context(cvs50, close, atr)
    avwap_ctx = _avwap_context(ytd_dist)

    # Verdict
    if is_extended:
        ext_note = f" ({ext_reasons})" if ext_reasons else ""
        verdict = f"Extended{ext_note} — wait for deeper pullback to ${_f(levels.entry_lo)}-${entry_hi}."
    elif action_label == "Actionable on pullback":
        verdict = (
            f"Constructive pullback. Entry zone: ${entry_lo}-${entry_hi} ({support_str}). "
            f"Stop: ${_f(levels.stop)}. T1: ${_f(levels.t1)}."
        )
    else:
        verdict = f"Pullback candidate — wait for entry zone ${entry_lo}-${entry_hi}. Stop: ${_f(levels.stop)}."

    trade_plan = (
        f"Entry zone: ${entry_lo}-${entry_hi}. "
        f"Stop: ${_f(levels.stop)}. "
        f"T1: ${_f(levels.t1)} | T2: ${_f(levels.t2)}. "
        f"R/R to T1: {_f(levels.risk_reward_t1, 1)}x."
    )

    return {
        "setup": setup, "why": why, "entry": entry, "risk": risk,
        "targets": targets, "ma_context": ma_ctx, "avwap_context": avwap_ctx,
        "verdict": verdict, "trade_plan": trade_plan,
    }


def _build_extended_leader_narrative(
    row: pd.Series,
    levels: TradeLevels,
) -> dict[str, str]:
    """Extended-leader narrative: healthy but too extended for fresh entry."""
    sym = str(row.get("user_symbol", row.get("symbol", "?")))
    state = str(row.get("state", "NONE"))
    days = int(row.get("days_in_state", 0) or 0)
    close = _frow_safe(row, "close")
    atr = _frow_safe(row, "atr14")
    dist = _frow_safe(row, "dist_to_pivot_atr")
    daily_rs = _frow_safe(row, "daily_rs_63")
    cvs50 = _frow_safe(row, "close_vs_sma50")
    ytd_dist = _frow_safe(row, "ytd_dist_atr")
    ext_reasons = str(row.get("extension_reasons", ""))

    # How extended
    dist_str = f"{_f(dist, 1)} ATR above pivot" if math.isfinite(dist) else "extended above pivot"
    ext_note = f" ({ext_reasons})" if ext_reasons else ""

    setup = (
        f"{sym} is extended ({dist_str}, Day {days} in {state}){ext_note}. "
        f"Strong leader, but current price is not a safe fresh-entry point."
    )

    why = (
        f"{sym} is {_rs_context(daily_rs)} — an outperforming leader. "
        f"{_weekly_context(row)} "
        f"Not actionable for new entries at current extension."
    )

    # What would make it actionable again
    add_lo = levels.pivot
    add_hi = levels.pivot + atr * 0.5 if math.isfinite(atr) else math.nan
    sma50_est = close / (1 + cvs50) if math.isfinite(cvs50) and abs(cvs50) < 1 else math.nan
    avwap_add_str = ""
    if math.isfinite(ytd_dist) and ytd_dist > 1.0 and math.isfinite(atr) and math.isfinite(close):
        avwap_approx = close - ytd_dist * atr
        avwap_add_str = f" or YTD AVWAP (~${_f(avwap_approx)})"

    if math.isfinite(add_lo):
        entry = (
            f"Do NOT enter at current levels. "
            f"Potential re-entry: pullback to ${_f(add_lo)}-${_f(add_hi) if math.isfinite(add_hi) else _f(add_lo)}"
            f"{avwap_add_str}"
            + (f" (near SMA50 ~${_f(sma50_est)})" if math.isfinite(sma50_est) else "")
            + ". Wait for a new base or meaningful vol-dry pullback."
        )
    else:
        entry = "Monitor only. Do not enter at current extension."

    risk = (
        f"If already holding: alert on close below ${_f(levels.stop)}. "
        f"Not a new-entry stop — that level is too wide for a fresh buy."
    )

    targets = (
        f"Already extended. Informational only: prior T1 was ~${_f(levels.t1)}, T2 ~${_f(levels.t2)}. "
        f"Hold if in position; new entries not advised here."
    )

    ma_ctx = _ma_context(cvs50, close, atr)
    avwap_ctx = _avwap_context(ytd_dist)

    verdict = (
        f"Extended leader — monitor, do not chase. "
        f"Add zone on pullback to ${_f(add_lo)}-${_f(add_hi) if math.isfinite(add_hi) else _f(add_lo)}"
        + avwap_add_str + "."
    )

    trade_plan = (
        f"No new entry. Watch zone: ${_f(add_lo)}-${_f(add_hi) if math.isfinite(add_hi) else _f(add_lo)}"
        + avwap_add_str + ". "
        f"If holding: stop ${_f(levels.stop)}."
    )

    return {
        "setup": setup, "why": why, "entry": entry, "risk": risk,
        "targets": targets, "ma_context": ma_ctx, "avwap_context": avwap_ctx,
        "verdict": verdict, "trade_plan": trade_plan,
    }


def _build_reversal_narrative(
    row: pd.Series,
    levels: TradeLevels,
    action_label: str,
) -> dict[str, str]:
    """Speculative-reversal narrative: structural weakness + reversal potential."""
    sym = str(row.get("user_symbol", row.get("symbol", "?")))
    state = str(row.get("state", "NONE"))
    days = int(row.get("days_in_state", 0) or 0)
    close = _frow_safe(row, "close")
    atr = _frow_safe(row, "atr14")
    daily_rs = _frow_safe(row, "daily_rs_63")
    ytd_dist = _frow_safe(row, "ytd_dist_atr")
    rejection_reasons = str(row.get("rejection_reasons", ""))

    setup = (
        f"{sym} is a speculative watch (Day {days} in {state}). "
        f"Structural issues: {rejection_reasons or 'see eligibility details'}. "
        f"Showing reversal / basing characteristics — treat as higher-risk."
    )

    why = (
        f"Failed primary long gates but showing potential for a base/reversal. "
        f"{sym} is {_rs_context(daily_rs)}. "
        + (f"YTD AVWAP: {_f(ytd_dist, 1)} ATR — pulled back significantly." if math.isfinite(ytd_dist) else "")
        + f" {_regime_context(row)}"
    )

    entry = (
        f"Speculative only: small position on close above ${_f(levels.pivot)} with volume. "
        f"Strict stop: ${_f(levels.stop)}. Size 25-50% of normal."
    )

    risk = (
        f"High-risk setup. Structural issues remain. "
        f"Stop: ${_f(levels.stop)}. Do not average down."
    )

    targets = _format_targets(levels, close)

    ma_ctx = _ma_context(_frow_safe(row, "close_vs_sma50"), close, atr)
    avwap_ctx = _avwap_context(ytd_dist)

    verdict = (
        f"Speculative watch — NOT in primary long list. "
        f"Entry only on ${_f(levels.pivot)} breakout with strict stop. "
        f"Reduce size — this is a lower-conviction setup."
    )

    trade_plan = (
        f"Speculative trigger: above ${_f(levels.pivot)} on volume. "
        f"Strict stop: ${_f(levels.stop)}. T1: ${_f(levels.t1)}. Reduced position size."
    )

    return {
        "setup": setup, "why": why, "entry": entry, "risk": risk,
        "targets": targets, "ma_context": ma_ctx, "avwap_context": avwap_ctx,
        "verdict": verdict, "trade_plan": trade_plan,
    }


# ── Shared helpers used by bucket narratives ──────────────────────────────────

def _format_targets(levels: TradeLevels, close: float, pullback_context: bool = False) -> str:
    """Format T1/T2/T3 targets with % gain and R/R."""
    if math.isfinite(levels.t1) and math.isfinite(close) and close > 0:
        t1_pct = (levels.t1 - close) / close * 100
        t2_pct = (levels.t2 - close) / close * 100 if math.isfinite(levels.t2) else math.nan
        t3_pct = (levels.t3 - close) / close * 100 if math.isfinite(levels.t3) else math.nan
        rr_str = f" (R/R {_f(levels.risk_reward_t1, 1)}x from entry mid)" if math.isfinite(levels.risk_reward_t1) else ""
        note = " Partial trim at T1 advised." if not pullback_context else " T1 is prior resistance / high."
        t1_str = f"T1: ${_f(levels.t1)} (+{t1_pct:.1f}%){rr_str}."
        t2_str = f" T2: ${_f(levels.t2)} (+{t2_pct:.1f}%)." if math.isfinite(t2_pct) else f" T2: ${_f(levels.t2)}."
        t3_str = f" T3: ${_f(levels.t3)} (+{t3_pct:.1f}%)." if math.isfinite(t3_pct) else f" T3: ${_f(levels.t3)}."
        return t1_str + t2_str + t3_str + note
    if math.isfinite(levels.t1):
        rr_str = f" (R/R: {_f(levels.risk_reward_t1, 1)}x)" if math.isfinite(levels.risk_reward_t1) else ""
        return f"T1: ${_f(levels.t1)}{rr_str}. T2: ${_f(levels.t2)}. T3: ${_f(levels.t3)}."
    return "Targets unavailable (pivot or ATR missing)."
