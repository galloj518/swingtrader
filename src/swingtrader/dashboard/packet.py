"""Canonical symbol packet — single source of truth for all downstream consumers.

Architecture
------------
The packet is the unit of analysis.  Everything downstream (selector, artifacts,
dashboard, AI notes) reads from completed packets.  No downstream layer invents
or repairs meaning; they only present what the packet already contains.

Two-tier packet model
---------------------
build_lightweight_packet(row)
    Builds a complete decision-relevant packet from a scored snapshot row.
    All analysis is computed internally — eligibility, freshness, action label,
    bucket, levels, trade plan, narrative.  No file I/O.  Fast enough to run on
    200+ symbols in the scoring pipeline.

enrich_with_context(pkt, row)
    Adds the expensive context block (MA table, AVWAP map, assessments,
    checklist, volume block) to a lightweight packet.  Involves file I/O
    (raw OHLCV + features parquet).  Called only for the top-N selected
    packets after selection.

The selector receives the list of lightweight packets and returns a
PacketSelections dict.  Only after selection are the selected packets enriched
with context.

Canonical fields
----------------
See _PACKET_FIELDS below for the full ordered field list.  Every packet
guarantees these keys regardless of data completeness.

Old-repo adaptations
--------------------
The following concepts were ported or adapted from the swing_engine repo:
- Trade plan fields: explicit entry_condition / stop_basis / time_stop / key_risk
  (adapted from swing_engine/checklist.py evaluate_actionability)
- Portfolio health: position_health / recommended_action / key_level
  (adapted from swing_engine/packets.py portfolio_guidance)
- Actionability verdict taxonomy: matches swing_engine's BUY NOW / WATCH BREAKOUT /
  WAIT PULLBACK labelling, mapped onto the current ACTION_* constants
- MA tomorrow / need_flat logic: in build_ma_table (context.py)
  (ported from swing_engine/features.py extract_ma_state)
- Assessment suite: base_quality / continuation / clean_air / overhead_supply /
  breakout_integrity / chart_quality in assessments.py
  (ported from swing_engine/features.py assess_* family)
- WTD / MTD AVWAP anchors: in build_avwap_table (context.py)
  (adapted from swing_engine/features.py get_dynamic_anchor_dates)

Intentionally NOT ported
------------------------
- Calibrated gating: swing_engine's _check_weekly_gate / _check_daily_gate are
  replaced by our fitted model scores + hard gates in eligibility.py
- AI-derived narrative: swing_engine's _decision_summary uses heuristic templates;
  we use build_narrative which is also template-based but bucket-routed
- Broker / OMS fields: execution_price, position_size, account_risk — out of scope
  per project constraints (analysis-only)
"""
from __future__ import annotations

import math
from typing import Any

import pandas as pd

from swingtrader.dashboard.action import (
    ACTION_AVOID,
    ACTION_BREAKOUT,
    ACTION_EXTENDED,
    ACTION_NOW,
    ACTION_PORTFOLIO,
    ACTION_PULLBACK,
)
from swingtrader.dashboard.buckets import (
    BREAKOUT_TRIGGER_DAYS,
    BUCKET_BREAKOUT,
    BUCKET_EXCLUDED,
    BUCKET_EXTENDED,
    BUCKET_NON_EQUITY,
    BUCKET_PORTFOLIO,
    BUCKET_PULLBACK,
    BUCKET_REVERSAL,
)
from swingtrader.dashboard.context import build_context
from swingtrader.dashboard.eligibility import assess_eligibility
from swingtrader.dashboard.freshness import FRESH_MAX_DAYS, SCORED_STATES, classify_row
from swingtrader.dashboard.levels import TradeLevels, compute_levels

__all__ = [
    "build_all_lightweight_packets",
    "build_lightweight_packet",
    "build_packet",
    "build_packets",
    "enrich_with_context",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _f(v: Any) -> float:
    try:
        fv = float(v)
        return fv if math.isfinite(fv) else math.nan
    except (TypeError, ValueError):
        return math.nan


def _s(v: Any, default: str = "—") -> str:
    if v is None:
        return default
    s = str(v)
    return s if s not in ("nan", "None", "") else default


def _num(row: pd.Series | dict, col: str, d: int = 2) -> str:
    v = row.get(col, math.nan)
    fv = _f(v)
    return f"{fv:.{d}f}" if math.isfinite(fv) else "—"


def _pct(row: pd.Series | dict, col: str) -> str:
    v = row.get(col, math.nan)
    fv = _f(v)
    return f"{fv * 100:.0f}%" if math.isfinite(fv) else "—"


# ---------------------------------------------------------------------------
# Structural analysis helpers (Tier 1 — snapshot fields only, no file I/O)
# ---------------------------------------------------------------------------

def _num_from_value(v: Any, d: int = 2) -> str:
    fv = _f(v)
    return f"{fv:.{d}f}" if math.isfinite(fv) else "â€”"


def _price(v: Any) -> str:
    fv = _f(v)
    return f"${fv:.2f}" if math.isfinite(fv) else "â€”"


def _rr(v: Any) -> str:
    fv = _f(v)
    return f"{fv:.1f}:1" if math.isfinite(fv) else "â€”"


def _is_missing(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, str):
        return v.strip() in ("", "â€”", "nan", "None")
    if isinstance(v, list):
        return len(v) == 0
    if isinstance(v, dict):
        return len(v) == 0
    if isinstance(v, (int, float)):
        return not math.isfinite(_f(v))
    return False


def _as_reason_list(raw: str) -> list[str]:
    text = str(raw or "").strip()
    if not text or text == "â€”":
        return []
    return [part.strip() for part in text.split(";") if part.strip()]


def _compute_daily_trend_state(row: pd.Series) -> str:
    """Classify daily trend from snapshot row fields only.

    Returns one of:
      "strong_uptrend"  — above SMA50 by >3%, positive RS, regime up
      "uptrend"         — above SMA50, regime supportive
      "neutral"         — near SMA50 (+/-2%), or mixed signals
      "weak"            — below SMA50 but not broken
      "broken"          — regime down and below SMA50 by >5%
      "unknown"         — required fields absent
    """
    cvs50        = _f(row.get("close_vs_sma50", math.nan))
    rs63         = _f(row.get("daily_rs_63", math.nan))
    regime_trend = _f(row.get("regime_spy_trend", math.nan))

    if not (math.isfinite(cvs50) or math.isfinite(rs63)):
        return "unknown"

    # Broken: regime down AND meaningfully below SMA50
    if (
        math.isfinite(regime_trend) and regime_trend < 0
        and math.isfinite(cvs50) and cvs50 < -0.05
    ):
        return "broken"

    if math.isfinite(cvs50):
        if cvs50 > 0.03 and math.isfinite(rs63) and rs63 > 0:
            return "strong_uptrend"
        if cvs50 > 0:
            return "uptrend"
        if cvs50 >= -0.02:
            return "neutral"
        if cvs50 >= -0.05:
            return "weak"
        return "broken"

    # Only RS available
    if math.isfinite(rs63):
        return "uptrend" if rs63 > 0.02 else "neutral" if rs63 > -0.03 else "weak"
    return "unknown"


def _compute_pullback_quality(
    row: pd.Series,
    bucket: str,
    state: str,
    days: int,
    dist_f: float,
) -> str:
    """Classify the quality of a pullback setup.

    Only meaningful for PULLBACK bucket names.  Returns one of:
      "at_pivot"       — price just below pivot, testing support
      "near_pivot"     — constructive retest, within 2 ATR
      "constructive"   — legitimate pullback in uptrend (2–3 ATR back)
      "deep"           — pulled back >3 ATR, significant recovery needed
      "old_trigger"    — triggered trade that has aged out of breakout window
      "far_base"       — BASE/ARMED too far from pivot yet
      "n/a"            — not a pullback bucket name
    """
    if bucket != BUCKET_PULLBACK:
        return "n/a"

    if state in {"TRIGGERED", "ACCEPTED"} and days > BREAKOUT_TRIGGER_DAYS:
        return "old_trigger"

    if math.isfinite(dist_f):
        if dist_f >= -0.5:
            return "at_pivot"
        if dist_f >= -2.0:
            return "near_pivot"
        if dist_f >= -3.0:
            return "constructive"
        return "deep"

    if state in {"BASE", "ARMED"}:
        return "far_base"

    return "n/a"


def _compute_demotion_reason(
    row: pd.Series,
    bucket: str,
    state: str,
    days: int,
    dist_f: float,
) -> str:
    """Return why a name was routed to pullback instead of breakout bucket.

    Returns a short reason string, or "" if the name is in its expected bucket.
    Only meaningful for PULLBACK bucket names.

    Reasons mirror the bucket routing logic in buckets.py:
      "far_from_pivot"  — dist_to_pivot_atr > 1.5 (pre-entry states)
      "below_sma50"     — close near pivot but below declining SMA50
      "old_trigger"     — TRIGGERED/ACCEPTED > BREAKOUT_TRIGGER_DAYS
      "stale_base"      — BASE/ARMED days_in_state exceeded freshness window
      "not_fresh"       — otherwise stale
    """
    if bucket != BUCKET_PULLBACK:
        return ""

    if state in {"TRIGGERED", "ACCEPTED"}:
        if days > BREAKOUT_TRIGGER_DAYS:
            return f"old_trigger ({days}d past {BREAKOUT_TRIGGER_DAYS}d breakout window)"
        return ""

    if state in {"BASE", "ARMED"}:
        if math.isfinite(dist_f) and abs(dist_f) > 1.0:
            return f"far_from_pivot ({abs(dist_f):.1f} ATR from trigger zone)"

        cvs50 = _f(row.get("close_vs_sma50", math.nan))
        if math.isfinite(cvs50) and cvs50 < -0.03:
            return f"below_sma50 ({cvs50 * 100:.1f}% below SMA50 near pivot)"

        max_days = FRESH_MAX_DAYS.get(state, 0)
        if max_days and days > max_days:
            return f"stale_base ({days}d exceeds {max_days}d freshness window)"

        return "not_fresh"

    return ""


# ---------------------------------------------------------------------------
# Trade plan builder (adapted from swing_engine/checklist.py)
# ---------------------------------------------------------------------------

def _is_reversal_candidate(row: pd.Series, rejection_reasons: list[str]) -> bool:
    rejection_blob = " ".join(rejection_reasons).lower()
    for hard_gate in ("non_equity", "invalid_state", "high_failure_risk"):
        if hard_gate in rejection_blob:
            return False

    composite = _f(row.get("composite_score", math.nan))
    if math.isfinite(composite) and composite < 0.22:
        return False

    base_length = int(row.get("base_length", 0) or 0)
    if base_length < 10:
        return False

    state = str(row.get("state", "NONE"))
    if state not in SCORED_STATES:
        return False

    ytd_dist = _f(row.get("ytd_dist_atr", math.nan))
    return math.isfinite(ytd_dist) and -4.0 <= ytd_dist <= -0.5


def _collect_extension_truth(row: pd.Series, fresh: dict) -> tuple[bool, str, str]:
    reasons: list[str] = []
    for item in _as_reason_list(str(fresh.get("extension_reasons", ""))):
        if item not in reasons:
            reasons.append(item)

    dist_pivot = _f(row.get("dist_to_pivot_atr", math.nan))
    close_vs_sma50 = _f(row.get("close_vs_sma50", math.nan))
    ytd_dist = _f(row.get("ytd_dist_atr", math.nan))
    swing_low_dist = _f(row.get("swing_low_dist_atr", math.nan))
    state = str(row.get("state", "NONE"))

    if state in {"LATE", "EXHAUSTED"}:
        reasons.append(f"{state.lower()} lifecycle state")
    if math.isfinite(dist_pivot) and dist_pivot >= 3.0:
        reasons.append(f"{dist_pivot:.1f} ATR above pivot")
    if math.isfinite(close_vs_sma50) and close_vs_sma50 >= 0.12:
        reasons.append(f"{close_vs_sma50 * 100:.1f}% above SMA50")
    if math.isfinite(ytd_dist) and ytd_dist >= 4.0:
        reasons.append(f"{ytd_dist:.1f} ATR above YTD AVWAP")
    if math.isfinite(swing_low_dist) and swing_low_dist >= 5.0:
        reasons.append(f"{swing_low_dist:.1f} ATR above swing-low AVWAP")

    unique_reasons: list[str] = []
    for reason in reasons:
        if reason not in unique_reasons:
            unique_reasons.append(reason)

    is_extended = bool(fresh.get("is_extended", False)) or bool(unique_reasons)
    if not is_extended:
        return False, "not_extended", ""

    extension_state = "mature_trend" if state in {"LATE", "EXHAUSTED"} else "extended_chase_risk"
    return True, extension_state, "; ".join(unique_reasons)


def _is_constructive_trend(daily_trend_state: str, row: pd.Series) -> bool:
    if daily_trend_state in {"strong_uptrend", "uptrend", "neutral"}:
        return True
    close_vs_sma50 = _f(row.get("close_vs_sma50", math.nan))
    daily_rs_63 = _f(row.get("daily_rs_63", math.nan))
    return bool(
        math.isfinite(close_vs_sma50)
        and close_vs_sma50 >= -0.02
        and math.isfinite(daily_rs_63)
        and daily_rs_63 >= 0
    )


def _route_packet(
    row: pd.Series,
    *,
    eligible: bool,
    rejection_reasons: list[str],
    is_fresh: bool,
    is_extended: bool,
    extension_state: str,
    daily_trend_state: str,
) -> dict:
    state = str(row.get("state", "NONE"))
    dist = _f(row.get("dist_to_pivot_atr", math.nan))
    days = int(row.get("days_in_state", 0) or 0)
    close_vs_sma50 = _f(row.get("close_vs_sma50", math.nan))
    constructive_trend = _is_constructive_trend(daily_trend_state, row)
    near_pivot = math.isfinite(dist) and abs(dist) <= 1.0

    if bool(row.get("is_non_equity", False)):
        return {
            "bucket": BUCKET_NON_EQUITY,
            "setup_key": "non_equity",
            "setup_classification": "Informational only",
            "action_label": ACTION_AVOID,
            "promotion_reason": "",
            "demotion_reason": "",
            "route_reason": "non-equity informational symbol",
            "operational_readiness": "not_actionable",
        }

    if bool(row.get("is_portfolio", False)):
        return {
            "bucket": BUCKET_PORTFOLIO,
            "setup_key": "portfolio_hold",
            "setup_classification": "Portfolio management",
            "action_label": ACTION_PORTFOLIO,
            "promotion_reason": "Existing holding managed separately from fresh-entry lists",
            "demotion_reason": "",
            "route_reason": "portfolio holdings stay outside fresh-entry sections",
            "operational_readiness": "manage_existing",
        }

    if not eligible:
        if _is_reversal_candidate(row, rejection_reasons):
            return {
                "bucket": BUCKET_REVERSAL,
                "setup_key": "speculative_reversal",
                "setup_classification": "Speculative reversal watch",
                "action_label": ACTION_AVOID,
                "promotion_reason": "",
                "demotion_reason": "kept out of primary long list due to failed core eligibility",
                "route_reason": "speculative reversal only",
                "operational_readiness": "speculative_only",
            }
        return {
            "bucket": BUCKET_EXCLUDED,
            "setup_key": "excluded",
            "setup_classification": "Rejected setup",
            "action_label": ACTION_AVOID,
            "promotion_reason": "",
            "demotion_reason": "",
            "route_reason": "failed eligibility gates",
            "operational_readiness": "rejected",
        }

    if is_extended:
        return {
            "bucket": BUCKET_EXTENDED,
            "setup_key": "extended_leader",
            "setup_classification": "Extended leader",
            "action_label": ACTION_EXTENDED,
            "promotion_reason": "",
            "demotion_reason": f"too_extended_for_fresh_entry ({extension_state})",
            "route_reason": "healthy but too late for a fresh entry",
            "operational_readiness": "too_extended",
        }

    if state not in SCORED_STATES:
        return {
            "bucket": BUCKET_EXCLUDED,
            "setup_key": "excluded",
            "setup_classification": "Rejected setup",
            "action_label": ACTION_AVOID,
            "promotion_reason": "",
            "demotion_reason": "",
            "route_reason": "state not eligible for scored setup routing",
            "operational_readiness": "rejected",
        }

    if state in {"TRIGGERED", "ACCEPTED"}:
        if math.isfinite(dist) and dist < 0:
            return {
                "bucket": BUCKET_PULLBACK,
                "setup_key": "reclaim_pullback",
                "setup_classification": "Pullback reclaim",
                "action_label": ACTION_PULLBACK,
                "promotion_reason": "Constructive pullback into prior breakout pivot",
                "demotion_reason": _compute_demotion_reason(row, BUCKET_PULLBACK, state, days, dist),
                "route_reason": "below pivot; requires reclaim rather than immediate breakout entry",
                "operational_readiness": "needs_reclaim",
            }
        if days <= BREAKOUT_TRIGGER_DAYS and constructive_trend and is_fresh:
            return {
                "bucket": BUCKET_BREAKOUT,
                "setup_key": "fresh_breakout",
                "setup_classification": "Fresh breakout",
                "action_label": ACTION_NOW,
                "promotion_reason": "Fresh breakout above pivot with entry risk still defined",
                "demotion_reason": "",
                "route_reason": "fresh trigger remains operationally actionable",
                "operational_readiness": "ready_now",
            }
        return {
            "bucket": BUCKET_PULLBACK,
            "setup_key": "aged_breakout_pullback",
            "setup_classification": "Aged breakout retest",
            "action_label": ACTION_PULLBACK,
            "promotion_reason": "Trend intact, but the initial breakout window has aged",
            "demotion_reason": _compute_demotion_reason(row, BUCKET_PULLBACK, state, days, dist),
            "route_reason": "better handled as a pullback or reclaim, not a fresh breakout",
            "operational_readiness": "watch_pullback_zone",
        }

    if state in {"ARMED", "BASE"}:
        if near_pivot and constructive_trend and (not math.isfinite(close_vs_sma50) or close_vs_sma50 >= -0.03):
            return {
                "bucket": BUCKET_BREAKOUT,
                "setup_key": "breakout_watch",
                "setup_classification": "Near breakout setup",
                "action_label": ACTION_BREAKOUT,
                "promotion_reason": "Tight constructive base close to the pivot",
                "demotion_reason": "",
                "route_reason": "close enough to treat as an operational breakout watch",
                "operational_readiness": "needs_breakout_trigger",
            }
        if constructive_trend:
            return {
                "bucket": BUCKET_PULLBACK,
                "setup_key": "constructive_pullback",
                "setup_classification": "Constructive pullback",
                "action_label": ACTION_PULLBACK,
                "promotion_reason": "Trend remains constructive while price works through a pullback",
                "demotion_reason": _compute_demotion_reason(row, BUCKET_PULLBACK, state, days, dist),
                "route_reason": "not in a clean breakout trigger zone today",
                "operational_readiness": "watch_pullback_zone",
            }

    if state == "CONFIRMED" and constructive_trend:
        return {
            "bucket": BUCKET_EXTENDED,
            "setup_key": "extended_leader",
            "setup_classification": "Mature trend leader",
            "action_label": ACTION_EXTENDED,
            "promotion_reason": "",
            "demotion_reason": "confirmed trend is no longer a fresh entry setup",
            "route_reason": "mature trend monitored separately from fresh entries",
            "operational_readiness": "too_extended",
        }

    return {
        "bucket": BUCKET_EXCLUDED,
        "setup_key": "excluded",
        "setup_classification": "Rejected setup",
        "action_label": ACTION_AVOID,
        "promotion_reason": "",
        "demotion_reason": "",
        "route_reason": "trend or structure is not constructive enough for primary long routing",
        "operational_readiness": "rejected",
    }


def _build_trade_plan(
    row: pd.Series,
    levels: TradeLevels,
    action: str,
    bucket: str,
    freshness: dict,
) -> dict:
    """Build a structured trade plan from packet components.

    Returns a dict with explicit fields for entry condition, stop basis, targets,
    time stop, and key risk.  These are deterministic rules, not model scores.

    Adapted from swing_engine/checklist.py evaluate_actionability() + trade_plan
    concept.  Maps swing_engine actionability verdicts to our ACTION_* taxonomy.
    """
    state = str(row.get("state", "NONE"))
    pivot_f = _f(row.get("pivot", math.nan))
    close_f = _f(row.get("close", math.nan))
    atr_f   = _f(row.get("atr14", math.nan))
    dist_f  = _f(row.get("dist_to_pivot_atr", math.nan))
    rr      = _f(levels.risk_reward_t1)
    days    = int(row.get("days_in_state", 0) or 0)
    fail    = _f(row.get("failure_risk", math.nan))

    pivot_str   = f"${pivot_f:.2f}"     if math.isfinite(pivot_f) else "pivot"
    entry_lo_s  = f"${levels.entry_lo:.2f}" if math.isfinite(levels.entry_lo) else "—"
    entry_hi_s  = f"${levels.entry_hi:.2f}" if math.isfinite(levels.entry_hi) else "—"
    stop_s      = f"${levels.stop:.2f}"  if math.isfinite(levels.stop)    else "—"
    t1_s        = f"${levels.t1:.2f}"    if math.isfinite(levels.t1)      else "—"
    t2_s        = f"${levels.t2:.2f}"    if math.isfinite(levels.t2)      else "—"
    rr_s        = f"{rr:.1f}:1"          if math.isfinite(rr)             else "—"

    # Stop risk as % of mid-entry
    stop_risk_pct = "—"
    if math.isfinite(levels.entry_lo) and math.isfinite(levels.stop) and levels.entry_lo > 0:
        mid_entry = (levels.entry_lo + levels.entry_hi) / 2 if math.isfinite(levels.entry_hi) else levels.entry_lo
        stop_risk_pct = f"{abs(mid_entry - levels.stop) / mid_entry * 100:.1f}%"

    # --- Entry condition (based on state / action / bucket) ---
    if state in {"TRIGGERED", "ACCEPTED"}:
        if dist_f >= 0:
            entry_condition = f"Price has cleared {pivot_str}. Entry zone {entry_lo_s}-{entry_hi_s}."
        else:
            entry_condition = f"Pulled back below {pivot_str}. Re-entry on reclaim of pivot with volume."
    elif state in {"BASE", "ARMED"}:
        if action == "Actionable on breakout":
            entry_condition = f"Buy on breakout through {pivot_str} on volume >= 1.5x average. Entry zone {entry_lo_s}-{entry_hi_s}."
        else:
            entry_condition = f"Watch {pivot_str} for breakout. Base building; wait for pivot attempt."
    else:
        entry_condition = f"Entry zone {entry_lo_s}-{entry_hi_s} if setup triggers."

    # --- Stop basis ---
    if math.isfinite(atr_f) and math.isfinite(levels.stop):
        stop_atr_dist = abs(levels.stop - (pivot_f if math.isfinite(pivot_f) else close_f)) / atr_f if atr_f > 0 else math.nan
        if math.isfinite(stop_atr_dist):
            stop_basis = f"1.0 ATR below pivot ({stop_atr_dist:.1f} ATR from entry). Daily close below stop = exit."
        else:
            stop_basis = f"Stop at {stop_s}. Daily close below stop = exit."
    else:
        stop_basis = f"Stop at {stop_s}. Daily close below stop = exit."

    # --- Time stop ---
    if state in {"TRIGGERED", "ACCEPTED"}:
        time_stop = f"If no follow-through within 7 trading days, reassess. Already {days} days in {state}."
    elif state in {"BASE", "ARMED"}:
        time_stop = f"Base can persist; re-evaluate if {days} days extends beyond 45 without progress."
    else:
        time_stop = "No active time stop — waiting for trigger."

    # --- Key risk (adapted from swing_engine/checklist.py risk note) ---
    risks: list[str] = []
    if math.isfinite(fail) and fail > 0.50:
        risks.append(f"High failure risk ({fail:.0%})")
    if math.isfinite(dist_f) and dist_f < -1.5:
        risks.append(f"Far from pivot ({dist_f:.1f} ATR) — breakout not imminent")
    if state in {"TRIGGERED", "ACCEPTED"} and days > 7:
        risks.append("Extended time in triggered state — momentum may be fading")
    if not math.isfinite(rr) or rr < 1.5:
        risks.append("R/R below 1.5:1 — unfavorable reward profile at current entry")
    if not risks:
        risks = ["Volume failure at pivot", "Gap fill below stop"]
    key_risk = "; ".join(risks[:2])

    # --- Actionability verdict ---
    if action == "Actionable now":
        verdict_code = "BUY_NOW"
    elif action == "Actionable on breakout":
        verdict_code = "WATCH_BREAKOUT"
    elif action == "Actionable on pullback":
        verdict_code = "WAIT_PULLBACK"
    elif action in ("Extended, wait",):
        verdict_code = "WAIT_ZONE"
    else:
        verdict_code = "BLOCK"

    # --- Why now / why not now (deterministic, dual-sided analysis) ---
    # Adapted from swing_engine/checklist.py "decision summary" philosophy.
    # Each is a list of short signal strings, not narrative prose.
    cvs50   = _f(row.get("close_vs_sma50", math.nan))
    rs63    = _f(row.get("daily_rs_63", math.nan))
    vd      = _f(row.get("volume_dryup", math.nan))
    atr_pct = _f(row.get("atr_compression_pct", math.nan))
    ytd     = _f(row.get("ytd_dist_atr", math.nan))
    sl_dist = _f(row.get("swing_low_dist_atr", math.nan))

    why_now: list[str] = []
    why_not: list[str] = []

    # -- Trend / structure signals --
    if state in {"TRIGGERED", "ACCEPTED"} and days <= 7:
        why_now.append(f"Fresh breakout — only {days} day(s) since trigger")
    elif state in {"BASE", "ARMED"} and math.isfinite(dist_f) and abs(dist_f) <= 0.5:
        why_now.append("Price right at the door — pivot test imminent")
    elif state in {"BASE", "ARMED"} and math.isfinite(dist_f) and abs(dist_f) <= 1.5:
        why_now.append(f"Setup within {abs(dist_f):.1f} ATR of pivot — watching for attempt")

    if math.isfinite(cvs50) and cvs50 > 0.02:
        why_now.append(f"Above SMA50 ({cvs50*100:.1f}%) — daily trend healthy")
    elif math.isfinite(cvs50) and cvs50 < -0.02:
        why_not.append(f"Below SMA50 ({cvs50*100:.1f}%) — daily trend degraded")

    if math.isfinite(rs63) and rs63 > 0.03:
        why_now.append(f"Outperforming SPY by {rs63*100:.0f}% over 63 days")
    elif math.isfinite(rs63) and rs63 < -0.02:
        why_not.append(f"Underperforming SPY by {abs(rs63)*100:.0f}% over 63 days")

    # -- Compression / coiling --
    if math.isfinite(atr_pct) and atr_pct < 30:
        why_now.append(f"ATR compressed to {atr_pct:.0f}th pct — coiling for move")
    elif math.isfinite(atr_pct) and atr_pct > 60:
        why_not.append(f"ATR elevated ({atr_pct:.0f}th pct) — not yet compressed")

    if math.isfinite(vd) and vd > 0.4:
        why_now.append(f"Volume dry-up score {vd:.2f} — supply exhausted")
    elif math.isfinite(vd) and vd < 0.1:
        why_not.append("Volume still active — base not fully digested")

    # -- AVWAP / cost basis --
    if math.isfinite(ytd) and ytd > 0.5:
        why_now.append(f"Accepted {ytd:.1f} ATR above YTD AVWAP — healthy cost basis")
    elif math.isfinite(ytd) and ytd < -0.5:
        why_not.append(f"{abs(ytd):.1f} ATR below YTD AVWAP — below year-open cost basis")

    if math.isfinite(sl_dist) and sl_dist > 1.0:
        why_now.append(f"{sl_dist:.1f} ATR above swing low — buffer from base support")

    # -- Failure risk --
    if math.isfinite(fail) and fail < 0.30:
        why_now.append(f"Low failure risk ({fail:.0%}) from model")
    elif math.isfinite(fail) and fail > 0.55:
        why_not.append(f"Elevated failure risk ({fail:.0%}) from model")

    if not why_now:
        why_now = ["Eligible setup in scored state — awaiting catalyst"]
    if not why_not:
        why_not = ["No specific concerns flagged at this time"]

    # --- Setup improves/weakens if (deterministic tomorrow scenarios) ---
    # Adapted from swing_engine's "what changes the picture" concept.
    improves: list[str] = []
    weakens: list[str] = []

    if state in {"BASE", "ARMED"}:
        improves.append(f"Tomorrow closes above {pivot_str} on heavy volume")
        if math.isfinite(atr_pct) and atr_pct > 40:
            improves.append("ATR compression continues — range narrows further")
        if math.isfinite(vd) and vd < 0.4:
            improves.append("Volume dries up further — supply absorbed")
        weakens.append(f"Tomorrow closes below stop {stop_s} on volume")
        if math.isfinite(cvs50) and cvs50 > -0.05:
            weakens.append("Price breaks below SMA50 on volume")
    elif state in {"TRIGGERED", "ACCEPTED"}:
        improves.append(f"Follow-through close above {t1_s} (T1) on heavy volume")
        improves.append("Orderly pullback to pivot then reclaim — adds to position")
        weakens.append(f"Close below stop {stop_s} — invalidates trigger")
        if days > 5:
            weakens.append("Continued sideways action without progression above entry")
    else:
        improves = ["Setup clarifies — state machine advances to scoreable state"]
        weakens  = ["Continued deterioration or non-equity exclusion"]

    return {
        "actionability_code":  verdict_code,
        "entry_condition":     entry_condition,
        "entry_range":         f"{entry_lo_s}-{entry_hi_s}",
        "stop":                stop_s,
        "stop_basis":          stop_basis,
        "stop_risk_pct":       stop_risk_pct,
        "target_1":            t1_s,
        "target_2":            t2_s,
        "risk_reward_t1":      rr_s,
        "time_stop":           time_stop,
        "key_risk":            key_risk,
        # Dual-sided analysis (adapted from swing_engine decision_summary philosophy)
        "why_now":            why_now,
        "why_not_now":        why_not,
        "setup_improves_if":  improves,
        "setup_weakens_if":   weakens,
    }


# ---------------------------------------------------------------------------
# Canonical packet helpers
# ---------------------------------------------------------------------------

def _build_trade_plan_canonical(
    row: pd.Series,
    levels: TradeLevels,
    route: dict,
) -> dict:
    state = str(row.get("state", "NONE"))
    days = int(row.get("days_in_state", 0) or 0)
    dist = _f(row.get("dist_to_pivot_atr", math.nan))
    failure = _f(row.get("failure_risk", math.nan))
    close_vs_sma50 = _f(row.get("close_vs_sma50", math.nan))
    daily_rs_63 = _f(row.get("daily_rs_63", math.nan))
    atr_pct = _f(row.get("atr_compression_pct", math.nan))
    volume_dryup = _f(row.get("volume_dryup", math.nan))
    ytd_dist = _f(row.get("ytd_dist_atr", math.nan))
    swing_low_dist = _f(row.get("swing_low_dist_atr", math.nan))

    setup_key = route["setup_key"]
    pivot_s = _price(row.get("pivot", math.nan))
    entry_lo_s = _price(levels.entry_lo)
    entry_hi_s = _price(levels.entry_hi)
    stop_s = _price(levels.stop)
    t1_s = _price(levels.t1)
    t2_s = _price(levels.t2)
    t3_s = _price(levels.t3)
    alt_pullback = _price(levels.s1 if math.isfinite(levels.s1) else levels.entry_lo)
    rr_s = _rr(levels.risk_reward_t1)

    stop_risk_pct = "â€”"
    if math.isfinite(levels.entry_lo) and math.isfinite(levels.stop) and levels.entry_lo > 0:
        mid_entry = (levels.entry_lo + levels.entry_hi) / 2 if math.isfinite(levels.entry_hi) else levels.entry_lo
        stop_risk_pct = f"{abs(mid_entry - levels.stop) / mid_entry * 100:.1f}%"

    why_now: list[str] = []
    why_not_now: list[str] = []

    if math.isfinite(close_vs_sma50):
        if close_vs_sma50 > 0.02:
            why_now.append(f"Above SMA50 by {close_vs_sma50 * 100:.1f}%")
        elif close_vs_sma50 < -0.02:
            why_not_now.append(f"Below SMA50 by {abs(close_vs_sma50) * 100:.1f}%")

    if math.isfinite(daily_rs_63):
        if daily_rs_63 > 0.03:
            why_now.append(f"RS-63 is +{daily_rs_63 * 100:.0f}% vs SPY")
        elif daily_rs_63 < -0.02:
            why_not_now.append(f"RS-63 is weak at {daily_rs_63 * 100:.0f}% vs SPY")

    if math.isfinite(atr_pct):
        if atr_pct <= 35:
            why_now.append(f"ATR compression is supportive at the {atr_pct:.0f}th percentile")
        elif atr_pct >= 65:
            why_not_now.append(f"ATR is elevated at the {atr_pct:.0f}th percentile")

    if math.isfinite(volume_dryup):
        if volume_dryup >= 0.40:
            why_now.append(f"Volume dry-up score {volume_dryup:.2f} supports absorption")
        elif volume_dryup <= 0.10:
            why_not_now.append("Volume is still too active for a clean setup")

    if math.isfinite(ytd_dist):
        if ytd_dist > 0.50:
            why_now.append(f"Accepted above YTD AVWAP by {ytd_dist:.1f} ATR")
        elif ytd_dist < -0.50:
            why_not_now.append(f"Still {abs(ytd_dist):.1f} ATR below YTD AVWAP")

    if math.isfinite(swing_low_dist) and swing_low_dist > 1.0:
        why_now.append(f"{swing_low_dist:.1f} ATR above swing-low AVWAP support")

    if math.isfinite(failure):
        if failure <= 0.30:
            why_now.append(f"Failure risk remains contained at {failure:.0%}")
        elif failure >= 0.55:
            why_not_now.append(f"Failure risk is elevated at {failure:.0%}")

    entry_style = "avoid"
    actionable_now = False
    entry_condition = "Do not treat this as a new long setup."
    entry_trigger = "â€”"
    invalidation = f"Daily close below {stop_s} invalidates the idea."
    time_stop = "No active time stop."
    setup_improves_if: list[str] = []
    setup_weakens_if: list[str] = []
    key_risk = "Model and structure do not currently justify a fresh entry."

    if setup_key == "fresh_breakout":
        entry_style = "breakout"
        actionable_now = True
        entry_trigger = pivot_s
        entry_condition = f"Fresh breakout above {pivot_s}; use {entry_lo_s}-{entry_hi_s} while the trigger remains intact."
        time_stop = f"If follow-through stalls for 3-5 sessions, reassess. Current trigger age: {days} day(s)."
        setup_improves_if = [
            f"Closes through T1 at {t1_s} with expanding volume",
            f"Holds the pivot at {pivot_s} on any shallow retest",
        ]
        setup_weakens_if = [
            f"Falls back below {pivot_s} and cannot reclaim it quickly",
            f"Closes below stop {stop_s}",
        ]
        key_risk = "Failed follow-through after the breakout is the main risk."
        why_now.insert(0, f"Fresh trigger still within the {BREAKOUT_TRIGGER_DAYS} day breakout window")
    elif setup_key == "breakout_watch":
        entry_style = "breakout"
        entry_trigger = pivot_s
        entry_condition = f"Buy only on a decisive break above {pivot_s} with volume support. Current watch zone: {entry_lo_s}-{entry_hi_s}."
        time_stop = f"Watch for a real trigger while the base remains valid. Current age: {days} day(s) in {state}."
        setup_improves_if = [
            f"Breaks above {pivot_s} with demand expansion",
            "Keeps tightening near the pivot rather than slipping away from it",
        ]
        setup_weakens_if = [
            f"Closes below support at {alt_pullback}",
            "Base broadens and loses tightness near the pivot",
        ]
        key_risk = "Premature entry before the actual breakout trigger."
        why_not_now.insert(0, f"Still needs a clean breakout through {pivot_s}")
    elif setup_key == "reclaim_pullback":
        entry_style = "reclaim"
        entry_trigger = pivot_s
        entry_condition = f"Wait for a reclaim of {pivot_s} before treating this as a fresh entry. Current pullback support watch: {alt_pullback}."
        time_stop = f"Reassess if the reclaim does not occur within 5-10 sessions. Current trigger age: {days} day(s)."
        setup_improves_if = [
            f"Reclaims {pivot_s} on volume and then holds above it",
            "Finds support cleanly at the alternate pullback area",
        ]
        setup_weakens_if = [
            f"Breaks below support at {alt_pullback}",
            f"Closes below stop {stop_s}",
        ]
        key_risk = "Below-pivot pullback can become a failed breakout if reclaim does not happen."
        if math.isfinite(dist):
            why_not_now.insert(0, f"Below pivot by {abs(dist):.1f} ATR and needs a reclaim")
    elif setup_key in {"constructive_pullback", "aged_breakout_pullback"}:
        entry_style = "pullback"
        entry_trigger = alt_pullback
        entry_condition = f"Treat as a pullback setup, not a breakout chase. Preferred add zone is near {alt_pullback}; confirm support before entry."
        time_stop = "Let support prove itself; no fresh-entry time stop until the pullback resolves."
        setup_improves_if = [
            f"Holds support around {alt_pullback} and turns back toward {pivot_s}",
            "Shows lighter downside volume on the pullback",
        ]
        setup_weakens_if = [
            f"Loses support near {alt_pullback}",
            f"Closes below stop {stop_s}",
        ]
        key_risk = "Pullback can deepen if support does not hold on the first test."
        if setup_key == "aged_breakout_pullback":
            why_not_now.insert(0, "Initial breakout window has aged; prefer a cleaner reset")
        else:
            why_not_now.insert(0, "Not in a clean breakout trigger zone today")
    elif setup_key == "extended_leader":
        entry_style = "wait"
        entry_trigger = alt_pullback
        entry_condition = f"Do not chase here. Wait for a more attractive reset toward {alt_pullback} or a fresh base."
        time_stop = "No fresh-entry timer while the name remains extended."
        setup_improves_if = [
            "Builds a new base or digests the extension without material damage",
            f"Offers a lower-risk reset closer to {alt_pullback}",
        ]
        setup_weakens_if = [
            "Keeps extending without offering a reset",
            f"Breaks below stop {stop_s}",
        ]
        key_risk = "Chasing an extended leader destroys reward-to-risk."
        why_not_now.insert(0, "Too extended for a trustworthy fresh entry")
    elif setup_key == "portfolio_hold":
        entry_style = "hold"
        entry_trigger = stop_s
        entry_condition = f"Manage the existing position rather than treating this as a new entry. Keep the key level at {stop_s} in view."
        time_stop = "Manage against the existing thesis and stop discipline."
        setup_improves_if = [
            f"Holds above pivot {pivot_s} or support {alt_pullback}",
            f"Extends toward T1 at {t1_s} without violating support",
        ]
        setup_weakens_if = [
            f"Closes below stop {stop_s}",
            "Loses trend support and fails to recover promptly",
        ]
        key_risk = "Portfolio management is about protecting the open thesis, not forcing a new entry."
        why_not_now.insert(0, "Already a holding; separate from fresh-entry routing")
    elif setup_key == "speculative_reversal":
        entry_style = "avoid"
        entry_trigger = pivot_s
        entry_condition = "Speculative reversal only. Keep isolated from primary long routing until core trend and eligibility improve."
        time_stop = "No action until the setup graduates out of reversal status."
        setup_improves_if = [
            "Repairs trend damage and re-passes the primary eligibility gates",
            f"Moves back above pivot {pivot_s} with real sponsorship",
        ]
        setup_weakens_if = [
            "Breaks fresh lows or keeps failing at resistance",
            f"Stays below support at {alt_pullback}",
        ]
        key_risk = "Speculative reversals fail often and should not contaminate the primary list."
        why_not_now.insert(0, "Speculative-only status keeps this out of decision-ready long setups")
    else:
        entry_style = "avoid"
        entry_condition = "Rejected setup. Do not surface this as an actionable idea."
        time_stop = "No action while hard blockers remain."
        setup_improves_if = [
            "Re-passes the hard eligibility gates",
            "Repairs trend and setup integrity enough to re-enter a valid bucket",
        ]
        setup_weakens_if = ["Accumulates more eligibility or structure failures"]
        key_risk = "Hard blockers still dominate the setup."
        why_not_now.insert(0, "Hard blockers still prevent a decision-ready setup")

    if not why_now:
        why_now = ["No bullish edge is strong enough to override the setup classification."]
    if not why_not_now:
        why_not_now = ["No immediate blocker beyond normal execution discipline."]

    code_map = {
        "breakout": "BUY_NOW" if actionable_now else "WATCH_BREAKOUT",
        "reclaim": "WAIT_PULLBACK",
        "pullback": "WAIT_PULLBACK",
        "wait": "WAIT_ZONE",
        "hold": "WAIT_ZONE",
        "avoid": "BLOCK",
    }

    return {
        "actionability_code": code_map.get(entry_style, "BLOCK"),
        "actionable_now": actionable_now,
        "entry_style": entry_style,
        "best_entry_style": entry_style,
        "entry_condition": entry_condition,
        "entry_trigger": entry_trigger,
        "entry_range": f"{entry_lo_s}-{entry_hi_s}",
        "alternate_pullback_entry": alt_pullback,
        "stop": stop_s,
        "stop_basis": f"Daily close below {stop_s} invalidates the setup.",
        "stop_risk_pct": stop_risk_pct,
        "target_1": t1_s,
        "target_2": t2_s,
        "target_3": t3_s,
        "risk_reward_t1": rr_s,
        "time_stop": time_stop,
        "key_risk": key_risk,
        "invalidation": invalidation,
        "why_now": why_now,
        "why_not_now": why_not_now,
        "setup_improves_if": setup_improves_if,
        "setup_weakens_if": setup_weakens_if,
        "what_improves_tomorrow": list(setup_improves_if),
        "what_weakens_tomorrow": list(setup_weakens_if),
    }


def _build_final_verdict(route: dict, trade_plan: dict) -> str:
    setup_key = route["setup_key"]
    entry_condition = str(trade_plan.get("entry_condition", ""))
    invalidation = str(trade_plan.get("invalidation", ""))

    if setup_key == "fresh_breakout":
        return f"Fresh breakout. Actionable now while the trigger holds. {invalidation}"
    if setup_key == "breakout_watch":
        return f"Valid breakout watch, but not actionable until price clears the trigger. {entry_condition}"
    if setup_key == "reclaim_pullback":
        return f"Constructive pullback, but not actionable now because it still needs a reclaim. {entry_condition}"
    if setup_key in {"constructive_pullback", "aged_breakout_pullback"}:
        return f"Constructive pullback watch. Treat this as a pullback plan, not a breakout card. {entry_condition}"
    if setup_key == "extended_leader":
        return f"Healthy leader, but too extended for a fresh entry. {entry_condition}"
    if setup_key == "portfolio_hold":
        return f"Portfolio management only. {entry_condition}"
    if setup_key == "speculative_reversal":
        return "Speculative reversal only. Keep isolated from primary long selection."
    if setup_key == "non_equity":
        return "Informational symbol only. Not part of primary setup selection."
    return "Excluded from surfaced setup lists."


def _build_narrative(route: dict, trade_plan: dict, final_verdict: str) -> dict:
    why_now = trade_plan.get("why_now", [])
    why_not_now = trade_plan.get("why_not_now", [])
    targets = [trade_plan.get("target_1"), trade_plan.get("target_2"), trade_plan.get("target_3")]
    targets = [str(target) for target in targets if not _is_missing(target)]

    return {
        "setup": route.get("promotion_reason") or route.get("setup_classification", ""),
        "why": "; ".join(why_now[:3]),
        "entry": trade_plan.get("entry_condition", ""),
        "risk": trade_plan.get("key_risk", ""),
        "targets": " | ".join(targets),
        "verdict": final_verdict,
        "why_not_now": "; ".join(why_not_now[:3]),
        "ma_context": "",
        "avwap_context": "",
    }


# ---------------------------------------------------------------------------
# Portfolio position health (adapted from swing_engine/packets.py)
# ---------------------------------------------------------------------------

def _build_portfolio_health(
    row: pd.Series,
    levels: TradeLevels,
    action: str,
) -> dict:
    """Build per-position portfolio health assessment.

    Adapted from swing_engine/packets.py portfolio_guidance() and related logic.
    Determines position health (healthy / at_risk / extended / recovering) and
    produces an explicit recommended_action.

    Only called for is_portfolio=True rows; returns {} for non-portfolio.
    """
    if not bool(row.get("is_portfolio", False)):
        return {}

    state  = str(row.get("state", "NONE"))
    dist_f = _f(row.get("dist_to_pivot_atr", math.nan))
    score  = _f(row.get("composite_score", math.nan))
    fail   = _f(row.get("failure_risk", math.nan))
    is_ext = bool(row.get("is_extended", False))

    # Position health classification
    if state in {"FAILED"}:
        health = "failed"
        rec_action = "EXIT — setup failed"
    elif is_ext or state in {"LATE", "EXHAUSTED"}:
        health = "extended"
        rec_action = "TRIM — extended from base, consider taking partial profits"
    elif math.isfinite(dist_f) and dist_f < -1.0:
        # Price pulled back more than 1 ATR below pivot
        health = "at_risk"
        rec_action = "WATCH STOP — below pivot, stop discipline required"
    elif math.isfinite(fail) and fail > 0.55:
        health = "at_risk"
        rec_action = "MONITOR closely — elevated failure risk"
    elif state in {"CONFIRMED", "ACCEPTED", "TRIGGERED"} and (not math.isfinite(fail) or fail < 0.45):
        health = "healthy"
        rec_action = "HOLD — confirmed uptrend, maintain stop below pivot"
    elif state in {"BASE", "ARMED"}:
        health = "recovering"
        rec_action = "HOLD — base forming; add on fresh breakout if it re-triggers"
    else:
        health = "neutral"
        rec_action = "HOLD — no specific action required"

    # Key level to watch
    if math.isfinite(levels.stop):
        key_level = f"Stop at ${levels.stop:.2f}"
        if math.isfinite(dist_f):
            key_level += f" ({abs(dist_f):.1f} ATR from pivot)"
    else:
        key_level = "Stop level unavailable"

    notes: list[str] = []
    if state in {"TRIGGERED", "ACCEPTED"}:
        days = int(row.get("days_in_state", 0) or 0)
        if days > 0:
            notes.append(f"{days} days since trigger")
    if math.isfinite(score):
        notes.append(f"Composite score: {score:.2f}")
    if math.isfinite(dist_f):
        notes.append(f"{'Above' if dist_f >= 0 else 'Below'} pivot by {abs(dist_f):.1f} ATR")

    return {
        "position_health":    health,
        "recommended_action": rec_action,
        "key_level":          key_level,
        "notes":              notes,
    }


def _coherence_issues(
    *,
    bucket: str,
    route: dict,
    trade_plan: dict,
    extension_state: str,
) -> list[str]:
    issues: list[str] = []
    setup_key = route["setup_key"]
    entry_style = str(trade_plan.get("entry_style", ""))
    action_label = str(route.get("action_label", ""))
    setup_classification = str(route.get("setup_classification", ""))
    entry_condition = str(trade_plan.get("entry_condition", "")).lower()
    actionable_now = bool(trade_plan.get("actionable_now", False))

    if bucket == BUCKET_BREAKOUT and setup_key not in {"fresh_breakout", "breakout_watch"}:
        issues.append("breakout_bucket_with_non_breakout_setup")
    if bucket == BUCKET_BREAKOUT and entry_style != "breakout":
        issues.append("breakout_bucket_with_non_breakout_entry_style")
    if bucket == BUCKET_BREAKOUT and "pullback" in setup_classification.lower():
        issues.append("breakout_bucket_with_pullback_classification")

    if bucket == BUCKET_PULLBACK and setup_key not in {"reclaim_pullback", "constructive_pullback", "aged_breakout_pullback"}:
        issues.append("pullback_bucket_with_non_pullback_setup")
    if bucket == BUCKET_PULLBACK and entry_style not in {"pullback", "reclaim"}:
        issues.append("pullback_bucket_with_wrong_entry_style")

    if action_label == ACTION_NOW and not actionable_now:
        issues.append("actionable_now_label_without_actionable_now_trade_plan")
    if action_label == ACTION_NOW and any(token in entry_condition for token in ("break above", "reclaim")):
        issues.append("actionable_now_label_with_future_condition")
    if action_label == ACTION_BREAKOUT and actionable_now:
        issues.append("breakout_watch_label_with_actionable_now_trade_plan")
    if action_label == ACTION_PULLBACK and entry_style == "breakout":
        issues.append("pullback_label_with_breakout_entry_style")

    if extension_state != "not_extended" and bucket in {BUCKET_BREAKOUT, BUCKET_PULLBACK}:
        issues.append("extended_name_in_fresh_entry_bucket")

    if bucket == BUCKET_PORTFOLIO and entry_style != "hold":
        issues.append("portfolio_bucket_without_hold_trade_plan")
    if bucket == BUCKET_EXCLUDED and action_label != ACTION_AVOID:
        issues.append("excluded_bucket_without_avoid_label")
    if bucket == BUCKET_EXTENDED and action_label != ACTION_EXTENDED:
        issues.append("extended_bucket_without_extended_label")

    return issues


def _packet_completeness_issues(
    *,
    bucket: str,
    route: dict,
    trade_plan: dict,
    final_verdict: str,
    levels: TradeLevels,
) -> list[str]:
    if bucket not in {BUCKET_BREAKOUT, BUCKET_PULLBACK}:
        return ["bucket_not_surface_eligible"]

    issues: list[str] = []
    top_level_required = {
        "setup_classification": route.get("setup_classification"),
        "action_label": route.get("action_label"),
        "promotion_reason": route.get("promotion_reason"),
        "final_verdict": final_verdict,
    }
    for field, value in top_level_required.items():
        if _is_missing(value):
            issues.append(f"missing_{field}")

    trade_required = {
        "entry_style": trade_plan.get("entry_style"),
        "entry_condition": trade_plan.get("entry_condition"),
        "entry_trigger": trade_plan.get("entry_trigger"),
        "alternate_pullback_entry": trade_plan.get("alternate_pullback_entry"),
        "invalidation": trade_plan.get("invalidation"),
        "why_now": trade_plan.get("why_now"),
        "why_not_now": trade_plan.get("why_not_now"),
        "what_improves_tomorrow": trade_plan.get("what_improves_tomorrow"),
        "what_weakens_tomorrow": trade_plan.get("what_weakens_tomorrow"),
        "target_1": trade_plan.get("target_1"),
        "target_2": trade_plan.get("target_2"),
        "target_3": trade_plan.get("target_3"),
    }
    for field, value in trade_required.items():
        if _is_missing(value):
            issues.append(f"missing_trade_plan_{field}")

    level_required = {
        "pivot": levels.pivot,
        "entry_lo": levels.entry_lo,
        "entry_hi": levels.entry_hi,
        "stop": levels.stop,
        "t1": levels.t1,
        "t2": levels.t2,
        "t3": levels.t3,
        "s1": levels.s1,
        "s2": levels.s2,
        "s3": levels.s3,
        "r1": levels.r1,
        "r2": levels.r2,
        "r3": levels.r3,
    }
    for field, value in level_required.items():
        if not math.isfinite(_f(value)):
            issues.append(f"missing_level_{field}")

    return issues


def _summarize_ma_context(ma_table: list[dict]) -> str:
    if not ma_table:
        return ""
    parts: list[str] = []
    for row in ma_table:
        name = str(row.get("name", ""))
        if name not in {"SMA10", "SMA20", "EMA20", "SMA50"}:
            continue
        slope = str(row.get("slope", ""))
        bias = str(row.get("tomorrow_bias") or row.get("bias") or "")
        bit = f"{name} {slope}".strip()
        if bias:
            bit = f"{bit} ({bias})"
        if bit:
            parts.append(bit)
    return "; ".join(parts[:3])


def _summarize_avwap_context(avwap_table: list[dict]) -> str:
    if not avwap_table:
        return ""
    parts: list[str] = []
    for row in avwap_table[:3]:
        if not row.get("supported", True):
            continue
        anchor = str(row.get("anchor", ""))
        status = str(row.get("status", ""))
        try:
            dist_f = float(row.get("dist_atr", math.nan))
            dist_str = f" ({dist_f:+.1f} ATR)" if math.isfinite(dist_f) else ""
        except (TypeError, ValueError):
            dist_str = ""
        if anchor or status:
            parts.append(f"{anchor}: {status}{dist_str}")
    return "; ".join(parts)


# ---------------------------------------------------------------------------
# Lightweight packet builder (no file I/O — the canonical source of truth)
# ---------------------------------------------------------------------------

def build_lightweight_packet(row: pd.Series) -> dict:
    """Build the canonical analysis packet from a scored snapshot row.

    This is the central function in the packet-first architecture.  It:
      1. Computes eligibility (calls assess_eligibility internally — not from row columns)
      2. Computes freshness (calls classify_row internally)
      3. Builds an enriched row with those results for downstream callers
      4. Assigns action label, setup classification, bucket
      5. Computes trade levels
      6. Builds trade plan (explicit entry/stop/target/risk)
      7. Builds portfolio health (for portfolio holdings)
      8. Builds narrative (bucket-routed templates)
      9. Assembles and returns the complete packet dict

    Parameters
    ----------
    row : one row from the scored snapshot (with model scores, state, features).
          Does NOT need pre-added eligibility/freshness/action columns — this
          function computes all of them internally.

    Returns
    -------
    Complete packet dict.  The ``context`` key is None; call
    enrich_with_context() to populate it for display/AI use.
    """
    # ── Step 1: Eligibility (assessed internally, not from row columns) ─────────
    elig = assess_eligibility(row)
    eligible: bool = elig["eligible"]
    rejection_reasons: list[str] = elig.get("rejection_reasons", [])
    eligibility_warnings: list[str] = elig.get("warnings", [])

    # ── Step 2: Freshness (assessed internally) ─────────────────────────────────
    fresh = classify_row(row)
    is_fresh         = bool(fresh.get("is_fresh", False))
    is_extended      = bool(fresh.get("is_extended", False))
    is_stale_confirmed = bool(fresh.get("is_stale_confirmed", False))
    freshness_label  = str(fresh.get("freshness_label", "—"))
    extension_reasons = str(fresh.get("extension_reasons", ""))

    # ── Step 3: Build enriched row for action/bucket callers ────────────────────
    # These callers read is_fresh / is_extended / eligible / rejection_reasons from the row.
    enriched = row.copy()
    enriched["eligible"]           = eligible
    enriched["rejection_reasons"]  = ", ".join(rejection_reasons)
    enriched["is_fresh"]           = is_fresh
    enriched["is_extended"]        = is_extended
    enriched["is_stale_confirmed"] = is_stale_confirmed

    # ── Step 4: Action label, setup classification, bucket ───────────────────────
    action    = assign_action(enriched)
    setup_cls = classify_setup(enriched)
    port_guidance_str = str(row.get("portfolio_guidance", portfolio_guidance(enriched)))
    bucket    = assign_bucket(enriched)

    # ── Step 4b: Structural analysis (snapshot fields only) ─────────────────────
    dist_f_raw   = _f(row.get("dist_to_pivot_atr", math.nan))
    days_raw     = int(row.get("days_in_state", 0) or 0)
    state_raw    = str(row.get("state", "NONE"))

    daily_trend_state = _compute_daily_trend_state(row)
    pullback_quality  = _compute_pullback_quality(row, bucket, state_raw, days_raw, dist_f_raw)
    demotion_reason   = _compute_demotion_reason(row, bucket, state_raw, days_raw, dist_f_raw)

    # ── Step 5: Trade levels ─────────────────────────────────────────────────────
    levels: TradeLevels = compute_levels(row)

    # ── Step 6: Trade plan (explicit entry/stop/target/risk) ────────────────────
    trade_plan = _build_trade_plan(row, levels, action, bucket, fresh)

    # ── Step 7: Portfolio health ─────────────────────────────────────────────────
    portfolio_health = _build_portfolio_health(row, levels, action)

    # ── Step 8: Narrative (bucket-routed templates) ──────────────────────────────
    narrative = build_narrative(row, levels, action, bucket=bucket)

    # ── Step 9: Provider symbol ──────────────────────────────────────────────────
    provider_sym = _s(row.get("provider_symbol", row.get("symbol")))

    # ── Score helpers ────────────────────────────────────────────────────────────
    rs_f    = _f(row.get("daily_rs_63", math.nan))
    cvs50_f = _f(row.get("close_vs_sma50", math.nan))
    rs_class = (
        "outperforming" if rs_f > 0.05
        else "underperforming" if rs_f < -0.05
        else "neutral"
    ) if math.isfinite(rs_f) else "unknown"
    ma_slope_direction = (
        "rising"  if cvs50_f > 0.005
        else "falling" if cvs50_f < -0.005
        else "flat"
    ) if math.isfinite(cvs50_f) else "unknown"

    return {
        # ── Identity ──────────────────────────────────────────────────────────
        "symbol":          _s(row.get("user_symbol", row.get("symbol"))),
        "provider_symbol": provider_sym,
        "state":           _s(row.get("state"), "NONE"),
        "groups":          _s(row.get("groups")),
        "is_portfolio":    bool(row.get("is_portfolio", False)),
        "is_watchlist":    bool(row.get("is_watchlist", True)),
        "is_non_equity":   bool(row.get("is_non_equity", False)),

        # ── Eligibility (SOURCE OF TRUTH — computed internally) ───────────────
        "eligible":               eligible,
        "rejection_reasons":      ", ".join(rejection_reasons),
        "rejection_reasons_list": rejection_reasons,
        "eligibility_warnings":   ", ".join(eligibility_warnings),

        # ── Freshness (SOURCE OF TRUTH — computed internally) ─────────────────
        "freshness_label":    freshness_label,
        "is_fresh":           is_fresh,
        "is_extended":        is_extended,
        "is_stale_confirmed": is_stale_confirmed,
        "extension_reasons":  extension_reasons,
        "days_in_state":      int(row.get("days_in_state", 0) or 0),

        # ── Bucket / action (SOURCE OF TRUTH — computed from above) ──────────
        "bucket":               bucket,
        "action_label":         action,
        "setup_classification": setup_cls,
        "portfolio_guidance":   port_guidance_str,

        # ── Structural state (Tier 1 — snapshot fields, no file I/O) ─────────
        # weekly_trend_state is None until enrich_with_context(); daily is immediate.
        "daily_trend_state":  daily_trend_state,
        "weekly_trend_state": None,    # populated by enrich_with_context()
        "pullback_quality":   pullback_quality,
        "demotion_reason":    demotion_reason,

        # ── Scores (from snapshot row, model-calibrated) ──────────────────────
        "composite_score": _num(row, "composite_score"),
        "setup_score":     _num(row, "setup_score"),
        "trade_score":     _num(row, "trade_score"),
        "failure_risk":    _num(row, "failure_risk"),
        "percentile_rank": _num(row, "percentile_rank", 0),

        # ── Price & structure ─────────────────────────────────────────────────
        "close":            _num(row, "close"),
        "pivot":            _num(row, "pivot"),
        "atr14":            _num(row, "atr14"),
        "dist_to_pivot_atr": _num(row, "dist_to_pivot_atr", 1),
        "base_length":      int(row.get("base_length", 0) or 0),

        # ── Trade levels (from compute_levels) ────────────────────────────────
        "entry_lo": f"{levels.entry_lo:.2f}" if math.isfinite(levels.entry_lo) else "—",
        "entry_hi": f"{levels.entry_hi:.2f}" if math.isfinite(levels.entry_hi) else "—",
        "stop":     f"{levels.stop:.2f}"     if math.isfinite(levels.stop)     else "—",
        "t1":       f"{levels.t1:.2f}"       if math.isfinite(levels.t1)       else "—",
        "t2":       f"{levels.t2:.2f}"       if math.isfinite(levels.t2)       else "—",
        "t3":       f"{levels.t3:.2f}"       if math.isfinite(levels.t3)       else "—",
        "s1":       f"{levels.s1:.2f}"       if math.isfinite(levels.s1)       else "—",
        "s2":       f"{levels.s2:.2f}"       if math.isfinite(levels.s2)       else "—",
        "s3":       f"{levels.s3:.2f}"       if math.isfinite(levels.s3)       else "—",
        "r1":       f"{levels.r1:.2f}"       if math.isfinite(levels.r1)       else "—",
        "r2":       f"{levels.r2:.2f}"       if math.isfinite(levels.r2)       else "—",
        "r3":       f"{levels.r3:.2f}"       if math.isfinite(levels.r3)       else "—",
        "risk_reward_t1": f"{levels.risk_reward_t1:.1f}x"
                          if math.isfinite(levels.risk_reward_t1) else "—",

        # ── Trade plan (adapted from swing_engine/checklist.py) ───────────────
        "trade_plan": trade_plan,

        # ── Portfolio position health (adapted from swing_engine/packets.py) ──
        "portfolio_health": portfolio_health,

        # ── Context metrics (snapshot row pass-through) ───────────────────────
        "atr_compression_pct":  _num(row, "atr_compression_pct", 0),
        "volume_dryup":         _num(row, "volume_dryup", 1),
        "daily_rs_63":          _num(row, "daily_rs_63", 3),
        "rs_class":             rs_class,
        "close_vs_sma50":       _num(row, "close_vs_sma50", 3),
        "ma_slope_direction":   ma_slope_direction,
        "ytd_dist_atr":         _num(row, "ytd_dist_atr", 1),
        "swing_low_dist_atr":   _num(row, "swing_low_dist_atr", 1),
        "regime_spy_trend":     _num(row, "regime_spy_trend", 0),

        # ── Narrative (bucket-routed) ─────────────────────────────────────────
        "narrative": narrative,

        # ── Deep context: None until enrich_with_context() is called ─────────
        # Keys are guaranteed present so downstream consumers don't KeyError.
        "context":     None,
        "assessments": None,

        # ── AI note (filled by ai_notes.enrich_packets_with_ai) ──────────────
        "ai_note": None,

        # ── Chart paths (filled by charts module) ─────────────────────────────
        "chart_weekly":   None,
        "chart_daily":    None,
        "chart_intraday": None,
    }


# ---------------------------------------------------------------------------
# Context enrichment (file I/O — called only for top-N selected packets)
# ---------------------------------------------------------------------------

def _build_lightweight_packet_canonical(row: pd.Series) -> dict:
    elig = assess_eligibility(row)
    eligible: bool = bool(elig.get("eligible", False))
    rejection_reasons: list[str] = list(elig.get("rejection_reasons", []))
    eligibility_warnings: list[str] = list(elig.get("warnings", []))

    fresh = classify_row(row)
    is_fresh = bool(fresh.get("is_fresh", False))
    is_stale_confirmed = bool(fresh.get("is_stale_confirmed", False))
    freshness_label = str(fresh.get("freshness_label", "â€”"))
    is_extended, extension_state, extension_reasons = _collect_extension_truth(row, fresh)
    daily_trend_state = _compute_daily_trend_state(row)

    route = _route_packet(
        row,
        eligible=eligible,
        rejection_reasons=rejection_reasons,
        is_fresh=is_fresh,
        is_extended=is_extended,
        extension_state=extension_state,
        daily_trend_state=daily_trend_state,
    )
    if route["bucket"] == BUCKET_EXCLUDED and route["route_reason"] not in rejection_reasons:
        rejection_reasons = rejection_reasons + [route["route_reason"]]

    dist_f_raw = _f(row.get("dist_to_pivot_atr", math.nan))
    days_raw = int(row.get("days_in_state", 0) or 0)
    state_raw = str(row.get("state", "NONE"))
    pullback_quality = _compute_pullback_quality(row, route["bucket"], state_raw, days_raw, dist_f_raw)
    demotion_reason = route.get("demotion_reason") or _compute_demotion_reason(
        row,
        route["bucket"],
        state_raw,
        days_raw,
        dist_f_raw,
    )

    levels: TradeLevels = compute_levels(row)
    trade_plan = _build_trade_plan_canonical(row, levels, route)
    portfolio_health = _build_portfolio_health(row, levels, route["action_label"])
    final_verdict = _build_final_verdict(route, trade_plan)
    narrative = _build_narrative(route, trade_plan, final_verdict)
    coherence_issues = _coherence_issues(
        bucket=route["bucket"],
        route=route,
        trade_plan=trade_plan,
        extension_state=extension_state,
    )
    completeness_issues = _packet_completeness_issues(
        bucket=route["bucket"],
        route=route,
        trade_plan=trade_plan,
        final_verdict=final_verdict,
        levels=levels,
    )

    provider_sym = _s(row.get("provider_symbol", row.get("symbol")))
    rs_f = _f(row.get("daily_rs_63", math.nan))
    cvs50_f = _f(row.get("close_vs_sma50", math.nan))
    rs_class = (
        "outperforming" if rs_f > 0.05
        else "underperforming" if rs_f < -0.05
        else "neutral"
    ) if math.isfinite(rs_f) else "unknown"
    ma_slope_direction = (
        "rising" if cvs50_f > 0.005
        else "falling" if cvs50_f < -0.005
        else "flat"
    ) if math.isfinite(cvs50_f) else "unknown"
    portfolio_guidance = portfolio_health.get("recommended_action", "") if portfolio_health else ""
    if route["bucket"] == BUCKET_NON_EQUITY and not portfolio_guidance:
        portfolio_guidance = "Non-equity informational symbol; treat as cash/informational only."

    return {
        "symbol": _s(row.get("user_symbol", row.get("symbol"))),
        "provider_symbol": provider_sym,
        "state": _s(row.get("state"), "NONE"),
        "groups": _s(row.get("groups")),
        "is_portfolio": bool(row.get("is_portfolio", False)),
        "is_watchlist": bool(row.get("is_watchlist", True)),
        "is_non_equity": bool(row.get("is_non_equity", False)),
        "eligible": eligible,
        "rejection_reasons": ", ".join(rejection_reasons),
        "rejection_reasons_list": rejection_reasons,
        "eligibility_warnings": ", ".join(eligibility_warnings),
        "eligibility_warnings_list": eligibility_warnings,
        "freshness_label": freshness_label,
        "is_fresh": is_fresh,
        "is_extended": is_extended,
        "is_stale_confirmed": is_stale_confirmed,
        "extension_state": extension_state,
        "extension_reasons": extension_reasons,
        "days_in_state": days_raw,
        "bucket": route["bucket"],
        "action_label": route["action_label"],
        "setup_classification": route["setup_classification"],
        "setup_key": route["setup_key"],
        "promotion_reason": route["promotion_reason"],
        "demotion_reason": demotion_reason,
        "route_reason": route["route_reason"],
        "operational_readiness": route["operational_readiness"],
        "daily_trend_state": daily_trend_state,
        "weekly_trend_state": None,
        "trend_health": "constructive" if _is_constructive_trend(daily_trend_state, row) else "fragile",
        "freshness_maturity": (
            "extended" if is_extended
            else "fresh_trigger" if state_raw in {"TRIGGERED", "ACCEPTED"} and days_raw <= BREAKOUT_TRIGGER_DAYS
            else "primed" if state_raw in {"BASE", "ARMED"} and is_fresh
            else "aging"
        ),
        "pullback_quality": pullback_quality,
        "composite_score": _num(row, "composite_score"),
        "setup_score": _num(row, "setup_score"),
        "trade_score": _num(row, "trade_score"),
        "failure_risk": _num(row, "failure_risk"),
        "percentile_rank": _num(row, "percentile_rank", 0),
        "close": _num(row, "close"),
        "pivot": _num(row, "pivot"),
        "atr14": _num(row, "atr14"),
        "dist_to_pivot_atr": _num(row, "dist_to_pivot_atr", 1),
        "base_length": int(row.get("base_length", 0) or 0),
        "entry_lo": _num_from_value(levels.entry_lo),
        "entry_hi": _num_from_value(levels.entry_hi),
        "stop": _num_from_value(levels.stop),
        "t1": _num_from_value(levels.t1),
        "t2": _num_from_value(levels.t2),
        "t3": _num_from_value(levels.t3),
        "s1": _num_from_value(levels.s1),
        "s2": _num_from_value(levels.s2),
        "s3": _num_from_value(levels.s3),
        "r1": _num_from_value(levels.r1),
        "r2": _num_from_value(levels.r2),
        "r3": _num_from_value(levels.r3),
        "risk_reward_t1": f"{_f(levels.risk_reward_t1):.1f}x" if math.isfinite(_f(levels.risk_reward_t1)) else "â€”",
        "trade_plan": trade_plan,
        "portfolio_health": portfolio_health,
        "portfolio_guidance": portfolio_guidance,
        "atr_compression_pct": _num(row, "atr_compression_pct", 0),
        "volume_dryup": _num(row, "volume_dryup", 2),
        "daily_rs_63": _num(row, "daily_rs_63", 3),
        "rs_class": rs_class,
        "close_vs_sma50": _num(row, "close_vs_sma50", 3),
        "ma_slope_direction": ma_slope_direction,
        "ytd_dist_atr": _num(row, "ytd_dist_atr", 1),
        "swing_low_dist_atr": _num(row, "swing_low_dist_atr", 1),
        "regime_spy_trend": _num(row, "regime_spy_trend", 0),
        "operational_fresh_entry": route["bucket"] in {BUCKET_BREAKOUT, BUCKET_PULLBACK},
        "final_verdict": final_verdict,
        "coherence_ok": len(coherence_issues) == 0,
        "coherence_issues": coherence_issues,
        "packet_complete_for_surface": len(completeness_issues) == 0,
        "packet_completeness_issues": completeness_issues,
        "narrative": narrative,
        "context": None,
        "assessments": None,
        "ma_table": None,
        "avwap_table": None,
        "volume_block": None,
        "confluence": None,
        "checklist": None,
        "score_drivers": None,
        "trend_state": None,
        "ai_note": None,
        "chart_weekly": None,
        "chart_daily": None,
        "chart_intraday": None,
        "intraday_policy": "daily_only",
        "intraday_available": False,
        "intraday_used_in_qualification": False,
        "intraday_note": "Intraday confirmation is not part of v1 qualification; surfaced setup truth is daily/weekly only.",
        "surfaced_in_top": False,
        "surface_section": None,
        "not_surfaced_reason": "",
        "selector_blockers": [],
    }


build_lightweight_packet = _build_lightweight_packet_canonical


def enrich_with_context(pkt: dict, row: pd.Series) -> dict:
    """Add expensive context block to a lightweight packet.

    Reads raw daily OHLCV and features parquet for the symbol.  Should only
    be called for the top-N packets selected for display; NOT for all symbols.

    Parameters
    ----------
    pkt : lightweight packet produced by build_lightweight_packet().
    row : the original snapshot row (provides close, atr14, pivot).

    Returns
    -------
    The same packet dict (mutated in-place) with ``context`` and
    ``assessments`` populated.
    """
    provider_sym = pkt.get("provider_symbol", pkt.get("symbol", ""))
    pivot_f = _f(row.get("pivot", math.nan))

    levels_dict = {
        "pivot":          pivot_f,
        "stop":           _f(pkt.get("stop", math.nan)),
        "t1":             _f(pkt.get("t1", math.nan)),
        "t2":             _f(pkt.get("t2", math.nan)),
        "t3":             _f(pkt.get("t3", math.nan)),
        "s1":             _f(pkt.get("s1", math.nan)),
        "s2":             _f(pkt.get("s2", math.nan)),
        "r1":             _f(pkt.get("r1", math.nan)),
        "r2":             _f(pkt.get("r2", math.nan)),
        "risk_reward_t1": _f(pkt.get("risk_reward_t1", math.nan)),
    }

    context = build_context(provider_sym, row, levels_dict)
    pkt["context"]     = context
    pkt["assessments"] = context.get("assessments", {})
    pkt["ma_table"] = context.get("ma_table", [])
    pkt["avwap_table"] = context.get("avwap_table", [])
    pkt["volume_block"] = context.get("volume_block", {})
    pkt["confluence"] = context.get("confluence", {})
    pkt["checklist"] = context.get("checklist", [])
    pkt["score_drivers"] = context.get("score_drivers", {})
    pkt["trend_state"] = context.get("trend_state", {})

    # Populate Tier-2 structural fields from context
    trend_state = context.get("trend_state", {})
    if isinstance(trend_state, dict):
        pkt["daily_trend_state"] = trend_state.get("daily_trend_state", pkt.get("daily_trend_state"))
        pkt["weekly_trend_state"] = trend_state.get("weekly_trend_state")

    narrative = pkt.get("narrative", {})
    if isinstance(narrative, dict):
        narrative["ma_context"] = _summarize_ma_context(pkt.get("ma_table") or [])
        narrative["avwap_context"] = _summarize_avwap_context(pkt.get("avwap_table") or [])
        pkt["narrative"] = narrative

    return pkt


# ---------------------------------------------------------------------------
# Batch builders (backward compat + convenience)
# ---------------------------------------------------------------------------

def build_packet(row: pd.Series) -> dict:
    """Build a complete packet (lightweight + context) for a single row.

    Convenience function for callers that need a fully enriched packet for
    one symbol.  Equivalent to::

        pkt = build_lightweight_packet(row)
        pkt = enrich_with_context(pkt, row)

    Use ``build_lightweight_packet`` when building packets for all symbols
    in the pipeline (much faster; skip context for non-selected symbols).
    """
    pkt = build_lightweight_packet(row)
    return enrich_with_context(pkt, row)


def build_packets(df: pd.DataFrame) -> list[dict]:
    """Build lightweight packets for all rows, then enrich each with context.

    Backward-compatible batch builder.  For large DataFrames, prefer the
    two-pass pattern:
      1. all_pkts = build_all_lightweight_packets(df)
      2. selected = select_packets(all_pkts)
      3. enrich only the top-N selected packets.
    """
    return [build_packet(row) for _, row in df.iterrows()]


def build_all_lightweight_packets(df: pd.DataFrame) -> list[dict]:
    """Build lightweight packets for every row in df (no file I/O).

    This is the first pass in the packet-first pipeline.  All symbols get
    a complete decision-relevant packet; expensive context is deferred to
    the second pass (enrich_with_context) for selected packets only.

    Parameters
    ----------
    df : scored snapshot DataFrame.

    Returns
    -------
    list[dict] — one lightweight packet per row, in df order.
    """
    packets = []
    for _, row in df.iterrows():
        try:
            packets.append(build_lightweight_packet(row))
        except Exception:
            # Never let one bad row break the entire batch.
            import logging
            logging.getLogger(__name__).warning(
                "build_lightweight_packet failed for symbol=%s",
                row.get("symbol", "?"),
            )
    return packets
