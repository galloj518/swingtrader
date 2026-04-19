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

from swingtrader.dashboard.action import assign_action, classify_setup, portfolio_guidance
from swingtrader.dashboard.buckets import (
    BREAKOUT_DIST_ATR,
    BREAKOUT_TRIGGER_DAYS,
    BUCKET_PULLBACK,
    assign_bucket,
)
from swingtrader.dashboard.context import build_context
from swingtrader.dashboard.eligibility import assess_eligibility
from swingtrader.dashboard.freshness import FRESH_MAX_DAYS, classify_row
from swingtrader.dashboard.levels import TradeLevels, compute_levels
from swingtrader.dashboard.narrative import build_narrative

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
        if math.isfinite(dist_f) and abs(dist_f) > BREAKOUT_DIST_ATR:
            return f"far_from_pivot ({abs(dist_f):.1f} ATR vs {BREAKOUT_DIST_ATR} threshold)"

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

    # Populate Tier-2 structural fields from context
    trend_state = context.get("trend_state", {})
    if isinstance(trend_state, dict):
        pkt["weekly_trend_state"] = trend_state.get("weekly_trend_state")

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
