"""Structured analysis packet for each top setup.

Assembles all computed data for one symbol into a single dict that is:
  - Human-readable (used by the HTML dashboard)
  - AI-review-ready (can be passed to an LLM for further analysis without
    any additional data fetching)
  - Serialisable (all values are plain Python scalars or strings)

The packet is intentionally verbose: it captures everything the dashboard
needs so that the rendering layer can be kept simple.
"""
from __future__ import annotations

import math
from typing import Any

import pandas as pd

from swingtrader.dashboard.action import assign_action
from swingtrader.dashboard.freshness import classify_row
from swingtrader.dashboard.levels import TradeLevels, compute_levels
from swingtrader.dashboard.narrative import build_narrative


def _f(v: Any) -> float:
    """Coerce to float; return nan if missing."""
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


def build_packet(row: pd.Series) -> dict:
    """Build the complete analysis packet for one snapshot row.

    Parameters
    ----------
    row : one row from the scored snapshot with freshness + action columns.

    Returns
    -------
    dict with all fields needed by the dashboard renderer.
    """
    # Freshness (may already be computed; recompute defensively)
    fresh = classify_row(row)
    action = str(row.get("action_label", assign_action(row)))
    levels: TradeLevels = compute_levels(row)
    narrative = build_narrative(row, levels, action)

    def _pct(col: str) -> str:
        """Format a 0-1 float as percent string."""
        v = _f(row.get(col, math.nan))
        return f"{v * 100:.0f}%" if math.isfinite(v) else "—"

    def _num(col: str, d: int = 2) -> str:
        v = _f(row.get(col, math.nan))
        return f"{v:.{d}f}" if math.isfinite(v) else "—"

    # MA slope context: projected direction based on close vs SMA50
    close_vs_sma50 = _f(row.get("close_vs_sma50", math.nan))
    ma_slope_direction = (
        "rising" if close_vs_sma50 > 0.005
        else "falling" if close_vs_sma50 < -0.005
        else "flat"
    ) if math.isfinite(close_vs_sma50) else "unknown"

    # RS classification
    rs = _f(row.get("daily_rs_63", math.nan))
    rs_class = (
        "outperforming" if rs > 0.05
        else "underperforming" if rs < -0.05
        else "neutral"
    ) if math.isfinite(rs) else "unknown"

    return {
        # ── Identity ──────────────────────────────────────────────────────────
        "symbol": _s(row.get("user_symbol", row.get("symbol"))),
        "provider_symbol": _s(row.get("provider_symbol", row.get("symbol"))),
        "state": _s(row.get("state"), "NONE"),
        "groups": _s(row.get("groups")),
        "is_portfolio": bool(row.get("is_portfolio", False)),
        "is_watchlist": bool(row.get("is_watchlist", True)),

        # ── Classification ────────────────────────────────────────────────────
        "action_label": action,
        "freshness_label": fresh.get("freshness_label", "—"),
        "is_fresh": fresh.get("is_fresh", False),
        "is_extended": fresh.get("is_extended", False),
        "is_stale_confirmed": fresh.get("is_stale_confirmed", False),

        # ── Scores ───────────────────────────────────────────────────────────
        "composite_score": _num("composite_score"),
        "setup_score": _num("setup_score"),
        "trade_score": _num("trade_score"),
        "failure_risk": _num("failure_risk"),
        "percentile_rank": _num("percentile_rank", 0),

        # ── Price & ATR ───────────────────────────────────────────────────────
        "close": _num("close"),
        "pivot": _num("pivot"),
        "atr14": _num("atr14"),
        "dist_to_pivot_atr": _num("dist_to_pivot_atr", 1),
        "base_length": int(row.get("base_length", 0) or 0),
        "days_in_state": int(row.get("days_in_state", 0) or 0),

        # ── Trade levels ──────────────────────────────────────────────────────
        "entry_lo": f"{levels.entry_lo:.2f}" if math.isfinite(levels.entry_lo) else "—",
        "entry_hi": f"{levels.entry_hi:.2f}" if math.isfinite(levels.entry_hi) else "—",
        "stop": f"{levels.stop:.2f}" if math.isfinite(levels.stop) else "—",
        "t1": f"{levels.t1:.2f}" if math.isfinite(levels.t1) else "—",
        "t2": f"{levels.t2:.2f}" if math.isfinite(levels.t2) else "—",
        "t3": f"{levels.t3:.2f}" if math.isfinite(levels.t3) else "—",
        "s1": f"{levels.s1:.2f}" if math.isfinite(levels.s1) else "—",
        "s2": f"{levels.s2:.2f}" if math.isfinite(levels.s2) else "—",
        "s3": f"{levels.s3:.2f}" if math.isfinite(levels.s3) else "—",
        "r1": f"{levels.r1:.2f}" if math.isfinite(levels.r1) else "—",
        "r2": f"{levels.r2:.2f}" if math.isfinite(levels.r2) else "—",
        "r3": f"{levels.r3:.2f}" if math.isfinite(levels.r3) else "—",
        "risk_reward_t1": f"{levels.risk_reward_t1:.1f}x" if math.isfinite(levels.risk_reward_t1) else "—",

        # ── Context ───────────────────────────────────────────────────────────
        "atr_compression_pct": _num("atr_compression_pct", 0),
        "volume_dryup": _num("volume_dryup", 1),
        "daily_rs_63": _num("daily_rs_63", 3),
        "rs_class": rs_class,
        "close_vs_sma50": _num("close_vs_sma50", 3),
        "ma_slope_direction": ma_slope_direction,
        "ytd_dist_atr": _num("ytd_dist_atr", 1),
        "swing_low_dist_atr": _num("swing_low_dist_atr", 1),
        "regime_spy_trend": _num("regime_spy_trend", 0),

        # ── Narrative ─────────────────────────────────────────────────────────
        "narrative": narrative,

        # ── Chart paths (filled in by charts module) ──────────────────────────
        "chart_weekly": None,
        "chart_daily": None,
        "chart_intraday": None,
    }


def build_packets(df: pd.DataFrame) -> list[dict]:
    """Build packets for all rows in df.

    Parameters
    ----------
    df : top setups DataFrame (output of selector.select_top_setups()).

    Returns
    -------
    list of packet dicts, one per row, in the same order.
    """
    packets = []
    for _, row in df.iterrows():
        packets.append(build_packet(row))
    return packets
