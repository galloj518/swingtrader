"""Machine-readable JSON artifacts for each daily run.

Writes stable structured outputs to an artifacts/ subdirectory alongside the
HTML dashboard. These files are designed for downstream tooling and AI review:

  artifacts/dashboard_summary.json  — top-level run metadata and regime snapshot
  artifacts/top_setups.json         — array of sanitised, section-labelled packets
  artifacts/portfolio_review.json   — portfolio-only review records
  artifacts/{SYM}_packet.json       — full raw packet per top setup

All values are JSON-serializable: NaN/inf → null, "—" sentinel → null,
floats rounded to 4 dp where numeric, strings preserved otherwise.

Usage::

    from swingtrader.reports.artifacts import write_artifacts

    paths = write_artifacts(packets, portfolio_df, snapshot_df, as_of, output_dir)
"""
from __future__ import annotations

import json
import math
import datetime
from pathlib import Path

import pandas as pd

from swingtrader.utils.logging import get_logger

log = get_logger(__name__)

# String sentinel used by packet.py for missing numeric fields.
_DASH = "—"


# ── Value cleaners ────────────────────────────────────────────────────────────

def _clean_value(v):
    """Return a JSON-serializable scalar.

    Rules
    -----
    - None / NaN / ±inf  → None
    - "—" sentinel str   → None
    - Finite float/int   → float (preserves precision up to JSON limits)
    - bool               → bool (must come before numeric check; bool is int subclass)
    - Everything else    → str
    """
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(v, int):
        return v
    if isinstance(v, str):
        if v in (_DASH, "nan", "None", "inf", "-inf", ""):
            return None
        # Attempt numeric coercion for string-formatted floats from packet.py
        # (e.g. "38.50", "0.72") so downstream consumers get proper JSON numbers.
        try:
            fv = float(v)
            if math.isnan(fv) or math.isinf(fv):
                return None
            return fv
        except ValueError:
            return v
    # Fallback: delegate to str for non-standard types (Timestamp, etc.)
    return str(v)


def _clean_dict(d: dict) -> dict:
    """Recursively clean a dict, returning a new dict with clean values."""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = _clean_dict(v)
        elif isinstance(v, list):
            out[k] = _clean_list(v)
        else:
            out[k] = _clean_value(v)
    return out


def _clean_list(lst: list) -> list:
    """Recursively clean a list, returning a new list with clean values."""
    out = []
    for v in lst:
        if isinstance(v, dict):
            out.append(_clean_dict(v))
        elif isinstance(v, list):
            out.append(_clean_list(v))
        else:
            out.append(_clean_value(v))
    return out


# ── Regime extraction ─────────────────────────────────────────────────────────

def _first_non_null(df: pd.DataFrame, col: str):
    """Return the first non-null value from a DataFrame column, or None."""
    if col not in df.columns:
        return None
    col_series = df[col].dropna()
    if col_series.empty:
        return None
    v = col_series.iloc[0]
    return _clean_value(v)


def _extract_regime(snapshot_df: pd.DataFrame) -> dict:
    """Build the regime sub-dict from snapshot_df regime columns."""
    spy_trend = _first_non_null(snapshot_df, "regime_spy_trend")
    above_200 = _first_non_null(snapshot_df, "regime_spy_above_200sma")
    vix = _first_non_null(snapshot_df, "regime_vix_level")

    # Normalise above_200 to a proper bool (it may arrive as 0/1 float)
    if above_200 is not None:
        above_200 = bool(above_200)

    return {
        "spy_trend": spy_trend,
        "spy_above_200sma": above_200,
        "vix_level": vix,
    }


# ── Action label helpers ──────────────────────────────────────────────────────

_BREAKOUT_LABELS = frozenset({
    "Actionable on breakout",
    "Actionable now",
})
_PULLBACK_LABELS = frozenset({
    "Actionable on pullback",
})


def _count_action(packets: list[dict], labels: frozenset[str]) -> int:
    return sum(1 for p in packets if p.get("action_label") in labels)


# ── Portfolio guidance ────────────────────────────────────────────────────────

def _derive_portfolio_guidance(row: pd.Series) -> str:
    """Rule-based portfolio guidance string when packet doesn't supply one."""
    state = str(row.get("state", "NONE"))
    action = str(row.get("action_label", "—"))
    sym = str(row.get("user_symbol", row.get("symbol", "?")))

    # Prefer the action label if it is descriptive
    if action and action not in (_DASH, "—", "None"):
        prefix = f"{sym} ({state})"
        return f"{prefix} — {action}."

    # Fallback rules by state
    if state == "CONFIRMED":
        return f"{sym} — Hold, confirmed uptrend. Monitor for extended conditions."
    if state == "TRIGGERED":
        return f"{sym} — Recently triggered. Monitor breakout follow-through."
    if state == "ACCEPTED":
        return f"{sym} — Accepted breakout. Hold with stop below pivot."
    if state == "ARMED":
        return f"{sym} — Armed near pivot. Watch for breakout catalyst."
    if state == "LATE":
        return f"{sym} — Extended from base. Do not add; let it consolidate."
    if state == "FAILED":
        return f"{sym} — Setup failed. Review exit criteria."
    return f"{sym} ({state}) — No specific guidance."


# ── Packet sectioning ─────────────────────────────────────────────────────────

def _section_packet(pkt: dict, as_of_str: str) -> dict:
    """Reorganise a flat packet dict into clearly labelled sections for JSON output."""
    sym = pkt.get("symbol", "?")
    provider = pkt.get("provider_symbol", sym)

    # ── identity ──────────────────────────────────────────────────────────────
    identity = {
        "state": _clean_value(pkt.get("state")),
        "action_label": _clean_value(pkt.get("action_label")),
        "setup_classification": _clean_value(
            pkt.get("setup_classification", pkt.get("freshness_label"))
        ),
        "freshness_label": _clean_value(pkt.get("freshness_label")),
        "days_in_state": _clean_value(pkt.get("days_in_state")),
    }

    # ── model scores ─────────────────────────────────────────────────────────
    model_scores = {
        "composite_score": _clean_value(pkt.get("composite_score")),
        "setup_score": _clean_value(pkt.get("setup_score")),
        "trade_score": _clean_value(pkt.get("trade_score")),
        "failure_risk": _clean_value(pkt.get("failure_risk")),
        "percentile_rank": _clean_value(pkt.get("percentile_rank")),
        "note": "Model-calibrated probabilities from fitted classifiers",
    }

    # ── price & levels ────────────────────────────────────────────────────────
    price_and_levels = {
        "close": _clean_value(pkt.get("close")),
        "pivot": _clean_value(pkt.get("pivot")),
        "atr14": _clean_value(pkt.get("atr14")),
        "dist_to_pivot_atr": _clean_value(pkt.get("dist_to_pivot_atr")),
        "entry_lo": _clean_value(pkt.get("entry_lo")),
        "entry_hi": _clean_value(pkt.get("entry_hi")),
        "stop": _clean_value(pkt.get("stop")),
        "t1": _clean_value(pkt.get("t1")),
        "t2": _clean_value(pkt.get("t2")),
        "t3": _clean_value(pkt.get("t3")),
        "s1": _clean_value(pkt.get("s1")),
        "s2": _clean_value(pkt.get("s2")),
        "r1": _clean_value(pkt.get("r1")),
        "r2": _clean_value(pkt.get("r2")),
        "risk_reward_t1": _clean_value(pkt.get("risk_reward_t1")),
        "level_method": "ATR-pivot: entry=pivot+0.10*ATR, stop=pivot-1.0*ATR, T1=pivot+2.0*ATR",
    }

    # ── context (narrative sub-dicts) ─────────────────────────────────────────
    narrative_raw = pkt.get("narrative", {})
    if not isinstance(narrative_raw, dict):
        narrative_raw = {}

    context = {
        "ma_table": _clean_value(pkt.get("ma_table")),
        "avwap_table": _clean_value(pkt.get("avwap_table")),
        "volume_block": _clean_value(pkt.get("volume_block")),
        "checklist": _clean_value(pkt.get("checklist")),
        "confluence": _clean_value(pkt.get("confluence")),
        # Convenience duplicates from narrative for AI consumption
        "ma_context": _clean_value(narrative_raw.get("ma_context")),
        "avwap_context": _clean_value(narrative_raw.get("avwap_context")),
    }

    # ── narrative ─────────────────────────────────────────────────────────────
    narrative = {
        "setup": _clean_value(narrative_raw.get("setup")),
        "why": _clean_value(narrative_raw.get("why")),
        "entry": _clean_value(narrative_raw.get("entry")),
        "risk": _clean_value(narrative_raw.get("risk")),
        "targets": _clean_value(narrative_raw.get("targets")),
        "verdict": _clean_value(narrative_raw.get("verdict")),
    }

    # ── chart paths ───────────────────────────────────────────────────────────
    chart_paths = {
        "daily": _clean_value(pkt.get("chart_daily")),
        "weekly": _clean_value(pkt.get("chart_weekly")),
        "intraday": _clean_value(pkt.get("chart_intraday")),
    }

    # ── freshness metadata ────────────────────────────────────────────────────
    freshness_metadata = {
        "is_fresh": _clean_value(pkt.get("is_fresh")),
        "is_extended": _clean_value(pkt.get("is_extended")),
        "is_stale_confirmed": _clean_value(pkt.get("is_stale_confirmed")),
        "days_since_trigger": _clean_value(pkt.get("days_since_trigger")),
        "days_since_confirmation": _clean_value(pkt.get("days_since_confirmation")),
        "last_actionable_check": as_of_str,
    }

    return {
        "symbol": _clean_value(sym),
        "provider_symbol": _clean_value(provider),
        "as_of": as_of_str,
        "identity": identity,
        "model_scores": model_scores,
        "price_and_levels": price_and_levels,
        "context": context,
        "narrative": narrative,
        "ai_note": _clean_value(pkt.get("ai_note")),
        "chart_paths": chart_paths,
        "freshness_metadata": freshness_metadata,
    }


# ── Public API ────────────────────────────────────────────────────────────────

def write_artifacts(
    packets: list[dict],
    portfolio_df: pd.DataFrame,
    snapshot_df: pd.DataFrame,
    as_of: pd.Timestamp,
    output_dir: Path,
) -> dict:
    """Write all JSON artifacts for this run.

    Parameters
    ----------
    packets      : list of packet dicts from dashboard.packet.build_packets()
                   for the top setups. May include ai_note if enriched.
    portfolio_df : subset of snapshot_df where is_portfolio == True.
    snapshot_df  : full scored + freshness + action-labelled snapshot DataFrame.
    as_of        : report date.
    output_dir   : same directory as dashboard.html
                   (e.g. docs/reports/daily/2026-04-17/).

    Returns
    -------
    dict of {"summary": str, "top_setups": str, "portfolio": str,
             "per_symbol": {sym: str, ...}} — all values are absolute path strings.
    """
    output_dir = Path(output_dir)
    artifacts_dir = output_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    as_of_str = str(as_of.date())
    generated_at = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # ── 1. Top setups ─────────────────────────────────────────────────────────
    top_setups_list = [_section_packet(p, as_of_str) for p in packets]
    top_setups_path = artifacts_dir / "top_setups.json"
    top_setups_path.write_text(
        json.dumps(top_setups_list, indent=2, default=str),
        encoding="utf-8",
    )
    log.info("Artifact written → %s (%d setups)", top_setups_path, len(top_setups_list))

    # ── 2. Per-symbol full packets ────────────────────────────────────────────
    per_symbol_paths: dict[str, str] = {}
    for pkt in packets:
        sym = str(pkt.get("symbol", "UNKNOWN"))
        sym_path = artifacts_dir / f"{sym}_packet.json"
        sym_path.write_text(
            json.dumps(_clean_dict(pkt), indent=2, default=str),
            encoding="utf-8",
        )
        per_symbol_paths[sym] = str(sym_path)

    # ── 3. Portfolio review ───────────────────────────────────────────────────
    portfolio_records = []
    if not portfolio_df.empty:
        # Build a lookup from packets for portfolio guidance
        pkt_by_sym = {str(p.get("symbol", "")): p for p in packets}

        for _, row in portfolio_df.iterrows():
            sym = str(row.get("user_symbol", row.get("symbol", "?")))
            pkt = pkt_by_sym.get(sym, {})

            # portfolio_guidance: from packet if available, else rule-based
            guidance = pkt.get("portfolio_guidance")
            if not guidance or guidance in (_DASH, "None", ""):
                guidance = _derive_portfolio_guidance(row)

            record = {
                "symbol": sym,
                "state": _clean_value(row.get("state")),
                "close": _clean_value(row.get("close")),
                "pivot": _clean_value(row.get("pivot")),
                "dist_to_pivot_atr": _clean_value(row.get("dist_to_pivot_atr")),
                "days_in_state": _clean_value(row.get("days_in_state")),
                "action_label": _clean_value(row.get("action_label")),
                "portfolio_guidance": guidance,
                "composite_score": _clean_value(
                    pkt.get("composite_score", row.get("composite_score"))
                ),
                "is_non_equity": bool(row.get("is_non_equity", False)),
            }
            portfolio_records.append(record)

    portfolio_path = artifacts_dir / "portfolio_review.json"
    portfolio_path.write_text(
        json.dumps(portfolio_records, indent=2, default=str),
        encoding="utf-8",
    )
    log.info("Artifact written → %s (%d holdings)", portfolio_path, len(portfolio_records))

    # ── 4. Dashboard summary ──────────────────────────────────────────────────
    n_scored = 0
    if not snapshot_df.empty and "state" in snapshot_df.columns:
        scored_states = {"BASE", "ARMED", "TRIGGERED", "ACCEPTED"}
        n_scored = int(snapshot_df["state"].isin(scored_states).sum())

    n_actionable = 0
    if not snapshot_df.empty and "action_label" in snapshot_df.columns:
        actionable_labels = {
            "Actionable now",
            "Actionable on breakout",
            "Actionable on pullback",
        }
        n_actionable = int(snapshot_df["action_label"].isin(actionable_labels).sum())

    n_breakout = _count_action(packets, _BREAKOUT_LABELS)
    n_pullback = _count_action(packets, _PULLBACK_LABELS)

    top_symbols = [str(p.get("symbol", "?")) for p in packets]

    summary_obj = {
        "as_of": as_of_str,
        "generated_at": generated_at,
        "n_top_setups": len(packets),
        "n_scored": n_scored,
        "n_actionable_now": n_actionable,
        "n_breakout": n_breakout,
        "n_pullback": n_pullback,
        "regime": _extract_regime(snapshot_df),
        "top_symbols": top_symbols,
        "artifact_paths": {
            "top_setups": "artifacts/top_setups.json",
            "portfolio_review": "artifacts/portfolio_review.json",
        },
    }

    summary_path = artifacts_dir / "dashboard_summary.json"
    summary_path.write_text(
        json.dumps(summary_obj, indent=2, default=str),
        encoding="utf-8",
    )
    log.info("Artifact written → %s", summary_path)

    return {
        "summary": str(summary_path),
        "top_setups": str(top_setups_path),
        "portfolio": str(portfolio_path),
        "per_symbol": per_symbol_paths,
    }
