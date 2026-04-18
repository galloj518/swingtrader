"""Machine-readable JSON artifacts for each daily run.

Writes stable structured outputs to an artifacts/ subdirectory alongside the
HTML dashboard. These files are designed for downstream tooling and AI review:

  artifacts/dashboard_summary.json    — top-level run metadata and regime snapshot
  artifacts/top_setups.json           — combined top-N packets (breakout + pullback)
  artifacts/breakout_top_setups.json  — breakout-bucket top candidates only
  artifacts/pullback_top_setups.json  — pullback-bucket top candidates only
  artifacts/extended_leaders.json     — extended-leader symbols (informational)
  artifacts/portfolio_review.json     — portfolio-only review records
  artifacts/eligibility_results.json  — eligibility gate results for all scored symbols
  artifacts/bucket_assignments.json   — bucket membership for all scored symbols
  artifacts/{SYM}_packet.json         — full raw packet per top setup

All values are JSON-serializable: NaN/inf → null, "—" sentinel → null,
floats rounded to 4 dp where numeric, strings preserved otherwise.

Usage::

    from swingtrader.reports.artifacts import write_artifacts

    paths = write_artifacts(
        packets, portfolio_df, snapshot_df, as_of, output_dir,
        breakout_df=breakout_df, pullback_df=pullback_df,
    )
"""
from __future__ import annotations

import datetime
import json
import math
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

def _df_to_simple_records(df: pd.DataFrame, cols: list[str]) -> list[dict]:
    """Convert a DataFrame subset to a list of cleaned dicts."""
    if df.empty:
        return []
    avail = [c for c in cols if c in df.columns]
    records = []
    for _, row in df[avail].iterrows():
        records.append({c: _clean_value(row.get(c)) for c in avail})
    return records


def write_artifacts(
    packets: list[dict],
    portfolio_df: pd.DataFrame,
    snapshot_df: pd.DataFrame,
    as_of: pd.Timestamp,
    output_dir: Path,
    *,
    breakout_df: pd.DataFrame | None = None,
    pullback_df: pd.DataFrame | None = None,
) -> dict:
    """Write all JSON artifacts for this run.

    Parameters
    ----------
    packets      : list of packet dicts (combined top setups).
    portfolio_df : portfolio holdings DataFrame.
    snapshot_df  : full ranked + eligibility + bucket snapshot.
    as_of        : report date.
    output_dir   : same directory as dashboard.html.
    breakout_df  : breakout-bucket top candidates (optional).
    pullback_df  : pullback-bucket top candidates (optional).

    Returns
    -------
    dict mapping artifact names to absolute path strings.
    """
    output_dir = Path(output_dir)
    artifacts_dir = output_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    as_of_str = str(as_of.date())
    generated_at = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

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

    # ── 5. Breakout top setups ────────────────────────────────────────────────
    breakout_list: list[dict] = []
    breakout_path = artifacts_dir / "breakout_top_setups.json"
    if breakout_df is not None and not breakout_df.empty:
        breakout_sym_col = "user_symbol" if "user_symbol" in breakout_df.columns else "symbol"
        breakout_syms = set(breakout_df[breakout_sym_col].tolist() if breakout_sym_col in breakout_df.columns else [])
        breakout_list = [p for p in top_setups_list if p.get("symbol") in breakout_syms]
    breakout_path.write_text(
        json.dumps(breakout_list, indent=2, default=str),
        encoding="utf-8",
    )
    log.info("Artifact written → %s (%d setups)", breakout_path, len(breakout_list))

    # ── 6. Pullback top setups ────────────────────────────────────────────────
    pullback_list: list[dict] = []
    pullback_path = artifacts_dir / "pullback_top_setups.json"
    if pullback_df is not None and not pullback_df.empty:
        pullback_sym_col = "user_symbol" if "user_symbol" in pullback_df.columns else "symbol"
        pullback_syms = set(pullback_df[pullback_sym_col].tolist() if pullback_sym_col in pullback_df.columns else [])
        pullback_list = [p for p in top_setups_list if p.get("symbol") in pullback_syms]
    pullback_path.write_text(
        json.dumps(pullback_list, indent=2, default=str),
        encoding="utf-8",
    )
    log.info("Artifact written → %s (%d setups)", pullback_path, len(pullback_list))

    # ── 7. Extended leaders ───────────────────────────────────────────────────
    extended_path = artifacts_dir / "extended_leaders.json"
    extended_cols = [
        "user_symbol", "symbol", "state", "bucket", "composite_score",
        "percentile_rank", "dist_to_pivot_atr", "days_in_state",
        "action_label", "is_extended", "close", "pivot",
    ]
    extended_records: list[dict] = []
    if not snapshot_df.empty and "bucket" in snapshot_df.columns:
        ext_df = snapshot_df[snapshot_df["bucket"] == "extended_leader"]
        extended_records = _df_to_simple_records(ext_df, extended_cols)
    extended_path.write_text(
        json.dumps(extended_records, indent=2, default=str),
        encoding="utf-8",
    )
    log.info("Artifact written → %s (%d leaders)", extended_path, len(extended_records))

    # ── 8. Eligibility results ────────────────────────────────────────────────
    elig_path = artifacts_dir / "eligibility_results.json"
    elig_cols = [
        "user_symbol", "symbol", "state", "eligible",
        "rejection_reasons", "eligibility_warnings",
        "composite_score", "failure_risk", "daily_rs_63",
        "close_vs_sma200", "close_vs_sma50", "regime_spy_trend",
    ]
    elig_records: list[dict] = []
    if not snapshot_df.empty and "eligible" in snapshot_df.columns:
        elig_records = _df_to_simple_records(snapshot_df, elig_cols)
    elig_path.write_text(
        json.dumps(elig_records, indent=2, default=str),
        encoding="utf-8",
    )
    log.info("Artifact written → %s (%d symbols)", elig_path, len(elig_records))

    # ── 9. Bucket assignments ─────────────────────────────────────────────────
    bucket_path = artifacts_dir / "bucket_assignments.json"
    bucket_cols = [
        "user_symbol", "symbol", "state", "bucket",
        "action_label", "is_fresh", "is_portfolio", "is_extended",
        "composite_score", "percentile_rank",
    ]
    bucket_records: list[dict] = []
    if not snapshot_df.empty and "bucket" in snapshot_df.columns:
        bucket_records = _df_to_simple_records(snapshot_df, bucket_cols)
    bucket_path.write_text(
        json.dumps(bucket_records, indent=2, default=str),
        encoding="utf-8",
    )
    log.info("Artifact written → %s (%d symbols)", bucket_path, len(bucket_records))

    # ── 10. Dashboard summary ─────────────────────────────────────────────────
    # Enrich bucket counts from snapshot
    bucket_summary: dict[str, int] = {}
    if not snapshot_df.empty and "bucket" in snapshot_df.columns:
        bucket_summary = snapshot_df["bucket"].value_counts().to_dict()

    summary_obj = {
        "as_of": as_of_str,
        "generated_at": generated_at,
        "n_top_setups": len(packets),
        "n_scored": n_scored,
        "n_actionable_now": n_actionable,
        "n_breakout": n_breakout,
        "n_pullback": n_pullback,
        "n_breakout_bucket": len(breakout_list),
        "n_pullback_bucket": len(pullback_list),
        "n_excluded": int(bucket_summary.get("excluded", 0)),
        "regime": _extract_regime(snapshot_df),
        "bucket_counts": {k: int(v) for k, v in bucket_summary.items()},
        "top_symbols": top_symbols,
        "artifact_paths": {
            "top_setups": "artifacts/top_setups.json",
            "breakout_top_setups": "artifacts/breakout_top_setups.json",
            "pullback_top_setups": "artifacts/pullback_top_setups.json",
            "extended_leaders": "artifacts/extended_leaders.json",
            "portfolio_review": "artifacts/portfolio_review.json",
            "eligibility_results": "artifacts/eligibility_results.json",
            "bucket_assignments": "artifacts/bucket_assignments.json",
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
        "breakout": str(breakout_path),
        "pullback": str(pullback_path),
        "extended": str(extended_path),
        "eligibility": str(elig_path),
        "buckets": str(bucket_path),
        "per_symbol": per_symbol_paths,
    }
