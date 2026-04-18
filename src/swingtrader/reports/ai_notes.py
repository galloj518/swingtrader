"""AI-powered analyst note generation with graceful fallback.

If OPENAI_API_KEY is set in the environment:
  - Calls the OpenAI Chat Completions API (gpt-4o-mini)
  - Generates a concise institutional-style trade note per setup
  - Uses only information from the structured packet (no hallucination)

If OPENAI_API_KEY is absent or the call fails:
  - Falls back to deterministic rule-based narrative (no failure)
  - The pipeline never fails due to missing API key

SECURITY:
  - Key is read ONLY from os.environ["OPENAI_API_KEY"]
  - Key is NEVER written to any file, log, HTML, or JSON output
  - Key is never passed to browser-side code
"""
from __future__ import annotations

import math

from swingtrader.utils.logging import get_logger

log = get_logger(__name__)


# ── Prompt helpers ────────────────────────────────────────────────────────────

def _fmt(v, decimals: int = 2) -> str:
    """Format a scalar for inclusion in a prompt. Returns '—' for missing values."""
    if v is None:
        return "—"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, float)):
        try:
            fv = float(v)
            if math.isnan(fv) or math.isinf(fv):
                return "—"
            return f"{fv:.{decimals}f}"
        except (TypeError, ValueError):
            return "—"
    s = str(v)
    return "—" if s in ("nan", "None", "—", "") else s


def _checklist_summary(packet: dict) -> str:
    """Build a compact checklist pass/fail summary line."""
    ctx = packet.get("context") or {}
    checklist = ctx.get("checklist") or packet.get("checklist")
    if not checklist or not isinstance(checklist, list):
        return ""

    passes = []
    fails = []
    for item in checklist:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", item.get("label", "")))
        passed = item.get("passed", item.get("ok", item.get("result")))
        if passed is True or passed == "pass":
            passes.append(name)
        elif passed is False or passed == "fail":
            fails.append(name)

    if not passes and not fails:
        return ""

    lines = [f"CHECKLIST: {len(passes)} pass / {len(fails)} fail"]
    if fails:
        failed_str = ", ".join(fails[:5])
        if len(fails) > 5:
            failed_str += f" (+{len(fails) - 5} more)"
        lines.append(f"FAILED CHECKS: {failed_str}")
    return "\n".join(lines)


def _ma_summary(packet: dict) -> str:
    """Summarise MA context from packet fields."""
    # Prefer a dedicated ma_table list (nested under context); fall back to narrative.
    ctx = packet.get("context") or {}
    ma_table = ctx.get("ma_table") or packet.get("ma_table")
    if ma_table and isinstance(ma_table, list):
        rows = []
        for item in ma_table[:4]:
            if isinstance(item, dict):
                label = item.get("label", item.get("name", ""))
                value = item.get("value", item.get("val", ""))
                if label and value:
                    rows.append(f"{label}: {value}")
        if rows:
            return "MA: " + " | ".join(rows)

    # Fallback: narrative field
    narrative = packet.get("narrative", {})
    if isinstance(narrative, dict):
        ctx = narrative.get("ma_context", "")
        if ctx and ctx not in ("—", ""):
            # Keep to one line to stay within prompt budget
            return f"MA CONTEXT: {ctx[:200]}"

    # Last resort: raw close_vs_sma50
    cvs = packet.get("close_vs_sma50")
    if cvs is not None:
        return f"CLOSE vs SMA50: {_fmt(cvs, 3)}"

    return ""


def _avwap_summary(packet: dict) -> str:
    """Summarise AVWAP context from packet fields."""
    ctx = packet.get("context") or {}
    avwap_table = ctx.get("avwap_table") or packet.get("avwap_table")
    if avwap_table and isinstance(avwap_table, list):
        rows = []
        for item in avwap_table[:3]:
            if isinstance(item, dict):
                label = item.get("label", item.get("name", ""))
                value = item.get("value", item.get("val", ""))
                if label and value:
                    rows.append(f"{label}: {value}")
        if rows:
            return "AVWAP: " + " | ".join(rows)

    narrative = packet.get("narrative", {})
    if isinstance(narrative, dict):
        ctx = narrative.get("avwap_context", "")
        if ctx and ctx not in ("—", ""):
            return f"AVWAP CONTEXT: {ctx[:200]}"

    ytd = packet.get("ytd_dist_atr")
    if ytd is not None:
        return f"YTD AVWAP dist (ATR): {_fmt(ytd, 2)}"

    return ""


# ── Core functions ────────────────────────────────────────────────────────────

def _build_prompt(packet: dict) -> str:
    """Build a structured prompt from the packet for gpt-4o-mini.

    Uses only data already present in the packet — no external lookups.
    """
    symbol = _fmt(packet.get("symbol"))
    state = _fmt(packet.get("state"))
    action = _fmt(packet.get("action_label"))
    close = _fmt(packet.get("close"))
    pivot = _fmt(packet.get("pivot"))
    entry_lo = _fmt(packet.get("entry_lo"))
    entry_hi = _fmt(packet.get("entry_hi"))
    stop = _fmt(packet.get("stop"))
    t1 = _fmt(packet.get("t1"))
    t2 = _fmt(packet.get("t2"))
    rr = _fmt(packet.get("risk_reward_t1"))
    composite = _fmt(packet.get("composite_score"))
    failure = _fmt(packet.get("failure_risk"))

    checklist_block = _checklist_summary(packet)
    ma_block = _ma_summary(packet)
    avwap_block = _avwap_summary(packet)

    extra_lines = "\n".join(
        line for line in [checklist_block, ma_block, avwap_block] if line
    )

    prompt = (
        "You are an institutional swing trade analyst reviewing a technical setup.\n"
        "Write a concise, professional trade note in 6 sections: "
        "Setup, Why It Matters Now, Entry Idea, Risk/Invalidation, Targets, Verdict.\n"
        "Use only the data provided. Do not invent facts. "
        "Be direct and trader-usable.\n"
        "Keep total length under 300 words.\n"
        "\n"
        f"SYMBOL: {symbol} | STATE: {state} | ACTION: {action}\n"
        f"CLOSE: {close} | PIVOT: {pivot} | ENTRY: {entry_lo}-{entry_hi}\n"
        f"STOP: {stop} | T1: {t1} | T2: {t2} | R/R to T1: {rr}\n"
        f"SCORE: {composite} | FAILURE RISK: {failure}\n"
    )

    if extra_lines:
        prompt += extra_lines + "\n"

    return prompt.strip()


def _rule_based_note(packet: dict) -> str:
    """Deterministic fallback analyst note built from existing narrative fields.

    Concatenates the pre-computed narrative sections from packet.py / narrative.py.
    Never fails; returns a generic message if narrative is entirely absent.
    """
    narrative = packet.get("narrative", {})
    sym = packet.get("symbol", "?")

    if not isinstance(narrative, dict) or not narrative:
        return (
            f"[{sym}] No narrative available. "
            "Run the full pipeline to generate trade narrative fields."
        )

    def _section(label: str, key: str) -> str:
        val = narrative.get(key, "")
        if not val or val in ("—", "None", ""):
            return ""
        return f"{label}\n{val}"

    sections = [
        _section("Setup", "setup"),
        _section("Why It Matters Now", "why"),
        _section("Entry Idea", "entry"),
        _section("Risk/Invalidation", "risk"),
        _section("Targets", "targets"),
        _section("Verdict", "verdict"),
    ]

    body = "\n\n".join(s for s in sections if s)
    if not body:
        return f"[{sym}] Narrative fields are empty. Ensure scoring and narrative pipeline have run."

    return f"[{sym} — Rule-based note]\n\n{body}"


def generate_ai_note(packet: dict) -> str:
    """Generate analyst note. Uses OpenAI if key present, else rule-based fallback.

    Parameters
    ----------
    packet : full packet dict for one symbol.

    Returns
    -------
    str — analyst note. Never raises; falls back to rule-based on any failure.
    """
    import os
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        return _rule_based_note(packet)
    try:
        import openai
        client = openai.OpenAI(api_key=key)
        prompt = _build_prompt(packet)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=450,
            temperature=0.25,
        )
        return resp.choices[0].message.content.strip()
    except ImportError:
        return _rule_based_note(packet)
    except Exception as exc:
        log.warning(
            "OpenAI note generation failed for %s: %s",
            packet.get("symbol"),
            exc,
        )
        return _rule_based_note(packet)


def enrich_packets_with_ai(packets: list[dict]) -> list[dict]:
    """Add 'ai_note' field to each packet. Modifies in-place and returns.

    Parameters
    ----------
    packets : list of packet dicts (from dashboard.packet.build_packets()).

    Returns
    -------
    Same list, with ai_note populated on every dict.
    """
    for pkt in packets:
        pkt["ai_note"] = generate_ai_note(pkt)
    return packets
