"""Trader-facing 3-layer daily dashboard renderer.

Produces a self-contained HTML file at:
  docs/reports/daily/{date}/dashboard.html

Layer 1  — Dashboard summary
  Market / regime summary, top actionable setup summary strip,
  portfolio-attention callouts, run timestamp.

Layer 2  — Top setup cards (5-7 symbols)
  Per-symbol: action label, charts (weekly + daily + intraday),
  trade levels (entry / stop / T1 / T2 / T3 / S-R ladder),
  narrative (setup / why / entry / risk / targets / verdict),
  AI-review-ready data panel.

Layer 3  — Full research tables (collapsible)
  Full state tables (ARMED / BASE / TRIGGERED / etc.),
  portfolio detail, skipped symbols.

All HTML is generated inline (no build tools, no template files).
The file is fully self-contained (inline CSS, no external resources)
so it renders correctly when viewed locally or via GitHub Pages.
"""
from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from swingtrader.utils.logging import get_logger

log = get_logger(__name__)

# ── Colour palette (mirrors site theme) ──────────────────────────────────────

_CSS = """
:root {
  --bg: #0d1117; --bg2: #161b22; --bg3: #21262d;
  --fg: #c9d1d9; --fg2: #8b949e; --fg3: #6e7681;
  --blue: #58a6ff; --green: #3fb950; --red: #f78166;
  --amber: #d29922; --purple: #bc8cff; --orange: #fb8f44;
  --border: #30363d;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: system-ui, -apple-system, sans-serif;
       background: var(--bg); color: var(--fg);
       padding: 1rem 1.5rem; font-size: 14px; }
a { color: var(--blue); text-decoration: none; }
a:hover { text-decoration: underline; }

/* ── Typography ────────────────────────────────────────────────────────────── */
h1 { color: var(--blue); font-size: 1.4rem; border-bottom: 1px solid var(--border);
     padding-bottom: .4rem; margin-bottom: .8rem; }
h2 { color: var(--fg); font-size: 1.1rem; margin: 1.6rem 0 .6rem; }
h3 { color: var(--fg2); font-size: .95rem; margin: .8rem 0 .4rem; }

/* ── Regime / summary bar ─────────────────────────────────────────────────── */
.summary-bar { display: flex; gap: 1rem; flex-wrap: wrap;
               background: var(--bg2); border: 1px solid var(--border);
               border-radius: 6px; padding: .6rem 1rem; margin-bottom: 1.2rem; }
.summary-pill { display: flex; flex-direction: column; align-items: center;
                min-width: 80px; }
.pill-label { font-size: .72rem; color: var(--fg3); text-transform: uppercase;
              letter-spacing: .05em; }
.pill-value { font-size: 1rem; font-weight: 600; color: var(--fg); }
.pill-value.up { color: var(--green); }
.pill-value.down { color: var(--red); }
.pill-value.neutral { color: var(--amber); }

/* ── Action label badges ──────────────────────────────────────────────────── */
.badge { display: inline-block; padding: .18rem .55rem; border-radius: 20px;
         font-size: .75rem; font-weight: 600; letter-spacing: .03em;
         white-space: nowrap; }
.badge-now      { background: #1a4f2a; color: var(--green); border: 1px solid var(--green); }
.badge-breakout { background: #0f2d4f; color: var(--blue);  border: 1px solid var(--blue); }
.badge-pullback { background: #3d2f0a; color: var(--amber); border: 1px solid var(--amber); }
.badge-extended { background: #2a2a2a; color: var(--fg3);   border: 1px solid var(--fg3); }
.badge-avoid    { background: #3d1212; color: var(--red);   border: 1px solid var(--red); }

/* ── Setup cards ──────────────────────────────────────────────────────────── */
.setup-cards { display: flex; flex-direction: column; gap: 1.4rem; }
.card { background: var(--bg2); border: 1px solid var(--border);
        border-radius: 8px; overflow: hidden; }
.card-header { display: flex; align-items: center; gap: .8rem;
               padding: .7rem 1rem; border-bottom: 1px solid var(--border);
               background: var(--bg3); flex-wrap: wrap; }
.card-symbol { font-size: 1.2rem; font-weight: 700; color: var(--fg); }
.card-state  { font-size: .85rem; font-weight: 600; }
.card-score  { margin-left: auto; font-size: .85rem; color: var(--fg2); }
.score-hi { color: var(--green); font-weight: 700; }
.score-md { color: var(--amber); font-weight: 700; }
.score-lo { color: var(--red); }

.card-body { display: grid; grid-template-columns: 1fr 1fr; gap: 0;
             padding: 0; }
@media (max-width: 900px) {
  .card-body { grid-template-columns: 1fr; }
}
.card-charts { padding: .8rem; border-right: 1px solid var(--border);
               display: flex; flex-direction: column; gap: .5rem; }
.card-charts img { width: 100%; border-radius: 4px;
                   border: 1px solid var(--border); }
.chart-na { background: var(--bg3); border: 1px solid var(--border);
            border-radius: 4px; padding: .5rem; text-align: center;
            color: var(--fg3); font-size: .78rem; }
.card-detail { padding: .8rem 1rem; display: flex; flex-direction: column; gap: .8rem; }

/* ── Levels table ─────────────────────────────────────────────────────────── */
.levels-grid { display: grid; grid-template-columns: 1fr 1fr; gap: .3rem .8rem; }
.lvl-row { display: flex; justify-content: space-between; align-items: baseline;
           font-size: .82rem; border-bottom: 1px solid var(--bg3); padding: .15rem 0; }
.lvl-label { color: var(--fg2); }
.lvl-value { font-family: monospace; font-size: .82rem; font-weight: 600; }
.lvl-pivot   { color: var(--purple); }
.lvl-entry   { color: var(--green); }
.lvl-stop    { color: var(--red); }
.lvl-target  { color: var(--orange); }
.lvl-rs      { color: var(--fg2); }
.lvl-sp      { color: var(--amber); }

/* ── Narrative ────────────────────────────────────────────────────────────── */
.narrative { font-size: .83rem; line-height: 1.55; }
.narrative dt { color: var(--fg2); font-weight: 600; font-size: .78rem;
                text-transform: uppercase; letter-spacing: .04em;
                margin-top: .5rem; }
.narrative dd { color: var(--fg); margin: .15rem 0 0 0; }
.verdict-box { background: var(--bg3); border-left: 3px solid var(--blue);
               padding: .4rem .7rem; border-radius: 0 4px 4px 0;
               font-size: .83rem; margin-top: .4rem; }

/* ── State colours ────────────────────────────────────────────────────────── */
.s-ARMED     { color: var(--green); font-weight: 700; }
.s-BASE      { color: var(--fg2); }
.s-TRIGGERED { color: var(--red); font-weight: 700; }
.s-ACCEPTED  { color: var(--blue); font-weight: 700; }
.s-CONFIRMED { color: var(--purple); font-weight: 700; }
.s-FAILED    { color: var(--fg3); text-decoration: line-through; }
.s-LATE      { color: var(--amber); }
.s-NONE      { color: var(--fg3); }

/* ── Tables ───────────────────────────────────────────────────────────────── */
table { border-collapse: collapse; width: 100%; font-size: .8rem; margin: .5rem 0; }
th { background: var(--bg3); color: var(--blue); text-align: left;
     padding: .3rem .5rem; border-bottom: 1px solid var(--border); }
td { padding: .25rem .5rem; border-bottom: 1px solid var(--bg3); }
tr:hover td { background: var(--bg2); }

/* ── Collapsible ─────────────────────────────────────────────────────────── */
details { border: 1px solid var(--border); border-radius: 6px;
          margin: .6rem 0; overflow: hidden; }
details > summary { padding: .6rem 1rem; background: var(--bg2);
                    cursor: pointer; font-weight: 600; font-size: .9rem;
                    list-style: none; user-select: none; color: var(--fg2); }
details > summary::before { content: "▶ "; font-size: .7rem; }
details[open] > summary::before { content: "▼ "; }
details > .details-body { padding: .5rem 1rem; }

/* ── Portfolio attention callouts ─────────────────────────────────────────── */
.portfolio-strip { display: flex; gap: .6rem; flex-wrap: wrap; margin-bottom: 1rem; }
.port-chip { background: var(--bg2); border: 1px solid var(--border);
             border-radius: 6px; padding: .4rem .7rem; font-size: .82rem;
             display: flex; flex-direction: column; gap: .1rem; min-width: 110px; }
.port-chip .pc-sym { font-weight: 700; color: var(--fg); }
.port-chip .pc-state { font-size: .75rem; }
.port-chip .pc-action { font-size: .72rem; color: var(--fg2); }

/* ── Bias note ────────────────────────────────────────────────────────────── */
.bias { background: var(--bg2); border-left: 3px solid var(--amber);
        padding: .4rem .8rem; border-radius: 0 4px 4px 0;
        font-size: .78rem; color: var(--fg3); margin-bottom: 1rem; }

footer { margin-top: 2rem; font-size: .75rem; color: var(--fg3);
         border-top: 1px solid var(--border); padding-top: .5rem; }

/* ── Rank bar ────────────────────────────────────────────────────────────── */
.rank-bar { display: inline-block; height: 6px; background: var(--blue);
            border-radius: 2px; vertical-align: middle; margin-left: 3px; }

/* ── Setup classification badges ─────────────────────────────────────────── */
.setup-class { font-size: .75rem; font-weight: 500; padding: .1rem .4rem;
               border-radius: 3px; margin-left: .3rem; }
.sc-active   { color: var(--green); background: #1a4f2a; }
.sc-early    { color: var(--blue);  background: #0f2d4f; }
.sc-near     { color: var(--purple); background: #2b1d4f; }
.sc-pullback { color: var(--amber); background: #3d2f0a; }
.sc-building { color: var(--fg2);  background: var(--bg3); }
.sc-extended { color: var(--fg3);  background: var(--bg3); }
.sc-failed   { color: var(--red);  background: #3d1212; }

/* ── MA table ─────────────────────────────────────────────────────────────── */
.ma-table table { font-size: .75rem; }
.slope-rising  { color: var(--green); font-weight: 600; }
.slope-falling { color: var(--red);   font-weight: 600; }
.slope-flat    { color: var(--fg2); }

/* ── AVWAP table ──────────────────────────────────────────────────────────── */
.avwap-support    { color: var(--green); font-weight: 600; }
.avwap-resistance { color: var(--red);   font-weight: 600; }
.avwap-testing    { color: var(--amber); font-weight: 600; }

/* ── Checklist ────────────────────────────────────────────────────────────── */
.checklist { list-style: none; font-size: .78rem; margin: .3rem 0; }
.checklist li { padding: .16rem 0; display: flex; gap: .4rem; align-items: flex-start;
                border-bottom: 1px solid var(--bg3); flex-wrap: wrap; }
.chk-icon { flex-shrink: 0; width: 1rem; text-align: center; font-weight: 700; }
.chk-pass    { color: var(--green); }
.chk-fail    { color: var(--red); }
.chk-neutral { color: var(--fg3); }
.chk-item    { color: var(--fg2); min-width: 160px; }
.chk-reason  { color: var(--fg3); font-size: .72rem; margin-left: auto; }

/* ── Confluence ───────────────────────────────────────────────────────────── */
.confluence-box { border-radius: 4px; padding: .35rem .6rem; font-size: .78rem;
                  margin: .35rem 0; background: var(--bg3); }
.conf-support    { border-left: 3px solid var(--green); }
.conf-resistance { border-left: 3px solid var(--red); }
.conf-trigger    { border-left: 3px solid var(--blue); }
.conf-scattered  { border-left: 3px solid var(--fg3); }

/* ── Volume block ─────────────────────────────────────────────────────────── */
.vol-block { display: grid; grid-template-columns: 1fr 1fr; gap: .15rem .8rem;
             font-size: .77rem; margin: .3rem 0; }
.vol-row { display: flex; justify-content: space-between;
           border-bottom: 1px solid var(--bg3); padding: .1rem 0; }
.vol-label { color: var(--fg3); }
.vol-value { color: var(--fg); font-family: monospace; }

/* ── AI note ──────────────────────────────────────────────────────────────── */
.ai-note { background: var(--bg3); border-left: 3px solid var(--purple);
           padding: .5rem .8rem; border-radius: 0 4px 4px 0;
           font-size: .8rem; line-height: 1.55; margin-top: .5rem; }
.ai-note-label { font-size: .7rem; color: var(--purple); text-transform: uppercase;
                 letter-spacing: .05em; margin-bottom: .3rem; font-weight: 600; }

/* ── Trade plan ───────────────────────────────────────────────────────────── */
.trade-plan { background: var(--bg3); border: 1px solid var(--border);
              border-radius: 4px; padding: .5rem .7rem; font-size: .8rem;
              line-height: 1.55; margin-top: .5rem; }
.tp-label { font-size: .7rem; color: var(--blue); text-transform: uppercase;
            letter-spacing: .05em; margin-bottom: .3rem; font-weight: 600; }

/* ── Portfolio guidance chip ──────────────────────────────────────────────── */
.port-chip .pc-guidance { font-size: .7rem; color: var(--amber);
                          margin-top: .1rem; font-style: italic; }

/* ── Regime environment pill ──────────────────────────────────────────────── */
.env-favorable { color: var(--green); }
.env-selective  { color: var(--amber); }
.env-risk-off   { color: var(--red); }
.env-mixed      { color: var(--fg2); }
"""

# ── Helpers ───────────────────────────────────────────────────────────────────

def _e(v: Any, decimals: int = 2) -> str:
    """Format float; return '—' if NaN/None."""
    if v is None:
        return "—"
    try:
        fv = float(v)
        return "—" if not math.isfinite(fv) else f"{fv:.{decimals}f}"
    except (TypeError, ValueError):
        s = str(v)
        return "—" if s in ("nan", "None", "") else s


def _badge(label: str) -> str:
    cls_map = {
        "Actionable now": "badge-now",
        "Actionable on breakout": "badge-breakout",
        "Actionable on pullback": "badge-pullback",
        "Extended, wait": "badge-extended",
        "Avoid / low quality": "badge-avoid",
    }
    cls = cls_map.get(label, "badge-extended")
    return f'<span class="badge {cls}">{label}</span>'


def _state_span(state: str) -> str:
    return f'<span class="s-{state}">{state}</span>'


def _score_cls(v: Any) -> str:
    try:
        fv = float(v)
        if not math.isfinite(fv):
            return ""
        return "score-hi" if fv >= 0.60 else ("score-md" if fv >= 0.30 else "score-lo")
    except (TypeError, ValueError):
        return ""


def _regime_pill(label: str, value: str, direction: str = "") -> str:
    cls = f" {direction}" if direction else ""
    return (
        f'<div class="summary-pill">'
        f'<span class="pill-label">{label}</span>'
        f'<span class="pill-value{cls}">{value}</span>'
        f'</div>'
    )


def _lvl_row(label: str, value: str, css_cls: str) -> str:
    return (
        f'<div class="lvl-row">'
        f'<span class="lvl-label">{label}</span>'
        f'<span class="lvl-value {css_cls}">{value}</span>'
        f'</div>'
    )


def _chart_img(path: str | None, alt: str) -> str:
    if path:
        return f'<img src="{path}" alt="{alt}" loading="lazy">'
    return f'<div class="chart-na">{alt}<br><small>Data unavailable</small></div>'


def _setup_class_badge(classification: str) -> str:
    """Render a coloured setup classification badge."""
    cls_map = {
        "Active breakout":        "sc-active",
        "Early breakout":         "sc-early",
        "Near breakout / poised": "sc-near",
        "Confirmed uptrend":      "sc-near",
        "Approaching pivot":      "sc-building",
        "Building base":          "sc-building",
        "Pullback entry":         "sc-pullback",
        "Extended / chase risk":  "sc-extended",
        "Mature trend":           "sc-extended",
        "Failed / avoid":         "sc-failed",
        "Watching":               "sc-building",
    }
    cls = cls_map.get(classification, "sc-building")
    return f'<span class="setup-class {cls}">{classification}</span>'


def _ma_table_html(ma_table: list[dict]) -> str:
    """Render the MA state table as HTML."""
    if not ma_table:
        return ""
    rows = ""
    for ma in ma_table:
        slope = ma.get("slope", "")
        slope_cls = f"slope-{slope}"
        pct = ma.get("pct_dist", math.nan)
        try:
            pct_str = f"{float(pct):+.2f}%" if math.isfinite(float(pct)) else "—"
        except (TypeError, ValueError):
            pct_str = "—"
        rows += (
            f"<tr>"
            f"<td><strong>{ma.get('name', '')}</strong></td>"
            f"<td style='font-family:monospace'>{_e(ma.get('value'), 2)}</td>"
            f"<td>{pct_str}</td>"
            f"<td class='{slope_cls}'>{slope}</td>"
            f"<td style='font-size:.7rem;color:var(--fg3)'>{ma.get('bias', '')}</td>"
            f"</tr>"
        )
    return (
        f'<div class="ma-table">'
        f'<table><tr>'
        f'<th>MA</th><th>Value</th><th>Dist%</th><th>Slope</th><th>Bias note</th>'
        f'</tr>{rows}</table></div>'
    )


def _avwap_table_html(avwap_table: list[dict]) -> str:
    """Render the AVWAP map table as HTML."""
    if not avwap_table:
        return ""
    rows = ""
    for av in avwap_table:
        role = av.get("role", "")
        role_cls = f"avwap-{role}"
        pct = av.get("pct_dist", math.nan)
        try:
            pct_str = f"{float(pct):+.2f}%" if math.isfinite(float(pct)) else "—"
        except (TypeError, ValueError):
            pct_str = "—"
        dist_atr = av.get("dist_atr", math.nan)
        try:
            dist_str = f"{float(dist_atr):+.2f}" if math.isfinite(float(dist_atr)) else "—"
        except (TypeError, ValueError):
            dist_str = "—"
        reclaim = av.get("reclaim")
        reclaim_str = "✓" if reclaim is True else ("✗" if reclaim is False else "—")
        rows += (
            f"<tr>"
            f"<td><strong>{av.get('anchor', '')}</strong></td>"
            f"<td style='font-family:monospace'>{_e(av.get('avwap'), 2)}</td>"
            f"<td>{pct_str}</td>"
            f"<td style='font-family:monospace'>{dist_str} ATR</td>"
            f"<td class='{role_cls}'>{av.get('status', '')}</td>"
            f"<td style='color:var(--fg2)'>{reclaim_str}</td>"
            f"</tr>"
        )
    return (
        f'<table><tr>'
        f'<th>Anchor</th><th>AVWAP</th><th>Dist%</th><th>ATR Dist</th>'
        f'<th>Status</th><th>Reclaim</th>'
        f'</tr>{rows}</table>'
    )


def _checklist_html(checklist: list[dict]) -> str:
    """Render the 15-item checklist as HTML."""
    if not checklist:
        return ""
    pass_count = sum(1 for c in checklist if c.get("result") == "pass")
    fail_count = sum(1 for c in checklist if c.get("result") == "fail")
    total = len(checklist)
    items = ""
    for c in checklist:
        result = c.get("result", "neutral")
        if result == "pass":
            icon, icon_cls = "✓", "chk-pass"
        elif result == "fail":
            icon, icon_cls = "✗", "chk-fail"
        else:
            icon, icon_cls = "-", "chk-neutral"
        items += (
            f'<li>'
            f'<span class="chk-icon {icon_cls}">{icon}</span>'
            f'<span class="chk-item">{c.get("item", "")}</span>'
            f'<span class="chk-reason">{c.get("reason", "")}</span>'
            f'</li>'
        )
    summary_line = (
        f'<div style="font-size:.74rem;color:var(--fg2);margin-bottom:.2rem">'
        f'<span class="chk-pass">✓ {pass_count}</span> / {total} pass &nbsp;'
        f'<span class="chk-fail">✗ {fail_count}</span> fail'
        f'</div>'
    )
    return f'{summary_line}<ul class="checklist">{items}</ul>'


def _confluence_html(confluence: dict) -> str:
    """Render the confluence block as HTML."""
    if not confluence:
        return ""
    n = confluence.get("nearby_count", 0)
    role = confluence.get("cluster_role", "scattered")
    nearby = confluence.get("nearby_levels", [])
    role_cls = {
        "support cluster":    "conf-support",
        "resistance cluster": "conf-resistance",
        "trigger zone":       "conf-trigger",
        "scattered":          "conf-scattered",
    }.get(role, "conf-scattered")
    if n == 0:
        detail = "No key levels within 0.5 ATR"
    else:
        level_strs = [
            f"{lv.get('name', '?')} @ {_e(lv.get('value'), 2)}"
            for lv in nearby[:5]
        ]
        detail = ", ".join(level_strs)
    return (
        f'<div class="confluence-box {role_cls}">'
        f'<strong>{n} nearby level{"s" if n != 1 else ""} — {role}</strong>'
        f'<br><small style="color:var(--fg3)">{detail}</small>'
        f'</div>'
    )


def _volume_block_html(volume_block: dict) -> str:
    """Render the volume/efficiency block as HTML."""
    if not volume_block:
        return ""

    def _vr(label: str, value: Any) -> str:
        v_str = str(value) if not isinstance(value, float) else _e(value, 2)
        return (
            f'<div class="vol-row">'
            f'<span class="vol-label">{label}</span>'
            f'<span class="vol-value">{v_str}</span>'
            f'</div>'
        )

    comp_label = volume_block.get("compression_label", "—")
    rel_vol    = volume_block.get("relative_vol_label", "—")
    atr_pct    = volume_block.get("atr_compression_pct", math.nan)
    vd_pct     = volume_block.get("volume_dryup_pct", math.nan)
    thrust     = volume_block.get("breakout_thrust_atr", math.nan)
    vc520      = volume_block.get("vol_contraction_5_20", math.nan)

    try:
        atr_pct_str = f"{float(atr_pct):.0f}th pct" if math.isfinite(float(atr_pct)) else "—"
    except (TypeError, ValueError):
        atr_pct_str = "—"
    try:
        vd_str = f"{float(vd_pct):.1f}%" if math.isfinite(float(vd_pct)) else "—"
    except (TypeError, ValueError):
        vd_str = "—"

    return (
        f'<div class="vol-block">'
        f'<div>'
        f'{_vr("Compression", comp_label)}'
        f'{_vr("Rel volume", rel_vol)}'
        f'{_vr("Vol dry-up", vd_str)}'
        f'</div>'
        f'<div>'
        f'{_vr("ATR pct", atr_pct_str)}'
        f'{_vr("Thrust ATR", _e(thrust, 3))}'
        f'{_vr("VC 5/20", _e(vc520, 3))}'
        f'</div>'
        f'</div>'
    )


def _ai_note_html(ai_note: str | None) -> str:
    """Render the AI analysis note block."""
    if not ai_note:
        return ""
    return (
        f'<div class="ai-note">'
        f'<div class="ai-note-label">AI Analysis</div>'
        f'{ai_note}'
        f'</div>'
    )


def _trade_plan_html(trade_plan: str | None) -> str:
    """Render the trade plan block."""
    if not trade_plan:
        return ""
    return (
        f'<div class="trade-plan">'
        f'<div class="tp-label">Trade Plan</div>'
        f'{trade_plan}'
        f'</div>'
    )


# ── Table rendering ───────────────────────────────────────────────────────────

_TABLE_COLS = [
    "user_symbol", "state", "close", "pivot", "atr14",
    "dist_to_pivot_atr", "base_length", "days_in_state",
    "atr_compression_pct", "volume_dryup", "daily_rs_63",
    "composite_score", "percentile_rank",
]

_PORTFOLIO_COLS = [
    "user_symbol", "state", "close", "pivot", "dist_to_pivot_atr",
    "days_in_state", "composite_score", "action_label",
]


def _html_table(df: pd.DataFrame, cols: list[str]) -> str:
    avail = [c for c in cols if c in df.columns]
    if not avail or df.empty:
        return "<p><em>No data.</em></p>"
    header = "".join(f"<th>{c}</th>" for c in avail)
    rows = []
    for _, row in df.iterrows():
        cells = []
        for c in avail:
            v = row[c]
            if c == "state":
                cells.append(f"<td>{_state_span(str(v))}</td>")
            elif c == "composite_score":
                cls = _score_cls(v)
                cells.append(f'<td class="{cls}">{_e(v)}</td>')
            elif c == "action_label":
                cells.append(f"<td>{_badge(str(v))}</td>")
            elif c == "percentile_rank" and isinstance(v, float) and math.isfinite(v):
                w = int(v * 0.45)
                cells.append(
                    f'<td>{_e(v, 0)}'
                    f'<span class="rank-bar" style="width:{w}px"></span></td>'
                )
            else:
                cells.append(f"<td>{_e(v)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><tr>{header}</tr>{''.join(rows)}</table>"


# ── Setup card rendering ──────────────────────────────────────────────────────

def _render_card(packet: dict) -> str:
    sym = packet.get("symbol", "?")
    state = packet.get("state", "NONE")
    action = packet.get("action_label", "—")
    setup_cls_label = packet.get("setup_classification", "")
    score_raw = packet.get("composite_score", "—")
    failure_raw = packet.get("failure_risk", "—")
    rank_raw = packet.get("percentile_rank", "—")
    score_cls = _score_cls(score_raw)
    freshness_label = packet.get("freshness_label", "")

    narrative = packet.get("narrative", {})
    context = packet.get("context", {}) or {}
    ai_note = packet.get("ai_note")
    is_portfolio = packet.get("is_portfolio", False)
    portfolio_tag = ' <small style="color:var(--purple)">[Portfolio]</small>' if is_portfolio else ""

    # Setup classification badge
    sc_badge = _setup_class_badge(setup_cls_label) if setup_cls_label else ""

    # Freshness tag
    fresh_tag = ""
    if freshness_label and freshness_label not in ("—", ""):
        fresh_color = "var(--green)" if packet.get("is_fresh") else "var(--fg3)"
        fresh_tag = f'<small style="color:{fresh_color};margin-left:.3rem">[{freshness_label}]</small>'

    # Header
    header = (
        f'<div class="card-header">'
        f'{_badge(action)}'
        f'<span class="card-symbol">{sym}{portfolio_tag}</span>'
        f'{_state_span(state)}'
        f'{sc_badge}'
        f'{fresh_tag}'
        f'<span class="card-score">'
        f'Score: <span class="{score_cls}">{score_raw}</span>'
        f' &nbsp;|&nbsp; Fail: {failure_raw}'
        f' &nbsp;|&nbsp; Rank: {rank_raw}'
        f'</span>'
        f'</div>'
    )

    # Charts
    charts_html = (
        f'<div class="card-charts">'
        f'<div style="font-size:.75rem;color:var(--fg3);margin-bottom:.3rem">Weekly</div>'
        f'{_chart_img(packet.get("chart_weekly"), f"{sym} Weekly")}'
        f'<div style="font-size:.75rem;color:var(--fg3);margin:.4rem 0 .3rem">Daily</div>'
        f'{_chart_img(packet.get("chart_daily"), f"{sym} Daily")}'
        f'<div style="font-size:.75rem;color:var(--fg3);margin:.4rem 0 .3rem">Intraday (5m)</div>'
        f'{_chart_img(packet.get("chart_intraday"), f"{sym} Intraday")}'
        f'</div>'
    )

    # Trade levels
    lvl_left = "".join([
        _lvl_row("Pivot",    packet.get("pivot", "—"),    "lvl-pivot"),
        _lvl_row("Entry lo", packet.get("entry_lo", "—"), "lvl-entry"),
        _lvl_row("Entry hi", packet.get("entry_hi", "—"), "lvl-entry"),
        _lvl_row("Stop",     packet.get("stop", "—"),     "lvl-stop"),
        _lvl_row("T1",       packet.get("t1", "—"),       "lvl-target"),
        _lvl_row("T2",       packet.get("t2", "—"),       "lvl-target"),
        _lvl_row("T3",       packet.get("t3", "—"),       "lvl-target"),
        _lvl_row("R/R T1",   packet.get("risk_reward_t1", "—"), "lvl-rs"),
    ])
    lvl_right = "".join([
        _lvl_row("R1", packet.get("r1", "—"), "lvl-rs"),
        _lvl_row("R2", packet.get("r2", "—"), "lvl-rs"),
        _lvl_row("R3", packet.get("r3", "—"), "lvl-rs"),
        _lvl_row("S1", packet.get("s1", "—"), "lvl-sp"),
        _lvl_row("S2", packet.get("s2", "—"), "lvl-sp"),
        _lvl_row("S3", packet.get("s3", "—"), "lvl-sp"),
        _lvl_row("ATR14", packet.get("atr14", "—"), "lvl-rs"),
        _lvl_row("Base len", str(packet.get("base_length", "—")), "lvl-rs"),
    ])
    levels_section = (
        f'<div>'
        f'<h3>Trade Levels</h3>'
        f'<div class="levels-grid">'
        f'<div>{lvl_left}</div>'
        f'<div>{lvl_right}</div>'
        f'</div>'
        f'</div>'
    )

    # Confluence (brief, shown inline before narrative)
    confluence_html = _confluence_html(context.get("confluence", {}))

    # Narrative
    n = narrative
    ma_ctx = n.get("ma_context", "")
    avwap_ctx = n.get("avwap_context", "")
    context_block = ""
    if ma_ctx or avwap_ctx:
        context_block = (
            f'<dt>MA &amp; AVWAP context</dt>'
            f'<dd>{ma_ctx}'
            f'{(" &nbsp;|&nbsp; " + avwap_ctx) if avwap_ctx else ""}</dd>'
        )

    narrative_section = (
        f'<div>'
        f'<h3>Narrative</h3>'
        f'<dl class="narrative">'
        f'<dt>Setup</dt><dd>{n.get("setup", "—")}</dd>'
        f'<dt>Why now</dt><dd>{n.get("why", "—")}</dd>'
        f'<dt>Entry</dt><dd>{n.get("entry", "—")}</dd>'
        f'<dt>Risk / invalidation</dt><dd>{n.get("risk", "—")}</dd>'
        f'<dt>Targets</dt><dd>{n.get("targets", "—")}</dd>'
        f'{context_block}'
        f'</dl>'
        f'<div class="verdict-box">{n.get("verdict", "—")}</div>'
        f'</div>'
    )

    # Trade plan
    trade_plan_section = _trade_plan_html(n.get("trade_plan"))

    # AI analysis note
    ai_note_section = _ai_note_html(ai_note)

    # Context deep-dives (collapsible)
    ma_html = _ma_table_html(context.get("ma_table", []))
    avwap_html = _avwap_table_html(context.get("avwap_table", []))
    checklist_html_str = _checklist_html(context.get("checklist", []))
    vol_html = _volume_block_html(context.get("volume_block", {}))

    context_details = ""
    if ma_html:
        context_details += (
            f'<details style="margin-top:.4rem">'
            f'<summary style="font-size:.78rem">MA State Table</summary>'
            f'<div class="details-body">{ma_html}</div>'
            f'</details>'
        )
    if avwap_html:
        context_details += (
            f'<details style="margin-top:.4rem">'
            f'<summary style="font-size:.78rem">AVWAP Map</summary>'
            f'<div class="details-body">{avwap_html}</div>'
            f'</details>'
        )
    if vol_html:
        context_details += (
            f'<details style="margin-top:.4rem">'
            f'<summary style="font-size:.78rem">Volume &amp; Efficiency</summary>'
            f'<div class="details-body">{vol_html}</div>'
            f'</details>'
        )
    if checklist_html_str:
        context_details += (
            f'<details style="margin-top:.4rem">'
            f'<summary style="font-size:.78rem">Trade Checklist</summary>'
            f'<div class="details-body">{checklist_html_str}</div>'
            f'</details>'
        )

    # Raw packet JSON (collapsible debug)
    ai_data = {k: v for k, v in packet.items() if k not in ("narrative", "context")}
    ai_json = json.dumps(ai_data, indent=2, default=str)
    ai_panel = (
        f'<details style="margin-top:.5rem">'
        f'<summary style="font-size:.73rem;color:var(--fg3)">Raw packet (JSON)</summary>'
        f'<div class="details-body"><pre style="font-size:.7rem;overflow:auto;max-height:180px">'
        f'{ai_json}</pre></div>'
        f'</details>'
    )

    detail = (
        f'<div class="card-detail">'
        f'{levels_section}'
        f'{confluence_html}'
        f'{narrative_section}'
        f'{trade_plan_section}'
        f'{ai_note_section}'
        f'{context_details}'
        f'{ai_panel}'
        f'</div>'
    )

    return (
        f'<div class="card">'
        f'{header}'
        f'<div class="card-body">{charts_html}{detail}</div>'
        f'</div>'
    )


# ── Layer 1: regime summary ───────────────────────────────────────────────────

def _regime_html(snapshot_df: pd.DataFrame) -> str:
    """Build the regime summary bar from available snapshot features."""
    spy_trend = math.nan
    spy_above_200 = math.nan
    vix_level = math.nan

    if not snapshot_df.empty:
        for col_name, dest in [
            ("regime_spy_trend", "spy_trend"),
            ("regime_spy_above_200sma", "spy_above_200"),
            ("regime_vix_level", "vix_level"),
        ]:
            if col_name in snapshot_df.columns:
                col = snapshot_df[col_name].dropna()
                if not col.empty:
                    val = float(col.iloc[0])
                    if dest == "spy_trend":
                        spy_trend = val
                    elif dest == "spy_above_200":
                        spy_above_200 = val
                    elif dest == "vix_level":
                        vix_level = val

    # SPY trend pill
    if math.isfinite(spy_trend):
        if spy_trend > 0:
            trend_val, trend_dir = "Uptrend", "up"
        elif spy_trend < 0:
            trend_val, trend_dir = "Downtrend", "down"
        else:
            trend_val, trend_dir = "Neutral", "neutral"
    else:
        trend_val, trend_dir = "—", ""

    # SPY vs 200SMA pill
    if math.isfinite(spy_above_200):
        above_200_val = "Above 200SMA" if spy_above_200 >= 0.5 else "Below 200SMA"
        above_200_dir = "up" if spy_above_200 >= 0.5 else "down"
    else:
        above_200_val, above_200_dir = "—", ""

    # VIX pill
    if math.isfinite(vix_level):
        if vix_level < 15:
            vix_str, vix_dir = f"Complacent ({vix_level:.0f})", "up"
        elif vix_level < 20:
            vix_str, vix_dir = f"Low ({vix_level:.0f})", "up"
        elif vix_level < 30:
            vix_str, vix_dir = f"Elevated ({vix_level:.0f})", "neutral"
        else:
            vix_str, vix_dir = f"High ({vix_level:.0f})", "down"
    else:
        vix_str, vix_dir = "—", ""

    # Breakout environment classification
    if math.isfinite(spy_trend) and math.isfinite(spy_above_200):
        if spy_trend > 0 and spy_above_200 >= 0.5:
            env_label = "Favors breakouts"
            env_cls = "env-favorable"
        elif spy_trend <= 0 and spy_above_200 < 0.5:
            if math.isfinite(vix_level) and vix_level >= 25:
                env_label = "Risk-off"
                env_cls = "env-risk-off"
            else:
                env_label = "Selective"
                env_cls = "env-selective"
        else:
            env_label = "Mixed"
            env_cls = "env-mixed"
    else:
        env_label, env_cls = "—", ""

    env_pill = (
        f'<div class="summary-pill">'
        f'<span class="pill-label">Environment</span>'
        f'<span class="pill-value {env_cls}">{env_label}</span>'
        f'</div>'
    )

    # Universe counts
    actionable_n = armed_n = triggered_n = 0
    if not snapshot_df.empty:
        if "action_label" in snapshot_df.columns:
            actionable_n = int((snapshot_df["action_label"].isin(
                ["Actionable now", "Actionable on breakout", "Actionable on pullback"]
            )).sum())
        if "state" in snapshot_df.columns:
            armed_n = int((snapshot_df["state"] == "ARMED").sum())
            triggered_n = int(
                snapshot_df["state"].isin(["TRIGGERED", "ACCEPTED"]).sum()
            )

    pills = "".join([
        _regime_pill("SPY Trend", trend_val, trend_dir),
        _regime_pill("SPY vs 200SMA", above_200_val, above_200_dir),
        _regime_pill("VIX", vix_str, vix_dir),
        env_pill,
        _regime_pill("Actionable", str(actionable_n)),
        _regime_pill("ARMED", str(armed_n)),
        _regime_pill("Triggered/Acc", str(triggered_n)),
    ])
    return f'<div class="summary-bar">{pills}</div>'


# ── Layer 1: portfolio strip ─────────────────────────────────────────────────

def _portfolio_strip_html(portfolio_df: pd.DataFrame) -> str:
    if portfolio_df.empty:
        return ""

    chips = []
    for _, row in portfolio_df.iterrows():
        sym = str(row.get("user_symbol", row.get("symbol", "?")))
        state = str(row.get("state", "NONE"))
        action = str(row.get("action_label", "—")) if "action_label" in portfolio_df.columns else "—"
        guidance = str(row.get("portfolio_guidance", "")) if "portfolio_guidance" in portfolio_df.columns else ""
        guidance_html = (
            f'<span class="pc-guidance">{guidance}</span>'
            if guidance and guidance not in ("—", "")
            else ""
        )
        chips.append(
            f'<div class="port-chip">'
            f'<span class="pc-sym">{sym}</span>'
            f'<span class="pc-state s-{state}">{state}</span>'
            f'<span class="pc-action">{action}</span>'
            f'{guidance_html}'
            f'</div>'
        )

    return (
        f'<h2>Portfolio Holdings</h2>'
        f'<div class="portfolio-strip">{"".join(chips)}</div>'
    )


# ── Public API ────────────────────────────────────────────────────────────────

def render_dashboard(
    snapshot_df: pd.DataFrame,
    packets: list[dict],
    as_of: pd.Timestamp,
    *,
    oos_metrics: dict | None = None,
) -> str:
    """Render the complete 3-layer dashboard as an HTML string.

    Parameters
    ----------
    snapshot_df : full scored + freshness + action-labelled snapshot.
    packets     : list of dicts from packet.build_packets() for the top setups.
    as_of       : report date.
    oos_metrics : optional OOS calibration metrics for the footer.

    Returns
    -------
    str — complete self-contained HTML.
    """
    generated_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    date_str = str(as_of.date())

    # Partition snapshot
    portfolio_df = pd.DataFrame()
    full_df = snapshot_df.copy() if not snapshot_df.empty else pd.DataFrame()

    if not full_df.empty and "is_portfolio" in full_df.columns:
        portfolio_df = full_df[full_df["is_portfolio"].astype(bool)].copy()

    # Layer 1 — Regime + portfolio strip
    regime_html = _regime_html(full_df)
    portfolio_html = _portfolio_strip_html(portfolio_df)

    # Layer 2 — Top setup cards
    if packets:
        cards_html = "\n".join(_render_card(p) for p in packets)
        cards_section = f'<h2>Top Actionable Setups</h2><div class="setup-cards">{cards_html}</div>'
    else:
        cards_section = (
            '<h2>Top Actionable Setups</h2>'
            '<p style="color:var(--fg3)"><em>'
            'No actionable setups today — either models are not yet fitted, '
            'no symbols are in BASE/ARMED/TRIGGERED/ACCEPTED state, '
            'or all candidates are extended / low-score.'
            '</em></p>'
        )

    # Layer 3 — Full tables (collapsible)
    scored_states = ["ARMED", "TRIGGERED", "ACCEPTED", "BASE"]
    unscored_states = ["CONFIRMED", "LATE", "FAILED", "NONE", "EXHAUSTED"]

    state_details = []
    for state in scored_states + unscored_states:
        if full_df.empty or "state" not in full_df.columns:
            break
        subset = full_df[full_df["state"] == state]
        if subset.empty:
            continue
        n = len(subset)
        tbl = _html_table(subset, _TABLE_COLS)
        state_details.append(
            f'<details><summary class="s-{state}">{state} — {n} symbol{"s" if n != 1 else ""}</summary>'
            f'<div class="details-body">{tbl}</div></details>'
        )

    # Skipped symbols
    skipped_section = ""
    if not full_df.empty and "state" in full_df.columns:
        skipped = full_df[full_df["state"] == "SKIPPED"]
        if not skipped.empty:
            skipped_rows = "".join(
                f'<tr><td>{r.get("user_symbol","—")}</td>'
                f'<td>{r.get("skip_reason","—")}</td>'
                f'<td>{"Y" if r.get("is_non_equity") else "N"}</td></tr>'
                for r in skipped.to_dict("records")
            )
            skipped_section = (
                f'<details><summary>Skipped symbols — {len(skipped)}</summary>'
                f'<div class="details-body">'
                f'<table><tr><th>Symbol</th><th>Reason</th><th>Non-equity</th></tr>'
                f'{skipped_rows}</table></div></details>'
            )

    # Portfolio detail table
    portfolio_detail = ""
    if not portfolio_df.empty:
        tbl = _html_table(portfolio_df, _PORTFOLIO_COLS)
        portfolio_detail = (
            f'<details><summary>Portfolio detail</summary>'
            f'<div class="details-body">{tbl}</div></details>'
        )

    # OOS metrics footer note
    if oos_metrics:
        ss = oos_metrics.get("setup_score", {})
        fr = oos_metrics.get("failure_risk", {})
        brier_ss = ss.get("brier_score_mean", math.nan)
        brier_fr = fr.get("brier_score_mean", math.nan)
        model_note = (
            f"Models fitted. Setup OOS Brier: {_e(brier_ss, 4)}. "
            f"Failure OOS Brier: {_e(brier_fr, 4)}."
        )
    else:
        model_note = "Models not yet fitted — composite scores are NaN."

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Swingtrader — {date_str}</title>
<style>{_CSS}</style>
</head>
<body>

<h1>Daily Dashboard &mdash; {date_str}</h1>

<div class="bias">
  <strong>Bias caveat:</strong> Universe membership uses monthly snapshots.
  Survivorship bias is present. Analysis-only — not investment advice.
  {model_note}
</div>

{regime_html}

{portfolio_html}

{cards_section}

<h2>Full Research Tables</h2>
<p style="font-size:.82rem;color:var(--fg3);margin-bottom:.5rem">
  Expand a state to see all symbols. Sorted by composite score descending.
</p>
{''.join(state_details)}
{portfolio_detail}
{skipped_section}

<footer>
  Generated {generated_at} &nbsp;|&nbsp;
  <a href="snapshot.md">Markdown version</a> &nbsp;|&nbsp;
  swingtrader v0.1
</footer>

</body>
</html>"""

    return html


def write_dashboard(
    snapshot_df: pd.DataFrame,
    packets: list[dict],
    as_of: pd.Timestamp,
    output_dir: Path,
    *,
    oos_metrics: dict | None = None,
) -> Path:
    """Write dashboard.html to output_dir and return the path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    html = render_dashboard(snapshot_df, packets, as_of, oos_metrics=oos_metrics)
    path = output_dir / "dashboard.html"
    path.write_text(html, encoding="utf-8")
    log.info("Dashboard written → %s", path)
    return path
