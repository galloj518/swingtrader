"""Trader-facing 3-layer daily dashboard renderer.

Produces a self-contained HTML file at:
  docs/reports/daily/{date}/dashboard.html

Layer 1  — Dashboard summary
  Market / regime summary, top actionable setup summary strip,
  portfolio-attention callouts, run timestamp.

Layer 2  — Top setup cards (5-7 symbols)
  Per-symbol: action label, charts (weekly + daily, with intraday policy note
  when intraday confirmation is not part of qualification),
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

/* ── Score drivers ────────────────────────────────────────────────────────── */
.score-drivers { background: var(--bg3); border-radius: 4px; padding: .4rem .6rem;
                 font-size: .77rem; margin: .35rem 0; }
.driver-row { display: flex; align-items: baseline; gap: .4rem; padding: .1rem 0;
              border-bottom: 1px solid var(--bg2); }
.driver-bull { color: var(--green); font-weight: 700; flex-shrink: 0; }
.driver-bear { color: var(--red);   font-weight: 700; flex-shrink: 0; }
.driver-text { color: var(--fg2); }
.driver-why  { color: var(--fg3); font-size: .72rem; margin-bottom: .25rem;
               border-bottom: 1px solid var(--border); padding-bottom: .2rem; }

/* ── Export links ─────────────────────────────────────────────────────────── */
.export-links { font-size: .72rem; color: var(--fg3); margin-top: .3rem; }

/* ── Portfolio guidance icons ─────────────────────────────────────────────── */
.pg-hold   { color: var(--green); }
.pg-trim   { color: var(--amber); }
.pg-defend { color: var(--orange); }
.pg-exit   { color: var(--red); }
.pg-info   { color: var(--fg3); }

/* ── MA direction brief (visible strip on card, not collapsible) ──────────── */
.ma-brief { display: flex; gap: .5rem; flex-wrap: wrap; align-items: center;
            font-size: .77rem; margin: .3rem 0; padding: .25rem .4rem;
            background: var(--bg3); border-radius: 4px; }
.ma-brief-label { color: var(--fg3); font-size: .72rem; margin-right: .2rem; }
.ma-pill { display: inline-flex; align-items: center; gap: .2rem;
           padding: .1rem .35rem; border-radius: 3px; font-weight: 600;
           font-size: .74rem; white-space: nowrap; }
.ma-rising  { color: var(--green); background: #1a4f2a; }
.ma-falling { color: var(--red);   background: #3d1212; }
.ma-flat    { color: var(--fg3);   background: var(--bg2); }
.ma-bias-note { font-size: .72rem; color: var(--fg3); margin-top: .15rem;
                padding-left: .4rem; border-left: 2px solid var(--border); }

/* ── Intraday unavailable inline note ────────────────────────────────────── */
.chart-na-inline { font-size: .74rem; color: var(--fg3); padding: .3rem .4rem;
                   border-left: 2px solid var(--bg3); margin-top: .3rem;
                   font-style: italic; }

/* ── Universe summary bar ─────────────────────────────────────────────────── */
.universe-bar { background: var(--bg2); border: 1px solid var(--border);
                border-radius: 6px; padding: .5rem .9rem; margin-bottom: 1.2rem;
                font-size: .78rem; color: var(--fg3); }
.universe-bar strong { color: var(--fg); }

/* ── Bucket section headings ─────────────────────────────────────────────── */
.bucket-section { margin-bottom: 2rem; }
.bucket-header  { display: flex; align-items: center; gap: .7rem; flex-wrap: wrap;
                  background: var(--bg2); border: 1px solid var(--border);
                  border-radius: 6px; padding: .55rem 1rem; margin-bottom: .8rem; }
.bh-title  { font-size: 1rem; font-weight: 700; color: var(--fg); }
.bh-count  { font-size: .8rem; color: var(--fg2); }
.bh-desc   { font-size: .75rem; color: var(--fg3); margin-left: auto; }
.bh-breakout { border-left: 3px solid var(--blue); }
.bh-pullback { border-left: 3px solid var(--amber); }
.bh-extended { border-left: 3px solid var(--fg3); }
.bh-reversal { border-left: 3px solid var(--red); }

/* ── "No setups" notice ──────────────────────────────────────────────────── */
.no-setups-note { background: var(--bg3); border: 1px solid var(--border);
                  border-radius: 4px; padding: .5rem .9rem;
                  font-size: .82rem; color: var(--fg3); }

/* ── Setup quality warning ────────────────────────────────────────────────── */
.top5-warn   { background: #3d2f0a; border: 1px solid var(--amber);
               border-radius: 4px; padding: .4rem .8rem; margin: .4rem 0;
               font-size: .82rem; color: var(--amber); }

/* ── Bucket tag on card header ────────────────────────────────────────────── */
.bucket-tag  { font-size: .72rem; padding: .1rem .35rem; border-radius: 3px;
               font-weight: 600; margin-left: .3rem; }
.bt-breakout { color: var(--blue);  background: #0f2d4f; }
.bt-pullback { color: var(--amber); background: #3d2f0a; }
.rank-num    { font-size: .85rem; color: var(--fg3); font-weight: 700;
               min-width: 1.4rem; }

/* ── Compact monitoring list (Extended Leaders / Speculative Watch) ─────── */
.monitor-list { background: var(--bg2); border: 1px solid var(--border);
                border-radius: 6px; overflow: hidden; margin-bottom: .6rem; }
.monitor-row  { display: flex; align-items: center; gap: .7rem; flex-wrap: wrap;
                padding: .4rem .8rem; border-bottom: 1px solid var(--bg3);
                font-size: .82rem; }
.monitor-row:last-child { border-bottom: none; }
.mr-sym   { font-weight: 700; color: var(--fg); min-width: 60px; }
.mr-state { font-size: .75rem; min-width: 70px; }
.mr-score { font-family: monospace; color: var(--fg2); font-size: .78rem; }
.mr-note  { color: var(--fg3); font-size: .76rem; flex: 1; }
.mr-warn  { background: #3d2f0a; color: var(--amber); font-size: .7rem;
            padding: .1rem .35rem; border-radius: 3px; white-space: nowrap; }

/* ── Speculative warning banner ───────────────────────────────────────────── */
.reversal-banner { background: #3d1212; border: 1px solid var(--red);
                   border-radius: 4px; padding: .4rem .8rem; margin-bottom: .6rem;
                   font-size: .78rem; color: var(--red); }
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
        "Portfolio hold": "badge-breakout",
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
        supported = bool(av.get("supported", True))
        anchor_date = av.get("anchor_date") or "—"
        priority = av.get("priority") or "—"
        if supported:
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
            rows += (
                f"<tr>"
                f"<td><strong>{av.get('anchor', '')}</strong></td>"
                f"<td style='font-family:monospace'>{anchor_date}</td>"
                f"<td style='font-family:monospace'>{_e(av.get('avwap'), 2)}</td>"
                f"<td>{pct_str}</td>"
                f"<td style='font-family:monospace'>{dist_str} ATR</td>"
                f"<td class='{role_cls}'>{av.get('role', '')}</td>"
                f"<td>{av.get('status', '')}</td>"
                f"<td style='color:var(--fg2)'>{priority}</td>"
                f"</tr>"
            )
        else:
            rows += (
                f"<tr>"
                f"<td><strong>{av.get('anchor', '')}</strong></td>"
                f"<td style='font-family:monospace'>{anchor_date}</td>"
                f"<td colspan='6' style='color:var(--fg3)'>"
                f"Unavailable: {av.get('unavailable_reason', 'Unavailable')}"
                f"</td>"
                f"</tr>"
            )
    return (
        f'<table><tr>'
        f'<th>Anchor</th><th>Date</th><th>AVWAP</th><th>Dist%</th><th>ATR Dist</th>'
        f'<th>Role</th><th>Status</th><th>Priority</th>'
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


def _trade_plan_html(trade_plan: dict | str | None) -> str:
    """Render the trade plan block."""
    if not trade_plan:
        return ""
    if isinstance(trade_plan, dict):
        def _line(label: str, value: Any) -> str:
            if value in (None, "", "—", [], {}):
                return ""
            if isinstance(value, list):
                value = "; ".join(str(v) for v in value if str(v).strip())
            return f'<div><strong>{label}:</strong> {value}</div>'

        body = "".join([
            _line("Actionability", trade_plan.get("actionability_code")),
            _line("Best entry", trade_plan.get("best_entry_style") or trade_plan.get("entry_style")),
            _line("Entry condition", trade_plan.get("entry_condition")),
            _line("Entry trigger", trade_plan.get("entry_trigger")),
            _line("Entry range", trade_plan.get("entry_range")),
            _line("Alternate pullback entry", trade_plan.get("alternate_pullback_entry") or trade_plan.get("alt_entry")),
            _line("Stop", trade_plan.get("stop")),
            _line("Invalidation", trade_plan.get("invalidation")),
            _line(
                "Targets",
                " | ".join(
                    str(v)
                    for v in (
                        trade_plan.get("target_1"),
                        trade_plan.get("target_2"),
                        trade_plan.get("target_3"),
                    )
                    if v not in (None, "", "—")
                ),
            ),
            _line("R/R to T1", trade_plan.get("risk_reward_t1")),
            _line("Why now", trade_plan.get("why_now")),
            _line("Why not now", trade_plan.get("why_not_now")),
            _line("Improves tomorrow", trade_plan.get("what_improves_tomorrow") or trade_plan.get("setup_improves_if")),
            _line("Weakens tomorrow", trade_plan.get("what_weakens_tomorrow") or trade_plan.get("setup_weakens_if")),
        ])
        if not body:
            return ""
        return (
            f'<div class="trade-plan">'
            f'<div class="tp-label">Trade Plan</div>'
            f'{body}'
            f'</div>'
        )
    return (
        f'<div class="trade-plan">'
        f'<div class="tp-label">Trade Plan</div>'
        f'{trade_plan}'
        f'</div>'
    )


def _score_drivers_html(score_drivers: dict) -> str:
    """Render the score transparency / signal drivers block."""
    if not score_drivers:
        return ""
    bullish = score_drivers.get("bullish_signals", [])
    bearish = score_drivers.get("bearish_signals", [])
    why = score_drivers.get("why_selected", "")
    if not bullish and not bearish and not why:
        return ""

    rows = ""
    for s in bullish:
        rows += (
            f'<div class="driver-row">'
            f'<span class="driver-bull">▲</span>'
            f'<span class="driver-text">{s}</span>'
            f'</div>'
        )
    for s in bearish:
        rows += (
            f'<div class="driver-row">'
            f'<span class="driver-bear">▼</span>'
            f'<span class="driver-text">{s}</span>'
            f'</div>'
        )

    why_html = ""
    if why:
        why_html = f'<div class="driver-why">{why}</div>'

    return (
        f'<div class="score-drivers">'
        f'{why_html}'
        f'{rows}'
        f'</div>'
    )


def _export_links_html(packet: dict, report_dir_rel: str = "artifacts") -> str:
    """Render compact export/artifact links for a symbol."""
    sym = packet.get("provider_symbol") or packet.get("symbol", "")
    if not sym or sym == "—":
        return ""
    json_path = f"{report_dir_rel}/{sym}_packet.json"
    return (
        f'<div class="export-links">'
        f'<span style="color:var(--fg3);font-size:.72rem">Export: </span>'
        f'<a href="{json_path}" style="font-size:.72rem">JSON packet</a>'
        f'</div>'
    )


def _ma_direction_brief_html(ma_table: list[dict]) -> str:
    """Render a compact always-visible MA direction strip for a card.

    Shows the 3 short MAs (SMA5, SMA10, SMA20) as rising/flat/falling pills,
    plus the SMA20/SMA50 bias notes (what close is needed to keep them rising).
    This lives in the visible card area — not inside a collapsible element.
    """
    if not ma_table:
        return ""

    # Pull out the MAs we care about most for next-day context
    short_mas = ["SMA5", "SMA10", "SMA20"]
    slope_pills = ""
    bias_notes: list[str] = []

    for ma in ma_table:
        name = ma.get("name", "")
        slope = ma.get("slope", "flat")
        bias = ma.get("bias", "")

        if name in short_mas:
            slope_cls = f"ma-{slope}"
            arrow = "▲" if slope == "rising" else ("▼" if slope == "falling" else "—")
            slope_pills += f'<span class="ma-pill {slope_cls}">{arrow} {name}</span>'

        # Surface SMA20 and SMA50 bias notes (the "close needed" threshold)
        if name in ("SMA20", "SMA50") and bias:
            bias_notes.append(f"<strong>{name}:</strong> {bias}")

    if not slope_pills and not bias_notes:
        return ""

    bias_html = ""
    if bias_notes:
        bias_html = (
            '<div class="ma-bias-note">'
            + " &nbsp;·&nbsp; ".join(bias_notes)
            + "</div>"
        )

    return (
        f'<div>'
        f'<div class="ma-brief">'
        f'<span class="ma-brief-label">MA direction:</span>'
        f'{slope_pills}'
        f'</div>'
        f'{bias_html}'
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

def _render_card(packet: dict, rank: int = 0) -> str:
    """Render a single setup card.

    Parameters
    ----------
    packet : analysis packet dict (from packet.build_packet + charts).
    rank   : 1-based position in the top-5 list; 0 means unranked.
    """
    sym = packet.get("symbol", "?")
    state = packet.get("state", "NONE")
    action = packet.get("action_label", "—")
    setup_cls_label = packet.get("setup_classification", "")
    score_raw = packet.get("composite_score", "—")
    failure_raw = packet.get("failure_risk", "—")
    rank_raw = packet.get("percentile_rank", "—")
    score_cls = _score_cls(score_raw)
    freshness_label = packet.get("freshness_label", "")
    bucket = packet.get("bucket", "")

    narrative = packet.get("narrative", {})
    context = packet.get("context", {}) or {}
    ai_note = packet.get("ai_note")
    is_portfolio = packet.get("is_portfolio", False)
    portfolio_tag = ' <small style="color:var(--purple)">[Portfolio]</small>' if is_portfolio else ""
    promotion_reason = str(packet.get("promotion_reason", "") or packet.get("route_reason", ""))

    # Setup classification badge
    sc_badge = _setup_class_badge(setup_cls_label) if setup_cls_label else ""

    # Bucket tag (breakout vs pullback label)
    bucket_tag = ""
    if bucket == "breakout_long":
        bucket_tag = '<span class="bucket-tag bt-breakout">BREAKOUT</span>'
    elif bucket == "pullback_long":
        bucket_tag = '<span class="bucket-tag bt-pullback">PULLBACK</span>'

    # Freshness tag
    fresh_tag = ""
    if freshness_label and freshness_label not in ("—", ""):
        fresh_color = "var(--green)" if packet.get("is_fresh") else "var(--fg3)"
        fresh_tag = f'<small style="color:{fresh_color};margin-left:.3rem">[{freshness_label}]</small>'

    # Extension warning badge (overrides other labels visually)
    ext_tag = ""
    if packet.get("is_extended"):
        ext_reasons = packet.get("extension_reasons", "")
        ext_tag = (
            f'<span style="color:var(--amber);font-size:.75rem;margin-left:.3rem" '
            f'title="{ext_reasons}">⚠ Extended</span>'
        )

    # Rank indicator
    rank_tag = ""
    if rank > 0:
        rank_tag = f'<span class="rank-num">#{rank}</span>'

    # Header
    header = (
        f'<div class="card-header">'
        f'{rank_tag}'
        f'{_badge(action)}'
        f'<span class="card-symbol">{sym}{portfolio_tag}</span>'
        f'{_state_span(state)}'
        f'{sc_badge}'
        f'{bucket_tag}'
        f'{fresh_tag}'
        f'{ext_tag}'
        f'<span class="card-score">'
        f'Score: <span class="{score_cls}">{score_raw}</span>'
        f' &nbsp;|&nbsp; Fail: {failure_raw}'
        f' &nbsp;|&nbsp; Rank: {rank_raw}'
        f'</span>'
        f'</div>'
    )

    # Charts: intraday is intentionally a compact policy note in v1.
    intraday_policy = packet.get("intraday_policy", "")
    intraday_available = packet.get("intraday_available", packet.get("chart_intraday") is not None)
    intraday_note = str(
        packet.get(
            "intraday_note",
            "Intraday confirmation is not part of v1 qualification; surfaced setup truth is daily/weekly only.",
        )
    )
    if intraday_policy != "daily_only" and intraday_available and packet.get("chart_intraday"):
        intraday_section = (
            f'<div style="font-size:.75rem;color:var(--fg3);margin:.4rem 0 .3rem">Intraday (5m)</div>'
            f'{_chart_img(packet.get("chart_intraday"), f"{sym} Intraday")}'
        )
    else:
        intraday_section = (
            '<div class="chart-na-inline">'
            f'{intraday_note}'
            '</div>'
        )

    charts_html = (
        f'<div class="card-charts">'
        f'<div style="font-size:.75rem;color:var(--fg3);margin-bottom:.3rem">Weekly</div>'
        f'{_chart_img(packet.get("chart_weekly"), f"{sym} Weekly")}'
        f'<div style="font-size:.75rem;color:var(--fg3);margin:.4rem 0 .3rem">Daily</div>'
        f'{_chart_img(packet.get("chart_daily"), f"{sym} Daily")}'
        f'{intraday_section}'
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

    qualification_html = ""
    if promotion_reason and promotion_reason not in ("—", ""):
        qualification_html = (
            f'<div class="trade-plan">'
            f'<div class="tp-label">Why Qualified</div>'
            f'<div>{promotion_reason}</div>'
            f'</div>'
        )

    # Confluence (brief, shown inline before narrative)
    confluence_html = _confluence_html(context.get("confluence", {}))

    # Score drivers transparency
    drivers_html = _score_drivers_html(context.get("score_drivers", {}))

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

    # MA direction brief (always-visible short strip)
    ma_brief_html = _ma_direction_brief_html(context.get("ma_table", []))

    narrative_section = (
        f'<div>'
        f'<h3>Narrative</h3>'
        f'{ma_brief_html}'
        f'<dl class="narrative">'
        f'<dt>Setup</dt><dd>{n.get("setup", "—")}</dd>'
        f'<dt>Why now</dt><dd>{n.get("why", "—")}</dd>'
        f'<dt>Why not now</dt><dd>{n.get("why_not_now", "—")}</dd>'
        f'<dt>Entry</dt><dd>{n.get("entry", "—")}</dd>'
        f'<dt>Risk / invalidation</dt><dd>{n.get("risk", "—")}</dd>'
        f'<dt>Targets</dt><dd>{n.get("targets", "—")}</dd>'
        f'{context_block}'
        f'</dl>'
        f'<div class="verdict-box">{n.get("verdict", "—")}</div>'
        f'</div>'
    )

    # Trade plan
    trade_plan_section = _trade_plan_html(packet.get("trade_plan"))

    # AI analysis note
    ai_note_section = _ai_note_html(ai_note)

    # Context deep-dives (collapsible — full detail available on demand)
    ma_html = _ma_table_html(context.get("ma_table", []))
    avwap_html = _avwap_table_html(context.get("avwap_table", []))
    checklist_html_str = _checklist_html(context.get("checklist", []))
    vol_html = _volume_block_html(context.get("volume_block", {}))

    context_details = ""
    if ma_html:
        context_details += (
            f'<details style="margin-top:.4rem">'
            f'<summary style="font-size:.78rem">Full MA Table</summary>'
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
        f'{qualification_html}'
        f'{levels_section}'
        f'{confluence_html}'
        f'{drivers_html}'
        f'{narrative_section}'
        f'{trade_plan_section}'
        f'{ai_note_section}'
        f'{context_details}'
        f'{_export_links_html(packet)}'
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

        # Score + percentile pill (model signal at a glance)
        score_v = row.get("composite_score", math.nan)
        pct_v = row.get("percentile_rank", math.nan)
        try:
            score_f = float(score_v)
            pct_f = float(pct_v)
            score_pill = (
                f'<span class="pc-score" title="composite_score · percentile">'
                f'{score_f:.2f} <span style="color:var(--fg3)">({pct_f:.0f}p)</span>'
                f'</span>'
            ) if math.isfinite(score_f) else ""
        except (TypeError, ValueError):
            score_pill = ""

        # Eligibility warnings (non-portfolio symbols can still have warnings)
        warn_str = str(row.get("eligibility_warnings", "")) if "eligibility_warnings" in portfolio_df.columns else ""
        warn_html = (
            f'<span class="pc-warn" title="{warn_str}">\u26a0 {warn_str[:40]}{"…" if len(warn_str) > 40 else ""}</span>'
        ) if warn_str and warn_str not in ("—", "") else ""

        if guidance and guidance not in ("—", ""):
            gl = guidance.lower()
            if "exit" in gl or "fail" in gl:
                icon, pg_cls = "✗", "pg-exit"
            elif "trim" in gl or "de-risk" in gl:
                icon, pg_cls = "↓", "pg-trim"
            elif "defend" in gl or "tighten" in gl:
                icon, pg_cls = "⚠", "pg-defend"
            elif "hold" in gl or "watch" in gl:
                icon, pg_cls = "✓", "pg-hold"
            else:
                icon, pg_cls = "\u2139", "pg-info"  # ℹ information source
            guidance_html = (
                f'<span class="pc-guidance {pg_cls}">'
                f'{icon} {guidance[:70]}{"…" if len(guidance) > 70 else ""}'
                f'</span>'
            )
        else:
            guidance_html = ""

        chips.append(
            f'<div class="port-chip">'
            f'<span class="pc-sym">{sym}</span>'
            f'<span class="pc-state s-{state}">{state}</span>'
            f'{score_pill}'
            f'<span class="pc-action">{action}</span>'
            f'{warn_html}'
            f'{guidance_html}'
            f'</div>'
        )

    return (
        f'<h2>Portfolio Holdings</h2>'
        f'<div class="portfolio-strip">{"".join(chips)}</div>'
    )


# ── Packet-first rendering helpers ───────────────────────────────────────────


def _portfolio_strip_html_from_packets(portfolio_pkts: list[dict]) -> str:
    """Render portfolio strip from packet list (packet-first path)."""
    if not portfolio_pkts:
        return ""
    chips = []
    for pkt in portfolio_pkts:
        sym     = str(pkt.get("symbol", "?"))
        state   = str(pkt.get("state", "NONE"))
        action  = str(pkt.get("action_label", "—"))
        guidance = str(pkt.get("portfolio_guidance", ""))
        score_v = pkt.get("composite_score", "—")
        pct_v   = pkt.get("percentile_rank", "—")

        try:
            score_f = float(score_v)
            pct_f   = float(pct_v)
            score_pill = (
                f'<span class="pc-score" title="composite_score · percentile">'
                f'{score_f:.2f} <span style="color:var(--fg3)">({pct_f:.0f}p)</span>'
                f'</span>'
            ) if math.isfinite(score_f) else ""
        except (TypeError, ValueError):
            score_pill = ""

        # Prefer portfolio_health.recommended_action if present
        ph = pkt.get("portfolio_health", {})
        if isinstance(ph, dict) and ph.get("recommended_action"):
            guidance = ph["recommended_action"]

        warn_str = str(pkt.get("eligibility_warnings", ""))
        warn_html = (
            f'<span class="pc-warn" title="{warn_str}">\u26a0 {warn_str[:40]}{"…" if len(warn_str) > 40 else ""}</span>'
        ) if warn_str and warn_str not in ("—", "") else ""

        if guidance and guidance not in ("—", ""):
            gl = guidance.lower()
            if "exit" in gl or "fail" in gl:
                icon, pg_cls = "✗", "pg-exit"
            elif "trim" in gl or "de-risk" in gl:
                icon, pg_cls = "↓", "pg-trim"
            elif "defend" in gl or "tighten" in gl or "stop" in gl:
                icon, pg_cls = "⚠", "pg-defend"
            elif "hold" in gl or "watch" in gl:
                icon, pg_cls = "✓", "pg-hold"
            else:
                icon, pg_cls = "\u2139", "pg-info"
            guidance_html = (
                f'<span class="pc-guidance {pg_cls}">'
                f'{icon} {guidance[:70]}{"…" if len(guidance) > 70 else ""}'
                f'</span>'
            )
        else:
            guidance_html = ""

        chips.append(
            f'<div class="port-chip">'
            f'<span class="pc-sym">{sym}</span>'
            f'<span class="pc-state s-{state}">{state}</span>'
            f'{score_pill}'
            f'<span class="pc-action">{action}</span>'
            f'{warn_html}'
            f'{guidance_html}'
            f'</div>'
        )

    return (
        f'<h2>Portfolio Holdings</h2>'
        f'<div class="portfolio-strip">{"".join(chips)}</div>'
    )


def _compact_monitor_rows_from_packets(pkts: list[dict], max_rows: int = 8) -> str:
    """Render compact monitor rows for extended leaders from packet list."""
    rows = []
    for pkt in pkts[:max_rows]:
        sym   = str(pkt.get("symbol", "—"))
        state = str(pkt.get("state", "—"))
        score_v = pkt.get("composite_score", "—")
        dist_v  = pkt.get("dist_to_pivot_atr", "—")
        ext_reason = str(pkt.get("extension_reasons", ""))
        try:
            score_str = f"{float(score_v):.3f}"
        except (TypeError, ValueError):
            score_str = "—"
        try:
            dist_f = float(dist_v)
            dist_str = f"+{dist_f:.1f} ATR extended" if math.isfinite(dist_f) and dist_f > 0 else ""
        except (TypeError, ValueError):
            dist_str = ""
        note = ext_reason if ext_reason and ext_reason not in ("—",) else "do not chase; watch for pullback to add zone"
        rows.append(
            f'<div class="monitor-row">'
            f'<span class="mr-sym">{sym}</span>'
            f'<span class="mr-state s-{state}">{state}</span>'
            f'<span class="mr-score">{score_str}</span>'
            f'<span class="mr-note">{dist_str}{" — " if dist_str else ""}{note}</span>'
            f'</div>'
        )
    return f'<div class="monitor-list">{"".join(rows)}</div>' if rows else ""


def _compact_reversal_rows_from_packets(pkts: list[dict], max_rows: int = 3) -> str:
    """Render compact monitor rows for reversal candidates from packet list."""
    rows = []
    for pkt in pkts[:max_rows]:
        sym       = str(pkt.get("symbol", "—"))
        state     = str(pkt.get("state", "—"))
        score_v   = pkt.get("composite_score", "—")
        rejection = str(pkt.get("rejection_reasons", ""))
        try:
            score_str = f"{float(score_v):.3f}"
        except (TypeError, ValueError):
            score_str = "—"
        rows.append(
            f'<div class="monitor-row">'
            f'<span class="mr-sym">{sym}</span>'
            f'<span class="mr-state s-{state}">{state}</span>'
            f'<span class="mr-score">{score_str}</span>'
            f'<span class="mr-warn">failed: {rejection[:60]}</span>'
            f'<span class="mr-note">speculative only; failed primary long eligibility</span>'
            f'</div>'
        )
    return f'<div class="monitor-list">{"".join(rows)}</div>' if rows else ""


# ── Public API ────────────────────────────────────────────────────────────────

def render_dashboard(
    snapshot_df: pd.DataFrame,
    packets: list[dict],
    as_of: pd.Timestamp,
    *,
    selections: dict | None = None,
    oos_metrics: dict | None = None,
) -> str:
    """Render the complete 3-layer dashboard as an HTML string.

    Parameters
    ----------
    snapshot_df : full scored snapshot (used for regime columns).  Pass an empty
                  DataFrame if not available; regime section will be blank.
    packets     : list of dicts for the top setups (breakout + pullback).
    as_of       : report date.
    selections  : optional PacketSelections dict from select_packets().  When
                  provided, extended/reversal/portfolio sections are rendered
                  from packet lists rather than from snapshot_df.  This is the
                  preferred packet-first path.
    oos_metrics : optional OOS calibration metrics for the footer.

    Returns
    -------
    str — complete self-contained HTML.
    """
    generated_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    date_str = str(as_of.date())

    # Partition snapshot (kept for regime columns)
    full_df = snapshot_df.copy() if not snapshot_df.empty else pd.DataFrame()

    # Layer 1 — Regime strip
    regime_html = _regime_html(full_df)

    # Portfolio strip: prefer packet-first (selections["portfolio"]) over DataFrame.
    # portfolio_df is always initialised so the detail table below can reference it.
    portfolio_df = pd.DataFrame()
    if not full_df.empty:
        if "bucket" in full_df.columns:
            portfolio_df = full_df[full_df["bucket"] == "portfolio_hold"].copy()
        elif "is_portfolio" in full_df.columns:
            portfolio_df = full_df[full_df["is_portfolio"].astype(bool)].copy()

    if selections is not None and selections.get("portfolio"):
        portfolio_html = _portfolio_strip_html_from_packets(selections["portfolio"])
    else:
        portfolio_html = _portfolio_strip_html(portfolio_df)

    # ── Layer 2 — Bucket-separated setup sections ────────────────────────────

    # Prefer packet-based partitioning when selections are available
    if selections is not None:
        breakout_packets = selections.get("breakout", [])
        pullback_packets  = selections.get("pullback", [])
        extended_pkts     = selections.get("extended", [])
        reversal_pkts     = selections.get("reversal", [])
    else:
        breakout_packets = [p for p in packets if p.get("bucket") == "breakout_long"]
        pullback_packets  = [p for p in packets if p.get("bucket") == "pullback_long"]
        extended_pkts     = []
        reversal_pkts     = []

    # Counts: from selections when available, else from snapshot_df
    if selections is not None:
        all_pkts = selections.get("all", packets)
        n_total    = len(all_pkts)
        n_eligible = sum(1 for p in all_pkts if bool(p.get("eligible", False)))
        n_fresh    = sum(1 for p in all_pkts if bool(p.get("is_fresh", False)))
        n_bo_cands = sum(1 for p in all_pkts if p.get("bucket") == "breakout_long")
        n_pb_cands = sum(1 for p in all_pkts if p.get("bucket") == "pullback_long")
        n_ext_cands = sum(1 for p in all_pkts if p.get("bucket") == "extended_leader")
        n_rev_cands = sum(1 for p in all_pkts if p.get("bucket") == "reversal_speculative")
        n_excl_cands = sum(1 for p in all_pkts if p.get("bucket") == "excluded")
        n_bo_blocked = sum(
            1 for p in all_pkts
            if p.get("bucket") == "breakout_long" and p.get("selector_blockers")
        )
        n_pb_blocked = sum(
            1 for p in all_pkts
            if p.get("bucket") == "pullback_long" and p.get("selector_blockers")
        )
    else:
        n_total = len(full_df) if not full_df.empty else 0
        n_eligible = 0
        n_fresh = 0
        n_bo_blocked = 0
        n_pb_blocked = 0
        snapshot_bucket_counts: dict[str, int] = {}
        if not full_df.empty:
            if "eligible" in full_df.columns:
                n_eligible = int(full_df["eligible"].astype(bool).sum())
            if "is_fresh" in full_df.columns:
                n_fresh = int(full_df["is_fresh"].astype(bool).sum())
            if "bucket" in full_df.columns:
                for bk, cnt in full_df["bucket"].value_counts().items():
                    snapshot_bucket_counts[str(bk)] = int(cnt)
        n_bo_cands   = snapshot_bucket_counts.get("breakout_long", 0)
        n_pb_cands   = snapshot_bucket_counts.get("pullback_long", 0)
        n_ext_cands  = snapshot_bucket_counts.get("extended_leader", 0)
        n_rev_cands  = snapshot_bucket_counts.get("reversal_speculative", 0)
        n_excl_cands = snapshot_bucket_counts.get("excluded", 0)

    # Universe summary bar
    universe_bar = (
        f'<div class="universe-bar">'
        f'<strong>{n_total}</strong> universe &nbsp;·&nbsp; '
        f'<strong>{n_eligible}</strong> eligible &nbsp;·&nbsp; '
        f'<strong>{n_fresh}</strong> fresh &nbsp;·&nbsp; '
        f'<strong style="color:var(--blue)">{n_bo_cands}</strong> breakout candidates &nbsp;·&nbsp; '
        f'<strong style="color:var(--amber)">{n_pb_cands}</strong> pullback candidates &nbsp;·&nbsp; '
        f'<strong style="color:var(--fg3)">{n_ext_cands}</strong> extended &nbsp;·&nbsp; '
        f'<strong style="color:var(--red)">{n_rev_cands}</strong> reversal watch &nbsp;·&nbsp; '
        f'<strong>{n_excl_cands}</strong> excluded by gates'
        f'</div>'
    )

    # ── Breakout section ──────────────────────────────────────────────────────
    bo_header = (
        f'<div class="bucket-header bh-breakout">'
        f'<span class="bh-title">Top Breakout Longs</span>'
        f'<span class="bh-count">{len(breakout_packets)} shown</span>'
        f'<span class="bh-desc">'
        f'Eligible, fresh setups at or near pivot. Ranked by setup quality score.'
        f'</span>'
        f'</div>'
    )
    if breakout_packets:
        bo_cards = "\n".join(
            _render_card(p, rank=i + 1) for i, p in enumerate(breakout_packets)
        )
        bo_body = f'<div class="setup-cards">{bo_cards}</div>'
    else:
        bo_body = (
            f'<div class="no-setups-note">'
            f'No breakout candidates qualify today. '
            f'{n_bo_cands} name{"s" if n_bo_cands != 1 else ""} landed in the breakout bucket, '
            f'but {n_bo_blocked} were blocked for packet coherence or completeness.'
            f'</div>'
        )
    breakout_section = (
        f'<div class="bucket-section">'
        f'{bo_header}'
        f'{bo_body}'
        f'</div>'
    )

    # ── Pullback section ──────────────────────────────────────────────────────
    pb_header = (
        f'<div class="bucket-header bh-pullback">'
        f'<span class="bh-title">Top Pullback / Re-entry Longs</span>'
        f'<span class="bh-count">{len(pullback_packets)} shown</span>'
        f'<span class="bh-desc">'
        f'Constructive pullbacks within confirmed uptrends. Secondary candidates.'
        f'</span>'
        f'</div>'
    )
    if pullback_packets:
        pb_cards = "\n".join(
            _render_card(p, rank=i + 1) for i, p in enumerate(pullback_packets)
        )
        pb_body = f'<div class="setup-cards">{pb_cards}</div>'
    else:
        pb_body = (
            '<div class="no-setups-note">'
            f'No pullback candidates qualify today. {n_pb_cands} name{"s" if n_pb_cands != 1 else ""} '
            f'landed in the pullback bucket, but {n_pb_blocked} were blocked for packet coherence or completeness.'
            '</div>'
        )
    pullback_section = (
        f'<div class="bucket-section">'
        f'{pb_header}'
        f'{pb_body}'
        f'</div>'
    )

    # ── Extended Leaders compact section ─────────────────────────────────────
    # Prefer packet list; fall back to DataFrame.
    ext_rows_html = ""
    if extended_pkts:
        ext_rows_html = _compact_monitor_rows_from_packets(extended_pkts, max_rows=8)
    elif not full_df.empty and "bucket" in full_df.columns:
        ext_df = full_df[full_df["bucket"] == "extended_leader"].copy()
        if not ext_df.empty:
            if "composite_score" in ext_df.columns:
                ext_df["_sc"] = ext_df["composite_score"].apply(
                    lambda v: float(v) if isinstance(v, (int, float)) and math.isfinite(float(v) if not isinstance(v, str) else -1) else -1.0
                )
                ext_df = ext_df.sort_values("_sc", ascending=False).drop(columns=["_sc"])
            sym_col = "user_symbol" if "user_symbol" in ext_df.columns else "symbol"
            rows = []
            for _, row in ext_df.head(8).iterrows():
                sym = str(row.get(sym_col, "—"))
                state = str(row.get("state", "—"))
                score = row.get("composite_score", math.nan)
                try:
                    score_str = f"{float(score):.3f}"
                except (TypeError, ValueError):
                    score_str = "—"
                dist = row.get("dist_to_pivot_atr", math.nan)
                try:
                    dist_str = f"+{float(dist):.1f} ATR extended" if math.isfinite(float(dist)) else ""
                except (TypeError, ValueError):
                    dist_str = ""
                rows.append(
                    f'<div class="monitor-row">'
                    f'<span class="mr-sym">{sym}</span>'
                    f'<span class="mr-state s-{state}">{state}</span>'
                    f'<span class="mr-score">{score_str}</span>'
                    f'<span class="mr-note">{dist_str} — do not chase; watch for pullback to add zone</span>'
                    f'</div>'
                )
            ext_rows_html = f'<div class="monitor-list">{"".join(rows)}</div>'

    ext_header = (
        f'<div class="bucket-header bh-extended">'
        f'<span class="bh-title">Extended Leaders</span>'
        f'<span class="bh-count">{n_ext_cands} total</span>'
        f'<span class="bh-desc">'
        f'Healthy names too extended for fresh entry. Monitor for pullback add zones.'
        f'</span>'
        f'</div>'
    )
    if ext_rows_html:
        extended_section = (
            f'<div class="bucket-section">'
            f'{ext_header}'
            f'{ext_rows_html}'
            f'</div>'
        )
    else:
        extended_section = ""

    # ── Speculative / Reversal Watch compact section ─────────────────────────
    # Prefer packet list; fall back to DataFrame.
    rev_rows_html = ""
    if reversal_pkts:
        rev_rows_html = _compact_reversal_rows_from_packets(reversal_pkts, max_rows=3)
    elif not full_df.empty and "bucket" in full_df.columns:
        rev_df = full_df[full_df["bucket"] == "reversal_speculative"].copy()
        if not rev_df.empty:
            if "composite_score" in rev_df.columns:
                rev_df["_sc"] = rev_df["composite_score"].apply(
                    lambda v: float(v) if isinstance(v, (int, float)) and math.isfinite(float(v) if not isinstance(v, str) else -1) else -1.0
                )
                rev_df = rev_df.sort_values("_sc", ascending=False).drop(columns=["_sc"])
            sym_col = "user_symbol" if "user_symbol" in rev_df.columns else "symbol"
            rows = []
            for _, row in rev_df.head(3).iterrows():
                sym = str(row.get(sym_col, "—"))
                state = str(row.get("state", "—"))
                score = row.get("composite_score", math.nan)
                rejection = str(row.get("rejection_reasons", ""))
                try:
                    score_str = f"{float(score):.3f}"
                except (TypeError, ValueError):
                    score_str = "—"
                rows.append(
                    f'<div class="monitor-row">'
                    f'<span class="mr-sym">{sym}</span>'
                    f'<span class="mr-state s-{state}">{state}</span>'
                    f'<span class="mr-score">{score_str}</span>'
                    f'<span class="mr-warn">failed: {rejection[:60]}</span>'
                    f'<span class="mr-note">speculative only; failed primary long eligibility</span>'
                    f'</div>'
                )
            rev_rows_html = f'<div class="monitor-list">{"".join(rows)}</div>'

    if rev_rows_html:
        reversal_section = (
            f'<div class="bucket-section">'
            f'<div class="bucket-header bh-reversal">'
            f'<span class="bh-title">Speculative / Reversal Watch</span>'
            f'<span class="bh-count">{n_rev_cands} total</span>'
            f'<span class="bh-desc">NOT in primary long list.</span>'
            f'</div>'
            f'<div class="reversal-banner">'
            f'<strong>⚠ These names failed primary eligibility gates.</strong> '
            f'They are shown for informational monitoring only. '
            f'Do not treat these as equivalent to breakout or pullback candidates. '
            f'They require separate discretionary judgment and should not be read as primary long ideas.'
            f'</div>'
            f'{rev_rows_html}'
            f'</div>'
        )
    else:
        reversal_section = ""

    # Combine into the main cards section
    cards_section = (
        f'<h2>Setups &mdash; {date_str}</h2>'
        f'{universe_bar}'
        f'{breakout_section}'
        f'{pullback_section}'
        f'{extended_section}'
        f'{reversal_section}'
    )

    # Excluded symbols section (compact, collapsible)
    excluded_section = ""
    if not full_df.empty and "eligible" in full_df.columns and "rejection_reasons" in full_df.columns:
        excl_df = full_df[~full_df["eligible"].astype(bool) & full_df["state"].isin(
            {"BASE", "ARMED", "TRIGGERED", "ACCEPTED"}
        )].copy()
        if not excl_df.empty:
            sym_col = "user_symbol" if "user_symbol" in excl_df.columns else "symbol"
            excl_rows = ""
            for _, row in excl_df.iterrows():
                sym = str(row.get(sym_col, "—"))
                state = str(row.get("state", "—"))
                reasons = str(row.get("rejection_reasons", ""))
                score = row.get("composite_score", math.nan)
                score_str = f"{float(score):.3f}" if (isinstance(score, (int, float)) and math.isfinite(float(score) if isinstance(score, str) else score)) else "—"
                excl_rows += (
                    f'<tr><td>{sym}</td><td class="s-{state}">{state}</td>'
                    f'<td style="color:var(--red);font-size:.75rem">{reasons}</td>'
                    f'<td>{score_str}</td></tr>'
                )
            excluded_section = (
                f'<details style="margin-top:1rem">'
                f'<summary style="color:var(--fg3);font-size:.85rem">'
                f'Excluded by eligibility gates — {len(excl_df)} symbols in scored states'
                f'</summary>'
                f'<div class="details-body">'
                f'<table><tr>'
                f'<th>Symbol</th><th>State</th><th>Rejection Reasons</th><th>Score</th>'
                f'</tr>{excl_rows}</table>'
                f'</div></details>'
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

{excluded_section}

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
    selections: dict | None = None,
    oos_metrics: dict | None = None,
) -> Path:
    """Write dashboard.html to output_dir and return the path.

    Parameters
    ----------
    selections : optional PacketSelections dict from select_packets().  When
                 provided, extended/reversal/portfolio sections are rendered
                 entirely from pre-built packets (packet-first path).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    html = render_dashboard(snapshot_df, packets, as_of,
                            selections=selections, oos_metrics=oos_metrics)
    path = output_dir / "dashboard.html"
    path.write_text(html, encoding="utf-8")
    log.info("Dashboard written -> %s", path)
    return path
