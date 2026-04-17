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
    score_raw = packet.get("composite_score", "—")
    failure_raw = packet.get("failure_risk", "—")
    rank_raw = packet.get("percentile_rank", "—")
    score_cls = _score_cls(score_raw)

    narrative = packet.get("narrative", {})
    is_portfolio = packet.get("is_portfolio", False)
    portfolio_tag = ' <small style="color:var(--purple)">[Portfolio]</small>' if is_portfolio else ""

    # Header
    header = (
        f'<div class="card-header">'
        f'{_badge(action)}'
        f'<span class="card-symbol">{sym}{portfolio_tag}</span>'
        f'{_state_span(state)}'
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

    # AI-review-ready data panel (collapsible)
    ai_data = {k: v for k, v in packet.items() if k != "narrative"}
    ai_json = json.dumps(ai_data, indent=2, default=str)
    ai_panel = (
        f'<details style="margin-top:.5rem">'
        f'<summary style="font-size:.75rem">AI-review packet (JSON)</summary>'
        f'<div class="details-body"><pre style="font-size:.72rem;overflow:auto;max-height:200px">'
        f'{ai_json}</pre></div>'
        f'</details>'
    )

    detail = (
        f'<div class="card-detail">'
        f'{levels_section}'
        f'{narrative_section}'
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

    if not snapshot_df.empty and "regime_spy_trend" in snapshot_df.columns:
        col = snapshot_df["regime_spy_trend"].dropna()
        if not col.empty:
            spy_trend = float(col.iloc[0])

    if math.isfinite(spy_trend):
        if spy_trend > 0:
            trend_val, trend_dir = "Uptrend", "up"
        elif spy_trend < 0:
            trend_val, trend_dir = "Downtrend", "down"
        else:
            trend_val, trend_dir = "Neutral", "neutral"
    else:
        trend_val, trend_dir = "—", ""

    actionable_n = 0
    armed_n = 0
    triggered_n = 0
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
        _regime_pill("Actionable", str(actionable_n)),
        _regime_pill("ARMED", str(armed_n)),
        _regime_pill("Triggered/Accepted", str(triggered_n)),
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
        chips.append(
            f'<div class="port-chip">'
            f'<span class="pc-sym">{sym}</span>'
            f'<span class="pc-state {f"s-{state}"}">{state}</span>'
            f'<span class="pc-action">{action}</span>'
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
