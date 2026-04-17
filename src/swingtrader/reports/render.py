"""Report rendering for daily snapshots.

Produces both Markdown (for GitHub display) and HTML (for local / Pages viewing).

Template variables are assembled in _build_context(); both output formats share
the same context dict — Jinja2 templates consume what they need.

Score column formatting:
  composite_score ≥ 0.6  → "high" CSS class (green)
  composite_score ≥ 0.3  → "mid"  CSS class (amber)
  composite_score < 0.3  → "low"  CSS class (red)
"""
from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from jinja2 import Environment, FileSystemLoader

from swingtrader.utils.logging import get_logger

log = get_logger(__name__)

_TEMPLATES_DIR = Path(__file__).parent.parent.parent.parent / "reports" / "templates"

# Priority order for display sections
_SCORED_STATES = ["ARMED", "TRIGGERED", "ACCEPTED", "BASE"]
_UNSCORED_STATES = ["CONFIRMED", "LATE", "FAILED", "NONE", "EXHAUSTED"]

_SNAPSHOT_COLS = [
    "user_symbol",
    "state",
    "close",
    "pivot",
    "dist_to_pivot_atr",
    "base_length",
    "days_in_state",
    "atr_compression_pct",
    "volume_dryup",
    "daily_rs_63",
    "composite_score",
    "percentile_rank",
]

_PORTFOLIO_COLS = [
    "user_symbol",
    "state",
    "close",
    "pivot",
    "dist_to_pivot_atr",
    "composite_score",
]


def _fmt(v: Any, decimals: int = 2) -> str:
    if v is None:
        return "—"
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return "—"
    if isinstance(v, float):
        return f"{v:.{decimals}f}"
    if isinstance(v, bool):
        return "Y" if v else "N"
    return str(v)


def _md_table(df: pd.DataFrame, cols: list[str]) -> str:
    """Render a subset of columns as a GitHub-flavoured Markdown table."""
    available = [c for c in cols if c in df.columns]
    sub = df[available].copy()

    header = "| " + " | ".join(available) + " |"
    sep = "| " + " | ".join("---" for _ in available) + " |"
    rows = []
    for _, row in sub.iterrows():
        cells = [_fmt(row[c]) for c in available]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep] + rows) if rows else ""


def _html_table(df: pd.DataFrame, cols: list[str]) -> str:
    """Render as an HTML table with score CSS classes."""
    available = [c for c in cols if c in df.columns]
    sub = df[available].copy()

    header_cells = "".join(f"<th>{c}</th>" for c in available)
    rows_html: list[str] = []
    for _, row in sub.iterrows():
        cells = []
        for c in available:
            v = row[c]
            css = ""
            if c == "composite_score" and isinstance(v, float) and math.isfinite(v):
                css = "score-high" if v >= 0.6 else ("score-mid" if v >= 0.3 else "score-low")
            if c == "state":
                css = f"state-{v}"
            if c == "percentile_rank" and isinstance(v, float) and math.isfinite(v):
                # tiny bar visualisation
                w = int(v * 0.5)  # max 50px
                cells.append(
                    f'<td>{_fmt(v, 0)}'
                    f'<span class="rank-bar" style="width:{w}px;margin-left:4px"></span></td>'
                )
                continue
            cells.append(f'<td class="{css}">{_fmt(v)}</td>' if css else f"<td>{_fmt(v)}</td>")
        rows_html.append("<tr>" + "".join(cells) + "</tr>")
    body = "".join(rows_html)
    return f"<table><tr>{header_cells}</tr>{body}</table>"


def _build_context(
    snapshot_df: pd.DataFrame,
    scores_df: pd.DataFrame | None,
    as_of: pd.Timestamp,
    oos_metrics: dict | None = None,
) -> dict:
    """Assemble the Jinja2 template context dict."""
    # Merge scores if provided
    df = snapshot_df.copy()
    if scores_df is not None and not scores_df.empty:
        key = "provider_symbol" if "provider_symbol" in df.columns else "user_symbol"
        scores_reset = scores_df.rename_axis("_sym").reset_index().rename(columns={"_sym": key})
        score_cols = [c for c in ["setup_score", "trade_score", "failure_risk", "composite_score", "percentile_rank"] if c in scores_reset.columns]
        df = df.merge(scores_reset[[key] + score_cols], on=key, how="left")

    # OOS metrics summary
    model_fitted = oos_metrics is not None or (scores_df is not None and not scores_df.empty)
    fit_date = "—"
    setup_brier = "—"
    failure_brier = "—"
    if oos_metrics:
        ss = oos_metrics.get("setup_score", {})
        fr = oos_metrics.get("failure_risk", {})
        setup_brier = f"{ss.get('brier_score_mean', float('nan')):.4f}" if isinstance(ss.get("brier_score_mean"), float) else "—"
        failure_brier = f"{fr.get('brier_score_mean', float('nan')):.4f}" if isinstance(fr.get("brier_score_mean"), float) else "—"

    # Partition rows
    portfolio = df[df.get("is_portfolio", pd.Series(False, index=df.index)).astype(bool)] if "is_portfolio" in df.columns else pd.DataFrame()
    watchlist = df[df.get("is_watchlist", pd.Series(True, index=df.index)).astype(bool)] if "is_watchlist" in df.columns else df
    skipped = df[df.get("state", pd.Series("", index=df.index)) == "SKIPPED"] if "state" in df.columns else pd.DataFrame()

    state_sections_md: dict[str, str] = {}
    state_sections_html: dict[str, str] = {}
    state_counts: dict[str, int] = {}

    all_states = _SCORED_STATES + _UNSCORED_STATES
    for state_name in all_states:
        subset = watchlist[watchlist.get("state", pd.Series("", index=watchlist.index)) == state_name] if "state" in watchlist.columns else pd.DataFrame()
        state_counts[state_name] = len(subset)
        if not subset.empty:
            state_sections_md[state_name] = _md_table(subset, _SNAPSHOT_COLS)
            state_sections_html[state_name] = _html_table(subset, _SNAPSHOT_COLS)
        else:
            state_sections_md[state_name] = ""
            state_sections_html[state_name] = ""

    skipped_records = skipped.to_dict("records") if not skipped.empty else []
    portfolio_md = _md_table(portfolio, _PORTFOLIO_COLS) if not portfolio.empty else ""
    portfolio_html = _html_table(portfolio, _PORTFOLIO_COLS) if not portfolio.empty else ""

    return {
        "as_of": str(as_of.date()),
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M"),
        "model_fitted": model_fitted,
        "fit_date": fit_date,
        "setup_brier": setup_brier,
        "failure_brier": failure_brier,
        "portfolio_rows": not portfolio.empty,
        "portfolio_table": portfolio_md,
        "portfolio_html": portfolio_html,
        "scored_states": _SCORED_STATES,
        "unscored_states": _UNSCORED_STATES,
        "state_sections": state_sections_md,
        "state_html": state_sections_html,
        "state_counts": state_counts,
        "skipped_rows": skipped_records,
    }


def _get_env() -> Environment:
    if _TEMPLATES_DIR.exists():
        return Environment(loader=FileSystemLoader(str(_TEMPLATES_DIR)), autoescape=False)
    # Fallback: inline minimal templates
    return Environment(loader=None, autoescape=False)


def render_daily_markdown(
    snapshot_df: pd.DataFrame,
    scores_df: pd.DataFrame | None = None,
    as_of: pd.Timestamp | None = None,
    oos_metrics: dict | None = None,
) -> str:
    """Render the daily snapshot as a Markdown string."""
    as_of = as_of or pd.Timestamp.today().normalize()
    ctx = _build_context(snapshot_df, scores_df, as_of, oos_metrics)

    env = _get_env()
    try:
        tmpl = env.get_template("snapshot.md.j2")
        return tmpl.render(**ctx)
    except Exception as exc:
        log.warning("Jinja2 template error (%s); using fallback markdown renderer", exc)
        return _fallback_markdown(ctx)


def render_daily_html(
    snapshot_df: pd.DataFrame,
    scores_df: pd.DataFrame | None = None,
    as_of: pd.Timestamp | None = None,
    oos_metrics: dict | None = None,
) -> str:
    """Render the daily snapshot as an HTML string."""
    as_of = as_of or pd.Timestamp.today().normalize()
    ctx = _build_context(snapshot_df, scores_df, as_of, oos_metrics)

    env = _get_env()
    try:
        tmpl = env.get_template("snapshot.html.j2")
        return tmpl.render(**ctx)
    except Exception as exc:
        log.warning("Jinja2 HTML template error (%s); skipping HTML output", exc)
        return f"<html><body><pre>Error rendering HTML: {exc}</pre></body></html>"


def write_daily_reports(
    snapshot_df: pd.DataFrame,
    scores_df: pd.DataFrame | None,
    as_of: pd.Timestamp,
    *,
    output_dir: Path,
    oos_metrics: dict | None = None,
) -> dict[str, Path]:
    """Write both markdown and HTML report files to output_dir."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}

    md = render_daily_markdown(snapshot_df, scores_df, as_of, oos_metrics)
    md_path = output_dir / "snapshot.md"
    md_path.write_text(md, encoding="utf-8")
    paths["markdown"] = md_path

    html = render_daily_html(snapshot_df, scores_df, as_of, oos_metrics)
    html_path = output_dir / "snapshot.html"
    html_path.write_text(html, encoding="utf-8")
    paths["html"] = html_path

    log.info("Reports written: %s, %s", md_path, html_path)
    return paths


# ── Fallback (no templates directory) ────────────────────────────────────────

def _fallback_markdown(ctx: dict) -> str:
    lines = [f"# Daily Snapshot — {ctx['as_of']}", ""]
    lines += ["> Survivorship bias present. See docs.", ""]
    if ctx["portfolio_table"]:
        lines += ["## Portfolio", "", ctx["portfolio_table"], ""]
    for st in ctx["scored_states"] + ctx["unscored_states"]:
        if ctx["state_sections"].get(st):
            lines += [f"## {st} ({ctx['state_counts'].get(st, 0)})", "", ctx["state_sections"][st], ""]
    lines += [f"*Generated {ctx['generated_at']} UTC*"]
    return "\n".join(lines)
