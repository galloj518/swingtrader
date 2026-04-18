"""GitHub Pages static site builder.

Generates docs/index.html — a rolling table of daily snapshot links with the
latest scored ranking inline.

GitHub Pages serves from the `docs/` folder on the `main` branch (configured in
repo Settings → Pages → Source: main / docs).

The generated site is intentionally minimal:
  - No build tools, no node, no bundlers
  - Pure HTML + inline CSS
  - Links to per-day snapshot.html reports already written by score_run

Run:
  python -m swingtrader.pipelines.pages_build
  (or called automatically at the end of score_run)
"""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from swingtrader.utils.config import REPO_ROOT
from swingtrader.utils.logging import get_logger

log = get_logger(__name__)

_DOCS_DIR = REPO_ROOT / "docs"
_REPORTS_DIR = REPO_ROOT / "docs" / "reports" / "daily"
_SCORES_DIR = REPO_ROOT / "data" / "scores"
_INDEX_PATH = _DOCS_DIR / "index.html"

# How many days of history to show in the index
_MAX_HISTORY_DAYS = 30


def _read_latest_scores() -> pd.DataFrame:
    """Load the latest scored snapshot."""
    latest = _SCORES_DIR / "latest.parquet"
    if not latest.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(latest)
    except Exception:
        return pd.DataFrame()


def _list_report_days() -> list[str]:
    """Return sorted list of YYYY-MM-DD strings that have a dashboard.html or snapshot.html."""
    if not _REPORTS_DIR.exists():
        return []
    days = sorted(
        d.name for d in _REPORTS_DIR.iterdir()
        if d.is_dir() and (
            (d / "dashboard.html").exists() or (d / "snapshot.html").exists()
        )
    )
    return days[-_MAX_HISTORY_DAYS:]


def _scores_table_html(scores_df: pd.DataFrame, max_rows: int = 50) -> str:
    """Render top-N scored rows (ARMED + BASE + TRIGGERED + ACCEPTED) as HTML."""
    if scores_df.empty:
        return "<p><em>No scores available yet. Run the scoring pipeline first.</em></p>"

    cols_to_show = [c for c in [
        "symbol", "state", "close", "pivot",
        "composite_score", "percentile_rank",
        "setup_score", "failure_risk", "trade_score",
        "dist_to_pivot_atr", "atr_compression_pct",
    ] if c in scores_df.columns]

    # Filter to scored states only
    scored_states = {"BASE", "ARMED", "TRIGGERED", "ACCEPTED"}
    if "state" in scores_df.columns:
        display = scores_df[scores_df["state"].isin(scored_states)].copy()
    else:
        display = scores_df.copy()

    if "percentile_rank" in display.columns:
        display = display.sort_values("percentile_rank", ascending=False)
    display = display.head(max_rows)[cols_to_show]

    def _fmt(v) -> str:
        if v is None:
            return "—"
        if isinstance(v, float):
            import math
            if math.isnan(v) or math.isinf(v):
                return "—"
            return f"{v:.3f}"
        if isinstance(v, bool):
            return "Y" if v else "N"
        return str(v)

    def _score_cls(v) -> str:
        if not isinstance(v, float):
            return ""
        import math
        if math.isnan(v):
            return ""
        return "score-high" if v >= 0.6 else ("score-mid" if v >= 0.3 else "score-low")

    header = "".join(f"<th>{c}</th>" for c in display.columns)
    rows_html = []
    for _, row in display.iterrows():
        cells = []
        for c in display.columns:
            v = row[c]
            css = ""
            if c == "composite_score":
                css = _score_cls(v)
            elif c == "state":
                css = f"state-{v}"
            cells.append(f'<td class="{css}">{_fmt(v)}</td>' if css else f"<td>{_fmt(v)}</td>")
        rows_html.append("<tr>" + "".join(cells) + "</tr>")

    return f"<table><tr>{header}</tr>{''.join(rows_html)}</table>"


def _regime_context_html(scores_df: pd.DataFrame) -> str:
    """Build a compact market context bar from latest scores."""
    if scores_df.empty:
        return ""

    import math

    def _first_val(col: str) -> float:
        if col not in scores_df.columns:
            return math.nan
        col_data = scores_df[col].dropna()
        if col_data.empty:
            return math.nan
        try:
            return float(col_data.iloc[0])
        except (TypeError, ValueError):
            return math.nan

    spy_trend    = _first_val("regime_spy_trend")
    spy_above    = _first_val("regime_spy_above_200sma")
    vix_level    = _first_val("regime_vix_level")

    # SPY trend
    if math.isfinite(spy_trend):
        trend_val = "Uptrend" if spy_trend > 0 else ("Downtrend" if spy_trend < 0 else "Neutral")
        trend_color = "#3fb950" if spy_trend > 0 else ("#f78166" if spy_trend < 0 else "#d29922")
    else:
        trend_val, trend_color = "—", "#6e7681"

    # SPY vs 200SMA
    if math.isfinite(spy_above):
        above_val = "Above 200SMA" if spy_above >= 0.5 else "Below 200SMA"
        above_color = "#3fb950" if spy_above >= 0.5 else "#f78166"
    else:
        above_val, above_color = "—", "#6e7681"

    # VIX
    if math.isfinite(vix_level):
        if vix_level < 15:
            vix_val, vix_color = f"VIX {vix_level:.0f} (low)", "#3fb950"
            env_val, env_color = "Favors breakouts", "#3fb950"
        elif vix_level < 20:
            vix_val, vix_color = f"VIX {vix_level:.0f} (neutral)", "#c9d1d9"
            env_val, env_color = "Selective", "#d29922"
        elif vix_level < 30:
            vix_val, vix_color = f"VIX {vix_level:.0f} (elevated)", "#d29922"
            env_val, env_color = "Cautious", "#d29922"
        else:
            vix_val, vix_color = f"VIX {vix_level:.0f} (high)", "#f78166"
            env_val, env_color = "Risk-off", "#f78166"
    else:
        vix_val, vix_color = "VIX —", "#6e7681"
        env_val, env_color = "—", "#6e7681"

    # Universe counts
    actionable_n = 0
    armed_n = 0
    if "action_label" in scores_df.columns:
        actionable_n = int(scores_df["action_label"].isin(
            ["Actionable now", "Actionable on breakout", "Actionable on pullback"]
        ).sum())
    if "state" in scores_df.columns:
        armed_n = int((scores_df["state"] == "ARMED").sum())

    def _pill(label, value, color):
        return (
            f'<div style="display:flex;flex-direction:column;align-items:center;min-width:80px">'
            f'<span style="font-size:.68rem;color:#6e7681;text-transform:uppercase;letter-spacing:.05em">{label}</span>'
            f'<span style="font-size:.95rem;font-weight:600;color:{color}">{value}</span>'
            f'</div>'
        )

    return (
        f'<div style="display:flex;gap:.8rem;flex-wrap:wrap;background:#161b22;'
        f'border:1px solid #30363d;border-radius:6px;padding:.6rem 1rem;margin-bottom:1rem">'
        f'{_pill("SPY Trend", trend_val, trend_color)}'
        f'{_pill("vs 200SMA", above_val, above_color)}'
        f'{_pill("Volatility", vix_val, vix_color)}'
        f'{_pill("Environment", env_val, env_color)}'
        f'{_pill("Actionable", str(actionable_n), "#58a6ff")}'
        f'{_pill("ARMED", str(armed_n), "#3fb950")}'
        f'</div>'
    )


def _top_setups_preview_html(scores_df: pd.DataFrame, latest_day: str) -> str:
    """Build a compact top-setups preview strip for the index page."""
    if scores_df.empty:
        return ""

    import math

    # Get top setups — action-labelled, scored, not avoid
    needed_cols = {"state", "composite_score", "percentile_rank"}
    if not needed_cols.issubset(set(scores_df.columns)):
        return ""

    scored_states = {"BASE", "ARMED", "TRIGGERED", "ACCEPTED"}
    df = scores_df[scores_df["state"].isin(scored_states)].copy()
    if df.empty:
        return ""

    if "action_label" in df.columns:
        df = df[df["action_label"] != "Avoid / low quality"]

    if "percentile_rank" in df.columns:
        df = df.sort_values("percentile_rank", ascending=False)
    elif "composite_score" in df.columns:
        df = df.sort_values("composite_score", ascending=False)

    df = df.head(7)
    if df.empty:
        return ""

    sym_col = next((c for c in ("user_symbol", "symbol") if c in df.columns), None)
    if sym_col is None:
        return ""

    # Action label color map
    action_colors = {
        "Actionable now":          "#3fb950",
        "Actionable on breakout":  "#58a6ff",
        "Actionable on pullback":  "#d29922",
        "Extended, wait":          "#6e7681",
    }
    state_colors = {
        "ARMED":     "#3fb950",
        "TRIGGERED": "#f78166",
        "ACCEPTED":  "#58a6ff",
        "BASE":      "#8b949e",
    }

    chips = []
    for _, row in df.iterrows():
        sym = str(row.get(sym_col, "?"))
        state = str(row.get("state", ""))
        action = str(row.get("action_label", "")) if "action_label" in df.columns else ""
        sc = row.get("composite_score", math.nan)
        setup_cls = str(row.get("setup_classification", "")) if "setup_classification" in df.columns else ""
        close_val = row.get("close", math.nan)

        try:
            sc_str = f"{float(sc):.2f}" if math.isfinite(float(sc)) else "—"
        except (TypeError, ValueError):
            sc_str = "—"
        try:
            close_str = f"${float(close_val):.2f}" if math.isfinite(float(close_val)) else ""
        except (TypeError, ValueError):
            close_str = ""

        s_color = state_colors.get(state, "#8b949e")
        a_color = action_colors.get(action, "#8b949e")

        # Dashboard link for this symbol's day
        detail_href = f"reports/daily/{latest_day}/dashboard.html#{sym}" if latest_day else "#"

        setup_cls_div = (
            f'<div style="font-size:.7rem;color:#8b949e">{setup_cls[:28]}</div>'
            if setup_cls else ""
        )
        close_div = (
            f'<div style="font-size:.7rem;color:#6e7681">{close_str}</div>'
            if close_str else ""
        )
        chip = (
            f'<a href="{detail_href}" style="text-decoration:none">'
            f'<div style="background:#161b22;border:1px solid #30363d;border-radius:6px;'
            f'padding:.5rem .7rem;min-width:130px;cursor:pointer;'
            f'transition:border-color .15s" '
            f'onmouseover="this.style.borderColor=\'#58a6ff\'" '
            f'onmouseout="this.style.borderColor=\'#30363d\'">'
            f'<div style="font-weight:700;font-size:.95rem;color:#c9d1d9">{sym}</div>'
            f'<div style="font-size:.75rem;margin:.15rem 0">'
            f'<span style="color:{s_color}">{state}</span>'
            f'</div>'
            f'{setup_cls_div}'
            f'<div style="display:flex;justify-content:space-between;margin-top:.25rem">'
            f'<span style="font-size:.72rem;color:{a_color}">{action[:20] if action else ""}</span>'
            f'<span style="font-size:.72rem;color:#8b949e">{sc_str}</span>'
            f'</div>'
            f'{close_div}'
            f'</div></a>'
        )
        chips.append(chip)

    return (
        f'<div style="margin-bottom:1rem">'
        f'<strong style="font-size:.85rem;color:#8b949e">Top Setups — {latest_day}</strong>'
        f'<div style="display:flex;gap:.5rem;flex-wrap:wrap;margin-top:.4rem">'
        f'{"".join(chips)}'
        f'</div>'
        f'</div>'
    )


def _artifacts_links_html(latest_day: str) -> str:
    """Build a compact artifact/export links section."""
    if not latest_day:
        return ""
    base = f"reports/daily/{latest_day}/artifacts"
    return (
        f'<div style="font-size:.8rem;color:#8b949e;margin-bottom:.8rem">'
        f'<strong>Today\'s machine-readable outputs:</strong> '
        f'<a href="{base}/dashboard_summary.json">summary.json</a> &nbsp;|&nbsp; '
        f'<a href="{base}/top_setups.json">top_setups.json</a> &nbsp;|&nbsp; '
        f'<a href="{base}/portfolio_review.json">portfolio_review.json</a>'
        f'</div>'
    )


def _history_links_html(days: list[str]) -> str:
    if not days:
        return "<p><em>No daily reports yet.</em></p>"
    items = []
    for d in reversed(days):
        day_dir = _REPORTS_DIR / d
        # Prefer dashboard.html (trader view); fall back to snapshot.html (research view)
        if (day_dir / "dashboard.html").exists():
            href = f"reports/daily/{d}/dashboard.html"
            label = f"{d} <small style='color:#8b949e'>[dashboard]</small>"
        else:
            href = f"reports/daily/{d}/snapshot.html"
            label = f"{d} <small style='color:#8b949e'>[snapshot]</small>"
        items.append(f'<li><a href="{href}">{label}</a></li>')
    return "<ul>" + "".join(items) + "</ul>"


_PAGE_STYLE = """
body { font-family: system-ui, sans-serif; margin: 0; padding: 1rem 2rem;
       background: #0d1117; color: #c9d1d9; }
h1 { color: #58a6ff; border-bottom: 1px solid #30363d; padding-bottom: .4rem; }
h2 { color: #8b949e; margin-top: 2rem; }
a { color: #58a6ff; }
.bias { background: #161b22; border-left: 3px solid #d29922; padding: .5rem 1rem;
        border-radius: 4px; font-size: .88rem; color: #8b949e; margin-bottom: 1rem; }
table { border-collapse: collapse; width: 100%; font-size: .83rem; }
th { background: #161b22; color: #58a6ff; text-align: left; padding: .35rem .5rem;
     border-bottom: 1px solid #30363d; }
td { padding: .3rem .5rem; border-bottom: 1px solid #21262d; }
tr:hover td { background: #161b22; }
.state-ARMED { color: #3fb950; font-weight: bold; }
.state-BASE { color: #8b949e; }
.state-TRIGGERED { color: #f78166; font-weight: bold; }
.state-ACCEPTED { color: #58a6ff; font-weight: bold; }
.state-CONFIRMED { color: #bc8cff; font-weight: bold; }
.state-FAILED { color: #6e7681; text-decoration: line-through; }
.state-LATE { color: #d29922; }
.score-high { color: #3fb950; font-weight: bold; }
.score-mid { color: #d29922; }
.score-low { color: #f78166; }
footer { margin-top: 2rem; font-size: .78rem; color: #6e7681;
         border-top: 1px solid #21262d; padding-top: .5rem; }
ul { padding-left: 1.2rem; }
li { margin: .2rem 0; }
"""


def build_index(
    docs_dir: Path | None = None,
    reports_dir: Path | None = None,
    scores_dir: Path | None = None,
    as_of: pd.Timestamp | None = None,
) -> Path:
    """Generate docs/index.html and return its path."""
    docs_dir = Path(docs_dir or _DOCS_DIR)
    docs_dir.mkdir(parents=True, exist_ok=True)

    scores_df = _read_latest_scores()
    days = _list_report_days()
    latest_day = days[-1] if days else (str(as_of.date()) if as_of else "—")

    scores_table = _scores_table_html(scores_df)
    history_links = _history_links_html(days)
    generated_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")

    # Market context bar (regime pills)
    regime_html = _regime_context_html(scores_df)

    # Top setups preview
    top_setups_preview = _top_setups_preview_html(scores_df, latest_day)

    # Artifact links
    artifacts_links = _artifacts_links_html(latest_day)

    # Latest dashboard link
    latest_link = ""
    if days:
        ld = days[-1]
        day_dir = _REPORTS_DIR / ld
        if (day_dir / "dashboard.html").exists():
            latest_link = (
                f'<p style="margin-bottom:.8rem">'
                f'<a href="reports/daily/{ld}/dashboard.html" '
                f'style="font-size:1rem;font-weight:600">'
                f'→ Open today\'s dashboard ({ld})</a></p>'
            )
        elif (day_dir / "snapshot.html").exists():
            latest_link = (
                f'<p style="margin-bottom:.8rem">'
                f'<a href="reports/daily/{ld}/snapshot.html">→ Open today\'s report ({ld})</a></p>'
            )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Swingtrader Research</title>
<style>{_PAGE_STYLE}</style>
</head>
<body>
<h1>Swingtrader Research</h1>

<div class="bias">
  <strong>Bias caveat:</strong> Universe membership uses monthly snapshots, not true point-in-time history.
  Survivorship bias is present. This is a personal research tool, not investment advice.
  No positions are taken automatically. All scores are exploratory.
</div>

{latest_link}
{regime_html}
{top_setups_preview}
{artifacts_links}

<details>
<summary style="font-size:.9rem;color:#8b949e;cursor:pointer;padding:.3rem 0">Full Score Table — {latest_day}</summary>
<p style="font-size:.8rem;color:#6e7681;margin:.3rem 0 .4rem">
  All BASE/ARMED/TRIGGERED/ACCEPTED symbols sorted by percentile rank.
</p>
{scores_table}
</details>

<h2>Daily Report Archive</h2>
{history_links}

<footer>Updated {generated_at} | <a href="https://github.com">Source</a></footer>
</body>
</html>"""

    index_path = docs_dir / "index.html"
    index_path.write_text(html, encoding="utf-8")
    log.info("Pages index written → %s", index_path)
    return index_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build GitHub Pages index")
    parser.add_argument("date", nargs="?", default=None, help="As-of date YYYY-MM-DD")
    args = parser.parse_args()
    as_of = pd.Timestamp(args.date) if args.date else pd.Timestamp.today().normalize()
    path = build_index(as_of=as_of)
    print(f"Pages index written: {path}")


if __name__ == "__main__":
    main()
