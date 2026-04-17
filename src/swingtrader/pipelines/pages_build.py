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
    """Return sorted list of YYYY-MM-DD strings that have a snapshot.html."""
    if not _REPORTS_DIR.exists():
        return []
    days = sorted(
        d.name for d in _REPORTS_DIR.iterdir()
        if d.is_dir() and (d / "snapshot.html").exists()
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


def _history_links_html(days: list[str]) -> str:
    if not days:
        return "<p><em>No daily reports yet.</em></p>"
    items = []
    for d in reversed(days):
        href = f"reports/daily/{d}/snapshot.html"
        items.append(f'<li><a href="{href}">{d}</a></li>')
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

<h2>Latest Scores &mdash; {latest_day}</h2>
<p>Top setups by composite score (BASE/ARMED/TRIGGERED/ACCEPTED states only).
   Score = calibrated model output &mdash; not a buy recommendation.</p>
{scores_table}

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
