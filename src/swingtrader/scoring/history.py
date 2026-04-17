"""Score history: append daily scores to a rolling parquet file.

schema: (date, symbol, state, composite_score, percentile_rank, setup_score,
          trade_score, failure_risk)

The history file enables:
  - Trend analysis: is a symbol's composite_score rising or falling?
  - Cohort analysis: how did symbols that were ARMED on date X perform?
  - Calibration monitoring: do high composite_scores actually break out more?

Append strategy: read existing, drop any rows for `as_of` (idempotent), append
new rows, write back.  This makes it safe to re-run score_run for the same date.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from swingtrader.utils.config import REPO_ROOT
from swingtrader.utils.logging import get_logger

log = get_logger(__name__)

_HISTORY_PATH = REPO_ROOT / "data" / "score_history" / "history.parquet"

_SCORE_COLS = [
    "date",
    "symbol",
    "state",
    "composite_score",
    "percentile_rank",
    "setup_score",
    "trade_score",
    "failure_risk",
]


def append_daily_scores(
    scores_df: pd.DataFrame,
    as_of: pd.Timestamp,
    path: Path | None = None,
) -> None:
    """Append today's scores to the rolling history parquet.

    scores_df is the output of score_all_symbols() (index = symbol, cols include
    state, composite_score, percentile_rank, etc.).
    """
    path = Path(path or _HISTORY_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)

    today_str = str(as_of.date())

    # Build today's rows
    rows = scores_df.copy().reset_index()
    rows.rename(columns={"index": "symbol"}, inplace=True)
    if "symbol" not in rows.columns and scores_df.index.name == "symbol":
        rows = scores_df.reset_index()
    rows["date"] = today_str

    available_cols = [c for c in _SCORE_COLS if c in rows.columns]
    rows = rows[available_cols]

    # Load existing and drop any existing rows for today (idempotent)
    if path.exists():
        try:
            existing = pd.read_parquet(path)
            existing = existing[existing["date"] != today_str]
        except Exception as exc:
            log.warning("score_history: could not read existing file (%s) — starting fresh", exc)
            existing = pd.DataFrame(columns=available_cols)
    else:
        existing = pd.DataFrame(columns=available_cols)

    combined = pd.concat([existing, rows], ignore_index=True)
    combined.to_parquet(path, index=False)
    log.info("score_history: appended %d rows for %s (total %d)", len(rows), today_str, len(combined))


def load_history(path: Path | None = None) -> pd.DataFrame:
    """Load the full score history."""
    path = Path(path or _HISTORY_PATH)
    if not path.exists():
        return pd.DataFrame(columns=_SCORE_COLS)
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["date", "symbol"])


def score_trend(
    symbol: str,
    history: pd.DataFrame | None = None,
    lookback_days: int = 20,
) -> dict:
    """Recent composite_score trend for a single symbol.

    Returns dict with: latest_score, score_20d_ago, slope (linear fit),
    days_consecutive_rising, days_above_50pct.
    """
    if history is None:
        history = load_history()

    sym_hist = history[history["symbol"] == symbol].sort_values("date")
    if sym_hist.empty:
        return {"symbol": symbol, "n_days": 0}

    recent = sym_hist.tail(lookback_days)
    scores = pd.to_numeric(recent["composite_score"], errors="coerce").dropna()
    if scores.empty:
        return {"symbol": symbol, "n_days": 0}

    # Linear slope (normalised by mean)
    if len(scores) >= 3:
        xs = range(len(scores))
        slope = float(pd.Series(list(xs)).cov(scores) / max(pd.Series(list(xs)).var(), 1e-9))
        mean_score = float(scores.mean())
        slope_norm = slope / mean_score if mean_score > 0 else slope
    else:
        slope_norm = float("nan")

    # Consecutive rising bars
    diffs = scores.diff().dropna()
    consec = 0
    for v in reversed(diffs.tolist()):
        if v > 0:
            consec += 1
        else:
            break

    return {
        "symbol": symbol,
        "n_days": len(scores),
        "latest_score": round(float(scores.iloc[-1]), 4),
        "score_20d_ago": round(float(scores.iloc[0]), 4),
        "slope_norm": round(slope_norm, 6) if isinstance(slope_norm, float) else float("nan"),
        "days_consecutive_rising": consec,
        "days_above_50pct": int((scores > 0.5).sum()),
    }
