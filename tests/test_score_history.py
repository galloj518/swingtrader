"""Tests for score history append/load and score_trend."""
from __future__ import annotations

import numpy as np
import pandas as pd

from swingtrader.scoring.history import append_daily_scores, load_history, score_trend


def _make_scores_df(syms=("AAPL", "MSFT", "NVDA"), seed=3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(syms)
    return pd.DataFrame({
        "state": rng.choice(["BASE", "ARMED", "TRIGGERED"], n),
        "composite_score": rng.uniform(0, 1, n),
        "percentile_rank": rng.uniform(0, 100, n),
        "setup_score": rng.uniform(0, 1, n),
        "trade_score": rng.normal(0, 1, n),
        "failure_risk": rng.uniform(0, 1, n),
    }, index=pd.Index(list(syms), name="symbol"))


def test_append_and_load_roundtrip(tmp_path) -> None:
    path = tmp_path / "history.parquet"
    scores = _make_scores_df()
    as_of = pd.Timestamp("2024-06-01")
    append_daily_scores(scores, as_of, path)
    hist = load_history(path)
    assert len(hist) == len(scores)
    assert "date" in hist.columns
    assert "symbol" in hist.columns


def test_append_idempotent(tmp_path) -> None:
    """Running append twice for the same date replaces, not duplicates."""
    path = tmp_path / "history.parquet"
    scores = _make_scores_df()
    as_of = pd.Timestamp("2024-06-01")
    append_daily_scores(scores, as_of, path)
    append_daily_scores(scores, as_of, path)
    hist = load_history(path)
    assert len(hist) == len(scores)


def test_append_accumulates_across_dates(tmp_path) -> None:
    path = tmp_path / "history.parquet"
    scores = _make_scores_df()
    for d in ["2024-06-01", "2024-06-02", "2024-06-03"]:
        append_daily_scores(scores, pd.Timestamp(d), path)
    hist = load_history(path)
    assert len(hist) == len(scores) * 3


def test_load_history_empty_when_no_file(tmp_path) -> None:
    hist = load_history(tmp_path / "nonexistent.parquet")
    assert isinstance(hist, pd.DataFrame)
    assert hist.empty


def test_load_history_sorted_by_date(tmp_path) -> None:
    path = tmp_path / "history.parquet"
    scores = _make_scores_df()
    for d in ["2024-06-03", "2024-06-01", "2024-06-02"]:
        append_daily_scores(scores, pd.Timestamp(d), path)
    hist = load_history(path)
    dates = hist["date"].tolist()
    assert dates == sorted(dates)


def test_score_trend_basic(tmp_path) -> None:
    path = tmp_path / "history.parquet"
    scores = _make_scores_df(("AAPL",), seed=1)
    for d in pd.bdate_range("2024-01-01", periods=25):
        # Gradually increasing score
        sc = scores.copy()
        sc["composite_score"] = 0.3 + (d - pd.Timestamp("2024-01-01")).days * 0.005
        append_daily_scores(sc, d, path)
    hist = load_history(path)
    trend = score_trend("AAPL", hist, lookback_days=20)
    assert trend["n_days"] >= 1
    assert "latest_score" in trend
    assert "slope_norm" in trend


def test_score_trend_symbol_not_in_history(tmp_path) -> None:
    path = tmp_path / "history.parquet"
    scores = _make_scores_df(("AAPL",))
    append_daily_scores(scores, pd.Timestamp("2024-06-01"), path)
    hist = load_history(path)
    trend = score_trend("ZZZZ", hist)
    assert trend["n_days"] == 0
