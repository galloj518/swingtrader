"""Tests for backtest replay and metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from swingtrader.backtest.metrics import (
    by_exit_reason,
    by_symbol,
    cumulative_pnl_series,
    full_report,
    trade_level_metrics,
)
from swingtrader.backtest.replay import (
    EXIT_CONFIRMED,
    EXIT_STOP,
    EXIT_TIMEOUT,
    ReplayTrade,
    replay_summary,
    replay_symbol,
)

# ── Synthetic state/price data ────────────────────────────────────────────────

def _make_states(n: int = 60, seed: int = 5) -> pd.DataFrame:
    """Synthetic states DataFrame with a single TRIGGERED event at bar 10."""
    dates = pd.bdate_range("2023-01-01", periods=n)
    states = ["BASE"] * n
    # Trigger at bar 10, ACCEPTED 11-12, CONFIRMED at 13
    states[10] = "TRIGGERED"
    states[11] = "ACCEPTED"
    states[12] = "ACCEPTED"
    states[13] = "CONFIRMED"
    state_changed = [False] * n
    state_changed[10] = True
    state_changed[13] = True

    pivot = 100.0
    atr = 2.0
    df = pd.DataFrame({
        "state": states,
        "state_changed": state_changed,
        "trigger_pivot": [np.nan] * 10 + [pivot] * (n - 10),
        "trigger_atr": [np.nan] * 10 + [atr] * (n - 10),
        "pivot": [pivot] * n,
    }, index=dates)
    return df


def _make_prices(n: int = 60, start: float = 100.0) -> pd.DataFrame:
    """Prices that confirm at bar 13 (above target)."""
    dates = pd.bdate_range("2023-01-01", periods=n)
    close = np.full(n, start)
    close[10] = 101.0    # trigger bar
    close[11] = 102.0
    close[12] = 103.0
    close[13] = 106.0    # confirmed — above trigger_pivot + 2×ATR = 100 + 4 = 104
    return pd.DataFrame({"close": close, "open": close, "high": close + 0.5, "low": close - 0.5}, index=dates)


def _make_stop_prices(n: int = 40, seed: int = 9) -> pd.DataFrame:
    """Prices that drop below stop at bar 15."""
    dates = pd.bdate_range("2023-01-01", periods=n)
    close = np.full(n, 100.0, dtype=float)
    close[10] = 101.0    # trigger
    close[15] = 96.0     # below stop = 100 - 2 = 98
    return pd.DataFrame({"close": close, "open": close, "high": close + 0.5, "low": close - 0.5}, index=dates)


# ── ReplayTrade ────────────────────────────────────────────────────────────────

def test_replay_trade_pnl_pct_winner() -> None:
    t = ReplayTrade(
        symbol="AAPL",
        trigger_date=pd.Timestamp("2023-01-10"),
        entry_price=100.0,
        trigger_pivot=100.0,
        atr_at_trigger=2.0,
        target_price=104.0,
        stop_price=98.0,
        exit_price=106.0,
        exit_reason=EXIT_CONFIRMED,
        exit_date=pd.Timestamp("2023-01-13"),
    )
    assert t.pnl_pct == pytest.approx(0.06, abs=1e-9)
    assert t.pnl_atr == pytest.approx(3.0, abs=1e-9)


def test_replay_trade_pnl_none_without_exit() -> None:
    t = ReplayTrade(
        symbol="X",
        trigger_date=pd.Timestamp("2023-01-01"),
        entry_price=50.0,
        trigger_pivot=50.0,
        atr_at_trigger=1.0,
        target_price=52.0,
        stop_price=49.0,
    )
    assert t.pnl_pct is None
    assert t.pnl_atr is None


def test_replay_trade_to_dict_has_all_keys() -> None:
    t = ReplayTrade(
        symbol="X",
        trigger_date=pd.Timestamp("2023-01-01"),
        entry_price=50.0,
        trigger_pivot=50.0,
        atr_at_trigger=1.0,
        target_price=52.0,
        stop_price=49.0,
    )
    d = t.to_dict()
    for key in ("symbol", "trigger_date", "entry_price", "exit_reason", "pnl_pct", "pnl_atr"):
        assert key in d


# ── replay_symbol ─────────────────────────────────────────────────────────────

def test_replay_symbol_confirmed_exit() -> None:
    states = _make_states()
    prices = _make_prices()
    trades = replay_symbol("SYM", states, prices, atr_stop=1.0, atr_target=2.0, max_bars=20)
    assert len(trades) >= 1
    confirmed = [t for t in trades if t.exit_reason == EXIT_CONFIRMED]
    assert len(confirmed) >= 1


def test_replay_symbol_stop_exit() -> None:
    # States: TRIGGERED at bar 10, no CONFIRMED — price drops below stop at bar 15
    n = 40
    dates = pd.bdate_range("2023-01-01", periods=n)
    states_list = ["BASE"] * n
    states_list[10] = "TRIGGERED"
    states_df = pd.DataFrame({
        "state": states_list,
        "state_changed": [i == 10 for i in range(n)],
        "trigger_pivot": [np.nan] * 10 + [100.0] * (n - 10),
        "trigger_atr": [np.nan] * 10 + [2.0] * (n - 10),
        "pivot": [100.0] * n,
    }, index=dates)
    close = np.full(n, 101.0, dtype=float)
    close[15] = 96.0   # below stop = 100 - 1.0 × 2.0 = 98
    prices_df = pd.DataFrame({"close": close}, index=dates)
    trades = replay_symbol("SYM", states_df, prices_df, atr_stop=1.0, atr_target=2.0, max_bars=20)
    stop_trades = [t for t in trades if t.exit_reason == EXIT_STOP]
    assert len(stop_trades) >= 1


def test_replay_symbol_timeout() -> None:
    """No CONFIRMED or stop hit — should timeout after max_bars."""
    n = 40
    dates = pd.bdate_range("2023-01-01", periods=n)
    states_data = ["BASE"] * n
    states_data[5] = "TRIGGERED"
    states_df = pd.DataFrame({
        "state": states_data,
        "state_changed": [i == 5 for i in range(n)],
        "trigger_pivot": [np.nan] * 5 + [100.0] * (n - 5),
        "trigger_atr": [np.nan] * 5 + [2.0] * (n - 5),
        "pivot": [100.0] * n,
    }, index=dates)
    prices_df = pd.DataFrame({"close": [101.0] * n}, index=dates)
    trades = replay_symbol("SYM", states_df, prices_df, atr_stop=1.0, max_bars=10)
    assert any(t.exit_reason == EXIT_TIMEOUT for t in trades)


def test_replay_symbol_empty_states_returns_empty() -> None:
    trades = replay_symbol("SYM", pd.DataFrame(), pd.DataFrame())
    assert trades == []


def test_replay_symbol_no_trigger_returns_empty() -> None:
    n = 30
    dates = pd.bdate_range("2023-01-01", periods=n)
    states_df = pd.DataFrame({"state": ["BASE"] * n, "state_changed": [False] * n}, index=dates)
    prices_df = pd.DataFrame({"close": [100.0] * n}, index=dates)
    trades = replay_symbol("SYM", states_df, prices_df)
    assert trades == []


# ── replay_summary ────────────────────────────────────────────────────────────

def _make_trade_log(n: int = 10, win_rate: float = 0.6) -> pd.DataFrame:
    winners = int(n * win_rate)
    pnl = [2.0] * winners + [-1.0] * (n - winners)
    pnl_pct = [p / 100.0 for p in pnl]
    reasons = [EXIT_CONFIRMED] * winners + [EXIT_STOP] * (n - winners)
    syms = [f"SYM{i % 5}" for i in range(n)]
    dates = pd.bdate_range("2023-01-01", periods=n)
    return pd.DataFrame({
        "symbol": syms,
        "trigger_date": dates,
        "entry_price": 100.0,
        "exit_reason": reasons,
        "pnl_atr": pnl,
        "pnl_pct": pnl_pct,
    })


def test_replay_summary_win_rate() -> None:
    log = _make_trade_log(10, 0.6)
    summ = replay_summary(log)
    assert summ["win_rate"] == pytest.approx(0.6, abs=1e-4)


def test_replay_summary_always_has_caveat() -> None:
    summ = replay_summary(_make_trade_log(5))
    assert summ["survivorship_bias_caveat"] is True


def test_replay_summary_empty_log() -> None:
    summ = replay_summary(pd.DataFrame())
    assert summ["n_trades"] == 0


# ── trade_level_metrics ───────────────────────────────────────────────────────

def test_trade_level_metrics_empty() -> None:
    m = trade_level_metrics(pd.Series([], dtype=float))
    assert m["n"] == 0


def test_trade_level_metrics_all_winners() -> None:
    m = trade_level_metrics(pd.Series([2.0, 3.0, 1.5]))
    assert m["win_rate"] == pytest.approx(1.0)
    assert m["n_losers"] == 0
    assert m["expectancy_atr"] == pytest.approx(2.1667, abs=0.001)


def test_trade_level_metrics_profit_factor() -> None:
    # 2 winners at 3.0, 1 loser at -1.0 → PF = 6/1 = 6
    m = trade_level_metrics(pd.Series([3.0, 3.0, -1.0]))
    assert m["profit_factor"] == pytest.approx(6.0, abs=1e-4)


def test_trade_level_metrics_max_drawdown_negative() -> None:
    m = trade_level_metrics(pd.Series([1.0, -2.0, -1.0, 3.0]))
    assert m["max_drawdown_atr"] < 0


def test_trade_level_metrics_kelly_fraction_positive_for_good_system() -> None:
    # Win rate 60%, avg win 2, avg loss -1 → positive Kelly
    arr = pd.Series([2.0] * 6 + [-1.0] * 4)
    m = trade_level_metrics(arr)
    assert m["kelly_fraction"] > 0


def test_by_exit_reason_groupby() -> None:
    log = _make_trade_log(10, 0.6)
    df = by_exit_reason(log)
    assert "exit_reason" in df.columns
    assert "count" in df.columns


def test_by_symbol_sorted_descending() -> None:
    log = _make_trade_log(20, 0.7)
    df = by_symbol(log)
    totals = df["total_pnl_atr"].tolist()
    assert totals == sorted(totals, reverse=True)


def test_cumulative_pnl_series_length() -> None:
    log = _make_trade_log(10)
    cum = cumulative_pnl_series(log)
    assert len(cum) == 10


def test_cumulative_pnl_series_monotone_for_all_winners() -> None:
    log = _make_trade_log(10, 1.0)  # all winners
    cum = cumulative_pnl_series(log)
    diffs = cum.diff().dropna()
    assert (diffs >= 0).all()


def test_full_report_has_required_keys() -> None:
    log = _make_trade_log(15, 0.6)
    report = full_report(log)
    assert "summary" in report
    assert "by_exit_reason" in report
    assert "by_symbol" in report
    assert report["survivorship_bias_caveat"] is True
