"""Backtest performance metrics derived from a replay trade log.

All metrics are trade-level (not bar-level equity curve), which avoids
the need for a position-sizing assumption. The inputs are the pnl_atr
column of a replay_universe() output DataFrame.

These metrics intentionally do NOT produce a time-weighted return or
compound equity curve — that would require a position sizing model that
this codebase deliberately omits.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def trade_level_metrics(pnl_atr: pd.Series | np.ndarray) -> dict:
    """Core trade-level statistics from an array of per-trade ATR P&Ls.

    Parameters
    ----------
    pnl_atr : one value per trade (positive = winner, negative = loser)

    Returns
    -------
    dict with descriptive stats and risk metrics.
    """
    arr = np.asarray(pd.to_numeric(pnl_atr, errors="coerce"), dtype=float)
    arr = arr[np.isfinite(arr)]

    if len(arr) == 0:
        return {"n": 0}

    winners = arr[arr > 0]
    losers = arr[arr <= 0]

    win_rate = len(winners) / len(arr)
    avg_win = float(winners.mean()) if len(winners) > 0 else float("nan")
    avg_loss = float(losers.mean()) if len(losers) > 0 else float("nan")
    gross_profit = float(winners.sum()) if len(winners) > 0 else 0.0
    gross_loss = float(abs(losers.sum())) if len(losers) > 0 else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("nan")
    expectancy = float(arr.mean())
    std = float(arr.std()) if len(arr) > 1 else float("nan")
    sharpe_trade = expectancy / std if np.isfinite(std) and std > 0 else float("nan")

    # Payoff ratio: avg_win / |avg_loss|
    payoff = abs(avg_win / avg_loss) if np.isfinite(avg_loss) and avg_loss != 0 else float("nan")

    # Kelly criterion (simplified): win_rate - (1 - win_rate) / payoff
    kelly = (win_rate - (1.0 - win_rate) / payoff) if np.isfinite(payoff) and payoff > 0 else float("nan")

    # Consecutive loss streaks
    signs = (arr > 0).astype(int)
    max_loss_streak = 0
    streak = 0
    for s in signs:
        if s == 0:
            streak += 1
            max_loss_streak = max(max_loss_streak, streak)
        else:
            streak = 0

    # Running drawdown (in cumulative ATR units)
    cumulative = np.cumsum(arr)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = cumulative - running_max
    max_drawdown_atr = float(drawdown.min())

    return {
        "n": len(arr),
        "n_winners": len(winners),
        "n_losers": len(losers),
        "win_rate": round(win_rate, 4),
        "avg_win_atr": round(avg_win, 4) if np.isfinite(avg_win) else float("nan"),
        "avg_loss_atr": round(avg_loss, 4) if np.isfinite(avg_loss) else float("nan"),
        "payoff_ratio": round(payoff, 4) if np.isfinite(payoff) else float("nan"),
        "expectancy_atr": round(expectancy, 4),
        "profit_factor": round(profit_factor, 4) if np.isfinite(profit_factor) else float("nan"),
        "std_atr": round(std, 4) if np.isfinite(std) else float("nan"),
        "sharpe_trade": round(sharpe_trade, 4) if np.isfinite(sharpe_trade) else float("nan"),
        "kelly_fraction": round(kelly, 4) if np.isfinite(kelly) else float("nan"),
        "max_consecutive_losses": max_loss_streak,
        "max_drawdown_atr": round(max_drawdown_atr, 4),
        "total_pnl_atr": round(float(arr.sum()), 4),
    }


def by_exit_reason(trade_log: pd.DataFrame) -> pd.DataFrame:
    """Per exit-reason breakdown of trade count and mean P&L."""
    if trade_log.empty or "exit_reason" not in trade_log.columns:
        return pd.DataFrame()
    groups = trade_log.groupby("exit_reason")["pnl_atr"].agg(
        count="count",
        mean_pnl_atr="mean",
        sum_pnl_atr="sum",
    ).reset_index()
    return groups


def by_symbol(trade_log: pd.DataFrame) -> pd.DataFrame:
    """Per-symbol trade summary sorted by total ATR P&L descending."""
    if trade_log.empty or "symbol" not in trade_log.columns:
        return pd.DataFrame()
    groups = (
        trade_log.groupby("symbol")["pnl_atr"]
        .agg(n="count", total_pnl_atr="sum", avg_pnl_atr="mean", win_rate=lambda x: (x > 0).mean())
        .reset_index()
        .sort_values("total_pnl_atr", ascending=False)
    )
    return groups


def cumulative_pnl_series(trade_log: pd.DataFrame) -> pd.Series:
    """Running cumulative ATR P&L, indexed by trigger_date.

    Useful for plotting an equity-like curve without position sizing.
    """
    if trade_log.empty:
        return pd.Series(dtype=float)
    log_sorted = trade_log.dropna(subset=["pnl_atr"]).sort_values("trigger_date")
    cum = log_sorted["pnl_atr"].cumsum()
    cum.index = pd.to_datetime(log_sorted["trigger_date"].values)
    return cum


def full_report(trade_log: pd.DataFrame) -> dict:
    """Return a structured dict with all metrics for reporting."""
    if trade_log is None or trade_log.empty:
        return {
            "summary": {"n": 0},
            "by_exit_reason": [],
            "by_symbol": [],
            "survivorship_bias_caveat": True,
        }

    pnl = pd.to_numeric(trade_log.get("pnl_atr", pd.Series()), errors="coerce")
    return {
        "summary": trade_level_metrics(pnl),
        "by_exit_reason": by_exit_reason(trade_log).to_dict("records"),
        "by_symbol": by_symbol(trade_log).head(30).to_dict("records"),
        "survivorship_bias_caveat": True,
    }
