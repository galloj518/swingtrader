"""Regime / market-context features.

These are computed once per day for the benchmark and broadcast to every symbol.
They are appended as constant columns to the symbol feature DataFrames.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from swingtrader.features.primitives import sma
from swingtrader.features.registry import register
from swingtrader.utils.config import load_config


@register("regime_spy_above_200sma", timeframe="daily", lookback_bars=200)
def regime_spy_above_200sma(df: pd.DataFrame, *, benchmark_df: pd.DataFrame | None = None, **_) -> pd.Series:
    """1 if benchmark (SPY) close is above its 200-SMA, 0 otherwise."""
    if benchmark_df is None or benchmark_df.empty:
        return pd.Series(np.nan, index=df.index)
    s200 = sma(benchmark_df["close"], 200)
    above = (benchmark_df["close"] > s200).astype(float)
    return above.reindex(df.index, method="ffill")


@register("regime_spy_trend", timeframe="daily", lookback_bars=200)
def regime_spy_trend(df: pd.DataFrame, *, benchmark_df: pd.DataFrame | None = None, **_) -> pd.Series:
    """Benchmark trend regime: 1=uptrend (close>200sma AND 50sma>200sma), -1=downtrend, 0=neutral."""
    if benchmark_df is None or benchmark_df.empty:
        return pd.Series(np.nan, index=df.index)
    c = benchmark_df["close"]
    s50 = sma(c, 50)
    s200 = sma(c, 200)
    regime = pd.Series(0.0, index=benchmark_df.index)
    regime[(c > s200) & (s50 > s200)] = 1.0
    regime[(c < s200) & (s50 < s200)] = -1.0
    return regime.reindex(df.index, method="ffill")


@register("regime_vix_level", timeframe="daily", lookback_bars=1)
def regime_vix_level(df: pd.DataFrame, *, vix_df: pd.DataFrame | None = None, **_) -> pd.Series:
    """VIX close level (0=low if <15, 1=mid if 15-25, 2=high if >25).

    vix_df is the VIX OHLCV DataFrame (^VIX from yfinance).
    When absent, returns NaN without error.
    """
    if vix_df is None or vix_df.empty:
        return pd.Series(np.nan, index=df.index)
    cfg = load_config("regimes")
    buckets = cfg["volatility_regime"]["buckets"]
    low_max = float(buckets["low"].get("max", 15))
    mid_max = float(buckets["mid"].get("max", 25))
    vix = vix_df["close"].reindex(df.index, method="ffill")
    result = pd.Series(2.0, index=df.index)  # default: high
    result[vix <= mid_max] = 1.0
    result[vix <= low_max] = 0.0
    return result


@register("regime_weekly_spy_above_200sma", timeframe="weekly", lookback_bars=200)
def regime_weekly_spy_above_200sma(df: pd.DataFrame, *, benchmark_df: pd.DataFrame | None = None, **_) -> pd.Series:
    """Weekly version: 1 if SPY weekly close > 200-week SMA."""
    if benchmark_df is None or benchmark_df.empty:
        return pd.Series(np.nan, index=df.index)
    s200 = sma(benchmark_df["close"], 200)
    above = (benchmark_df["close"] > s200).astype(float)
    return above.reindex(df.index, method="ffill")
