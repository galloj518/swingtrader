"""Weekly-bar feature functions.

Input: weekly OHLCV DataFrame (typically produced by ingest.yfinance_source.resample_weekly).
All features follow the same contract as daily features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from swingtrader.features.primitives import atr_wilder, linear_slope, sma
from swingtrader.features.registry import register


@register("weekly_trend_slope_26", timeframe="weekly", lookback_bars=26)
def weekly_trend_slope_26(df: pd.DataFrame, **_) -> pd.Series:
    """OLS slope of weekly close over 26 weeks, normalised by mean close."""
    return linear_slope(df["close"], 26)


@register("weekly_dist_wma10", timeframe="weekly", lookback_bars=10)
def weekly_dist_wma10(df: pd.DataFrame, **_) -> pd.Series:
    """(close / SMA10_weekly) - 1. Positive = above 10-week MA."""
    s = sma(df["close"], 10)
    return df["close"] / s.replace(0, np.nan) - 1


@register("weekly_dist_wma40", timeframe="weekly", lookback_bars=40)
def weekly_dist_wma40(df: pd.DataFrame, **_) -> pd.Series:
    s = sma(df["close"], 40)
    return df["close"] / s.replace(0, np.nan) - 1


@register("weekly_pct_from_52w_high", timeframe="weekly", lookback_bars=52)
def weekly_pct_from_52w_high(df: pd.DataFrame, **_) -> pd.Series:
    h52 = df["high"].rolling(52, min_periods=20).max()
    return df["close"] / h52.replace(0, np.nan) - 1


@register("weekly_atr_compression", timeframe="weekly", lookback_bars=40)
def weekly_atr_compression(df: pd.DataFrame, **_) -> pd.Series:
    """Weekly ATR percentile rank within its own 40-week history (0–100)."""
    atr = atr_wilder(df, 10)
    return atr.rolling(40).apply(
        lambda x: float(np.sum(x[:-1] <= x[-1]) / max(len(x) - 1, 1) * 100),
        raw=True,
    )


@register("weekly_range_compression", timeframe="weekly", lookback_bars=20)
def weekly_range_compression(df: pd.DataFrame, **_) -> pd.Series:
    """4-week average range / 20-week average range. < 0.7 = compressing."""
    rng = df["high"] - df["low"]
    return rng.rolling(4).mean() / rng.rolling(20).mean().replace(0, np.nan)


@register("weekly_volume_dryup", timeframe="weekly", lookback_bars=26)
def weekly_volume_dryup(df: pd.DataFrame, **_) -> pd.Series:
    """Current weekly volume / 26-week mean volume."""
    avg = df["volume"].rolling(26, min_periods=10).mean()
    return df["volume"] / avg.replace(0, np.nan)


@register("weekly_closes_above_sma10", timeframe="weekly", lookback_bars=10)
def weekly_closes_above_sma10(df: pd.DataFrame, **_) -> pd.Series:
    """Fraction of last 10 weekly closes above the 10-week SMA."""
    s = sma(df["close"], 10)
    above = (df["close"] > s).rolling(10).mean()
    return above
