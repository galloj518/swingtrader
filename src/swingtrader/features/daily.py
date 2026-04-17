"""Daily OHLCV feature functions.

Each function is registered in the feature REGISTRY and must:
  - Accept df: pd.DataFrame (daily OHLCV) as first positional arg
  - Accept **kwargs (for optional args like benchmark_df)
  - Return pd.Series aligned to df.index
  - Be computable from information available at bar t (no lookahead)
  - Return NaN for the warm-up period (first lookback_bars bars)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from swingtrader.features.primitives import atr_wilder, ema, linear_slope, sma
from swingtrader.features.registry import register

# ─── Volatility / compression ────────────────────────────────────────────────


@register("atr_14", timeframe="daily", lookback_bars=15)
def atr_14(df: pd.DataFrame, **_) -> pd.Series:
    """Wilder's ATR(14) — absolute volatility level."""
    return atr_wilder(df, 14)


@register("atr_pct", timeframe="daily", lookback_bars=15)
def atr_pct(df: pd.DataFrame, **_) -> pd.Series:
    """ATR(14) as a % of close — normalised volatility."""
    return atr_wilder(df, 14) / df["close"] * 100


@register("atr_compression_pct", timeframe="daily", lookback_bars=50)
def atr_compression_pct(df: pd.DataFrame, **_) -> pd.Series:
    """Current ATR percentile rank within its own 50-bar history (0–100).

    Low values indicate compressed / contracting volatility — a base signal.
    """
    atr = atr_wilder(df, 14)
    return atr.rolling(50).apply(
        lambda x: float(np.sum(x[:-1] <= x[-1]) / max(len(x) - 1, 1) * 100),
        raw=True,
    )


@register("range_compression", timeframe="daily", lookback_bars=20)
def range_compression(df: pd.DataFrame, **_) -> pd.Series:
    """Ratio of recent 5-bar average daily range to 20-bar average daily range.

    Values < 0.7 indicate range is contracting meaningfully.
    """
    daily_range = df["high"] - df["low"]
    return daily_range.rolling(5).mean() / daily_range.rolling(20).mean().replace(0, np.nan)


@register("volatility_contraction_5_20", timeframe="daily", lookback_bars=20)
def volatility_contraction_5_20(df: pd.DataFrame, **_) -> pd.Series:
    """Ratio of 5-bar close std-dev to 20-bar close std-dev.

    < 1 means recent vol is below the longer baseline — classic VCP / base signal.
    """
    std5 = df["close"].rolling(5).std(ddof=1)
    std20 = df["close"].rolling(20).std(ddof=1)
    return std5 / std20.replace(0, np.nan)


@register("volatility_contraction_10_50", timeframe="daily", lookback_bars=50)
def volatility_contraction_10_50(df: pd.DataFrame, **_) -> pd.Series:
    std10 = df["close"].rolling(10).std(ddof=1)
    std50 = df["close"].rolling(50).std(ddof=1)
    return std10 / std50.replace(0, np.nan)


# ─── Volume / participation ───────────────────────────────────────────────────


@register("volume_dryup", timeframe="daily", lookback_bars=50)
def volume_dryup(df: pd.DataFrame, **_) -> pd.Series:
    """Current volume / 50-bar mean volume. Values < 0.7 = volume drying up."""
    avg = df["volume"].rolling(50, min_periods=20).mean()
    return df["volume"] / avg.replace(0, np.nan)


@register("volume_dryup_10", timeframe="daily", lookback_bars=50)
def volume_dryup_10(df: pd.DataFrame, **_) -> pd.Series:
    """10-bar mean volume / 50-bar mean volume — longer dry-up signal."""
    vol10 = df["volume"].rolling(10).mean()
    vol50 = df["volume"].rolling(50, min_periods=20).mean()
    return vol10 / vol50.replace(0, np.nan)


@register("dollar_volume_log", timeframe="daily", lookback_bars=1)
def dollar_volume_log(df: pd.DataFrame, **_) -> pd.Series:
    """Log10 of daily dollar volume — proxy for liquidity."""
    dv = df["close"] * df["volume"]
    return np.log10(dv.replace(0, np.nan))


# ─── Trend / structure ───────────────────────────────────────────────────────


@register("close_vs_sma50", timeframe="daily", lookback_bars=50)
def close_vs_sma50(df: pd.DataFrame, **_) -> pd.Series:
    """(close / SMA50) - 1 as a fraction. Positive = above 50-DMA."""
    s = sma(df["close"], 50)
    return df["close"] / s.replace(0, np.nan) - 1


@register("close_vs_sma200", timeframe="daily", lookback_bars=200)
def close_vs_sma200(df: pd.DataFrame, **_) -> pd.Series:
    s = sma(df["close"], 200)
    return df["close"] / s.replace(0, np.nan) - 1


@register("close_vs_ema20", timeframe="daily", lookback_bars=20)
def close_vs_ema20(df: pd.DataFrame, **_) -> pd.Series:
    e = ema(df["close"], 20)
    return df["close"] / e.replace(0, np.nan) - 1


@register("pct_from_52w_high", timeframe="daily", lookback_bars=252)
def pct_from_52w_high(df: pd.DataFrame, **_) -> pd.Series:
    """Distance from 52-week high as a fraction (0 = at high, -0.15 = 15% below)."""
    h52 = df["high"].rolling(252, min_periods=50).max()
    return df["close"] / h52.replace(0, np.nan) - 1


@register("slope_50", timeframe="daily", lookback_bars=50)
def slope_50(df: pd.DataFrame, **_) -> pd.Series:
    """OLS slope of close over 50 bars, normalised by mean close (fraction per bar)."""
    return linear_slope(df["close"], 50)


# ─── Bar quality ─────────────────────────────────────────────────────────────


@register("close_vs_high", timeframe="daily", lookback_bars=1)
def close_vs_high(df: pd.DataFrame, **_) -> pd.Series:
    """Close location in the day's range: 1.0 = closed at high, 0.0 = closed at low.

    High values on breakout day signal strong acceptance.
    """
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    return (df["close"] - df["low"]) / rng


@register("gap_up_pct", timeframe="daily", lookback_bars=2)
def gap_up_pct(df: pd.DataFrame, **_) -> pd.Series:
    """Today's open vs yesterday's close, as a fraction. Positive = gap up."""
    return df["open"] / df["close"].shift(1) - 1
