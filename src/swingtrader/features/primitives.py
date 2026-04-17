"""Low-level feature primitives shared across feature modules.

All functions are pure (no side effects) and return pd.Series aligned to df.index.
NaN is returned for the warm-up period where the lookback window is not yet full.

ATR method: Wilder's smoothed ATR (the market standard).
  TR_t = max(H-L, |H-C_{t-1}|, |L-C_{t-1}|)
  ATR_t = (ATR_{t-1} * (n-1) + TR_t) / n   (Wilder's EMA with α = 1/n)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def true_range(df: pd.DataFrame) -> pd.Series:
    """True range per bar.  Requires columns: high, low, close."""
    hi = df["high"]
    lo = df["low"]
    prev_c = df["close"].shift(1)
    tr = pd.concat([hi - lo, (hi - prev_c).abs(), (lo - prev_c).abs()], axis=1).max(axis=1)
    tr.name = "tr"
    return tr


def atr_wilder(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder's ATR. First value uses simple mean of the first `period` TRs."""
    tr = true_range(df)
    atr = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    atr.name = f"atr_{period}"
    return atr


def ema(series: pd.Series, period: int) -> pd.Series:
    """Standard EMA (α = 2/(n+1))."""
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=period).mean()


def rolling_percentile_rank(series: pd.Series, window: int) -> pd.Series:
    """Percentile rank of the current value within its own rolling window (0–100)."""
    return series.rolling(window).apply(
        lambda x: float(np.sum(x[:-1] <= x[-1]) / max(len(x) - 1, 1) * 100),
        raw=True,
    )


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    m = series.rolling(window, min_periods=window).mean()
    s = series.rolling(window, min_periods=window).std(ddof=1)
    return (series - m) / s.replace(0, np.nan)


def linear_slope(series: pd.Series, window: int) -> pd.Series:
    """Rolling OLS slope of series on a 0…window-1 time index, normalised by series mean.

    Returns slope as fraction-per-bar (e.g. 0.002 = ~+0.2% per bar).
    """
    x = np.arange(window, dtype=float)
    x -= x.mean()
    ss_x = (x ** 2).sum()

    def _slope(y: np.ndarray) -> float:
        y_m = y.mean()
        if y_m == 0:
            return np.nan
        return float(np.dot(x, y - y_m) / ss_x / y_m)

    return series.rolling(window, min_periods=window).apply(_slope, raw=True)
