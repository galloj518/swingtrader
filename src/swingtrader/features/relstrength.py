"""Relative-strength features (symbol vs benchmark).

These features require a benchmark DataFrame passed as ``benchmark_df`` kwarg.
When benchmark_df is None the features return a Series of NaN without raising —
the runner logs a warning and continues.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from swingtrader.features.primitives import linear_slope
from swingtrader.features.registry import register

_LOG_WARN_ONCE: set[str] = set()


def _check_benchmark(benchmark_df: pd.DataFrame | None, name: str) -> bool:
    if benchmark_df is None or benchmark_df.empty:
        if name not in _LOG_WARN_ONCE:
            _LOG_WARN_ONCE.add(name)
            from swingtrader.utils.logging import get_logger
            get_logger(__name__).warning(
                "benchmark_df missing; feature %r will be NaN", name
            )
        return False
    return True


def _log_return(close: pd.Series, window: int) -> pd.Series:
    return np.log(close / close.shift(window))


@register("daily_rs_63", timeframe="daily", lookback_bars=63)
def daily_rs_63(df: pd.DataFrame, *, benchmark_df: pd.DataFrame | None = None, **_) -> pd.Series:
    """63-day log-return differential vs benchmark (symbol return − SPY return).

    Positive = outperforming benchmark over the quarter.
    """
    if not _check_benchmark(benchmark_df, "daily_rs_63"):
        return pd.Series(np.nan, index=df.index)
    sym_ret = _log_return(df["close"], 63)
    bmk_ret = _log_return(benchmark_df["close"], 63).reindex(df.index, method="ffill")
    return sym_ret - bmk_ret


@register("daily_rs_21", timeframe="daily", lookback_bars=21)
def daily_rs_21(df: pd.DataFrame, *, benchmark_df: pd.DataFrame | None = None, **_) -> pd.Series:
    """21-day relative return vs benchmark (monthly)."""
    if not _check_benchmark(benchmark_df, "daily_rs_21"):
        return pd.Series(np.nan, index=df.index)
    sym_ret = _log_return(df["close"], 21)
    bmk_ret = _log_return(benchmark_df["close"], 21).reindex(df.index, method="ffill")
    return sym_ret - bmk_ret


@register("daily_rs_slope_50", timeframe="daily", lookback_bars=50)
def daily_rs_slope_50(df: pd.DataFrame, *, benchmark_df: pd.DataFrame | None = None, **_) -> pd.Series:
    """Slope of the 63-day RS ratio over 50 bars — is RS trend improving?"""
    if not _check_benchmark(benchmark_df, "daily_rs_slope_50"):
        return pd.Series(np.nan, index=df.index)
    bmk_close = benchmark_df["close"].reindex(df.index, method="ffill")
    rs_ratio = df["close"] / bmk_close.replace(0, np.nan)
    return linear_slope(rs_ratio, 50)


@register("weekly_rs_26", timeframe="weekly", lookback_bars=26)
def weekly_rs_26(df: pd.DataFrame, *, benchmark_df: pd.DataFrame | None = None, **_) -> pd.Series:
    """26-week relative return vs benchmark on weekly bars."""
    if not _check_benchmark(benchmark_df, "weekly_rs_26"):
        return pd.Series(np.nan, index=df.index)
    sym_ret = _log_return(df["close"], 26)
    bmk_ret = _log_return(benchmark_df["close"], 26).reindex(df.index, method="ffill")
    return sym_ret - bmk_ret
