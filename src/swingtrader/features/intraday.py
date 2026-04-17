"""Intraday (5-minute bar) feature module.

Features here are a same-day overlay: they describe what has happened in the
current session up to the most recent 5m bar. They are NOT used in historical
model training (labels are end-of-day; intraday bars don't exist for past dates
beyond the ~60-day yfinance window). They are appended to the features DataFrame
at score-time only.

All functions accept a 5m OHLCV DataFrame with a DatetimeIndex and return a
scalar float representing the latest available value for the current session.

Registry entries use timeframe="intraday" and allows_realtime=True.
lookback_bars is measured in 5-minute bars.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from swingtrader.features.registry import register
from swingtrader.utils.logging import get_logger

log = get_logger(__name__)

# Opening range window in bars (30 min / 5 min = 6 bars)
_OR_BARS = 6


def _today_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Slice the most recent trading date's 5m bars."""
    if df.empty:
        return df
    last_date = df.index.normalize().max()
    return df[df.index.normalize() == last_date]


def _intraday_vwap(df: pd.DataFrame) -> pd.Series:
    """Running VWAP for the session (typical price weighted by volume)."""
    if df.empty or not {"high", "low", "close", "volume"}.issubset(df.columns):
        return pd.Series(dtype=float)
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    cumvol = df["volume"].cumsum()
    cumvol_tp = (tp * df["volume"]).cumsum()
    vwap = cumvol_tp / cumvol.where(cumvol > 0)
    return vwap


# ── Registered intraday features ─────────────────────────────────────────────

@register("intraday_rvol", timeframe="intraday", lookback_bars=78, allows_realtime=True)
def _intraday_rvol(df: pd.DataFrame, **_) -> pd.Series:
    """Relative volume: session-to-date volume vs 20-session average at same bar count.

    Returns a ratio ≥ 0. 1.0 = average. 2.0 = double normal pace.
    Requires at least 5 sessions of intraday history to be meaningful.
    """
    if df.empty:
        return pd.Series(dtype=float)

    today = _today_bars(df)
    if today.empty:
        return pd.Series(np.nan, index=df.index)

    today_cumvol = today["volume"].cumsum()
    n_bars_today = len(today)

    # For each past session, compute volume through the same bar count
    unique_dates = df.index.normalize().unique().sort_values()
    past_dates = unique_dates[unique_dates < today.index.normalize().min()]

    past_vols: list[float] = []
    for d in past_dates[-20:]:
        session = df[df.index.normalize() == d]
        if len(session) >= n_bars_today:
            past_vols.append(float(session["volume"].iloc[:n_bars_today].sum()))

    if not past_vols:
        return pd.Series(np.nan, index=df.index)

    avg_past_vol = float(np.mean(past_vols))
    if avg_past_vol <= 0:
        return pd.Series(np.nan, index=df.index)

    result = pd.Series(np.nan, index=df.index)
    today_rvol = today_cumvol / avg_past_vol
    result.loc[today.index] = today_rvol.values
    return result


@register("intraday_vwap_dist_pct", timeframe="intraday", lookback_bars=78, allows_realtime=True)
def _intraday_vwap_dist_pct(df: pd.DataFrame, **_) -> pd.Series:
    """(close - session_VWAP) / session_VWAP as a fraction.

    Positive = above VWAP (constructive); negative = below.
    """
    if df.empty:
        return pd.Series(dtype=float)

    result = pd.Series(np.nan, index=df.index)
    for d in df.index.normalize().unique():
        session = df[df.index.normalize() == d]
        if session.empty:
            continue
        vwap = _intraday_vwap(session)
        dist = (session["close"] - vwap) / vwap.where(vwap > 0)
        result.loc[session.index] = dist.values
    return result


@register("intraday_or_high", timeframe="intraday", lookback_bars=6, allows_realtime=True)
def _intraday_or_high(df: pd.DataFrame, **_) -> pd.Series:
    """Opening range high (first 30 min = 6 bars of 5m).

    Returns a constant (the OR high) for all bars in the session.
    NaN for bars before the OR is established.
    """
    if df.empty:
        return pd.Series(dtype=float)

    result = pd.Series(np.nan, index=df.index)
    for d in df.index.normalize().unique():
        session = df[df.index.normalize() == d]
        if len(session) < _OR_BARS:
            continue
        or_high = float(session["high"].iloc[:_OR_BARS].max())
        result.loc[session.index[_OR_BARS:]] = or_high
    return result


@register("intraday_or_low", timeframe="intraday", lookback_bars=6, allows_realtime=True)
def _intraday_or_low(df: pd.DataFrame, **_) -> pd.Series:
    """Opening range low (first 30 min)."""
    if df.empty:
        return pd.Series(dtype=float)

    result = pd.Series(np.nan, index=df.index)
    for d in df.index.normalize().unique():
        session = df[df.index.normalize() == d]
        if len(session) < _OR_BARS:
            continue
        or_low = float(session["low"].iloc[:_OR_BARS].min())
        result.loc[session.index[_OR_BARS:]] = or_low
    return result


@register("intraday_close_above_or_high", timeframe="intraday", lookback_bars=78, allows_realtime=True)
def _intraday_close_above_or_high(df: pd.DataFrame, **_) -> pd.Series:
    """1 if the latest close is above the opening range high, else 0. NaN before OR established."""
    if df.empty or "close" not in df.columns:
        return pd.Series(dtype=float)
    or_high = _intraday_or_high(df)
    result = pd.Series(np.nan, index=df.index)
    valid = or_high.notna()
    if valid.any():
        result[valid] = (df.loc[valid, "close"] > or_high[valid]).astype(float)
    return result


@register("intraday_momentum_30m", timeframe="intraday", lookback_bars=6, allows_realtime=True)
def _intraday_momentum_30m(df: pd.DataFrame, **_) -> pd.Series:
    """Return of the most recent 30-minute window (6 bars): close[t] / close[t-6] - 1."""
    if df.empty:
        return pd.Series(dtype=float)
    close = df["close"]
    return (close / close.shift(6) - 1.0).where(close.shift(6) > 0)


@register("intraday_gap_pct", timeframe="intraday", lookback_bars=1, allows_realtime=True)
def _intraday_gap_pct(df: pd.DataFrame, **_) -> pd.Series:
    """Today's open vs prior session's close: (open[0] - prev_close) / prev_close.

    Constant for all bars within the session.
    """
    if df.empty:
        return pd.Series(dtype=float)

    result = pd.Series(np.nan, index=df.index)
    unique_dates = df.index.normalize().unique().sort_values()
    for i, d in enumerate(unique_dates):
        if i == 0:
            continue
        prev_d = unique_dates[i - 1]
        prev_session = df[df.index.normalize() == prev_d]
        curr_session = df[df.index.normalize() == d]
        if prev_session.empty or curr_session.empty:
            continue
        prev_close = float(prev_session["close"].iloc[-1])
        curr_open = float(curr_session["open"].iloc[0])
        if prev_close > 0:
            gap = (curr_open - prev_close) / prev_close
            result.loc[curr_session.index] = gap
    return result


@register("intraday_high_of_day_pct", timeframe="intraday", lookback_bars=78, allows_realtime=True)
def _intraday_high_of_day_pct(df: pd.DataFrame, **_) -> pd.Series:
    """(close - session_high) / session_high — how close to the high of the day.

    0.0 = at the high; negative = below it.
    """
    if df.empty:
        return pd.Series(dtype=float)

    result = pd.Series(np.nan, index=df.index)
    for d in df.index.normalize().unique():
        session = df[df.index.normalize() == d]
        rolling_high = session["high"].cummax()
        dist = (session["close"] - rolling_high) / rolling_high.where(rolling_high > 0)
        result.loc[session.index] = dist.values
    return result


def load_intraday(symbol: str, cfg=None) -> pd.DataFrame:
    """Load 5m intraday bars for a symbol from the intraday parquet store.

    Returns an empty DataFrame if the file does not exist.
    """
    from swingtrader.utils.config import REPO_ROOT, load_config
    cfg = cfg or load_config("data_sources")
    intraday_dir = REPO_ROOT / cfg["storage"]["intraday_dir"]
    path = intraday_dir / f"{symbol}.parquet"
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        return df
    except Exception as exc:
        log.warning("load_intraday(%s): %s", symbol, exc)
        return pd.DataFrame()
