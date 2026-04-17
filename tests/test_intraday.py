"""Tests for intraday (5-minute) feature module."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from swingtrader.features.intraday import (
    _intraday_vwap,
    _today_bars,
    load_intraday,
)
from swingtrader.features.registry import REGISTRY, load_all_feature_modules


def _make_5m_bars(n_sessions: int = 3, bars_per_session: int = 78, seed: int = 1) -> pd.DataFrame:
    """Synthetic 5-minute OHLCV: n_sessions × bars_per_session bars."""
    rng = np.random.default_rng(seed)
    times = []
    for day_offset in range(n_sessions):
        base = pd.Timestamp("2024-06-03") + pd.Timedelta(days=day_offset)
        for bar in range(bars_per_session):
            times.append(base + pd.Timedelta(minutes=bar * 5))
    idx = pd.DatetimeIndex(times)
    n = len(idx)
    close = 100.0 + rng.normal(0, 1, n).cumsum()
    high = close + rng.uniform(0, 0.5, n)
    low = close - rng.uniform(0, 0.5, n)
    return pd.DataFrame({
        "open": close - rng.uniform(-0.2, 0.2, n),
        "high": high,
        "low": low,
        "close": close,
        "volume": rng.integers(1000, 50000, n),
    }, index=idx)


# ── Registry registration ─────────────────────────────────────────────────────

def test_intraday_features_registered() -> None:
    load_all_feature_modules()
    intraday_keys = [k for k, v in REGISTRY.items() if v.timeframe == "intraday"]
    assert len(intraday_keys) >= 6, f"Expected ≥6 intraday features, got {intraday_keys}"


def test_intraday_registry_entries_have_correct_timeframe() -> None:
    load_all_feature_modules()
    for name, spec in REGISTRY.items():
        if name.startswith("intraday_"):
            assert spec.timeframe == "intraday"


# ── _today_bars ───────────────────────────────────────────────────────────────

def test_today_bars_returns_last_date_only() -> None:
    df = _make_5m_bars(3, 10)
    today = _today_bars(df)
    unique_dates = today.index.normalize().unique()
    assert len(unique_dates) == 1
    assert unique_dates[0] == df.index.normalize().max()


def test_today_bars_empty_on_empty_input() -> None:
    today = _today_bars(pd.DataFrame())
    assert today.empty


# ── _intraday_vwap ────────────────────────────────────────────────────────────

def test_intraday_vwap_shape() -> None:
    df = _make_5m_bars(1, 20)
    vwap = _intraday_vwap(df)
    assert len(vwap) == len(df)


def test_intraday_vwap_positive() -> None:
    df = _make_5m_bars(1, 20)
    vwap = _intraday_vwap(df)
    assert (vwap > 0).all()


def test_intraday_vwap_bounded_by_session_extremes() -> None:
    """Running VWAP (cumulative) must stay within the session's full price range."""
    df = _make_5m_bars(1, 40)
    vwap = _intraday_vwap(df)
    session_low = float(df["low"].min())
    session_high = float(df["high"].max())
    assert (vwap >= session_low * 0.99).all()   # small tolerance for floating point
    assert (vwap <= session_high * 1.01).all()


# ── Feature output shape and NaN discipline ───────────────────────────────────

def _run_intraday_feature(name: str, df: pd.DataFrame) -> pd.Series:
    load_all_feature_modules()
    spec = REGISTRY[name]
    return spec.fn(df)


@pytest.mark.parametrize("feature_name", [
    "intraday_rvol",
    "intraday_vwap_dist_pct",
    "intraday_or_high",
    "intraday_or_low",
    "intraday_close_above_or_high",
    "intraday_momentum_30m",
    "intraday_gap_pct",
    "intraday_high_of_day_pct",
])
def test_intraday_feature_returns_series(feature_name: str) -> None:
    load_all_feature_modules()
    if feature_name not in REGISTRY:
        pytest.skip(f"{feature_name} not registered")
    df = _make_5m_bars(3, 78)
    result = _run_intraday_feature(feature_name, df)
    assert isinstance(result, pd.Series)


@pytest.mark.parametrize("feature_name", [
    "intraday_rvol",
    "intraday_vwap_dist_pct",
    "intraday_momentum_30m",
    "intraday_gap_pct",
])
def test_intraday_feature_has_finite_values(feature_name: str) -> None:
    load_all_feature_modules()
    if feature_name not in REGISTRY:
        pytest.skip(f"{feature_name} not registered")
    df = _make_5m_bars(5, 78)
    result = _run_intraday_feature(feature_name, df)
    n_finite = result.dropna().apply(np.isfinite).sum()
    assert n_finite > 0, f"{feature_name} returned no finite values"


def test_intraday_rvol_positive() -> None:
    load_all_feature_modules()
    df = _make_5m_bars(5, 78)
    rvol = _run_intraday_feature("intraday_rvol", df)
    finite = rvol.dropna()
    if len(finite) > 0:
        assert (finite >= 0).all()


def test_intraday_close_above_or_high_is_binary() -> None:
    load_all_feature_modules()
    df = _make_5m_bars(3, 78)
    result = _run_intraday_feature("intraday_close_above_or_high", df)
    valid = result.dropna()
    assert set(valid.unique()).issubset({0.0, 1.0})


def test_intraday_or_high_greater_than_or_low() -> None:
    load_all_feature_modules()
    df = _make_5m_bars(3, 78)
    or_high = _run_intraday_feature("intraday_or_high", df)
    or_low = _run_intraday_feature("intraday_or_low", df)
    valid_h = or_high.dropna()
    valid_l = or_low.dropna()
    common_idx = valid_h.index.intersection(valid_l.index)
    if len(common_idx) > 0:
        assert (valid_h[common_idx] >= valid_l[common_idx]).all()


def test_intraday_features_empty_input_returns_series() -> None:
    load_all_feature_modules()
    empty = pd.DataFrame()
    for name, spec in REGISTRY.items():
        if spec.timeframe != "intraday":
            continue
        result = spec.fn(empty)
        assert isinstance(result, pd.Series), f"{name} did not return Series on empty input"


# ── load_intraday ─────────────────────────────────────────────────────────────

def test_load_intraday_returns_empty_on_missing_file() -> None:
    df = load_intraday("NONEXISTENT_SYMBOL_XYZ")
    assert isinstance(df, pd.DataFrame)
    assert df.empty
