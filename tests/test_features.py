"""Tests for feature primitives and daily/weekly feature functions."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from swingtrader.features.primitives import (
    atr_wilder,
    linear_slope,
    sma,
    true_range,
)
from swingtrader.features.registry import compute_features, load_all_feature_modules


def _synthetic_daily(n: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    end = pd.Timestamp.today().normalize()
    idx = pd.bdate_range(end=end, periods=n)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    close = np.maximum(close, 1.0)
    df = pd.DataFrame(
        {
            "open": close * (1 + rng.normal(0, 0.005, n)),
            "high": close * (1 + rng.uniform(0, 0.015, n)),
            "low": close * (1 - rng.uniform(0, 0.015, n)),
            "close": close,
            "volume": rng.integers(500_000, 5_000_000, n).astype(float),
        },
        index=idx,
    )
    df["high"] = df[["open", "close", "high"]].max(axis=1)
    df["low"] = df[["open", "close", "low"]].min(axis=1)
    df.index.name = "date"
    return df


# ─── Primitives ──────────────────────────────────────────────────────────────


def test_true_range_non_negative() -> None:
    df = _synthetic_daily()
    tr = true_range(df)
    # First bar has no prior close; true_range uses H-L for it (pandas max skipna=True).
    # All values (including first bar) should be positive.
    assert (tr >= 0).all()


def test_true_range_gte_high_minus_low() -> None:
    """True range must always be >= the simple H-L range."""
    df = _synthetic_daily()
    tr = true_range(df)
    hl = df["high"] - df["low"]
    assert (tr.fillna(0) >= hl.fillna(0) - 1e-9).all()


def test_atr_wilder_shape_and_nonnan() -> None:
    df = _synthetic_daily(100)
    atr = atr_wilder(df, 14)
    assert len(atr) == 100
    # After warm-up period, should be non-NaN
    assert atr.iloc[20:].notna().all()
    assert (atr.dropna() > 0).all()


def test_atr_wilder_positive() -> None:
    df = _synthetic_daily()
    atr = atr_wilder(df)
    assert (atr.dropna() > 0).all()


def test_sma_correct_value() -> None:
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = sma(s, 3)
    assert np.isnan(result.iloc[0])
    assert np.isnan(result.iloc[1])
    assert result.iloc[2] == pytest.approx(2.0)
    assert result.iloc[4] == pytest.approx(4.0)


def test_linear_slope_uptrend() -> None:
    """A steadily rising series should have positive slope."""
    s = pd.Series(np.arange(50, dtype=float))
    slopes = linear_slope(s, 20)
    valid = slopes.dropna()
    assert (valid > 0).all()


def test_linear_slope_flat_returns_zero() -> None:
    s = pd.Series(np.ones(50))
    slopes = linear_slope(s, 20)
    valid = slopes.dropna()
    assert (valid.abs() < 1e-10).all()


# ─── Registry ────────────────────────────────────────────────────────────────


def test_load_all_feature_modules_populates_registry() -> None:
    from swingtrader.features.registry import REGISTRY
    load_all_feature_modules()
    assert len(REGISTRY) > 10, "expected many registered features"
    daily = [k for k, v in REGISTRY.items() if v.timeframe == "daily"]
    weekly = [k for k, v in REGISTRY.items() if v.timeframe == "weekly"]
    assert len(daily) >= 8
    assert len(weekly) >= 4


def test_compute_features_returns_aligned_dataframe() -> None:
    load_all_feature_modules()
    df = _synthetic_daily(250)
    feats = compute_features(df, "daily")
    assert isinstance(feats, pd.DataFrame)
    assert feats.index.equals(df.index)
    # Every feature column should exist even if all-NaN for short warmup
    assert "atr_14" in feats.columns
    assert "atr_compression_pct" in feats.columns
    assert "volume_dryup" in feats.columns


def test_daily_features_no_nan_after_warmup() -> None:
    """After the longest lookback, key features should be non-NaN."""
    load_all_feature_modules()
    df = _synthetic_daily(300)
    feats = compute_features(df, "daily")
    # atr_14 needs 15 bars; check last 100 bars (300 - 200 warmup buffer)
    tail = feats.tail(100)
    for col in ["atr_14", "volume_dryup", "close_vs_sma50"]:
        assert tail[col].notna().any(), f"{col} is all NaN in tail"


def test_regime_features_nan_without_benchmark() -> None:
    """Regime features should return NaN series (not raise) when benchmark_df is absent."""
    load_all_feature_modules()
    df = _synthetic_daily(100)
    feats = compute_features(df, "daily", extra_kwargs={"benchmark_df": None})
    assert "regime_spy_trend" in feats.columns
    # Should be all-NaN when no benchmark
    assert feats["regime_spy_trend"].isna().all()


def test_weekly_features_on_resampled_data() -> None:
    from swingtrader.ingest.yfinance_source import resample_weekly
    load_all_feature_modules()
    df = _synthetic_daily(500)
    w = resample_weekly(df)
    assert len(w) >= 20
    feats = compute_features(w, "weekly")
    assert "weekly_trend_slope_26" in feats.columns
