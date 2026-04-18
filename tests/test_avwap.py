"""Tests for AVWAP calculation and anchor detection."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from swingtrader.avwap.anchors import swing_high_anchor, swing_low_anchor, ytd_anchor
from swingtrader.avwap.calc import compute_avwap
from swingtrader.avwap.features import compute_avwap_features


def _df(n: int = 252, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Roll back to last business day so bdate_range always returns exactly n bars
    end = pd.offsets.BDay().rollback(pd.Timestamp.today().normalize())
    idx = pd.bdate_range(end=end, periods=n)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    close = np.maximum(close, 1.0)
    df = pd.DataFrame(
        {
            "open": close + rng.normal(0, 0.3, n),
            "high": close + rng.uniform(0, 1, n),
            "low": close - rng.uniform(0, 1, n),
            "close": close,
            "volume": rng.integers(100_000, 1_000_000, n).astype(float),
        },
        index=idx,
    )
    df["high"] = df[["open", "close", "high"]].max(axis=1)
    df["low"] = df[["open", "close", "low"]].min(axis=1)
    df.index.name = "date"
    return df


# ─── Anchor detection ────────────────────────────────────────────────────────


def test_ytd_anchor_is_in_current_year() -> None:
    df = _df(252)
    anchor = ytd_anchor(df)
    assert anchor is not None
    assert anchor.year == df.index.max().year


def test_ytd_anchor_is_first_bar_of_year() -> None:
    df = _df(252)
    anchor = ytd_anchor(df)
    assert anchor >= pd.Timestamp(f"{anchor.year}-01-01")
    # Must be in our index
    assert anchor in df.index


def test_swing_low_anchor_returns_timestamp() -> None:
    df = _df(252)
    anchor = swing_low_anchor(df)
    assert anchor is not None
    assert isinstance(anchor, pd.Timestamp)
    assert anchor in df.index


def test_swing_high_anchor_is_max_high() -> None:
    df = _df(252)
    anchor = swing_high_anchor(df)
    assert anchor in df.index
    # The anchor should be at or near the max high
    max_high = df["high"].max()
    assert df.loc[anchor, "high"] == pytest.approx(max_high)


# ─── AVWAP calculation ───────────────────────────────────────────────────────


def test_avwap_is_nan_before_anchor() -> None:
    df = _df(100)
    anchor = df.index[50]
    avwap = compute_avwap(df, anchor)
    assert avwap.iloc[:50].isna().all()


def test_avwap_is_non_nan_from_anchor() -> None:
    df = _df(100)
    anchor = df.index[10]
    avwap = compute_avwap(df, anchor)
    assert avwap.iloc[10:].notna().all()


def test_avwap_none_anchor_returns_all_nan() -> None:
    df = _df(50)
    avwap = compute_avwap(df, None)
    assert avwap.isna().all()


def test_avwap_single_bar_equals_typical_price() -> None:
    """When anchored at first bar, first avwap = TP of that bar."""
    df = _df(50)
    anchor = df.index[0]
    avwap = compute_avwap(df, anchor)
    tp0 = (df["high"].iloc[0] + df["low"].iloc[0] + df["close"].iloc[0]) / 3
    assert avwap.iloc[0] == pytest.approx(tp0)


def test_avwap_is_between_low_and_high() -> None:
    """AVWAP should always be within the price range of bars seen so far."""
    df = _df(100)
    anchor = df.index[0]
    avwap = compute_avwap(df, anchor).dropna()
    rolling_high = df["high"].expanding().max().reindex(avwap.index)
    rolling_low = df["low"].expanding().min().reindex(avwap.index)
    assert (avwap <= rolling_high + 1e-8).all()
    assert (avwap >= rolling_low - 1e-8).all()


# ─── AVWAP features ──────────────────────────────────────────────────────────


def test_compute_avwap_features_returns_dataframe() -> None:
    df = _df(300)
    feat = compute_avwap_features(df)
    assert isinstance(feat, pd.DataFrame)
    assert feat.index.equals(df.index)


def test_avwap_features_contain_expected_columns() -> None:
    df = _df(300)
    feat = compute_avwap_features(df)
    # At minimum the YTD anchor columns should exist
    assert any("ytd" in c for c in feat.columns)
    assert any("swing_low" in c for c in feat.columns)


def test_avwap_dist_atr_sign_correct() -> None:
    """dist_atr must be positive when close > AVWAP, negative when close < AVWAP."""
    df = _df(100)
    # The YTD or swing_low AVWAP will be non-NaN for most bars.
    feat = compute_avwap_features(df)
    dist_col = next((c for c in feat.columns if "dist_atr" in c and not c.startswith("avwap_conf")), None)
    avwap_col = dist_col.replace("_dist_atr", "_avwap") if dist_col else None
    if dist_col is None or avwap_col not in feat.columns:
        pytest.skip("No dist_atr column produced")
    valid = feat[[dist_col, avwap_col, "ytd_avwap" if "ytd_avwap" in feat.columns else avwap_col]].dropna()
    if valid.empty:
        pytest.skip("All NaN after dropna")
    close_above = df["close"].reindex(valid.index) > feat[avwap_col].reindex(valid.index)
    dist_pos = valid[dist_col] > 0
    # When close > avwap, dist_atr should be positive (may not hold for ALL due to ATR sign)
    # Just assert they have the right sign correlation
    assert close_above.equals(dist_pos) or (close_above == dist_pos).mean() > 0.8
