"""Smoke tests for the ingest layer. Does NOT hit the network."""
from __future__ import annotations

import numpy as np
import pandas as pd

from swingtrader.ingest.quality import check_daily
from swingtrader.ingest.universe import load_universe


def _synthetic_daily(n_days: int = 260, end: pd.Timestamp | None = None) -> pd.DataFrame:
    end = end or pd.Timestamp.today().normalize()
    idx = pd.bdate_range(end=end, periods=n_days)
    rng = np.random.default_rng(42)
    n = len(idx)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    df = pd.DataFrame(
        {
            "open": close + rng.normal(0, 0.5, n),
            "high": close + rng.uniform(0, 1, n),
            "low": close - rng.uniform(0, 1, n),
            "close": close,
            "volume": rng.integers(100_000, 1_000_000, n).astype(float),
        },
        index=idx,
    )
    df.index.name = "date"
    return df


def test_load_universe_returns_deduped_list() -> None:
    # load_universe() now delegates to resolve_universe() which uses the new config schema
    symbols = load_universe()
    assert isinstance(symbols, list)
    assert len(symbols) > 0, "expected non-empty universe from default config"
    assert "SPY" in symbols  # benchmarks are always active in default config
    assert len(symbols) == len(set(symbols)), "universe must be deduped"


def test_load_universe_preserves_source_priority() -> None:
    symbols = load_universe()
    # Benchmarks come first; provider symbol for SPY is SPY (no alias)
    assert symbols[0] in {"SPY", "QQQ", "IWM", "DIA", "MDY", "RSP"}


def test_quality_passes_on_clean_series() -> None:
    df = _synthetic_daily()
    report = check_daily("FAKE", df)
    assert report.missing_pct <= 0.02
    assert report.max_gap_bars <= 5
    assert not report.missing_cols


def test_quality_flags_large_gap() -> None:
    df = _synthetic_daily()
    df = df.drop(df.index[100:120])  # carve a 20-bar hole
    report = check_daily("FAKE", df)
    assert report.max_gap_bars >= 15
    assert not report.ok


def test_quality_flags_missing_columns() -> None:
    df = _synthetic_daily()
    df = df.drop(columns=["volume"])
    report = check_daily("FAKE", df)
    assert "volume" in report.missing_cols
    assert not report.ok


def test_quality_handles_empty_frame() -> None:
    report = check_daily("FAKE", pd.DataFrame())
    assert not report.ok
    assert report.n_rows == 0
