"""Smoke test for the daily pipeline using synthetic data (no network)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from swingtrader.avwap.features import compute_avwap_features
from swingtrader.bases.base_detect import detect_bases
from swingtrader.features.pivot_features import compute_pivot_features
from swingtrader.features.primitives import atr_wilder
from swingtrader.features.registry import compute_features, load_all_feature_modules
from swingtrader.ingest.yfinance_source import resample_weekly
from swingtrader.labels.generators import compute_all_labels
from swingtrader.states.machine import compute_states


def _make_df(n: int = 300, seed: int = 99) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=pd.offsets.BDay().rollback(pd.Timestamp.today().normalize()), periods=n)
    close = 50 + np.cumsum(rng.normal(0.05, 1.0, n))
    close = np.maximum(close, 5.0)
    df = pd.DataFrame(
        {
            "open": close * (1 + rng.normal(0, 0.003, n)),
            "high": close * (1 + rng.uniform(0, 0.012, n)),
            "low": close * (1 - rng.uniform(0, 0.012, n)),
            "close": close,
            "volume": rng.integers(1_000_000, 10_000_000, n).astype(float),
        },
        index=idx,
    )
    df["high"] = df[["open", "close", "high"]].max(axis=1)
    df["low"] = df[["open", "close", "low"]].min(axis=1)
    df.index.name = "date"
    return df


def test_full_pipeline_runs_without_error() -> None:
    """End-to-end pipeline on synthetic data: features → bases → states → labels."""
    load_all_feature_modules()
    df = _make_df(300)
    w_df = resample_weekly(df)

    # Features
    feat_daily = compute_features(df, "daily")
    feat_weekly = compute_features(w_df, "weekly").reindex(df.index, method="ffill")

    # Bases
    bases = detect_bases(df, min_days=15, max_days=120, max_depth_pct=0.25)

    # Pivot features
    atr = atr_wilder(df, 14)
    feat_pivot = compute_pivot_features(df, bases, atr=atr)

    # States
    states = compute_states(df, bases)

    # AVWAP
    feat_avwap = compute_avwap_features(df, state_history=states["state"])

    # Labels
    labels = compute_all_labels(df, states)

    # All outputs must be DataFrames aligned to df.index
    for name, out in [
        ("feat_daily", feat_daily),
        ("feat_weekly", feat_weekly),
        ("feat_pivot", feat_pivot),
        ("feat_avwap", feat_avwap),
        ("states", states),
        ("labels", labels),
    ]:
        assert isinstance(out, pd.DataFrame), f"{name} is not a DataFrame"
        assert len(out) == len(df), f"{name} length mismatch: {len(out)} vs {len(df)}"


def test_pipeline_outputs_are_not_empty() -> None:
    load_all_feature_modules()
    df = _make_df(300)
    feat_daily = compute_features(df, "daily")
    # At minimum the atr_14 column should have non-NaN values
    assert feat_daily["atr_14"].notna().any()


def test_states_column_contains_valid_strings() -> None:
    df = _make_df(300)
    bases = detect_bases(df, min_days=15, max_days=120, max_depth_pct=0.25)
    states = compute_states(df, bases)
    valid_states = {"NONE", "BASE", "ARMED", "TRIGGERED", "ACCEPTED", "CONFIRMED", "FAILED", "LATE", "EXHAUSTED"}
    assert set(states["state"].unique()).issubset(valid_states)


def test_labels_fwd_ret_nan_guard() -> None:
    """Forward returns must be NaN for the last H bars."""
    df = _make_df(100)
    bases = detect_bases(df, min_days=10, max_days=60, max_depth_pct=0.25)
    states = compute_states(df, bases)
    labels = compute_all_labels(df, states)
    assert labels["fwd_ret_h20"].iloc[-20:].isna().all()


def test_write_and_read_artifacts(tmp_path: Path) -> None:
    """Pipeline artifacts can be written to parquet and read back intact."""
    from swingtrader.utils.io import read_parquet, write_parquet
    load_all_feature_modules()
    df = _make_df(150)
    bases = detect_bases(df, min_days=10, max_days=60, max_depth_pct=0.25)
    states = compute_states(df, bases)

    path = tmp_path / "states.parquet"
    write_parquet(states, path)
    recovered = read_parquet(path)

    assert list(recovered.columns) == list(states.columns)
    assert len(recovered) == len(states)
    assert (recovered["state"] == states["state"]).all()
