"""Tests for label generators — correctness and no-lookahead leakage."""
from __future__ import annotations

import numpy as np
import pandas as pd

from swingtrader.bases.base_detect import detect_bases
from swingtrader.labels.generators import _forward_any, compute_all_labels
from swingtrader.states.machine import compute_states


def _make_inputs(n: int = 100, seed: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=pd.offsets.BDay().rollback(pd.Timestamp.today().normalize()), periods=n)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    close = np.maximum(close, 10.0)
    df = pd.DataFrame(
        {
            "open": close * (1 + rng.normal(0, 0.003, n)),
            "high": close * (1 + rng.uniform(0, 0.01, n)),
            "low": close * (1 - rng.uniform(0, 0.01, n)),
            "close": close,
            "volume": rng.integers(500_000, 5_000_000, n).astype(float),
        },
        index=idx,
    )
    df["high"] = df[["open", "close", "high"]].max(axis=1)
    df["low"] = df[["open", "close", "low"]].min(axis=1)
    df.index.name = "date"
    bases = detect_bases(df, min_days=10, max_days=60, max_depth_pct=0.20)
    states = compute_states(df, bases)
    return df, states


# ─── _forward_any ─────────────────────────────────────────────────────────────


def test_forward_any_simple() -> None:
    mask = pd.Series([0, 0, 0, 1, 0, 0], dtype=float)
    result = _forward_any(mask, horizon=3)
    # Bar 0: looks ahead at bars 1,2,3 → bar 3 is 1 → should be 1
    assert result.iloc[0] == 1.0
    # Bar 4 and 5 are NaN (no full horizon)
    assert np.isnan(result.iloc[4]) or np.isnan(result.iloc[5])


def test_forward_any_tail_is_nan() -> None:
    mask = pd.Series(np.zeros(20, dtype=float))
    result = _forward_any(mask, horizon=5)
    # Last 5 bars cannot have a full forward window
    assert result.iloc[-5:].isna().all()


def test_forward_any_zeros() -> None:
    mask = pd.Series(np.zeros(20, dtype=float))
    result = _forward_any(mask, horizon=3)
    valid = result.dropna()
    assert (valid == 0).all()


# ─── compute_all_labels ────────────────────────────────────────────────────────


def test_label_dataframe_shape() -> None:
    df, states = _make_inputs(120)
    labels = compute_all_labels(df, states)
    assert isinstance(labels, pd.DataFrame)
    assert labels.index.equals(df.index)


def test_label_columns_present() -> None:
    df, states = _make_inputs(120)
    labels = compute_all_labels(df, states)
    expected = [
        "setup_candidate", "triggered_breakout", "accepted_breakout",
        "failed_within_10", "followthrough_confirmed_20",
        "fwd_ret_h5", "fwd_ret_h10", "fwd_ret_h20",
    ]
    for col in expected:
        assert col in labels.columns, f"missing label column: {col}"


def test_setup_candidate_binary() -> None:
    df, states = _make_inputs(120)
    labels = compute_all_labels(df, states)
    vals = labels["setup_candidate"].dropna()
    assert vals.isin([0.0, 1.0]).all()


def test_forward_return_tail_is_nan() -> None:
    """Last H bars of forward returns must be NaN (no lookahead)."""
    df, states = _make_inputs(120)
    labels = compute_all_labels(df, states)
    assert labels["fwd_ret_h20"].iloc[-20:].isna().all()
    assert labels["fwd_ret_h5"].iloc[-5:].isna().all()


def test_no_future_leakage_in_labels() -> None:
    """Labels at bar t must not use data beyond bar t in features.

    This tests the labeling logic specifically: recomputing labels on a
    truncated history must produce identical values up to the cut point.
    """
    df, states = _make_inputs(120)
    labels_full = compute_all_labels(df, states)

    cut = 80
    df_cut = df.iloc[:cut]
    states_cut = states.iloc[:cut]
    labels_cut = compute_all_labels(df_cut, states_cut)

    # setup_candidate is not forward-looking — must match exactly up to cut
    for i in range(min(cut, len(labels_cut))):
        full_v = labels_full["setup_candidate"].iloc[i]
        cut_v = labels_cut["setup_candidate"].iloc[i]
        if not (np.isnan(full_v) and np.isnan(cut_v)):
            assert full_v == cut_v, f"setup_candidate differs at bar {i}"
