"""Tests for the breakout state machine.

Focuses on correct state transitions and the no-lookahead property
(state at t can only depend on data up to and including t).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from swingtrader.bases.base_detect import detect_bases
from swingtrader.states.machine import (
    ACCEPTED,
    ARMED,
    BASE,
    CONFIRMED,
    FAILED,
    NONE,
    TRIGGERED,
    compute_states,
)


def _build_df(prices: list[float], volumes: list[float] | None = None) -> pd.DataFrame:
    """Create a minimal OHLCV DataFrame from a close-price list."""
    n = len(prices)
    idx = pd.bdate_range(end=pd.offsets.BDay().rollback(pd.Timestamp.today().normalize()), periods=n)
    c = np.array(prices, dtype=float)
    vol = np.array(volumes, dtype=float) if volumes else np.full(n, 1_000_000.0)
    df = pd.DataFrame(
        {
            "open": c * 0.999,
            "high": c * 1.005,
            "low": c * 0.995,
            "close": c,
            "volume": vol,
        },
        index=idx,
    )
    df.index.name = "date"
    return df


def _run_machine(prices: list[float], **detect_kwargs) -> pd.DataFrame:
    df = _build_df(prices)
    bases = detect_bases(df, **detect_kwargs)
    return compute_states(df, bases)


# ─── Basic state presence ────────────────────────────────────────────────────


def test_no_data_returns_none() -> None:
    df = _build_df([100.0] * 5)
    bases = detect_bases(df, min_days=15)
    states = compute_states(df, bases)
    assert (states["state"] == NONE).all()


def test_flat_base_produces_base_state() -> None:
    """After enough flat bars, state should enter BASE."""
    prices = [100.0] * 50  # perfectly flat
    states = _run_machine(prices, min_days=15, max_days=60, max_depth_pct=0.15)
    # After the warmup (15 bars), the rest should be in BASE or ARMED
    tail = states["state"].iloc[20:]
    assert tail.isin([BASE, ARMED]).any()


def test_trigger_when_price_breaks_pivot() -> None:
    """A large close above the base's max_high should trigger TRIGGERED state.

    Note: base_detect at bar t uses today's high in the window (pivot = max_high including
    today). The state machine therefore compares close against YESTERDAY's pivot to allow
    today's breakout to fire. The breakout bar must have close clearly above yesterday's
    max_high — a +5% move on a 1% ATR base does this cleanly.
    """
    base_price = 100.0
    # 30 flat bars: high = base_price * 1.005 ≈ 100.5  → yesterday's pivot ≈ 100.5
    # Breakout bars: close = 106 >> 100.5 + 0.10 * ~1.0 = 100.6 → TRIGGERED
    base = [base_price] * 30
    breakout = [base_price * 1.06] * 10  # +6% above base, well over threshold
    states = _run_machine(base + breakout, min_days=15, max_days=60, max_depth_pct=0.15)
    after_break = states["state"].iloc[30:]
    assert after_break.isin([TRIGGERED, ACCEPTED, CONFIRMED]).any(), (
        f"Expected TRIGGERED/ACCEPTED/CONFIRMED; got: {after_break.unique().tolist()}"
    )


def test_state_changes_are_monotone_in_lifecycle() -> None:
    """Once TRIGGERED, state must not go back to BASE without FAILED first."""
    base_price = 100.0
    base = [base_price] * 30
    breakout = [base_price * 1.05] * 20
    states = _run_machine(base + breakout, min_days=15, max_days=60, max_depth_pct=0.15)
    in_lifecycle = False
    for state in states["state"]:
        if state == TRIGGERED:
            in_lifecycle = True
        if in_lifecycle and state == BASE:
            pytest.fail("Returned to BASE from an active lifecycle without FAILED")
        if state == FAILED:
            in_lifecycle = False


def test_failed_after_pullback_below_pivot() -> None:
    """Sharp reversal back below trigger_pivot - stop × ATR should produce FAILED."""
    base_price = 100.0
    # 30 flat bars (ATR ≈ 1.0), trigger at +6%, then crash to 90 (well below stop)
    # trigger_pivot ≈ 100.5 (yesterday's max_high), atr_trigger ≈ 1.0
    # FAILED when close ≤ trigger_pivot - 1.0 × atr = ~99.5 → close=90 fires this
    base = [base_price] * 30
    trigger_bar = [base_price * 1.06]     # +6%: above yesterday's pivot + breach
    crash = [base_price * 0.90] * 15     # -10%: well below stop level
    states = _run_machine(base + trigger_bar + crash, min_days=15, max_days=60, max_depth_pct=0.15)
    assert FAILED in states["state"].values, (
        f"Expected FAILED; got states: {states['state'].unique().tolist()}"
    )


def test_days_in_state_increments() -> None:
    """days_in_state should count consecutive bars in the same state."""
    prices = [100.0] * 50
    states = _run_machine(prices, min_days=15, max_days=60, max_depth_pct=0.15)
    # Find a run of BASE
    base_runs = states[states["state"] == BASE]["days_in_state"]
    if not base_runs.empty:
        assert base_runs.max() >= 2


def test_state_changed_flag() -> None:
    prices = [100.0] * 50
    states = _run_machine(prices, min_days=15, max_days=60, max_depth_pct=0.15)
    # First BASE bar should have state_changed=True
    first_base = states[states["state"] == BASE].index[0] if BASE in states["state"].values else None
    if first_base is not None:
        assert bool(states.loc[first_base, "state_changed"]) is True


# ─── No-lookahead property ────────────────────────────────────────────────────


def test_state_at_t_independent_of_future_bars() -> None:
    """State at bar t must be identical whether computed on full history or truncated to t."""
    prices = [100.0] * 30 + [100.0 * 1.04] * 10 + [100.0 * 0.92] * 10
    df_full = _build_df(prices)
    bases_full = detect_bases(df_full, min_days=15, max_days=60, max_depth_pct=0.15)
    states_full = compute_states(df_full, bases_full)

    # Check several mid-points
    for cut in [25, 35, 45]:
        df_cut = df_full.iloc[:cut]
        bases_cut = detect_bases(df_cut, min_days=15, max_days=60, max_depth_pct=0.15)
        states_cut = compute_states(df_cut, bases_cut)
        # States at all bars up to cut-1 must match
        for i in range(min(cut, len(states_cut))):
            full_s = states_full["state"].iloc[i]
            cut_s = states_cut["state"].iloc[i]
            assert full_s == cut_s, (
                f"Lookahead detected: bar {i} state differs "
                f"({full_s!r} full vs {cut_s!r} truncated at {cut})"
            )
