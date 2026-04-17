"""Tests for score generation and ranking."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from swingtrader.models.estimators import (
    ModelBundle,
    feature_cols,
    fit_failure_risk,
    fit_setup_score,
    fit_trade_score,
)
from swingtrader.scoring.generator import _sigmoid, score_features_row
from swingtrader.scoring.ranking import rank_within_state, summary_by_state, top_n_per_state

# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_dataset(n: int = 300, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-01", periods=n)
    df = pd.DataFrame(index=dates)
    df["symbol"] = "SYM"
    df["atr_14"] = rng.uniform(0.5, 3.0, n)
    df["atr_compression_pct"] = rng.uniform(0, 100, n)
    df["volume_dryup"] = rng.uniform(0, 1, n)
    df["close_vs_sma50"] = rng.normal(0, 0.05, n)
    df["dist_to_pivot_atr"] = rng.uniform(0, 3, n)
    df["state"] = rng.choice(["BASE", "ARMED", "TRIGGERED", "ACCEPTED", "NONE"], n, p=[0.25, 0.25, 0.2, 0.2, 0.1])
    df["triggered_breakout"] = (rng.random(n) > 0.7).astype(float)
    df["accepted_breakout"] = (rng.random(n) > 0.6).astype(float)
    df["failed_within_10"] = (rng.random(n) > 0.5).astype(float)
    df["followthrough_confirmed_20"] = (rng.random(n) > 0.4).astype(float)
    df["setup_candidate"] = (rng.random(n) > 0.5).astype(float)
    df["fwd_ret_h5"] = rng.normal(0.1, 1.0, n)
    df["fwd_ret_h10"] = rng.normal(0.15, 1.2, n)
    df["fwd_ret_h20"] = rng.normal(0.2, 1.5, n)
    df.loc[df.index[-10:], ["triggered_breakout", "failed_within_10", "fwd_ret_h20"]] = np.nan
    return df


def _make_bundle(df: pd.DataFrame) -> ModelBundle:
    return ModelBundle(
        setup_score=fit_setup_score(df),
        trade_score=fit_trade_score(df),
        failure_risk=fit_failure_risk(df),
        feature_names=feature_cols(df),
    )


# ── _sigmoid ─────────────────────────────────────────────────────────────────

def test_sigmoid_at_zero_is_half() -> None:
    assert _sigmoid(np.array([0.0]))[0] == pytest.approx(0.5, abs=1e-9)


def test_sigmoid_output_in_0_1() -> None:
    x = np.array([-100, -1, 0, 1, 100], dtype=float)
    out = _sigmoid(x)
    assert np.all((out >= 0) & (out <= 1))


# ── score_features_row ───────────────────────────────────────────────────────

def test_score_features_row_base_state_returns_composite() -> None:
    df = _make_dataset()
    bundle = _make_bundle(df)
    feat = df[bundle.feature_names].iloc[-20].to_dict()
    result = score_features_row(feat, "BASE", bundle, bundle.feature_names)
    assert "composite_score" in result
    # composite = setup_score * (1 - failure_risk) → in [0, 1]
    cs = result["composite_score"]
    if np.isfinite(cs):
        assert 0.0 <= cs <= 1.0


def test_score_features_row_armed_state_has_setup_score() -> None:
    df = _make_dataset()
    bundle = _make_bundle(df)
    feat = df[bundle.feature_names].iloc[-20].to_dict()
    result = score_features_row(feat, "ARMED", bundle, bundle.feature_names)
    assert np.isfinite(result["setup_score"])
    assert np.isnan(result["trade_score"])


def test_score_features_row_triggered_state_has_trade_score() -> None:
    df = _make_dataset()
    bundle = _make_bundle(df)
    feat = df[bundle.feature_names].iloc[-20].to_dict()
    result = score_features_row(feat, "TRIGGERED", bundle, bundle.feature_names)
    assert np.isfinite(result["trade_score"])
    assert np.isnan(result["setup_score"])


def test_score_features_row_none_state_all_nan() -> None:
    df = _make_dataset()
    bundle = _make_bundle(df)
    feat = df[bundle.feature_names].iloc[-20].to_dict()
    result = score_features_row(feat, "NONE", bundle, bundle.feature_names)
    for k in ("setup_score", "trade_score", "failure_risk", "composite_score"):
        assert np.isnan(result[k])


def test_score_features_row_confirmed_state_all_nan() -> None:
    df = _make_dataset()
    bundle = _make_bundle(df)
    feat = df[bundle.feature_names].iloc[-20].to_dict()
    result = score_features_row(feat, "CONFIRMED", bundle, bundle.feature_names)
    assert np.isnan(result["composite_score"])


def test_score_features_row_unfitted_bundle_all_nan() -> None:
    bundle = ModelBundle()  # no models fitted
    feat = {"atr_14": 1.5, "volume_dryup": 0.3}
    result = score_features_row(feat, "BASE", bundle, ["atr_14", "volume_dryup"])
    for k in ("setup_score", "trade_score", "failure_risk", "composite_score"):
        assert np.isnan(result[k])


# ── rank_within_state ─────────────────────────────────────────────────────────

def _make_scores_df(n: int = 20, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    syms = [f"SYM{i}" for i in range(n)]
    states = rng.choice(["BASE", "ARMED", "TRIGGERED", "NONE"], n)
    composites = rng.random(n)
    composites[composites < 0.1] = np.nan  # some NaN
    df = pd.DataFrame({
        "state": states,
        "composite_score": composites,
        "setup_score": rng.random(n),
        "failure_risk": rng.random(n),
        "trade_score": rng.random(n),
    }, index=syms)
    df.index.name = "symbol"
    return df


def test_rank_within_state_adds_column() -> None:
    df = _make_scores_df()
    ranked = rank_within_state(df)
    assert "percentile_rank" in ranked.columns


def test_rank_within_state_none_state_is_nan() -> None:
    df = _make_scores_df(30)
    ranked = rank_within_state(df)
    none_rows = ranked[ranked["state"] == "NONE"]
    assert none_rows["percentile_rank"].isna().all()


def test_rank_within_state_range_0_to_100() -> None:
    df = _make_scores_df(30)
    ranked = rank_within_state(df)
    valid = ranked["percentile_rank"].dropna()
    if len(valid) > 0:
        assert valid.min() >= 0.0
        assert valid.max() <= 100.0


def test_rank_within_state_preserves_row_count() -> None:
    df = _make_scores_df(20)
    ranked = rank_within_state(df)
    assert len(ranked) == len(df)


def test_top_n_per_state_limits_rows() -> None:
    df = _make_scores_df(40)
    ranked = rank_within_state(df)
    top = top_n_per_state(ranked, n=3)
    for state in top["state"].unique():
        assert (top["state"] == state).sum() <= 3


def test_top_n_per_state_sorted_descending() -> None:
    df = _make_scores_df(40)
    ranked = rank_within_state(df)
    top = top_n_per_state(ranked, n=10)
    for state, group in top.groupby("state"):
        ranks = group["percentile_rank"].dropna().tolist()
        assert ranks == sorted(ranks, reverse=True), f"Not sorted for state {state}"


def test_summary_by_state_has_expected_cols() -> None:
    df = _make_scores_df(30)
    ranked = rank_within_state(df)
    summ = summary_by_state(ranked)
    assert {"state", "count", "mean", "median"}.issubset(summ.columns)


# ── composite formula correctness ─────────────────────────────────────────────

def test_composite_base_equals_setup_times_one_minus_failure() -> None:
    """composite = setup_score × (1 − failure_risk) for BASE/ARMED."""
    df = _make_dataset()
    bundle = _make_bundle(df)
    feat = df[bundle.feature_names].iloc[-20].to_dict()
    result = score_features_row(feat, "BASE", bundle, bundle.feature_names)
    ss = result["setup_score"]
    fr = result["failure_risk"]
    cs = result["composite_score"]
    if np.isfinite(ss) and np.isfinite(fr) and np.isfinite(cs):
        assert cs == pytest.approx(ss * (1.0 - fr), abs=1e-9)
