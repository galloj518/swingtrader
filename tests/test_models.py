"""Tests for model estimators, calibration, and training pipeline."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from swingtrader.models.calibration import (
    aggregate_oos_reports,
    brier_score,
    calibration_report,
    ece,
    reliability_diagram,
)
from swingtrader.models.estimators import (
    LABEL_COLUMNS,
    ModelBundle,
    feature_cols,
    fit_failure_risk,
    fit_setup_score,
    fit_trade_score,
    predict_failure_risk,
    predict_setup_score,
    predict_trade_score,
)

# ── Synthetic dataset ────────────────────────────────────────────────────────

def _make_dataset(n: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    df = pd.DataFrame(index=dates)
    df["symbol"] = "SYM"
    # Features
    df["atr_14"] = rng.uniform(0.5, 3.0, n)
    df["atr_compression_pct"] = rng.uniform(0, 100, n)
    df["volume_dryup"] = rng.uniform(0, 1, n)
    df["close_vs_sma50"] = rng.normal(0, 0.05, n)
    df["daily_rs_63"] = rng.normal(0, 0.1, n)
    df["dist_to_pivot_atr"] = rng.uniform(0, 3, n)
    df["base_length"] = rng.integers(15, 80, n)
    # States — mix for training subset selection
    states = rng.choice(["BASE", "ARMED", "TRIGGERED", "ACCEPTED", "NONE"], n, p=[0.2, 0.2, 0.2, 0.2, 0.2])
    df["state"] = states
    # Labels
    df["triggered_breakout"] = (rng.random(n) > 0.7).astype(float)
    df["accepted_breakout"] = (rng.random(n) > 0.6).astype(float)
    df["failed_within_10"] = (rng.random(n) > 0.5).astype(float)
    df["followthrough_confirmed_20"] = (rng.random(n) > 0.4).astype(float)
    df["setup_candidate"] = (rng.random(n) > 0.5).astype(float)
    df["fwd_ret_h5"] = rng.normal(0.1, 1.0, n)
    df["fwd_ret_h10"] = rng.normal(0.15, 1.2, n)
    df["fwd_ret_h20"] = rng.normal(0.2, 1.5, n)
    # Make last 10 rows have NaN forward labels (tail)
    for col in ["triggered_breakout", "failed_within_10", "fwd_ret_h20"]:
        df.loc[df.index[-10:], col] = np.nan
    return df


# ── feature_cols ─────────────────────────────────────────────────────────────

def test_feature_cols_excludes_labels() -> None:
    df = _make_dataset(50)
    fc = feature_cols(df)
    for lbl in LABEL_COLUMNS:
        assert lbl not in fc, f"Label {lbl} leaked into feature_cols"


def test_feature_cols_excludes_state() -> None:
    df = _make_dataset(50)
    fc = feature_cols(df)
    assert "state" not in fc
    assert "symbol" not in fc


def test_feature_cols_returns_numeric_only() -> None:
    df = _make_dataset(50)
    fc = feature_cols(df)
    for c in fc:
        assert pd.api.types.is_numeric_dtype(df[c]), f"{c} is not numeric"


# ── fit / predict ────────────────────────────────────────────────────────────

def test_fit_setup_score_returns_model() -> None:
    df = _make_dataset(300)
    model = fit_setup_score(df)
    assert model is not None


def test_predict_setup_score_shape_and_range() -> None:
    df = _make_dataset(300)
    model = fit_setup_score(df)
    fc = feature_cols(df)
    x_arr = df[fc].iloc[:10].to_numpy(dtype=float)
    proba = predict_setup_score(model, x_arr)
    assert proba.shape == (10,)
    assert np.all((proba >= 0) & (proba <= 1) | np.isnan(proba))


def test_fit_trade_score_returns_model() -> None:
    df = _make_dataset(300)
    model = fit_trade_score(df)
    assert model is not None


def test_predict_trade_score_shape() -> None:
    df = _make_dataset(300)
    model = fit_trade_score(df)
    fc = feature_cols(df)
    x_arr = df[fc].iloc[:10].to_numpy(dtype=float)
    preds = predict_trade_score(model, x_arr)
    assert preds.shape == (10,)
    assert np.isfinite(preds).all()


def test_fit_failure_risk_returns_model() -> None:
    df = _make_dataset(300)
    model = fit_failure_risk(df)
    assert model is not None


def test_predict_failure_risk_range() -> None:
    df = _make_dataset(300)
    model = fit_failure_risk(df)
    fc = feature_cols(df)
    x_arr = df[fc].iloc[:10].to_numpy(dtype=float)
    proba = predict_failure_risk(model, x_arr)
    assert np.all((proba >= 0) & (proba <= 1) | np.isnan(proba))


def test_predict_with_none_model_returns_nan() -> None:
    x_arr = np.ones((5, 3))
    assert np.all(np.isnan(predict_setup_score(None, x_arr)))
    assert np.all(np.isnan(predict_trade_score(None, x_arr)))
    assert np.all(np.isnan(predict_failure_risk(None, x_arr)))


# ── ModelBundle ──────────────────────────────────────────────────────────────

def test_model_bundle_is_fitted_false_when_empty() -> None:
    bundle = ModelBundle()
    assert not bundle.is_fitted


def test_model_bundle_is_fitted_true_when_model_set() -> None:
    df = _make_dataset(300)
    bundle = ModelBundle(setup_score=fit_setup_score(df))
    assert bundle.is_fitted


def test_model_bundle_save_load(tmp_path) -> None:
    df = _make_dataset(300)
    bundle = ModelBundle(
        setup_score=fit_setup_score(df),
        trade_score=fit_trade_score(df),
        failure_risk=fit_failure_risk(df),
        fit_date="2024-01-01",
        feature_names=feature_cols(df),
    )
    bundle.save(tmp_path)
    loaded = ModelBundle.load(tmp_path)
    assert loaded.is_fitted
    assert loaded.fit_date == "2024-01-01"
    assert loaded.feature_names == feature_cols(df)


# ── Calibration metrics ──────────────────────────────────────────────────────

def test_brier_score_perfect() -> None:
    y = np.array([0.0, 1.0, 0.0, 1.0])
    assert brier_score(y, y) == pytest.approx(0.0, abs=1e-9)


def test_brier_score_worst() -> None:
    y = np.array([0.0, 1.0, 0.0, 1.0])
    y_pred = 1.0 - y
    assert brier_score(y, y_pred) == pytest.approx(1.0, abs=1e-9)


def test_brier_score_with_nan() -> None:
    y = np.array([0.0, np.nan, 1.0])
    p = np.array([0.1, 0.5, 0.9])
    score = brier_score(y, p)
    assert np.isfinite(score)  # NaN rows skipped


def test_ece_perfect_calibration() -> None:
    rng = np.random.default_rng(0)
    probs = rng.uniform(0, 1, 1000)
    labels = (rng.uniform(0, 1, 1000) < probs).astype(float)
    val = ece(labels, probs)
    assert val < 0.1  # approximately calibrated


def test_reliability_diagram_shape() -> None:
    y = np.array([0, 1, 0, 1, 1, 0])
    p = np.array([0.1, 0.9, 0.2, 0.8, 0.7, 0.3])
    rd = reliability_diagram(y, p, n_bins=5)
    assert len(rd["bin_centers"]) == 5
    assert len(rd["fraction_positive"]) == 5
    assert len(rd["counts"]) == 5


def test_calibration_report_keys() -> None:
    y = np.array([0, 1, 0, 1, 0, 1])
    p = np.array([0.2, 0.8, 0.3, 0.7, 0.1, 0.9])
    rep = calibration_report(y, p, label="test")
    assert "brier_score" in rep
    assert "ece" in rep
    assert "prevalence" in rep
    assert "brier_skill_score" in rep
    assert rep["label"] == "test"


def test_aggregate_oos_reports() -> None:
    reports = [
        calibration_report(np.array([0, 1, 0, 1]), np.array([0.2, 0.8, 0.3, 0.7])),
        calibration_report(np.array([1, 0, 1, 0]), np.array([0.7, 0.3, 0.8, 0.2])),
    ]
    agg = aggregate_oos_reports(reports)
    assert "n_folds" in agg
    assert "brier_score_mean" in agg
    assert agg["n_folds"] == 2
    assert np.isfinite(agg["brier_score_mean"])
