"""Scikit-learn model definitions for the three scoring targets.

Models
------
setup_score    — P(triggered breakout within W bars | current state BASE/ARMED)
                 LogisticRegression + isotonic calibration
trade_score    — E(forward_return_h20_atr | current state TRIGGERED/ACCEPTED)
                 Ridge regression
failure_risk   — P(failed within K bars | state TRIGGERED/ACCEPTED)
                 LogisticRegression + isotonic calibration

Anti-leakage discipline:
  LABEL_COLUMNS is the exhaustive set of columns written by labels/generators.py.
  _feature_cols() excludes ALL of them plus the state/metadata columns so they
  can never accidentally end up in X.

Persistence: models are saved as joblib files to models/ directory.
  models/setup_score.joblib
  models/trade_score.joblib
  models/failure_risk.joblib
  models/meta.json   — fit date, OOS metrics, sklearn version
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from swingtrader.utils.logging import get_logger

log = get_logger(__name__)

# ── Anti-leakage: never let these become features ──────────────────────────

LABEL_COLUMNS: frozenset[str] = frozenset(
    [
        "setup_candidate",
        "triggered_breakout",
        "accepted_breakout",
        "failed_within_10",
        "followthrough_confirmed_20",
        "fwd_ret_h5",
        "fwd_ret_h10",
        "fwd_ret_h20",
    ]
)

# State machine metadata columns (not features)
_META_COLUMNS: frozenset[str] = frozenset(
    [
        "state",
        "trigger_pivot",
        "trigger_atr",
        "trigger_date",
        "consecutive_above",
        "days_in_state",
        "state_changed",
        "pivot",
        # Symbol / universe metadata
        "symbol",
        "user_symbol",
        "provider_symbol",
        "groups",
        "is_portfolio",
        "is_watchlist",
        "is_non_equity",
        "score_eligible",
        "is_benchmark",
        "is_etf",
        "is_ok",
        "error",
    ]
)

# States for each model's training subset
_SETUP_STATES = {"BASE", "ARMED"}
_TRADE_STATES = {"TRIGGERED", "ACCEPTED"}


def feature_cols(df: pd.DataFrame) -> list[str]:
    """Return numeric columns that are safe to use as model features.

    Excludes label columns, state metadata, and non-numeric columns.
    """
    excluded = LABEL_COLUMNS | _META_COLUMNS
    cols = []
    for c in df.columns:
        if c in excluded:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def _build_classifier_pipeline(regularization_c: float = 1.0) -> Pipeline:
    """LogisticRegression with median imputation and standard scaling."""
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=regularization_c, max_iter=1000, solver="lbfgs")),
        ]
    )


def _build_regressor_pipeline(alpha: float = 1.0) -> Pipeline:
    """Ridge regression with median imputation and standard scaling."""
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("reg", Ridge(alpha=alpha)),
        ]
    )


def _prep_xy(
    df: pd.DataFrame,
    target: str,
    *,
    state_filter: set[str] | None = None,
    state_col: str = "state",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract (x_arr, y, row_mask) from the pooled training DataFrame.

    Parameters
    ----------
    df : pooled panel DataFrame with feature + label + state columns
    target : label column name
    state_filter : keep only rows where state is in this set; None = keep all
    state_col : name of the state column in df

    Returns
    -------
    x_arr : (n, p) float array
    y : (n,) float array
    mask : boolean array indicating which rows of df were kept
    """
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found in dataset. Available: {list(df.columns)}")

    feat_names = feature_cols(df)
    if not feat_names:
        raise ValueError("No valid feature columns found after excluding labels and metadata.")

    mask = np.ones(len(df), dtype=bool)

    if state_filter is not None and state_col in df.columns:
        mask &= df[state_col].isin(state_filter).to_numpy()

    # Drop rows where target is NaN (tail of forward-looking labels)
    target_arr = pd.to_numeric(df[target], errors="coerce").to_numpy(dtype=float)
    mask &= np.isfinite(target_arr)

    x_arr = df[feat_names].to_numpy(dtype=float)[mask]
    y = target_arr[mask]
    return x_arr, y, mask


# ── Model fit functions ─────────────────────────────────────────────────────

def fit_setup_score(df: pd.DataFrame, cfg: dict | None = None) -> CalibratedClassifierCV:
    """Fit setup score model: P(triggered within W bars | BASE or ARMED).

    Returns a calibrated classifier that outputs probabilities.
    """
    cfg = cfg or {}
    model_cfg = cfg.get("models", {}).get("setup_score", {})
    reg_c = float(model_cfg.get("C", 1.0))

    x_arr, y, _ = _prep_xy(df, "triggered_breakout", state_filter=_SETUP_STATES)
    log.info("fit_setup_score: %d training rows, %.1f%% positive", len(y), 100 * y.mean())

    base_clf = _build_classifier_pipeline(regularization_c=reg_c)
    calibrated = CalibratedClassifierCV(base_clf, method="isotonic", cv=5)
    calibrated.fit(x_arr, y)
    return calibrated


def fit_trade_score(df: pd.DataFrame, cfg: dict | None = None):  # -> Pipeline
    """Fit trade score model: E(fwd_ret_h20_atr | TRIGGERED or ACCEPTED).

    Returns a Ridge regression pipeline.
    """
    cfg = cfg or {}
    model_cfg = cfg.get("models", {}).get("trade_score", {})
    alpha = float(model_cfg.get("alpha", 1.0))

    # Use h20 return as primary target; fall back to h10 if h20 unavailable
    target = "fwd_ret_h20" if "fwd_ret_h20" in df.columns else "fwd_ret_h10"
    x_arr, y, _ = _prep_xy(df, target, state_filter=_TRADE_STATES)
    log.info("fit_trade_score: %d training rows, mean_y=%.4f", len(y), y.mean())

    reg = _build_regressor_pipeline(alpha=alpha)
    reg.fit(x_arr, y)
    return reg


def fit_failure_risk(df: pd.DataFrame, cfg: dict | None = None) -> CalibratedClassifierCV:
    """Fit failure risk model: P(failed within 10 bars | TRIGGERED or ACCEPTED).

    Returns a calibrated classifier.
    """
    cfg = cfg or {}
    model_cfg = cfg.get("models", {}).get("failure_risk", {})
    reg_c = float(model_cfg.get("C", 1.0))

    x_arr, y, _ = _prep_xy(df, "failed_within_10", state_filter=_TRADE_STATES)
    log.info("fit_failure_risk: %d training rows, %.1f%% positive", len(y), 100 * y.mean())

    base_clf = _build_classifier_pipeline(regularization_c=reg_c)
    calibrated = CalibratedClassifierCV(base_clf, method="isotonic", cv=5)
    calibrated.fit(x_arr, y)
    return calibrated


# ── Persistence ─────────────────────────────────────────────────────────────

def save_model(model: Any, path: Path) -> None:
    """Persist a fitted model to disk using joblib."""
    import joblib
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    log.info("Saved model → %s", path)


def load_model(path: Path) -> Any:
    """Load a joblib model from disk."""
    import joblib
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)


@dataclass
class ModelBundle:
    """Container for all three production models + metadata."""

    setup_score: Any = None
    trade_score: Any = None
    failure_risk: Any = None
    fit_date: str = ""
    oos_metrics: dict = field(default_factory=dict)
    feature_names: list[str] = field(default_factory=list)

    _SETUP_FILE = "setup_score.joblib"
    _TRADE_FILE = "trade_score.joblib"
    _FAILURE_FILE = "failure_risk.joblib"
    _META_FILE = "meta.json"

    def save(self, models_dir: Path) -> None:
        models_dir = Path(models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)
        if self.setup_score is not None:
            save_model(self.setup_score, models_dir / self._SETUP_FILE)
        if self.trade_score is not None:
            save_model(self.trade_score, models_dir / self._TRADE_FILE)
        if self.failure_risk is not None:
            save_model(self.failure_risk, models_dir / self._FAILURE_FILE)
        meta = {
            "fit_date": self.fit_date,
            "oos_metrics": self.oos_metrics,
            "feature_names": self.feature_names,
        }
        (models_dir / self._META_FILE).write_text(json.dumps(meta, indent=2))
        log.info("ModelBundle saved to %s", models_dir)

    @classmethod
    def load(cls, models_dir: Path) -> ModelBundle:
        models_dir = Path(models_dir)
        bundle = cls()
        for attr, fname in [
            ("setup_score", cls._SETUP_FILE),
            ("trade_score", cls._TRADE_FILE),
            ("failure_risk", cls._FAILURE_FILE),
        ]:
            fpath = models_dir / fname
            if fpath.exists():
                setattr(bundle, attr, load_model(fpath))
            else:
                log.warning("Model file missing: %s — will be None", fpath)
        meta_path = models_dir / cls._META_FILE
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            bundle.fit_date = meta.get("fit_date", "")
            bundle.oos_metrics = meta.get("oos_metrics", {})
            bundle.feature_names = meta.get("feature_names", [])
        return bundle

    @property
    def is_fitted(self) -> bool:
        return any(m is not None for m in [self.setup_score, self.trade_score, self.failure_risk])


def predict_setup_score(model: Any, x_arr: np.ndarray) -> np.ndarray:
    """P(setup triggers) for each row. Returns probability of positive class."""
    if model is None:
        return np.full(len(x_arr), np.nan)
    proba = model.predict_proba(x_arr)
    # CalibratedClassifierCV returns shape (n, 2); col 1 = P(class=1)
    return proba[:, 1] if proba.ndim == 2 else proba


def predict_trade_score(model: Any, x_arr: np.ndarray) -> np.ndarray:
    """Expected ATR-normalised 20-bar forward return for each row."""
    if model is None:
        return np.full(len(x_arr), np.nan)
    return model.predict(x_arr)


def predict_failure_risk(model: Any, x_arr: np.ndarray) -> np.ndarray:
    """P(failure within 10 bars) for each row."""
    if model is None:
        return np.full(len(x_arr), np.nan)
    proba = model.predict_proba(x_arr)
    return proba[:, 1] if proba.ndim == 2 else proba
