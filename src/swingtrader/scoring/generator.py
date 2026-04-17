"""Score generation: apply fitted models to the latest features per symbol.

Composite score formula
-----------------------
We avoid hand-picked weights.  The composite is the mathematically natural
product of the two relevant signals, which equals the joint probability under
conditional independence:

  BASE / ARMED    : composite = setup_score × (1 − failure_risk)
  TRIGGERED / ACCEPTED: composite = softmax(trade_score) × (1 − failure_risk)
  All others      : composite = NaN

``softmax(trade_score)`` maps the unbounded Ridge regression output onto [0, 1]
using a logistic transformation centred at 0 (zero ATR-normalised return maps to
0.5; positive returns → >0.5).  This makes it compatible with the multiplicative
composite.

No weights are introduced.  If the product needs calibration, Phase 5 can
stack a thin isotonic layer on top of the composite without introducing
arbitrary human choices.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from swingtrader.models.estimators import (
    ModelBundle,
    feature_cols,
    predict_failure_risk,
    predict_setup_score,
    predict_trade_score,
)
from swingtrader.utils.config import REPO_ROOT
from swingtrader.utils.logging import get_logger

log = get_logger(__name__)

_FEATURES_DIR = REPO_ROOT / "data" / "features"
_STATES_DIR = REPO_ROOT / "data" / "states"
_MODELS_DIR = REPO_ROOT / "models"

_SETUP_STATES = {"BASE", "ARMED"}
_TRADE_STATES = {"TRIGGERED", "ACCEPTED"}

# States that should carry a non-NaN composite
_SCORED_STATES = _SETUP_STATES | _TRADE_STATES


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Logistic function mapping ℝ → (0, 1)."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def load_models(models_dir: Path | None = None) -> ModelBundle:
    """Load the latest production ModelBundle from disk."""
    models_dir = Path(models_dir or _MODELS_DIR)
    return ModelBundle.load(models_dir)


def score_features_row(
    features: dict | pd.Series,
    state: str,
    bundle: ModelBundle,
    feat_names: list[str],
) -> dict:
    """Score a single feature vector (dict or Series) given the current state.

    Returns
    -------
    dict with keys: setup_score, trade_score, failure_risk, composite_score
    All values are float; NaN when model unavailable or state not scored.
    """
    result = {
        "setup_score": float("nan"),
        "trade_score": float("nan"),
        "failure_risk": float("nan"),
        "composite_score": float("nan"),
    }

    if state not in _SCORED_STATES:
        return result

    if isinstance(features, pd.Series):
        x_dict = features.to_dict()
    else:
        x_dict = dict(features)

    x_arr = np.array([[x_dict.get(f, np.nan) for f in feat_names]], dtype=float)

    if state in _SETUP_STATES:
        result["setup_score"] = float(predict_setup_score(bundle.setup_score, x_arr)[0])
        result["failure_risk"] = float(predict_failure_risk(bundle.failure_risk, x_arr)[0])
        ss = result["setup_score"]
        fr = result["failure_risk"]
        if np.isfinite(ss) and np.isfinite(fr):
            result["composite_score"] = ss * (1.0 - fr)

    elif state in _TRADE_STATES:
        raw_trade = float(predict_trade_score(bundle.trade_score, x_arr)[0])
        result["trade_score"] = raw_trade
        result["failure_risk"] = float(predict_failure_risk(bundle.failure_risk, x_arr)[0])
        ts = result["trade_score"]
        fr = result["failure_risk"]
        if np.isfinite(ts) and np.isfinite(fr):
            result["composite_score"] = float(_sigmoid(np.array([ts]))[0]) * (1.0 - fr)

    return result


def score_all_symbols(
    features_dir: Path | None = None,
    states_dir: Path | None = None,
    bundle: ModelBundle | None = None,
    models_dir: Path | None = None,
) -> pd.DataFrame:
    """Apply models to the latest bar of every symbol.

    Returns a DataFrame with columns:
        symbol, state, setup_score, trade_score, failure_risk, composite_score
    Indexed by symbol.
    """
    features_dir = Path(features_dir or _FEATURES_DIR)
    states_dir = Path(states_dir or _STATES_DIR)

    if bundle is None:
        bundle = load_models(models_dir)

    if not bundle.is_fitted:
        log.warning("No fitted models found — all scores will be NaN.")

    feat_names = bundle.feature_names or []

    rows: list[dict] = []
    for feat_path in sorted(features_dir.glob("*.parquet")):
        sym = feat_path.stem
        states_path = states_dir / feat_path.name

        try:
            feat_df = pd.read_parquet(feat_path)
            if feat_df.empty:
                continue
            last_feat = feat_df.iloc[-1]

            state = "NONE"
            if states_path.exists():
                st_df = pd.read_parquet(states_path)
                if not st_df.empty and "state" in st_df.columns:
                    state = str(st_df["state"].iloc[-1])

            # Use bundle's recorded feature names; fall back to what's in the file
            names = feat_names if feat_names else feature_cols(feat_df)
            scores = score_features_row(last_feat, state, bundle, names)
            rows.append({"symbol": sym, "state": state, **scores})

        except Exception as exc:
            log.warning("score_all_symbols: error for %s — %s", sym, exc)
            rows.append({
                "symbol": sym,
                "state": "ERROR",
                "setup_score": float("nan"),
                "trade_score": float("nan"),
                "failure_risk": float("nan"),
                "composite_score": float("nan"),
            })

    if not rows:
        return pd.DataFrame(columns=["symbol", "state", "setup_score", "trade_score", "failure_risk", "composite_score"])

    df = pd.DataFrame(rows).set_index("symbol")
    log.info("Scored %d symbols", len(df))
    return df
