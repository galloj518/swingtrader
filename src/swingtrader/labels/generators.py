"""Label generators for model training.

All labels look forward in time and MUST NOT be used as inference features.
Each function takes aligned daily df + states DataFrame and returns a pd.DataFrame
of label columns, indexed like df.

Leakage contract:
  - Labels at time t depend only on data at times t, t+1, …, t+horizon.
  - Features at time t depend only on data at times t, t-1, …, t-lookback.
  - The pipeline writes features and labels to SEPARATE parquet files.
  - The model fitting step (Phase 4) loads both; inference loads only features.

Forward-looking labels return NaN for the last max(horizon) bars where the
full window is not yet available. This is correct — do not fill or impute.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from swingtrader.features.primitives import atr_wilder
from swingtrader.states.machine import ACCEPTED, CONFIRMED, FAILED, TRIGGERED
from swingtrader.utils.config import load_config


def compute_all_labels(
    df: pd.DataFrame,
    states: pd.DataFrame,
    *,
    cfg=None,
) -> pd.DataFrame:
    """Compute the full label set for one symbol.

    Parameters
    ----------
    df : daily OHLCV, indexed by date.
    states : output of states.machine.compute_states(), same index.
    cfg : optional pre-loaded labels config.

    Returns
    -------
    DataFrame with columns:
        setup_candidate, triggered_breakout, accepted_breakout,
        failed_within_10, followthrough_confirmed_20,
        fwd_ret_h5, fwd_ret_h10, fwd_ret_h20
    """
    cfg = cfg or load_config("labels")
    atr = atr_wilder(df, 14)

    labels: dict[str, pd.Series] = {}

    # ── Setup candidate ───────────────────────────────────────────────────────
    # 1 if a valid base exists at this bar (BASE or ARMED).
    sc_cfg = cfg.get("setup_candidate", {})
    max_from_high = float(sc_cfg.get("max_pct_from_high", 0.15))
    h52 = df["high"].rolling(252, min_periods=50).max()
    pct_from_high = (df["close"] / h52.replace(0, np.nan)) - 1

    labels["setup_candidate"] = (
        (states["state"].isin(["BASE", "ARMED"]))
        & (states["pivot"].notna())
        & (pct_from_high >= -max_from_high)
    ).astype(float)

    # ── Triggered breakout ────────────────────────────────────────────────────
    # 1 on the bar where state first becomes TRIGGERED.
    labels["triggered_breakout"] = (
        (states["state"] == TRIGGERED) & states["state_changed"]
    ).astype(float)

    # ── Accepted breakout ─────────────────────────────────────────────────────
    # 1 on the bar where state first becomes ACCEPTED.
    labels["accepted_breakout"] = (
        (states["state"] == ACCEPTED) & states["state_changed"]
    ).astype(float)

    # ── Failed within K bars (forward-looking) ────────────────────────────────
    fail_cfg = cfg.get("failed_breakout", {})
    k_bars = int(fail_cfg.get("horizon_bars", 10))
    failed_mask = (states["state"] == FAILED).astype(float)
    labels["failed_within_10"] = _forward_any(failed_mask, k_bars)

    # ── Follow-through confirmed within H bars (forward-looking) ──────────────
    conf_cfg = cfg.get("followthrough_confirmed", {})
    h_bars = int(conf_cfg.get("horizon_bars", 20))
    confirmed_mask = (states["state"] == CONFIRMED).astype(float)
    labels["followthrough_confirmed_20"] = _forward_any(confirmed_mask, h_bars)

    # ── Forward returns normalised by ATR ─────────────────────────────────────
    fwd_cfg = cfg.get("forward_return", {})
    horizons = list(fwd_cfg.get("horizons_bars", [5, 10, 20]))
    normalize = bool(fwd_cfg.get("normalize_by_atr", True))
    for h in horizons:
        fwd_log = np.log(df["close"].shift(-h) / df["close"])
        labels[f"fwd_ret_h{h}"] = (fwd_log / atr.replace(0, np.nan)) if normalize else fwd_log

    return pd.DataFrame(labels, index=df.index)


def _forward_any(mask: pd.Series, horizon: int) -> pd.Series:
    """Return 1 if mask is 1 at any point in (t, t+horizon], else 0. NaN at the tail."""
    result = np.full(len(mask), np.nan)
    mask_arr = mask.to_numpy(dtype=float)
    for t in range(len(mask) - horizon):
        window = mask_arr[t + 1: t + horizon + 1]
        result[t] = float(np.any(window > 0))
    return pd.Series(result, index=mask.index)
