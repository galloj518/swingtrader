"""AVWAP-derived feature generation.

Produces a wide DataFrame of AVWAP features for one symbol, keyed by anchor name.
Each anchor contributes the following columns (prefix = anchor label):
  {anchor}_avwap              — the AVWAP level itself
  {anchor}_dist_atr           — (close - AVWAP) / ATR(14)   [signed; >0 = above]
  {anchor}_stretch_atr        — abs((close - AVWAP) / ATR)  [magnitude only]
  {anchor}_slope_20           — (AVWAP.iloc[-1] - AVWAP.iloc[-20]) / AVWAP.mean × 100
  {anchor}_closes_above_20    — fraction of closes above AVWAP in last 20 bars (0–1)
  {anchor}_reclaim_flag       — 1 if price is above AVWAP and was below 5 bars ago

AVWAP features are not registered in the global feature REGISTRY because they depend on
anchor detection and state history, which are computed separately by the pipeline.
The pipeline calls compute_avwap_features() directly and appends the result.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from swingtrader.avwap.anchors import (
    breakout_day_anchor,
    swing_high_anchor,
    swing_low_anchor,
    ytd_anchor,
)
from swingtrader.avwap.calc import compute_avwap
from swingtrader.features.primitives import atr_wilder
from swingtrader.utils.config import load_config
from swingtrader.utils.logging import get_logger

log = get_logger(__name__)


def compute_avwap_features(
    df: pd.DataFrame,
    state_history: pd.Series | None = None,
    cfg=None,
) -> pd.DataFrame:
    """Return a wide DataFrame of AVWAP features, indexed like df.

    ``state_history`` is a date-indexed Series of state strings used to find
    the breakout-day anchor. Pass None when states are not yet computed.

    Returns an empty DataFrame (with only df's index) if AVWAP config is disabled
    or df is too short.
    """
    cfg = cfg or load_config("avwap_anchors")
    anchor_cfg = cfg.get("anchors", {})
    feature_cfg = cfg.get("features", {})

    if len(df) < 5:
        return pd.DataFrame(index=df.index)

    atr = atr_wilder(df, 14)

    # Build anchor map: label → anchor_date
    anchors: dict[str, pd.Timestamp | None] = {}

    if anchor_cfg.get("ytd", {}).get("enabled", True):
        anchors["ytd"] = ytd_anchor(df)

    if anchor_cfg.get("swing_low", {}).get("enabled", True):
        lookback = int(anchor_cfg["swing_low"].get("lookback_days", 252))
        prom = float(anchor_cfg["swing_low"].get("min_swing_prominence_atr", 1.5))
        anchors["swing_low"] = swing_low_anchor(df, lookback=lookback, min_prominence_atr=prom)

    if anchor_cfg.get("swing_high", {}).get("enabled", True):
        lookback = int(anchor_cfg["swing_high"].get("lookback_days", 252))
        prom = float(anchor_cfg["swing_high"].get("min_swing_prominence_atr", 1.5))
        anchors["swing_high"] = swing_high_anchor(df, lookback=lookback, min_prominence_atr=prom)

    if anchor_cfg.get("breakout_day", {}).get("enabled", True):
        anchors["breakout_day"] = breakout_day_anchor(state_history)

    # Derive features for each anchor
    all_cols: dict[str, pd.Series] = {}

    for label, anchor_date in anchors.items():
        if anchor_date is None:
            # Emit NaN columns so downstream code sees consistent schema
            for suffix in ["avwap", "dist_atr", "stretch_atr", "slope_20", "closes_above_20", "reclaim_flag"]:
                all_cols[f"{label}_{suffix}"] = pd.Series(np.nan, index=df.index)
            continue

        avwap = compute_avwap(df, anchor_date)
        all_cols[f"{label}_avwap"] = avwap

        if feature_cfg.get("distance_atr", {}).get("enabled", True):
            dist = (df["close"] - avwap) / atr.replace(0, np.nan)
            all_cols[f"{label}_dist_atr"] = dist

        if feature_cfg.get("stretch_atr", {}).get("enabled", True):
            all_cols[f"{label}_stretch_atr"] = ((df["close"] - avwap) / atr.replace(0, np.nan)).abs()

        if feature_cfg.get("slope", {}).get("enabled", True):
            slope_window = int(feature_cfg["slope"].get("window", 20))
            avwap_mean = avwap.rolling(slope_window, min_periods=2).mean()
            slope = (avwap - avwap.shift(slope_window)) / avwap_mean.replace(0, np.nan) * 100
            all_cols[f"{label}_slope_20"] = slope

        if feature_cfg.get("closes_above_ratio", {}).get("enabled", True):
            lb = int(feature_cfg["closes_above_ratio"].get("lookback_days", 20))
            above = (df["close"] > avwap).astype(float)
            all_cols[f"{label}_closes_above_20"] = above.rolling(lb, min_periods=1).mean()

        if feature_cfg.get("reclaim_event", {}).get("enabled", True):
            above_now = df["close"] > avwap
            above_5ago = (df["close"] > avwap).shift(5).fillna(False)
            reclaim = (above_now & ~above_5ago).astype(float)
            reclaim[avwap.isna()] = np.nan
            all_cols[f"{label}_reclaim_flag"] = reclaim

    # Confluence: count of anchors where close is above AVWAP
    if feature_cfg.get("confluence", {}).get("enabled", True):
        dist_cols = [c for c in all_cols if c.endswith("_dist_atr")]
        if dist_cols:
            above_count = sum(
                (all_cols[c] > 0).astype(float).fillna(0) for c in dist_cols
            )
            all_cols["avwap_confluence_count"] = above_count

    return pd.DataFrame(all_cols, index=df.index)
