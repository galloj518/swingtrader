"""AVWAP calculation.

AVWAP (Anchored Volume-Weighted Average Price) is computed as:
    AVWAP_t = Σ(TP_i × V_i, i=anchor..t) / Σ(V_i, i=anchor..t)
where TP_i = (H_i + L_i + C_i) / 3 (typical price).

Results are aligned to df.index (NaN before the anchor date).

Assumptions:
  - Bars with zero volume are skipped (cumulative denominator is not advanced).
  - Bars with any NaN OHLCV are skipped.
  - If anchor_date is after df.index.max() or not found in df, returns all NaN.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_avwap(df: pd.DataFrame, anchor_date: pd.Timestamp | None) -> pd.Series:
    """Return AVWAP series anchored at ``anchor_date``, aligned to df.index.

    Returns a Series of NaN if anchor_date is None or outside df's date range.
    """
    if anchor_date is None:
        return pd.Series(np.nan, index=df.index, name="avwap")
    # Find the first bar on or after anchor_date
    valid_idx = df.index[df.index >= anchor_date]
    if valid_idx.empty:
        return pd.Series(np.nan, index=df.index, name="avwap")

    sub = df.loc[valid_idx].copy()
    tp = (sub["high"] + sub["low"] + sub["close"]) / 3
    vol = sub["volume"].copy()
    # Zero or NaN volume → skip those bars (don't advance cumulative sum)
    mask_valid = (vol > 0) & tp.notna() & vol.notna()
    tp_v = (tp * vol).where(mask_valid, 0.0)
    vol = vol.where(mask_valid, 0.0)

    cum_tp_v = tp_v.cumsum()
    cum_vol = vol.cumsum()
    avwap_sub = cum_tp_v / cum_vol.replace(0, np.nan)
    avwap_sub.name = "avwap"

    return avwap_sub.reindex(df.index)


def compute_avwap_std(df: pd.DataFrame, anchor_date: pd.Timestamp | None) -> pd.Series:
    """Anchored VWAP standard deviation (1σ band reference).

    σ_t = sqrt(Σ(V_i × (TP_i - AVWAP_t)^2) / Σ(V_i))
    """
    if anchor_date is None:
        return pd.Series(np.nan, index=df.index, name="avwap_std")
    valid_idx = df.index[df.index >= anchor_date]
    if valid_idx.empty:
        return pd.Series(np.nan, index=df.index, name="avwap_std")

    sub = df.loc[valid_idx].copy()
    tp = (sub["high"] + sub["low"] + sub["close"]) / 3
    vol = sub["volume"].copy()
    mask = (vol > 0) & tp.notna()
    tp_v = (tp * vol).where(mask, 0.0)
    vol_c = vol.where(mask, 0.0)

    cum_tp_v = tp_v.cumsum()
    cum_vol = vol_c.cumsum()
    avwap = cum_tp_v / cum_vol.replace(0, np.nan)

    deviation_sq = vol_c * (tp - avwap) ** 2
    cum_dev_sq = deviation_sq.where(mask, 0.0).cumsum()
    std = np.sqrt(cum_dev_sq / cum_vol.replace(0, np.nan))
    std.name = "avwap_std"
    return std.reindex(df.index)
