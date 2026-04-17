"""Features that depend on base-detection output.

These are computed after detect_bases() is called and appended to the feature DataFrame.
Not registered in the global REGISTRY (they require the bases DataFrame as extra input).
The pipeline calls compute_pivot_features() directly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from swingtrader.bases.base_detect import resistance_touches
from swingtrader.features.primitives import atr_wilder


def compute_pivot_features(
    df: pd.DataFrame,
    bases: pd.DataFrame,
    atr: pd.Series | None = None,
) -> pd.DataFrame:
    """Return a DataFrame of pivot-dependent features aligned to df.index.

    Columns produced:
      dist_to_pivot_atr   — (close - pivot) / ATR; <0 inside base, >0 above
      dist_to_pivot_pct   — (close - pivot) / pivot; price vs resistance
      base_length         — pass-through from bases for convenience
      base_depth_pct      — pass-through
      resistance_touches  — rolling count of touches in last 60 bars
      pivot_flatness      — std of daily highs near pivot level (last 30 bars, top-20%)
      breakout_thrust_atr — (close - pivot) / ATR when close > pivot, else NaN
    """
    if atr is None:
        atr = atr_wilder(df, 14)

    pivot = bases["pivot"]
    close = df["close"]

    results: dict[str, pd.Series] = {}

    # Distance to pivot
    atr_safe = atr.replace(0, np.nan)
    results["dist_to_pivot_atr"] = (close - pivot) / atr_safe
    results["dist_to_pivot_pct"] = (close - pivot) / pivot.replace(0, np.nan)

    # Pass-throughs
    results["base_length"] = bases["base_length"].astype(float)
    results["base_depth_pct"] = bases["base_depth_pct"]

    # Resistance touches — computed rolling (one value per bar)
    touch_counts: list[float] = []
    for i in range(len(df)):
        p = pivot.iloc[i]
        if np.isnan(p):
            touch_counts.append(np.nan)
            continue
        start = max(0, i - 60 + 1)
        sub_df = df.iloc[start: i + 1]
        sub_atr = atr.iloc[start: i + 1]
        touch_counts.append(float(resistance_touches(sub_df, p, atr=sub_atr)))
    results["resistance_touches"] = pd.Series(touch_counts, index=df.index)

    # Breakout thrust: (close - pivot) / ATR — only meaningful when close > pivot
    thrust = (close - pivot) / atr_safe
    thrust[close <= pivot] = np.nan
    results["breakout_thrust_atr"] = thrust

    # Pivot flatness: std-dev of highs within 5% of pivot in the last 30 bars
    flatness: list[float] = []
    for i in range(len(df)):
        p = pivot.iloc[i]
        if np.isnan(p):
            flatness.append(np.nan)
            continue
        start = max(0, i - 30 + 1)
        sub = df.iloc[start: i + 1]
        near_top = sub["high"][sub["high"] >= p * 0.95]
        flatness.append(float(near_top.std()) if len(near_top) >= 2 else np.nan)
    results["pivot_flatness"] = pd.Series(flatness, index=df.index)

    return pd.DataFrame(results, index=df.index)
