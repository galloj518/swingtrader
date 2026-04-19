"""Dashboard context builder.

Builds MA state tables, AVWAP map tables, volume/efficiency blocks,
confluence counts, and checklist items for each top setup.

All values are computed deterministically from available data:
  - data/raw/daily/{SYM}.parquet  — OHLCV with DatetimeIndex
  - data/features/{SYM}.parquet  — 64 feature columns (last row = today)

No values are fabricated; every function catches all exceptions and returns
safe defaults. Callers receive plain Python scalars, strings, lists, or dicts.
"""
from __future__ import annotations

import math
from typing import Any

import pandas as pd

from swingtrader.dashboard.assessments import run_all_assessments
from swingtrader.utils.config import REPO_ROOT
from swingtrader.utils.logging import get_logger

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NAN = float("nan")

FRESH_MAX_DAYS: dict[str, int] = {
    "TRIGGERED": 10,
    "ACCEPTED": 15,
    "ARMED": 30,
    "BASE": 60,
}

_AVWAP_ANCHORS: list[tuple[str, str, str, str | None, str]] = [
    # (label, avwap_key, dist_key, reclaim_key, priority)
    ("YTD",          "ytd_avwap",          "ytd_dist_atr",          "ytd_reclaim_flag",          "primary"),
    ("Swing Low",    "swing_low_avwap",    "swing_low_dist_atr",    "swing_low_reclaim_flag",    "primary"),
    ("Swing High",   "swing_high_avwap",   "swing_high_dist_atr",   "swing_high_reclaim_flag",   "secondary"),
    ("Breakout Day", "breakout_day_avwap", "breakout_day_dist_atr", "breakout_day_reclaim_flag", "secondary"),
]

# Per-anchor enrichment columns available in features parquet
_AVWAP_EXTRAS: dict[str, dict[str, str]] = {
    "YTD": {
        "stretch_atr":     "ytd_stretch_atr",
        "slope_20":        "ytd_slope_20",
        "closes_above_20": "ytd_closes_above_20",
    },
    "Swing Low": {
        "stretch_atr":     "swing_low_stretch_atr",
        "slope_20":        "swing_low_slope_20",
        "closes_above_20": "swing_low_closes_above_20",
    },
    "Swing High": {
        "stretch_atr":     "swing_high_stretch_atr",
        "slope_20":        "swing_high_slope_20",
        "closes_above_20": "swing_high_closes_above_20",
    },
    "Breakout Day": {
        "stretch_atr":     "breakout_day_stretch_atr",
        "slope_20":        "breakout_day_slope_20",
        "closes_above_20": "breakout_day_closes_above_20",
    },
}

# State sets
_CONSTRUCTIVE_STATES = frozenset({"BASE", "ARMED", "TRIGGERED", "ACCEPTED"})

# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _sf(row_or_series: Any, key: str, default: float = _NAN) -> float:
    """Safe float extraction from a Series/dict-like.

    Returns *default* (NaN by default) on any error or non-finite result.
    """
    try:
        if isinstance(row_or_series, (pd.Series, dict)):
            v = row_or_series.get(key, default)
        else:
            v = getattr(row_or_series, key, default)
        fv = float(v)  # type: ignore[arg-type]
        return fv if math.isfinite(fv) else default
    except Exception:
        return default


def _load_features(provider_symbol: str) -> pd.Series | None:
    """Load ``data/features/{sym}.parquet`` and return the last row as a Series.

    Returns None on any error (file missing, empty frame, parse error, …).
    """
    path = REPO_ROOT / "data" / "features" / f"{provider_symbol}.parquet"
    try:
        df = pd.read_parquet(path)
        if df.empty:
            _log.warning("features empty for %s", provider_symbol)
            return None
        return df.iloc[-1]
    except Exception as exc:
        _log.warning("cannot load features for %s: %s", provider_symbol, exc)
        return None


def _load_raw_daily(provider_symbol: str) -> pd.DataFrame | None:
    """Load ``data/raw/daily/{sym}.parquet`` and return a DataFrame with DatetimeIndex.

    Returns None on any error.
    """
    path = REPO_ROOT / "data" / "raw" / "daily" / f"{provider_symbol}.parquet"
    try:
        df = pd.read_parquet(path)
        if df.empty:
            _log.warning("raw daily empty for %s", provider_symbol)
            return None
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        return df.sort_index()
    except Exception as exc:
        _log.warning("cannot load raw daily for %s: %s", provider_symbol, exc)
        return None


# ---------------------------------------------------------------------------
# MA table
# ---------------------------------------------------------------------------


def _sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n).mean()


def _ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()


def build_ma_table(provider_symbol: str, close: float) -> list[dict]:
    """Compute daily MA state table from raw OHLCV.

    Returns a list of dicts, one per MA that has enough history:
      - name       : str  (e.g. "SMA50")
      - value      : float (last-bar MA value)
      - pct_dist   : float ((close - ma) / close * 100; negative = below)
      - slope      : str  ("rising" | "falling" | "flat")
      - bias       : str  (plain-English note about direction if price stays flat)

    Returns [] if data unavailable or close is not finite.
    """
    if not math.isfinite(close):
        return []

    df = _load_raw_daily(provider_symbol)
    if df is None:
        return []

    try:
        closes = df["close"].dropna()
        n_bars = len(closes)
    except Exception as exc:
        _log.warning("MA table - cannot read close column for %s: %s", provider_symbol, exc)
        return []

    # Define MAs: (name, kind, period)
    ma_specs: list[tuple[str, str, int]] = [
        ("SMA5",   "sma", 5),
        ("SMA10",  "sma", 10),
        ("SMA20",  "sma", 20),
        ("SMA50",  "sma", 50),
        ("SMA200", "sma", 200),
        ("EMA20",  "ema", 20),
    ]

    rows: list[dict] = []
    for name, kind, period in ma_specs:
        if n_bars < period:
            continue
        try:
            if kind == "sma":
                ma_series = _sma(closes, period)
            else:
                ma_series = _ema(closes, period)

            ma_vals = ma_series.dropna()
            if len(ma_vals) < 4:
                continue

            ma_now = float(ma_vals.iloc[-1])
            ma_lag = float(ma_vals.iloc[-4])   # 3 bars ago

            # Slope classification
            delta = ma_now - ma_lag
            if delta > 0.001:
                slope = "rising"
            elif delta < -0.001:
                slope = "falling"
            else:
                slope = "flat"

            # Bias: for SMA(n), next MA = MA_now + (new_close - oldest_close) / n
            # If new_close = current close, MA moves by (close - oldest) / n
            # We report the "break-even" close that keeps the MA flat or rising.
            if kind == "sma":
                # The bar that rolls off next is the bar n periods ago
                oldest_idx = closes.index.get_loc(ma_vals.index[-1])  # position of last MA bar
                roll_off_pos = oldest_idx - period + 1
                if roll_off_pos >= 0:
                    oldest_bar = float(closes.iloc[roll_off_pos])
                    if slope == "rising":
                        bias = f"Stays rising if close > {oldest_bar:.2f}"
                    elif slope == "falling":
                        bias = f"Stays falling unless close > {oldest_bar:.2f}"
                    else:
                        bias = f"Flat; rises if close > {oldest_bar:.2f}"
                else:
                    bias = f"Slope: {slope}"
            else:
                # EMA: slope is driven by current price vs EMA.
                # EMA_next = close * k + EMA_now * (1 - k), stays flat when close = EMA_now.
                flat_level = ma_now
                if slope == "rising":
                    bias = f"Stays rising if close > {flat_level:.2f}"
                elif slope == "falling":
                    bias = f"Stays falling if close < {flat_level:.2f}"
                else:
                    bias = f"Flat; needs close > {flat_level:.2f} to start rising"

            pct_dist = (close - ma_now) / close * 100

            # ---- Tomorrow-bias -----------------------------------------------
            # For SMA: the bar rolling off tomorrow is `period` bars ago.
            # need_tomorrow = that bar's close (the price that must be replaced).
            # If need_tomorrow >> close → rolling off a big value → MA headwind.
            # If need_tomorrow << close → rolling off a small value → MA tailwind.
            need_tomorrow: float = math.nan
            tomorrow_bias: str = "neutral"

            if kind == "sma":
                # Position of oldest bar (will roll off when tomorrow's bar is added)
                # ma_vals.index[-1] is the last date with a valid MA.
                # The bar at position (last_ma_pos - period + 1) rolls off next.
                try:
                    last_ma_pos = int(closes.index.get_loc(ma_vals.index[-1]))
                    roll_off_pos = last_ma_pos - period + 1
                    if 0 <= roll_off_pos < len(closes):
                        need_tomorrow = float(closes.iloc[roll_off_pos])
                        if math.isfinite(need_tomorrow) and close > 0:
                            pct_diff = (need_tomorrow - close) / close
                            if pct_diff > 0.015:
                                # Old bar much higher → it's been holding MA up;
                                # dropping it will pull MA down (headwind).
                                tomorrow_bias = "headwind"
                            elif pct_diff < -0.015:
                                # Old bar much lower → dropping it lets MA rise
                                # even if tomorrow closes flat (tailwind).
                                tomorrow_bias = "tailwind"
                            else:
                                tomorrow_bias = "neutral"
                except Exception:
                    pass
            else:
                # EMA: keeps rising as long as close > EMA.
                tomorrow_bias = (
                    "tailwind" if close > ma_now * 1.005
                    else "headwind" if close < ma_now * 0.995
                    else "neutral"
                )
                need_tomorrow = ma_now  # flat-level for EMA is the EMA itself

            rows.append({
                "name":          name,
                "value":         round(ma_now, 2),
                "pct_dist":      round(pct_dist, 2),
                "slope":         slope,
                "bias":          bias,
                "need_tomorrow": round(need_tomorrow, 2) if math.isfinite(need_tomorrow) else None,
                "tomorrow_bias": tomorrow_bias,
            })

        except Exception as exc:
            _log.warning("MA table - error on %s/%s: %s", provider_symbol, name, exc)
            continue

    return rows


# ---------------------------------------------------------------------------
# AVWAP table
# ---------------------------------------------------------------------------


def build_avwap_table(provider_symbol: str, close: float, atr: float) -> list[dict]:
    """Build AVWAP map from feature values.

    Returns a list of dicts, one per anchor where the AVWAP value is finite:
      - anchor    : str
      - avwap     : float
      - pct_dist  : float  ((close - avwap) / close * 100)
      - dist_atr  : float  (signed ATR distance from feature)
      - role      : str    ("support" | "resistance" | "testing")
      - status    : str    ("Accepted above" | "Accepted below" | "Testing")
      - reclaim   : bool | None

    Returns [] if features unavailable or close is not finite.
    """
    if not math.isfinite(close):
        return []

    feat = _load_features(provider_symbol)
    if feat is None:
        return []

    rows: list[dict] = []
    for anchor, avwap_key, dist_key, reclaim_key, priority in _AVWAP_ANCHORS:
        try:
            avwap = _sf(feat, avwap_key)
            if not math.isfinite(avwap) or avwap <= 0:
                continue

            dist_atr = _sf(feat, dist_key)  # may be nan
            pct_dist = (close - avwap) / close * 100

            # Role / status — tighter threshold for "testing" zone (0.25 ATR)
            if math.isfinite(dist_atr) and abs(dist_atr) < 0.25:
                role = "testing"
                status = "Testing"
            elif math.isfinite(dist_atr) and dist_atr > 0:
                role = "support"
                status = "Accepted above"
            elif math.isfinite(dist_atr) and dist_atr < 0:
                role = "resistance"
                status = "Accepted below"
            else:
                # Fall back to raw price comparison if dist_atr missing
                if close > avwap:
                    role = "support"
                    status = "Accepted above"
                else:
                    role = "resistance"
                    status = "Accepted below"

            # Reclaim flag
            reclaim: bool | None = None
            if reclaim_key is not None:
                rv = _sf(feat, reclaim_key)
                if math.isfinite(rv):
                    reclaim = bool(rv >= 0.5)

            # Extra enrichment columns: stretch_atr, slope_20, closes_above_20
            extras = _AVWAP_EXTRAS.get(anchor, {})
            stretch_atr     = _sf(feat, extras.get("stretch_atr", ""))
            slope_20        = _sf(feat, extras.get("slope_20", ""))
            closes_above_20 = _sf(feat, extras.get("closes_above_20", ""))

            # Slope label for the AVWAP anchor
            if math.isfinite(slope_20):
                if slope_20 > 0.002:
                    slope_label = "rising"
                elif slope_20 < -0.002:
                    slope_label = "falling"
                else:
                    slope_label = "flat"
            else:
                slope_label = None

            rows.append({
                "anchor":           anchor,
                "priority":         priority,
                "avwap":            round(avwap, 2),
                "pct_dist":         round(pct_dist, 2),
                "dist_atr":         round(dist_atr, 3) if math.isfinite(dist_atr) else _NAN,
                "role":             role,
                "status":           status,
                "reclaim":          reclaim,
                # Enrichment
                "stretch_atr":     round(stretch_atr, 3)     if math.isfinite(stretch_atr)     else None,
                "slope_20":        round(slope_20, 4)        if math.isfinite(slope_20)        else None,
                "slope_label":     slope_label,
                "closes_above_20": round(closes_above_20, 3) if math.isfinite(closes_above_20) else None,
            })

        except Exception as exc:
            _log.warning("AVWAP table - error on %s/%s: %s", provider_symbol, anchor, exc)
            continue

    # ── Dynamic anchors (WTD, MTD) computed from raw daily ────────────────────
    # These are not stored in features parquet; compute on-the-fly from raw OHLCV.
    try:
        raw_df = _load_raw_daily(provider_symbol)
        if raw_df is not None and not raw_df.empty and math.isfinite(atr) and atr > 0:
            _dynamic_anchors: list[tuple[str, pd.Timestamp | None]] = []

            latest_date = raw_df.index.max()

            # WTD: first trading day of the current ISO week
            week_start = latest_date - pd.Timedelta(days=latest_date.dayofweek)
            wtd_dates = raw_df.index[raw_df.index >= week_start]
            wtd_anchor: pd.Timestamp | None = pd.Timestamp(wtd_dates[0]) if len(wtd_dates) > 1 else None
            if wtd_anchor is not None:
                _dynamic_anchors.append(("WTD", wtd_anchor))

            # MTD: first trading day of the current calendar month
            month_start = latest_date.replace(day=1)
            mtd_dates = raw_df.index[raw_df.index >= month_start]
            mtd_anchor: pd.Timestamp | None = pd.Timestamp(mtd_dates[0]) if len(mtd_dates) > 1 else None
            if mtd_anchor is not None:
                _dynamic_anchors.append(("MTD", mtd_anchor))

            for dyn_label, anchor_date in _dynamic_anchors:
                try:
                    # Compute AVWAP from anchor_date to latest using close x volume / cumvol
                    since = raw_df.loc[anchor_date:]
                    if len(since) < 1:
                        continue
                    vp = (since["close"] * since["volume"]).cumsum()
                    cv = since["volume"].cumsum()
                    avwap_dyn = float(vp.iloc[-1] / cv.iloc[-1]) if float(cv.iloc[-1]) > 0 else math.nan
                    if not math.isfinite(avwap_dyn) or avwap_dyn <= 0:
                        continue

                    dist_atr_dyn = (close - avwap_dyn) / atr
                    pct_dist_dyn = (close - avwap_dyn) / close * 100

                    if abs(dist_atr_dyn) < 0.25:
                        dyn_role, dyn_status = "testing", "Testing"
                    elif dist_atr_dyn > 0:
                        dyn_role, dyn_status = "support", "Accepted above"
                    else:
                        dyn_role, dyn_status = "resistance", "Accepted below"

                    rows.append({
                        "anchor":           dyn_label,
                        "priority":         "dynamic",
                        "anchor_date":      str(anchor_date.date()),
                        "avwap":            round(avwap_dyn, 2),
                        "pct_dist":         round(pct_dist_dyn, 2),
                        "dist_atr":         round(dist_atr_dyn, 3),
                        "role":             dyn_role,
                        "status":           dyn_status,
                        "reclaim":          None,
                        "stretch_atr":      None,
                        "slope_20":         None,
                        "slope_label":      None,
                        "closes_above_20":  None,
                    })
                except Exception as exc:
                    _log.debug("AVWAP dynamic anchor %s/%s: %s", provider_symbol, dyn_label, exc)
                    continue

    except Exception as exc:
        _log.debug("AVWAP dynamic anchors error for %s: %s", provider_symbol, exc)

    return rows


# ---------------------------------------------------------------------------
# Volume / efficiency block
# ---------------------------------------------------------------------------


def build_volume_block(provider_symbol: str, close: float, atr: float) -> dict:
    """Build volume/efficiency metrics block.

    Returns a dict with floats and labelled strings.  All values that cannot be
    computed are returned as NaN (floats) or descriptive strings ("—").
    """
    defaults: dict[str, Any] = {
        "volume_dryup_pct":        _NAN,
        "atr_compression_pct":     _NAN,
        "range_compression":       _NAN,
        "vol_contraction_5_20":    _NAN,
        "vol_contraction_10_50":   _NAN,
        "dollar_volume_log":       _NAN,
        "breakout_thrust_atr":     _NAN,
        "weekly_vol_dryup":        _NAN,
        "weekly_atr_compression":  _NAN,
        "down_close_ratio":        _NAN,
        "relative_vol_label":      "—",
        "compression_label":       "—",
    }

    feat = _load_features(provider_symbol)
    if feat is None:
        return defaults

    try:
        vd            = _sf(feat, "volume_dryup")
        atr_pct       = _sf(feat, "atr_compression_pct")
        rng_comp      = _sf(feat, "range_compression")
        vc_5_20       = _sf(feat, "volatility_contraction_5_20")
        vc_10_50      = _sf(feat, "volatility_contraction_10_50")
        dv_log        = _sf(feat, "dollar_volume_log")
        thrust        = _sf(feat, "breakout_thrust_atr")
        wk_vd         = _sf(feat, "weekly_volume_dryup")
        wk_atr        = _sf(feat, "weekly_atr_compression")

        # Relative volume label based on volume_dryup (higher = more dryup)
        if math.isfinite(vd):
            if vd >= 0.7:
                rel_vol_label = "Very Low"
            elif vd >= 0.4:
                rel_vol_label = "Low"
            elif vd >= 0.1:
                rel_vol_label = "Normal"
            else:
                rel_vol_label = "High"
        else:
            rel_vol_label = "—"

        # Compression label based on atr_compression_pct (0-100 percentile)
        if math.isfinite(atr_pct):
            if atr_pct < 30:
                comp_label = f"Tight (<30th pct) - {atr_pct:.0f}th percentile"
            elif atr_pct < 60:
                comp_label = f"Moderate - {atr_pct:.0f}th percentile"
            else:
                comp_label = f"Elevated - {atr_pct:.0f}th percentile"
        else:
            comp_label = "—"

        # --- Additional metrics from raw daily OHLCV ---
        down_close_ratio = _NAN
        try:
            df_raw = _load_raw_daily(provider_symbol)
            if df_raw is not None and len(df_raw) >= 20:
                recent = df_raw.tail(20)
                n_down = int((recent["close"] < recent["open"]).sum())
                down_close_ratio = n_down / 20
        except Exception:
            pass

        return {
            "volume_dryup_pct":        round(vd * 100, 1)   if math.isfinite(vd)      else _NAN,
            "atr_compression_pct":     round(atr_pct, 1)    if math.isfinite(atr_pct) else _NAN,
            "range_compression":       round(rng_comp, 3)   if math.isfinite(rng_comp) else _NAN,
            "vol_contraction_5_20":    round(vc_5_20, 3)    if math.isfinite(vc_5_20)  else _NAN,
            "vol_contraction_10_50":   round(vc_10_50, 3)   if math.isfinite(vc_10_50) else _NAN,
            "dollar_volume_log":       round(dv_log, 3)     if math.isfinite(dv_log)   else _NAN,
            "breakout_thrust_atr":     round(thrust, 3)     if math.isfinite(thrust)   else _NAN,
            "weekly_vol_dryup":        round(wk_vd, 3)      if math.isfinite(wk_vd)    else _NAN,
            "weekly_atr_compression":  round(wk_atr, 1)     if math.isfinite(wk_atr)   else _NAN,
            "down_close_ratio":        round(down_close_ratio, 3) if math.isfinite(down_close_ratio) else _NAN,
            "relative_vol_label":      rel_vol_label,
            "compression_label":       comp_label,
        }

    except Exception as exc:
        _log.warning("volume block - error for %s: %s", provider_symbol, exc)
        return defaults


# ---------------------------------------------------------------------------
# Confluence
# ---------------------------------------------------------------------------


def build_confluence(
    close: float,
    pivot: float,
    atr: float,
    avwap_table: list[dict],
    levels: dict,
) -> dict:
    """Count key levels that cluster within 0.5 ATR of current price.

    Returns:
      nearby_count  : int
      nearby_levels : list of {name, value, dist_atr}
      cluster_role  : str ("support cluster" | "resistance cluster" |
                           "trigger zone" | "scattered")
    """
    empty: dict = {
        "nearby_count": 0,
        "nearby_levels": [],
        "cluster_role": "scattered",
    }

    try:
        if not (math.isfinite(close) and math.isfinite(atr) and atr > 0):
            return empty

        threshold = 0.5 * atr

        # Collect named levels from the levels dict
        named_levels: list[tuple[str, float]] = []

        level_keys = [
            ("pivot",    "Pivot"),
            ("stop",     "Stop"),
            ("t1",       "T1"),
            ("t2",       "T2"),
            ("s1",       "S1"),
            ("s2",       "S2"),
            ("r1",       "R1"),
            ("r2",       "R2"),
        ]
        for key, label in level_keys:
            raw = levels.get(key, _NAN)
            try:
                val = float(raw)
                if math.isfinite(val):
                    named_levels.append((label, val))
            except (TypeError, ValueError):
                pass

        # Add AVWAP levels
        for row in avwap_table:
            try:
                av = float(row.get("avwap", _NAN))
                if math.isfinite(av):
                    named_levels.append((f"AVWAP {row.get('anchor', '?')}", av))
            except Exception:
                continue

        # Find nearby
        nearby: list[dict] = []
        support_count = 0
        resistance_count = 0

        for name, val in named_levels:
            dist = abs(val - close)
            if dist <= threshold:
                dist_atr = (close - val) / atr  # positive = price above level
                nearby.append({
                    "name":     name,
                    "value":    round(val, 2),
                    "dist_atr": round(dist_atr, 3),
                })
                if val < close:
                    support_count += 1
                else:
                    resistance_count += 1

        # Cluster role
        n = len(nearby)
        if n == 0:
            cluster_role = "scattered"
        elif support_count >= 2 and support_count > resistance_count:
            cluster_role = "support cluster"
        elif resistance_count >= 2 and resistance_count > support_count:
            cluster_role = "resistance cluster"
        elif n >= 2:
            cluster_role = "trigger zone"
        else:
            cluster_role = "scattered"

        return {
            "nearby_count":  n,
            "nearby_levels": nearby,
            "cluster_role":  cluster_role,
        }

    except Exception as exc:
        _log.warning("confluence - unexpected error: %s", exc)
        return empty


# ---------------------------------------------------------------------------
# Checklist
# ---------------------------------------------------------------------------


def build_checklist(
    provider_symbol: str,
    snapshot_row: pd.Series,
    levels: dict,
    assessments: dict | None = None,
) -> list[dict]:
    """Build structured trade checklist (15 core items + up to 3 assessment items).

    Each item: {"item": str, "result": "pass"|"fail"|"neutral", "reason": str}

    Core items (1-15) use features parquet and snapshot columns.
    Assessment items (16-18) use the pre-computed assessments dict if provided;
    these cover base quality, continuation pattern, and clean air.

    Parameters
    ----------
    assessments : optional dict from run_all_assessments(); if None, the
                  assessment items are skipped (not shown as failures).
    """
    try:
        feat = _load_features(provider_symbol)
        # Feature helper: falls back gracefully if feat is None
        def _ff(key: str, default: float = _NAN) -> float:
            if feat is None:
                return default
            return _sf(feat, key, default)

        def _sr(key: str, default: float = _NAN) -> float:
            return _sf(snapshot_row, key, default)

        def _item(item_text: str, result: str, reason: str) -> dict:
            return {"item": item_text, "result": result, "reason": reason}

        items: list[dict] = []

        # 1. Higher timeframe trend
        # Prefer snapshot_row values (always present); features is secondary source.
        spy_above = _sr("regime_spy_above_200sma")
        if not math.isfinite(spy_above):
            spy_above = _ff("regime_spy_above_200sma")
        spy_trend = _sr("regime_spy_trend")
        if not math.isfinite(spy_trend):
            spy_trend = _ff("regime_spy_trend")
        if spy_above >= 0.5 or spy_trend > 0:
            items.append(_item(
                "Higher timeframe trend",
                "pass",
                "SPY above 200SMA" if spy_above >= 0.5 else f"SPY trend score: {spy_trend:.2f}",
            ))
        elif math.isfinite(spy_above) or math.isfinite(spy_trend):
            items.append(_item(
                "Higher timeframe trend",
                "fail",
                f"SPY below 200SMA (regime_spy_above_200sma={spy_above:.2f})",
            ))
        else:
            items.append(_item("Higher timeframe trend", "neutral", "Regime data unavailable"))

        # 2. Weekly trend constructive
        wk_dist = _ff("weekly_dist_wma10")
        if math.isfinite(wk_dist):
            if wk_dist > 0:
                items.append(_item(
                    "Weekly trend constructive",
                    "pass",
                    f"Price {wk_dist * 100:.1f}% above 10-week WMA",
                ))
            else:
                items.append(_item(
                    "Weekly trend constructive",
                    "fail",
                    f"Price {abs(wk_dist) * 100:.1f}% below 10-week WMA",
                ))
        else:
            items.append(_item("Weekly trend constructive", "neutral", "Weekly WMA data unavailable"))

        # 3. Daily structure
        state = str(snapshot_row.get("state", "NONE"))
        dist_pivot_atr = _sr("dist_to_pivot_atr")
        if state in _CONSTRUCTIVE_STATES and math.isfinite(dist_pivot_atr) and -2.0 <= dist_pivot_atr <= 0.5:
            items.append(_item(
                "Daily structure",
                "pass",
                f"State={state}, {dist_pivot_atr:.2f} ATR from pivot",
            ))
        elif state not in _CONSTRUCTIVE_STATES:
            items.append(_item(
                "Daily structure",
                "fail",
                f"State={state} — not a constructive setup state",
            ))
        else:
            result = "neutral"
            reason = (
                f"State={state} but dist_to_pivot={dist_pivot_atr:.2f} ATR (out of -2 to +0.5 range)"
                if math.isfinite(dist_pivot_atr)
                else f"State={state}, pivot distance unavailable"
            )
            items.append(_item("Daily structure", result, reason))

        # 4. Pivot clarity
        res_touches = _ff("resistance_touches")
        flatness = _ff("pivot_flatness")
        if math.isfinite(res_touches) and res_touches >= 2:
            items.append(_item(
                "Pivot clarity",
                "pass",
                f"{res_touches:.0f} resistance touches at pivot",
            ))
        elif math.isfinite(flatness) and flatness >= 0.6:
            items.append(_item(
                "Pivot clarity",
                "pass",
                f"Pivot flatness score: {flatness:.2f}",
            ))
        elif math.isfinite(res_touches) or math.isfinite(flatness):
            touches_str = f"{res_touches:.0f}" if math.isfinite(res_touches) else "?"
            flat_str = f"{flatness:.2f}" if math.isfinite(flatness) else "?"
            items.append(_item(
                "Pivot clarity",
                "fail",
                f"Only {touches_str} touches; flatness={flat_str}",
            ))
        else:
            items.append(_item("Pivot clarity", "neutral", "Pivot data unavailable"))

        # 5. Compression present
        atr_pct = _ff("atr_compression_pct")
        if math.isfinite(atr_pct):
            if atr_pct < 40:
                items.append(_item(
                    "Compression present",
                    "pass",
                    f"ATR at {atr_pct:.0f}th percentile (tight)",
                ))
            else:
                items.append(_item(
                    "Compression present",
                    "fail",
                    f"ATR at {atr_pct:.0f}th percentile (not compressed)",
                ))
        else:
            items.append(_item("Compression present", "neutral", "ATR compression data unavailable"))

        # 6. Volume dry-up
        vd = _ff("volume_dryup")
        if math.isfinite(vd):
            if vd > 0.3:
                items.append(_item(
                    "Volume dry-up",
                    "pass",
                    f"Volume dry-up score: {vd:.2f}",
                ))
            else:
                items.append(_item(
                    "Volume dry-up",
                    "fail",
                    f"Volume dry-up score: {vd:.2f} (below threshold 0.30)",
                ))
        else:
            items.append(_item("Volume dry-up", "neutral", "Volume dry-up data unavailable"))

        # 7. Relative strength
        rs63 = _ff("daily_rs_63")
        if math.isfinite(rs63):
            if rs63 > 0:
                items.append(_item(
                    "Relative strength",
                    "pass",
                    f"RS-63: +{rs63 * 100:.1f}% vs SPY",
                ))
            elif rs63 < -0.03:
                items.append(_item(
                    "Relative strength",
                    "fail",
                    f"RS-63: {rs63 * 100:.1f}% vs SPY (underperforming)",
                ))
            else:
                items.append(_item(
                    "Relative strength",
                    "neutral",
                    f"RS-63: {rs63 * 100:.1f}% vs SPY (near flat)",
                ))
        else:
            items.append(_item("Relative strength", "neutral", "RS data unavailable"))

        # 8. YTD AVWAP acceptance
        ytd_dist = _ff("ytd_dist_atr")
        if math.isfinite(ytd_dist):
            if ytd_dist > 0.2:
                items.append(_item(
                    "YTD AVWAP acceptance",
                    "pass",
                    f"Price {ytd_dist:.2f} ATR above YTD AVWAP",
                ))
            elif ytd_dist < -0.5:
                items.append(_item(
                    "YTD AVWAP acceptance",
                    "fail",
                    f"Price {abs(ytd_dist):.2f} ATR below YTD AVWAP",
                ))
            else:
                items.append(_item(
                    "YTD AVWAP acceptance",
                    "neutral",
                    f"Testing YTD AVWAP ({ytd_dist:.2f} ATR)",
                ))
        else:
            items.append(_item("YTD AVWAP acceptance", "neutral", "YTD AVWAP data unavailable"))

        # 9. MA alignment (above SMA50)
        cvs50 = _ff("close_vs_sma50")
        if math.isfinite(cvs50):
            if cvs50 > 0:
                items.append(_item(
                    "MA alignment (above SMA50)",
                    "pass",
                    f"Price {cvs50 * 100:.1f}% above SMA50",
                ))
            elif cvs50 < -0.05:
                items.append(_item(
                    "MA alignment (above SMA50)",
                    "fail",
                    f"Price {abs(cvs50) * 100:.1f}% below SMA50",
                ))
            else:
                items.append(_item(
                    "MA alignment (above SMA50)",
                    "neutral",
                    f"Price near SMA50 ({cvs50 * 100:.1f}%)",
                ))
        else:
            items.append(_item("MA alignment (above SMA50)", "neutral", "SMA50 data unavailable"))

        # 10. Not overextended
        d2p = _sr("dist_to_pivot_atr")
        if math.isfinite(d2p):
            if d2p < 3.0:
                items.append(_item(
                    "Not overextended",
                    "pass",
                    f"{d2p:.2f} ATR from pivot (within 3 ATR)",
                ))
            elif d2p >= 5.0:
                items.append(_item(
                    "Not overextended",
                    "fail",
                    f"{d2p:.2f} ATR from pivot (overextended ≥5 ATR)",
                ))
            else:
                items.append(_item(
                    "Not overextended",
                    "neutral",
                    f"{d2p:.2f} ATR from pivot (borderline 3-5 ATR)",
                ))
        else:
            items.append(_item("Not overextended", "neutral", "Pivot distance unavailable"))

        # 11. Failure risk acceptable
        fail_risk_raw = snapshot_row.get("failure_risk", _NAN)
        try:
            fail_risk = float(fail_risk_raw)
            if not math.isfinite(fail_risk):
                raise ValueError
        except (TypeError, ValueError):
            fail_risk = _NAN

        if math.isfinite(fail_risk):
            if fail_risk < 0.50:
                items.append(_item(
                    "Failure risk acceptable",
                    "pass",
                    f"Failure risk: {fail_risk:.0%}",
                ))
            elif fail_risk > 0.65:
                items.append(_item(
                    "Failure risk acceptable",
                    "fail",
                    f"Failure risk: {fail_risk:.0%} (elevated)",
                ))
            else:
                items.append(_item(
                    "Failure risk acceptable",
                    "neutral",
                    f"Failure risk: {fail_risk:.0%} (borderline 50-65%)",
                ))
        else:
            items.append(_item("Failure risk acceptable", "neutral", "Failure risk unavailable"))

        # 12. Market regime supportive
        reg_trend = _sr("regime_spy_trend")
        if not math.isfinite(reg_trend):
            reg_trend = _ff("regime_spy_trend")
        if math.isfinite(reg_trend):
            if reg_trend > 0:
                items.append(_item(
                    "Market regime supportive",
                    "pass",
                    f"SPY trend score: {reg_trend:.2f} (positive)",
                ))
            elif reg_trend == 0:
                items.append(_item(
                    "Market regime supportive",
                    "neutral",
                    "SPY trend score: 0 (neutral)",
                ))
            else:
                items.append(_item(
                    "Market regime supportive",
                    "fail",
                    f"SPY trend score: {reg_trend:.2f} (negative)",
                ))
        else:
            items.append(_item("Market regime supportive", "neutral", "Regime trend data unavailable"))

        # 13. R/R acceptable
        rr_raw = levels.get("risk_reward_t1", _NAN)
        try:
            rr = float(rr_raw)
            if not math.isfinite(rr):
                raise ValueError
        except (TypeError, ValueError):
            rr = _NAN

        if math.isfinite(rr):
            if rr >= 1.5:
                items.append(_item(
                    "R/R acceptable",
                    "pass",
                    f"Risk/reward to T1: {rr:.2f}:1",
                ))
            else:
                items.append(_item(
                    "R/R acceptable",
                    "fail",
                    f"Risk/reward to T1: {rr:.2f}:1 (below 1.5:1 minimum)",
                ))
        else:
            items.append(_item("R/R acceptable", "neutral", "R/R not computable (missing levels)"))

        # 14. Base freshness
        days = _sr("days_in_state")
        state_for_fresh = str(snapshot_row.get("state", "NONE"))
        max_days = FRESH_MAX_DAYS.get(state_for_fresh)

        if max_days is not None and math.isfinite(days):
            days_int = int(days)
            if days_int <= max_days:
                items.append(_item(
                    "Base freshness",
                    "pass",
                    f"{days_int} days in {state_for_fresh} (max {max_days})",
                ))
            else:
                items.append(_item(
                    "Base freshness",
                    "fail",
                    f"{days_int} days in {state_for_fresh} (exceeds max {max_days})",
                ))
        elif max_days is None:
            items.append(_item(
                "Base freshness",
                "neutral",
                f"State {state_for_fresh!r} has no freshness threshold",
            ))
        else:
            items.append(_item("Base freshness", "neutral", "Days-in-state data unavailable"))

        # 15. Score quality
        cscore_raw = snapshot_row.get("composite_score", _NAN)
        try:
            cscore = float(cscore_raw)
            if not math.isfinite(cscore):
                raise ValueError
        except (TypeError, ValueError):
            cscore = _NAN

        if math.isfinite(cscore):
            if cscore >= 0.40:
                items.append(_item(
                    "Score quality",
                    "pass",
                    f"Composite score: {cscore:.2f}",
                ))
            elif cscore >= 0.20:
                items.append(_item(
                    "Score quality",
                    "neutral",
                    f"Composite score: {cscore:.2f} (marginal 0.20-0.40)",
                ))
            else:
                items.append(_item(
                    "Score quality",
                    "fail",
                    f"Composite score: {cscore:.2f} (below 0.20)",
                ))
        else:
            items.append(_item("Score quality", "neutral", "Composite score unavailable"))

        # ── Assessment items (16-18) — only if assessments were pre-computed ────
        if assessments:
            # 16. Base quality
            bq = assessments.get("base_quality", {})
            bq_grade = bq.get("grade", "")
            bq_score = bq.get("score", math.nan)
            if isinstance(bq_score, float) and math.isfinite(bq_score):
                bq_note = bq.get("notes", [""])[0] if bq.get("notes") else bq.get("label", "")
                if bq_grade in ("A", "B"):
                    items.append(_item("Base quality", "pass", f"Grade {bq_grade}: {bq_note}"))
                elif bq_grade == "D":
                    items.append(_item("Base quality", "fail", f"Grade {bq_grade}: {bq_note}"))
                else:
                    items.append(_item("Base quality", "neutral", f"Grade {bq_grade}: {bq_note}"))

            # 17. Continuation pattern
            cp = assessments.get("continuation", {})
            patterns = cp.get("patterns", [])
            strongest = cp.get("strongest_pattern", "none")
            cp_notes = cp.get("notes", [])
            if patterns:
                cp_desc = cp_notes[0] if cp_notes else strongest
                items.append(_item("Continuation pattern", "pass", cp_desc))
            else:
                items.append(_item("Continuation pattern", "neutral",
                                   "No tight continuation pattern detected (NR7/3WT/tight5d)"))

            # 18. Clean air above
            ca = assessments.get("clean_air", {})
            ca_score = ca.get("score", math.nan)
            ca_atrs  = ca.get("clean_air_atrs", math.nan)
            ca_notes = ca.get("notes", [])
            if isinstance(ca_score, float) and math.isfinite(ca_score):
                ca_desc = ca_notes[0] if ca_notes else f"Clean air: {ca_atrs:.1f} ATR"
                if ca_score >= 0.70:
                    items.append(_item("Clean air above pivot", "pass", ca_desc))
                elif ca_score <= 0.30:
                    items.append(_item("Clean air above pivot", "fail", ca_desc))
                else:
                    items.append(_item("Clean air above pivot", "neutral", ca_desc))
            else:
                items.append(_item("Clean air above pivot", "neutral", "AVWAP resistance data unavailable"))

        return items

    except Exception as exc:
        _log.warning("checklist - unexpected error for %s: %s", provider_symbol, exc)
        return []


# ---------------------------------------------------------------------------
# Trend state (Tier 2 — from features parquet)
# ---------------------------------------------------------------------------


def build_trend_state(provider_symbol: str, snapshot_row: pd.Series) -> dict:
    """Build trend state from features parquet (requires file I/O).

    Returns a dict:
      daily_trend_state   : str  — more precise than snapshot-only version
      weekly_trend_state  : str  — weekly structure classification
      daily_detail        : dict — raw values used for classification
      weekly_detail       : dict — raw weekly values
    """
    feat = _load_features(provider_symbol)

    def _ff(key: str) -> float:
        if feat is None:
            return _NAN
        return _sf(feat, key, _NAN)

    def _sr(key: str) -> float:
        return _sf(snapshot_row, key, _NAN)

    # Daily: use features for more complete picture
    cvs50    = _ff("close_vs_sma50")  if not math.isfinite(_sr("close_vs_sma50"))  else _sr("close_vs_sma50")
    cvs200   = _ff("close_vs_sma200")
    slope50   = _ff("slope_50")
    cvs_ema20 = _ff("close_vs_ema20")
    rs63      = _sr("daily_rs_63")
    if not math.isfinite(rs63):
        rs63 = _ff("daily_rs_63")

    # Weekly
    wk_dist  = _ff("weekly_dist_wma10")
    wk_slope = _ff("weekly_trend_slope_26")
    wk_rs26  = _ff("weekly_rs_26")
    wk_dist40 = _ff("weekly_dist_wma40")

    # --- Daily classification ---
    if not math.isfinite(cvs50) and not math.isfinite(cvs200):
        daily = "unknown"
    elif math.isfinite(cvs200) and cvs200 < -0.06:
        daily = "broken"  # below 200 SMA — major structural breakdown
    elif math.isfinite(cvs200) and cvs200 < 0:
        daily = "below_200sma"
    elif math.isfinite(cvs50) and cvs50 < -0.05:
        if math.isfinite(slope50) and slope50 < 0:
            daily = "below_sma50_declining"
        else:
            daily = "below_sma50"
    elif math.isfinite(cvs50) and cvs50 < 0:
        daily = "near_sma50"  # within 5% below SMA50
    elif math.isfinite(cvs50) and cvs50 >= 0:
        if math.isfinite(slope50) and slope50 > 0.002:
            if math.isfinite(rs63) and rs63 > 0.03:
                daily = "strong_uptrend"  # above rising SMA50, outperforming
            else:
                daily = "uptrend_sma50_rising"
        elif math.isfinite(slope50) and slope50 < -0.002:
            daily = "above_sma50_declining"  # above but MA heading down — late stage
        else:
            daily = "above_sma50_flat"
    else:
        daily = "unknown"

    # EMA20 note
    ema20_note = None
    if math.isfinite(cvs_ema20):
        if cvs_ema20 > 0:
            ema20_note = f"Above EMA20 (+{cvs_ema20*100:.1f}%)"
        else:
            ema20_note = f"Below EMA20 ({cvs_ema20*100:.1f}%)"

    # --- Weekly classification ---
    if not math.isfinite(wk_dist) and not math.isfinite(wk_slope):
        weekly = "unknown"
    elif math.isfinite(wk_dist) and wk_dist < -0.05 and math.isfinite(wk_slope) and wk_slope < 0:
        weekly = "broken_weekly"
    elif math.isfinite(wk_dist) and wk_dist < -0.03:
        weekly = "below_wma10_weekly"
    elif math.isfinite(wk_dist) and wk_dist >= 0:
        if math.isfinite(wk_slope) and wk_slope > 0.001:
            if math.isfinite(wk_rs26) and wk_rs26 > 0.03:
                weekly = "strong_weekly_uptrend"
            else:
                weekly = "weekly_uptrend"
        elif math.isfinite(wk_slope) and wk_slope < -0.001:
            weekly = "above_wma10_declining"
        else:
            weekly = "above_wma10_flat"
    else:
        weekly = "neutral_weekly"

    # 40-week check (long-term health)
    wma40_note = None
    if math.isfinite(wk_dist40):
        if wk_dist40 > 0:
            wma40_note = f"Above 40-week WMA (+{wk_dist40*100:.1f}%)"
        else:
            wma40_note = f"Below 40-week WMA ({wk_dist40*100:.1f}%)"

    # Summary note
    trend_notes: list[str] = []
    if math.isfinite(cvs50):
        trend_notes.append(f"SMA50 dist: {cvs50*100:+.1f}%")
    if math.isfinite(cvs200):
        trend_notes.append(f"SMA200 dist: {cvs200*100:+.1f}%")
    if math.isfinite(wk_dist):
        trend_notes.append(f"WMA10 wk dist: {wk_dist*100:+.1f}%")

    return {
        "daily_trend_state":  daily,
        "weekly_trend_state": weekly,
        "daily_detail": {
            "close_vs_sma50":  round(cvs50, 4)   if math.isfinite(cvs50)    else None,
            "close_vs_sma200": round(cvs200, 4)  if math.isfinite(cvs200)   else None,
            "slope_50":        round(slope50, 5) if math.isfinite(slope50)  else None,
            "ema20_note":      ema20_note,
        },
        "weekly_detail": {
            "weekly_dist_wma10":      round(wk_dist, 4)   if math.isfinite(wk_dist)   else None,
            "weekly_trend_slope_26":  round(wk_slope, 5)  if math.isfinite(wk_slope)  else None,
            "weekly_rs_26":           round(wk_rs26, 4)   if math.isfinite(wk_rs26)   else None,
            "wma40_note":             wma40_note,
        },
        "trend_summary": trend_notes,
    }


# ---------------------------------------------------------------------------
# Score drivers / transparency
# ---------------------------------------------------------------------------


def build_score_drivers(provider_symbol: str, snapshot_row: pd.Series) -> dict:
    """Extract top bullish/bearish feature signals and score transparency info.

    Returns a dict:
      bullish_signals  : list[str] - top bullish signals present
      bearish_signals  : list[str] - top bearish signals / red flags
      model_based      : list[str] - columns produced by fitted models
      rule_based       : list[str] - columns assigned by rules
      why_selected     : str       - brief text explaining why this name made top list
    """
    def _sr(key: str, default: float = _NAN) -> float:
        return _sf(snapshot_row, key, default)

    feat = _load_features(provider_symbol)

    def _ff(key: str, default: float = _NAN) -> float:
        if feat is None:
            return default
        return _sf(feat, key, default)

    def _fb(feat_key: str, row_key: str | None = None) -> float:
        """Features-first fallback: use snapshot_row when features value is NaN."""
        v = _ff(feat_key)
        if not math.isfinite(v):
            v = _sr(row_key or feat_key)
        return v

    bullish: list[str] = []
    bearish: list[str] = []

    # --- Trend alignment ---
    cvs50 = _fb("close_vs_sma50")
    if math.isfinite(cvs50):
        if cvs50 > 0.02:
            bullish.append(f"Price {cvs50*100:.1f}% above SMA50")
        elif cvs50 < -0.03:
            bearish.append(f"Price {abs(cvs50)*100:.1f}% below SMA50")

    # --- ATR compression ---
    atr_pct = _fb("atr_compression_pct")
    if math.isfinite(atr_pct):
        if atr_pct < 30:
            bullish.append(f"ATR compressed to {atr_pct:.0f}th pct - coiled")
        elif atr_pct > 60:
            bearish.append(f"ATR elevated ({atr_pct:.0f}th pct) - not compressed")

    # --- Volume dry-up ---
    vd = _fb("volume_dryup")
    if math.isfinite(vd):
        if vd > 0.4:
            bullish.append(f"Volume dry-up score {vd:.2f} - demand absorbed")
        elif vd < 0.1:
            bearish.append(f"High volume activity ({vd:.2f}) - volatile")

    # --- Relative strength ---
    rs63 = _fb("daily_rs_63")
    if math.isfinite(rs63):
        if rs63 > 0.05:
            bullish.append(f"RS-63: +{rs63*100:.0f}% vs SPY - outperforming")
        elif rs63 < -0.05:
            bearish.append(f"RS-63: {rs63*100:.0f}% vs SPY - underperforming")

    # --- Pivot touches / base quality ---
    res_touches = _ff("resistance_touches")
    if math.isfinite(res_touches) and res_touches >= 2:
        bullish.append(f"{res_touches:.0f}x resistance touches at pivot - tested level")

    flatness = _ff("pivot_flatness")
    if math.isfinite(flatness):
        if flatness >= 0.7:
            bullish.append(f"Pivot flatness {flatness:.2f} - tight, well-defined base")
        elif flatness < 0.3:
            bearish.append(f"Pivot flatness {flatness:.2f} - loose structure")

    # --- YTD AVWAP ---
    ytd_dist = _fb("ytd_dist_atr")
    if math.isfinite(ytd_dist):
        if ytd_dist > 0.5:
            bullish.append(f"YTD AVWAP +{ytd_dist:.1f} ATR - accepted above cost basis")
        elif ytd_dist < -1.0:
            bearish.append(f"YTD AVWAP {ytd_dist:.1f} ATR - below year-open cost basis")

    # --- Failure risk ---
    fail = _sr("failure_risk")
    if math.isfinite(fail):
        if fail < 0.35:
            bullish.append(f"Failure risk {fail:.0%} - low model estimate")
        elif fail > 0.60:
            bearish.append(f"Failure risk {fail:.0%} - elevated model estimate")

    # --- Weekly trend ---
    wk_dist = _ff("weekly_dist_wma10")
    if math.isfinite(wk_dist):
        if wk_dist > 0:
            bullish.append(f"Weekly: {wk_dist:.1f} ATR above 10-week WMA")
        elif wk_dist < -1.0:
            bearish.append(f"Weekly: {abs(wk_dist):.1f} ATR below 10-week WMA")

    # --- Model/rule classification ---
    model_based = [
        "composite_score", "setup_score", "trade_score", "failure_risk",
        "percentile_rank",
    ]
    rule_based = [
        "action_label", "freshness_label", "setup_classification",
        "portfolio_guidance", "state",
    ]

    # --- Why selected ---
    score = _sr("composite_score")
    rank = _sr("percentile_rank")
    state = str(snapshot_row.get("state", ""))
    action = str(snapshot_row.get("action_label", ""))

    why_parts = []
    if math.isfinite(score):
        why_parts.append(f"composite_score={score:.2f}")
    if math.isfinite(rank):
        why_parts.append(f"rank={rank:.0f}th pct within {state}")
    if action and action not in ("--", ""):
        why_parts.append(f"action={action}")
    why_parts += bullish[:2]  # top 2 bullish signals in summary

    why_selected = (
        f"Selected because: {'; '.join(why_parts)}."
        if why_parts else "Selection criteria unavailable."
    )

    return {
        "bullish_signals": bullish[:6],   # cap at 6 for display
        "bearish_signals": bearish[:4],
        "model_based": model_based,
        "rule_based": rule_based,
        "why_selected": why_selected,
    }


# ---------------------------------------------------------------------------
# Master builder
# ---------------------------------------------------------------------------


def build_context(
    provider_symbol: str,
    snapshot_row: pd.Series,
    levels: dict,
) -> dict:
    """Master context builder — calls all sub-builders and returns a unified dict.

    Parameters
    ----------
    provider_symbol : str
        Symbol identifier used to resolve data file paths.
    snapshot_row : pd.Series
        One row from the scored snapshot (state, close, pivot, atr14, …).
    levels : dict
        Trade levels keyed by: pivot, entry_lo, entry_hi, stop, t1, t2, t3,
        s1, s2, r1, r2, risk_reward_t1.  Values may be floats or "—" strings.

    Returns
    -------
    dict with keys: ma_table, avwap_table, volume_block, confluence, checklist.
    All sub-builders catch their own exceptions; this function also wraps the
    entire call to guarantee a safe return.
    """
    try:
        close = _sf(snapshot_row, "close")
        atr   = _sf(snapshot_row, "atr14")

        # Pivot from levels dict (may be "—" or float)
        pivot_raw = levels.get("pivot", _NAN)
        try:
            pivot = float(pivot_raw)
            if not math.isfinite(pivot):
                pivot = _NAN
        except (TypeError, ValueError):
            pivot = _NAN

        ma_table     = build_ma_table(provider_symbol, close)
        avwap_table  = build_avwap_table(provider_symbol, close, atr)
        volume_block = build_volume_block(provider_symbol, close, atr)
        confluence   = build_confluence(close, pivot, atr, avwap_table, levels)

        # Run deterministic pattern assessments from raw daily OHLCV.
        raw_df = _load_raw_daily(provider_symbol)
        assessments = run_all_assessments(
            df=raw_df if raw_df is not None else pd.DataFrame(),
            close=close,
            atr=atr,
            avwap_table=avwap_table,
            pivot=pivot,
        )

        checklist     = build_checklist(provider_symbol, snapshot_row, levels, assessments)
        score_drivers = build_score_drivers(provider_symbol, snapshot_row)
        trend_state   = build_trend_state(provider_symbol, snapshot_row)

        return {
            "ma_table":      ma_table,
            "avwap_table":   avwap_table,
            "volume_block":  volume_block,
            "confluence":    confluence,
            "checklist":     checklist,
            "score_drivers": score_drivers,
            "assessments":   assessments,
            "trend_state":   trend_state,
        }

    except Exception as exc:
        _log.warning("build_context - unexpected error for %s: %s", provider_symbol, exc)
        return {
            "ma_table":     [],
            "avwap_table":  [],
            "volume_block": {},
            "confluence":   {"nearby_count": 0, "nearby_levels": [], "cluster_role": "scattered"},
            "checklist":    [],
            "score_drivers": {"bullish_signals": [], "bearish_signals": [], "model_based": [], "rule_based": [], "why_selected": "Error"},
            "assessments":  {},
            "trend_state":  {"daily_trend_state": "unknown", "weekly_trend_state": "unknown",
                             "daily_detail": {}, "weekly_detail": {}, "trend_summary": []},
        }
