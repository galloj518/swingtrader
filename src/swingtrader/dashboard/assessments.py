"""Deterministic pattern assessments for trade-decision support.

Each function accepts raw OHLCV (or derived) data and returns a structured
result dict with a grade, numeric score, and plain-English notes.  None of
these produce probabilistic model scores — they are rule-based, transparent,
and directly useful for reviewing a setup before acting.

Assessment results are added to the canonical packet via build_context() so
they appear in the checklist, score-drivers block, and narrative.

Functions
---------
assess_base_quality(df)
    Tightness, ATR compression, NR-day count, down-close ratio.

assess_continuation_pattern(df, weekly_df)
    3WT, tight5d, NR7/NR4 detection.

assess_overhead_supply(close, atr, df)
    How many of the last 252 bars closed above current price.

assess_breakout_integrity(df)
    Volume and close-location quality on the most recent trigger bar.

assess_clean_air(close, atr, avwap_table, pivot)
    Distance to the nearest resistance AVWAP above current price.

assess_chart_quality(df)
    Trend efficiency ratio and directional consistency.
"""
from __future__ import annotations

import math

import pandas as pd

# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _sf(s: pd.Series | None, key: str, default: float = math.nan) -> float:
    if s is None:
        return default
    try:
        v = float(s[key])
        return v if math.isfinite(v) else default
    except Exception:
        return default


def _grade(score: float) -> str:
    """Convert 0-1 score to A/B/C/D letter grade."""
    if score >= 0.75:
        return "A"
    if score >= 0.55:
        return "B"
    if score >= 0.35:
        return "C"
    return "D"


# ---------------------------------------------------------------------------
# Base quality
# ---------------------------------------------------------------------------


def assess_base_quality(df: pd.DataFrame, base_window: int = 30) -> dict:
    """Evaluate the quality of the current base / consolidation pattern.

    Parameters
    ----------
    df : daily OHLCV DataFrame, DatetimeIndex, sorted ascending.
         Must have 'close', 'high', 'low', 'open', 'volume' columns.
    base_window : bars to use for base analysis (default 30 ≈ 6 weeks).

    Returns
    -------
    dict with keys: grade, score, label, notes, components.
    """
    empty = {"grade": "D", "score": 0.0, "label": "Insufficient data", "notes": [], "components": {}}
    if df is None or len(df) < max(base_window, 15):
        return empty

    try:
        base = df.tail(base_window).copy()
        closes = base["close"].dropna()

        if len(closes) < 10:
            return empty

        mean_close = float(closes.mean())
        if mean_close <= 0:
            return empty

        # --- 1. Close tightness (std of closes / mean) -------------------------
        # Lower is better; <2% is very tight.  Score 1→0 as pct goes 0%→5%.
        close_std_pct = float(closes.std(ddof=1)) / mean_close * 100
        if close_std_pct <= 2.0:
            tightness_score = 1.0
        elif close_std_pct >= 5.0:
            tightness_score = 0.0
        else:
            tightness_score = 1.0 - (close_std_pct - 2.0) / 3.0

        # --- 2. ATR compression (current ATR vs 20-bar average earlier) ---------
        if len(df) >= base_window + 20:
            def _atr_series(d: pd.DataFrame) -> pd.Series:
                tr = pd.concat([
                    d["high"] - d["low"],
                    (d["high"] - d["close"].shift()).abs(),
                    (d["low"]  - d["close"].shift()).abs(),
                ], axis=1).max(axis=1)
                return tr.rolling(14).mean()

            prior_atr = float(_atr_series(df.iloc[-(base_window + 20):]).iloc[-(base_window + 1)])
            curr_atr  = float(_atr_series(df).iloc[-1])

            if math.isfinite(prior_atr) and prior_atr > 0 and math.isfinite(curr_atr):
                compression_ratio = curr_atr / prior_atr  # < 1 = compressed
                if compression_ratio <= 0.65:
                    atr_score = 1.0
                elif compression_ratio >= 1.10:
                    atr_score = 0.0
                else:
                    atr_score = 1.0 - (compression_ratio - 0.65) / 0.45
            else:
                atr_score = 0.5
        else:
            atr_score = 0.5
            compression_ratio = math.nan

        # --- 3. NR days (inside or narrow-range) --------------------------------
        # Count bars where (high-low) < 80% of 14-bar ATR range.
        recent_15 = df.tail(15)
        tr15 = (recent_15["high"] - recent_15["low"]).dropna()
        if len(tr15) >= 10:
            median_range = float(tr15.median())
            if median_range > 0:
                nr_days = int((tr15 < 0.80 * median_range).sum())
            else:
                nr_days = 0
        else:
            nr_days = 0
        # 4+ NR days in 15 = constructive; 0 = no compression event
        nr_score = min(nr_days / 5.0, 1.0)

        # --- 4. Down-close ratio in base ----------------------------------------
        # Fraction of bars that closed below their open.  < 40% is healthy.
        n_down = int((closes < base["open"].dropna()).sum())
        down_ratio = n_down / len(closes) if len(closes) > 0 else 0.5
        if down_ratio <= 0.35:
            down_score = 1.0
        elif down_ratio >= 0.60:
            down_score = 0.0
        else:
            down_score = 1.0 - (down_ratio - 0.35) / 0.25

        # --- 5. Base not too long -----------------------------------------------
        base_len = len(closes)
        if base_len <= 40:
            age_score = 1.0
        elif base_len >= 70:
            age_score = 0.3
        else:
            age_score = 1.0 - (base_len - 40) / 45.0

        # --- Composite ----------------------------------------------------------
        weights = {"tightness": 0.30, "atr": 0.25, "nr": 0.20, "down": 0.15, "age": 0.10}
        score = (
            weights["tightness"] * tightness_score
            + weights["atr"]      * atr_score
            + weights["nr"]       * nr_score
            + weights["down"]     * down_score
            + weights["age"]      * age_score
        )
        score = max(0.0, min(1.0, score))

        # --- Notes --------------------------------------------------------------
        notes: list[str] = []
        if tightness_score >= 0.7:
            notes.append(f"Tight closes ({close_std_pct:.1f}% std)")
        elif tightness_score < 0.3:
            notes.append(f"Loose closes ({close_std_pct:.1f}% std) — wide base")
        if atr_score >= 0.7:
            notes.append("ATR compressed vs prior range")
        elif atr_score < 0.3:
            notes.append("ATR expanding — base still volatile")
        if nr_days >= 4:
            notes.append(f"{nr_days} NR days in last 15 — coiling")
        if down_ratio > 0.50:
            notes.append(f"{down_ratio:.0%} down closes — distribution signal")
        elif down_ratio < 0.35:
            notes.append(f"Healthy up/down close ratio ({down_ratio:.0%} down)")

        grade = _grade(score)
        label = f"Base quality: {grade} ({score:.2f})"

        return {
            "grade": grade,
            "score": round(score, 3),
            "label": label,
            "notes": notes,
            "components": {
                "tightness_score":   round(tightness_score, 3),
                "atr_score":         round(atr_score, 3),
                "nr_score":          round(nr_score, 3),
                "down_score":        round(down_score, 3),
                "age_score":         round(age_score, 3),
                "close_std_pct":     round(close_std_pct, 2),
                "nr_days":           nr_days,
                "down_ratio":        round(down_ratio, 3),
            },
        }

    except Exception:
        return empty


# ---------------------------------------------------------------------------
# Continuation patterns
# ---------------------------------------------------------------------------


def assess_continuation_pattern(
    df: pd.DataFrame,
    weekly_df: pd.DataFrame | None = None,
) -> dict:
    """Detect continuation candlestick / bar patterns.

    Patterns checked
    ----------------
    nr7      : today's range is the narrowest of the last 7 days.
    nr4      : today's range is the narrowest of the last 4 days.
    tight5d  : last 5 closes have std < 1% of price (pocket pivot setup).
    inside_day : today's high < yesterday's high and today's low > yesterday's low.
    3wt      : last 3 weekly closes within 1.5% of each other (3-week tight).

    Parameters
    ----------
    df         : daily OHLCV, sorted ascending.
    weekly_df  : weekly OHLCV for 3WT check; if None, derived from df.

    Returns
    -------
    dict: patterns (list of detected pattern strings), strongest_pattern,
          score (0-1), notes, grade.
    """
    empty = {"grade": "D", "score": 0.0, "strongest_pattern": "none",
             "patterns": [], "notes": []}
    if df is None or len(df) < 8:
        return empty

    try:
        highs  = df["high"].dropna()
        lows   = df["low"].dropna()
        closes = df["close"].dropna()

        ranges = (highs - lows).dropna()

        patterns: list[str] = []
        notes: list[str] = []

        # --- NR7 -----------------------------------------------------------------
        if len(ranges) >= 7:
            today_range = float(ranges.iloc[-1])
            past7_min   = float(ranges.iloc[-7:].min())
            if today_range <= past7_min * 1.001:  # tiny tolerance for float equality
                patterns.append("NR7")
                notes.append("NR7: today is the narrowest range in 7 days — compression event")

        # --- NR4 -----------------------------------------------------------------
        if len(ranges) >= 4 and "NR7" not in patterns:
            today_range = float(ranges.iloc[-1])
            past4_min   = float(ranges.iloc[-4:].min())
            if today_range <= past4_min * 1.001:
                patterns.append("NR4")
                notes.append("NR4: narrowest range in 4 days")

        # --- Inside day ----------------------------------------------------------
        if len(highs) >= 2 and float(highs.iloc[-1]) < float(highs.iloc[-2]) and float(lows.iloc[-1]) > float(lows.iloc[-2]):
            patterns.append("inside_day")
            notes.append("Inside day: range fully within prior bar")

        # --- Tight 5-day close cluster -------------------------------------------
        if len(closes) >= 5:
            last5 = closes.iloc[-5:]
            mean5 = float(last5.mean())
            if mean5 > 0:
                std_pct = float(last5.std(ddof=1)) / mean5 * 100
                if std_pct < 1.0:
                    patterns.append("tight5d")
                    notes.append(f"Tight5d: last 5 closes within {std_pct:.2f}% — spring loaded")

        # --- 3-week tight (3WT) --------------------------------------------------
        try:
            if weekly_df is not None and len(weekly_df) >= 3:
                wk_closes = weekly_df["close"].dropna()
            else:
                # Derive weekly from daily by resampling
                wk_temp = df["close"].resample("W-FRI").last().dropna()
                wk_closes = wk_temp

            if len(wk_closes) >= 3:
                last3w = wk_closes.iloc[-3:]
                mean3w = float(last3w.mean())
                if mean3w > 0:
                    rng_pct = (float(last3w.max()) - float(last3w.min())) / mean3w * 100
                    if rng_pct < 1.5:
                        patterns.append("3wt")
                        notes.append(f"3WT: last 3 weekly closes within {rng_pct:.2f}% — textbook tight consolidation")
        except Exception:
            pass

        # --- Score --------------------------------------------------------------
        # Each pattern adds to the score, with more selective patterns worth more.
        pattern_weights = {
            "3wt":        0.90,
            "tight5d":    0.75,
            "NR7":        0.65,
            "inside_day": 0.50,
            "NR4":        0.40,
        }
        # Use the highest single-pattern score (patterns don't stack additively
        # because they're often co-occurring expressions of the same condition).
        score = max((pattern_weights.get(p, 0.0) for p in patterns), default=0.0)

        strongest = max(patterns, key=lambda p: pattern_weights.get(p, 0.0)) if patterns else "none"

        return {
            "grade":            _grade(score),
            "score":            round(score, 3),
            "strongest_pattern": strongest,
            "patterns":         patterns,
            "notes":            notes,
        }

    except Exception:
        return empty


# ---------------------------------------------------------------------------
# Overhead supply
# ---------------------------------------------------------------------------


def assess_overhead_supply(
    close: float,
    atr: float,
    df: pd.DataFrame,
    lookback: int = 252,
) -> dict:
    """Estimate overhead supply as fraction of prior bars that closed above current price.

    High supply means many prior closes above → potential distribution / resistance.
    Low supply means price is in relatively clear territory.

    Parameters
    ----------
    close    : current close price.
    atr      : 14-day ATR (used to define "just above" zone).
    df       : daily OHLCV, sorted ascending.
    lookback : bars to examine (default 1 year).

    Returns
    -------
    dict: supply_pct (fraction of bars above), supply_zone,
          supply_score (0=no supply, 1=heavy supply), notes, grade.
    """
    empty = {"grade": "D", "score": 0.5, "supply_pct": math.nan,
             "supply_zone": "unknown", "notes": []}

    if not math.isfinite(close) or close <= 0:
        return empty
    if df is None or len(df) < 20:
        return empty

    try:
        window = df.tail(lookback)
        prior_closes = window["close"].dropna()
        # Exclude the most recent 5 bars (current base) from the supply scan
        if len(prior_closes) > 5:
            prior_closes = prior_closes.iloc[:-5]

        n_total = len(prior_closes)
        if n_total < 10:
            return empty

        # Bars that closed above current close → overhead supply
        n_above = int((prior_closes > close).sum())
        supply_pct = n_above / n_total

        # "Just above" zone: bars within 1 ATR of current close
        if math.isfinite(atr) and atr > 0:
            n_just_above = int(((prior_closes > close) & (prior_closes <= close + atr)).sum())
            just_above_pct = n_just_above / n_total
        else:
            just_above_pct = 0.0

        # Supply score: 0 = no overhead, 1 = heavy overhead.
        # Weight "just above" bars more heavily (they're the near-term problem).
        supply_score = min(supply_pct * 0.5 + just_above_pct * 0.5, 1.0)

        # Zone label
        if supply_pct < 0.10:
            supply_zone = "clear air"
        elif supply_pct < 0.30:
            supply_zone = "light supply"
        elif supply_pct < 0.55:
            supply_zone = "moderate supply"
        else:
            supply_zone = "heavy supply"

        notes: list[str] = []
        if supply_pct < 0.10:
            notes.append(f"Only {supply_pct:.0%} of prior year closed above — clear air above")
        elif supply_pct > 0.50:
            notes.append(f"{supply_pct:.0%} of prior year closed above — significant overhead supply")
        if just_above_pct > 0.15:
            notes.append(f"{just_above_pct:.0%} of bars within 1 ATR above — near-term resistance cluster")

        # For this assessment, lower supply = better grade.
        grade_score = 1.0 - supply_score

        return {
            "grade":       _grade(grade_score),
            "score":       round(supply_score, 3),   # higher = more supply (worse)
            "supply_pct":  round(supply_pct, 3),
            "supply_zone": supply_zone,
            "notes":       notes,
        }

    except Exception:
        return empty


# ---------------------------------------------------------------------------
# Breakout integrity
# ---------------------------------------------------------------------------


def assess_breakout_integrity(df: pd.DataFrame, lookback_vol: int = 50) -> dict:
    """Evaluate the quality of the most recent breakout bar.

    Specifically checks the most recent bar with volume >= 1.5× average
    as the "trigger bar".  Falls back to the last bar if no high-volume bar
    is found in the last 10 days.

    Parameters
    ----------
    df           : daily OHLCV, sorted ascending.
    lookback_vol : bars for average volume baseline.

    Returns
    -------
    dict: integrity_score (0-1), volume_ratio, close_location,
          close_location_score, volume_label, grade, notes.
    """
    empty = {"grade": "D", "score": 0.0, "volume_ratio": math.nan,
             "close_location": math.nan, "volume_label": "—", "notes": []}

    if df is None or len(df) < max(lookback_vol, 10):
        return empty

    try:
        closes  = df["close"].dropna()
        highs   = df["high"].dropna()
        lows    = df["low"].dropna()
        volumes = df["volume"].dropna()

        avg_vol = float(volumes.iloc[-lookback_vol:-1].mean())
        if not math.isfinite(avg_vol) or avg_vol <= 0:
            return empty

        # Find trigger bar: most recent bar in last 10 with volume >= 1.5× avg
        recent_vol = volumes.iloc[-10:]
        trigger_idx = None
        for i in range(len(recent_vol) - 1, -1, -1):
            if float(recent_vol.iloc[i]) >= 1.5 * avg_vol:
                trigger_idx = i - len(recent_vol)  # negative offset from end
                break
        if trigger_idx is None:
            trigger_idx = -1  # use last bar as fallback

        t_close  = float(closes.iloc[trigger_idx])
        t_high   = float(highs.iloc[trigger_idx])
        t_low    = float(lows.iloc[trigger_idx])
        t_vol    = float(volumes.iloc[trigger_idx])

        vol_ratio = t_vol / avg_vol

        # Close location in range: 0=bottom, 1=top
        t_range = t_high - t_low
        if t_range > 0:
            close_location = (t_close - t_low) / t_range
        else:
            close_location = 0.5

        # Volume score: ratio of 1.0 = base, 2.0 = best
        vol_score = min((vol_ratio - 1.0) / 1.5, 1.0) if vol_ratio >= 1.0 else 0.0

        # Close location score: close in top 40% = full score
        if close_location >= 0.70:
            loc_score = 1.0
        elif close_location >= 0.40:
            loc_score = (close_location - 0.40) / 0.30
        else:
            loc_score = 0.0

        integrity_score = 0.55 * vol_score + 0.45 * loc_score

        # Labels
        if vol_ratio >= 2.5:
            vol_label = f"Surge ({vol_ratio:.1f}x avg)"
        elif vol_ratio >= 1.5:
            vol_label = f"Elevated ({vol_ratio:.1f}x avg)"
        elif vol_ratio >= 1.0:
            vol_label = f"Normal ({vol_ratio:.1f}x avg)"
        else:
            vol_label = f"Below avg ({vol_ratio:.1f}x)"

        notes: list[str] = []
        if vol_ratio >= 2.0 and close_location >= 0.70:
            notes.append("Strong breakout: high volume + close near high")
        elif vol_ratio < 1.3:
            notes.append("Weak volume on breakout bar — low conviction")
        if close_location < 0.30:
            notes.append("Closed in lower third of range — rejection risk")
        elif close_location >= 0.80:
            notes.append(f"Closed in top {(1 - close_location):.0%} of range")

        return {
            "grade":               _grade(integrity_score),
            "score":               round(integrity_score, 3),
            "volume_ratio":        round(vol_ratio, 2),
            "close_location":      round(close_location, 3),
            "close_location_score": round(loc_score, 3),
            "volume_label":        vol_label,
            "notes":               notes,
        }

    except Exception:
        return empty


# ---------------------------------------------------------------------------
# Clean air
# ---------------------------------------------------------------------------


def assess_clean_air(
    close: float,
    atr: float,
    avwap_table: list[dict],
    pivot: float,
) -> dict:
    """Measure how much clear space exists above current price.

    Resistance sources checked:
    1. AVWAP levels above close (from avwap_table).
    2. The pivot level itself (if above close — shouldn't happen in breakout
       context but handled defensively).

    Distance is expressed in ATR units.

    Returns
    -------
    dict: clean_air_atrs, nearest_resistance_atrs, nearest_resistance_label,
          grade, score, notes.
    """
    empty = {"grade": "D", "score": 0.0, "clean_air_atrs": math.nan,
             "nearest_resistance_atrs": math.nan, "nearest_resistance_label": "unknown",
             "notes": []}

    if not math.isfinite(close) or close <= 0:
        return empty
    if not math.isfinite(atr) or atr <= 0:
        # Rough fallback: 1% of price as ATR
        atr = close * 0.01

    try:
        resistance_levels: list[tuple[float, str]] = []

        # AVWAP levels above close
        for row in avwap_table:
            avwap_val = row.get("avwap", math.nan)
            if not isinstance(avwap_val, float):
                try:
                    avwap_val = float(avwap_val)
                except Exception:
                    continue
            if math.isfinite(avwap_val) and avwap_val > close:
                label = str(row.get("anchor", "AVWAP"))
                resistance_levels.append((avwap_val, label))

        # Pivot above close
        if math.isfinite(pivot) and pivot > close:
            resistance_levels.append((pivot, "Pivot"))

        if not resistance_levels:
            # No identified resistance above — price in clear air (or at 52wk high)
            notes = ["No identified AVWAP resistance above — clear air"]
            return {
                "grade": "A",
                "score": 1.0,
                "clean_air_atrs": 999.0,
                "nearest_resistance_atrs": 999.0,
                "nearest_resistance_label": "none identified",
                "notes": notes,
            }

        # Nearest resistance
        resistance_levels.sort(key=lambda x: x[0])
        nearest_val, nearest_label = resistance_levels[0]
        nearest_atrs = (nearest_val - close) / atr

        # Score: ≥ 3 ATR to nearest resistance = clean; < 1 ATR = crowded.
        if nearest_atrs >= 3.0:
            clean_score = 1.0
        elif nearest_atrs <= 1.0:
            clean_score = 0.0
        else:
            clean_score = (nearest_atrs - 1.0) / 2.0

        notes: list[str] = []
        if nearest_atrs >= 3.0:
            notes.append(f"Clear air: {nearest_atrs:.1f} ATR to nearest resistance ({nearest_label})")
        elif nearest_atrs < 1.5:
            notes.append(f"Resistance close: only {nearest_atrs:.1f} ATR to {nearest_label}")
        else:
            notes.append(f"{nearest_atrs:.1f} ATR to nearest resistance ({nearest_label})")

        return {
            "grade":                    _grade(clean_score),
            "score":                    round(clean_score, 3),
            "clean_air_atrs":           round(nearest_atrs, 2),
            "nearest_resistance_atrs":  round(nearest_atrs, 2),
            "nearest_resistance_label": nearest_label,
            "notes":                    notes,
        }

    except Exception:
        return empty


# ---------------------------------------------------------------------------
# Chart quality (trend efficiency)
# ---------------------------------------------------------------------------


def assess_chart_quality(df: pd.DataFrame, lookback: int = 60) -> dict:
    """Evaluate the efficiency and directionality of the recent price trend.

    Trend Efficiency Ratio (TER):
        TER = |close[-1] - close[-N]| / sum(|daily close changes|)
    A TER near 1 means price moved in a straight line (very efficient).
    A TER near 0 means lots of whipsawing with no net progress.

    Also checks: up-close consistency over 10 days, weekly trend quality.

    Parameters
    ----------
    df      : daily OHLCV, sorted ascending.
    lookback: bars for TER calculation (default 60 ≈ 3 months).

    Returns
    -------
    dict: ter, up_close_pct, grade, score, notes.
    """
    empty = {"grade": "D", "score": 0.0, "ter": math.nan,
             "up_close_pct": math.nan, "notes": []}

    if df is None or len(df) < 15:
        return empty

    try:
        closes = df["close"].dropna()
        window = closes.tail(lookback)

        if len(window) < 10:
            return empty

        # --- Trend Efficiency Ratio -------------------------------------------
        net_move = abs(float(window.iloc[-1]) - float(window.iloc[0]))
        daily_moves = window.diff().dropna().abs()
        path_length = float(daily_moves.sum())
        if path_length > 0:
            ter = net_move / path_length
        else:
            ter = 0.0

        # TER score: 0.50+ is good, 0.20 is choppy
        if ter >= 0.50:
            ter_score = 1.0
        elif ter <= 0.15:
            ter_score = 0.0
        else:
            ter_score = (ter - 0.15) / 0.35

        # --- Up-close ratio (recent 10 bars) ----------------------------------
        recent10 = df.tail(10)
        n_up = int((recent10["close"] > recent10["open"]).sum())
        up_close_pct = n_up / 10
        if up_close_pct >= 0.65:
            up_score = 1.0
        elif up_close_pct <= 0.35:
            up_score = 0.0
        else:
            up_score = (up_close_pct - 0.35) / 0.30

        # --- Composite --------------------------------------------------------
        score = 0.60 * ter_score + 0.40 * up_score
        score = max(0.0, min(1.0, score))

        notes: list[str] = []
        if ter >= 0.50:
            notes.append(f"Efficient trend: TER {ter:.2f} (low chop)")
        elif ter < 0.20:
            notes.append(f"Choppy chart: TER {ter:.2f} — whipsaw risk")
        if up_close_pct >= 0.70:
            notes.append(f"{up_close_pct:.0%} up-closes in last 10 days")
        elif up_close_pct <= 0.30:
            notes.append(f"Only {up_close_pct:.0%} up-closes in last 10 days — weak demand")

        return {
            "grade":        _grade(score),
            "score":        round(score, 3),
            "ter":          round(ter, 3),
            "up_close_pct": round(up_close_pct, 3),
            "notes":        notes,
        }

    except Exception:
        return empty


# ---------------------------------------------------------------------------
# Convenience: run all assessments for a symbol
# ---------------------------------------------------------------------------


def run_all_assessments(
    df: pd.DataFrame,
    close: float,
    atr: float,
    avwap_table: list[dict],
    pivot: float,
    weekly_df: pd.DataFrame | None = None,
) -> dict:
    """Run all assessments and return a combined dict keyed by assessment name.

    Parameters
    ----------
    df          : daily OHLCV, sorted ascending.
    close       : current close.
    atr         : 14-day ATR.
    avwap_table : from build_avwap_table() — list of AVWAP anchor dicts.
    pivot       : current pivot level.
    weekly_df   : optional weekly OHLCV for 3WT check.

    Returns
    -------
    dict with keys: base_quality, continuation, overhead_supply,
                    breakout_integrity, clean_air, chart_quality.
    """
    return {
        "base_quality":       assess_base_quality(df),
        "continuation":       assess_continuation_pattern(df, weekly_df),
        "overhead_supply":    assess_overhead_supply(close, atr, df),
        "breakout_integrity": assess_breakout_integrity(df),
        "clean_air":          assess_clean_air(close, atr, avwap_table, pivot),
        "chart_quality":      assess_chart_quality(df),
    }


__all__ = [
    "assess_base_quality",
    "assess_breakout_integrity",
    "assess_chart_quality",
    "assess_clean_air",
    "assess_continuation_pattern",
    "assess_overhead_supply",
    "run_all_assessments",
]
