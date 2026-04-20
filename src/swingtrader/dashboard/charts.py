"""Static chart generation for top setups — OHLC candlestick edition.

Generates PNG charts using matplotlib (already a project dependency).
Charts are saved to:
  {output_dir}/charts/{SYMBOL}_weekly.png
  {output_dir}/charts/{SYMBOL}_daily.png
  {output_dir}/charts/{SYMBOL}_intraday.png  (only if 5m data exists)

Design
------
- Dark background to match the site theme (#0d1117).
- Full OHLC candlestick bodies (FancyBboxPatch) + wick lines.
- Each chart shows: candlesticks, key MAs, AVWAP overlays, trade levels, volume.
- Deliberately simple: no interactive JS, no 3rd-party charting libs (no mplfinance).
- Charts are generated only for symbols present in data/raw/{timeframe}/.
  If data is missing, the chart is skipped and the path is not returned.

All matplotlib imports are local to the functions so the module can be
imported without matplotlib in testing environments where it is unavailable.
"""
from __future__ import annotations

import contextlib
import math
from pathlib import Path

import pandas as pd

from swingtrader.utils.config import REPO_ROOT
from swingtrader.utils.logging import get_logger

log = get_logger(__name__)

_RAW_DAILY_DIR    = REPO_ROOT / "data" / "raw" / "daily"
_RAW_WEEKLY_DIR   = REPO_ROOT / "data" / "raw" / "weekly"
_RAW_INTRADAY_DIR = REPO_ROOT / "data" / "raw" / "intraday"
_FEATURES_DIR     = REPO_ROOT / "data" / "features"

# ── Dark theme colours (match site palette) ───────────────────────────────────
BG      = "#0d1117"
GRID    = "#21262d"
TEXT    = "#c9d1d9"
BLUE    = "#58a6ff"
GREEN   = "#3fb950"
RED     = "#f78166"
AMBER   = "#d29922"
PURPLE  = "#bc8cff"
ORANGE  = "#fb8f44"
UP_BODY   = "#3fb950"
DOWN_BODY = "#f78166"

# Max bars shown on each timeframe chart
DAILY_BARS    = 130
WEEKLY_BARS   = 60
INTRADAY_BARS = 78  # ~6.5 hours at 5-min intervals

# ── Feature helpers ───────────────────────────────────────────────────────────

def _load_features_row(provider_symbol: str) -> pd.Series | None:
    """Load the last row of the features parquet for *provider_symbol*, or None."""
    path = _FEATURES_DIR / f"{provider_symbol}.parquet"
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        if df.empty:
            return None
        return df.iloc[-1]
    except Exception:
        return None


def _safe_f(feat_row: pd.Series | None, key: str) -> float:
    """Return float value from *feat_row* for *key*, or nan if missing/non-finite."""
    if feat_row is None or key not in feat_row.index:
        return math.nan
    try:
        v = float(feat_row[key])
        return v if math.isfinite(v) else math.nan
    except (TypeError, ValueError):
        return math.nan


# ── Shared drawing utilities ──────────────────────────────────────────────────

def _sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n, min_periods=1).mean()


def _ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()


def _setup_dark_axes(fig, height_ratios=(4, 1)):
    """Add two vertically stacked subplots to *fig* with the dark theme applied."""
    import matplotlib.gridspec as gridspec

    gs = gridspec.GridSpec(
        2, 1, height_ratios=list(height_ratios), hspace=0.05, figure=fig
    )
    ax_price = fig.add_subplot(gs[0])
    ax_vol   = fig.add_subplot(gs[1], sharex=ax_price)

    for ax in (ax_price, ax_vol):
        ax.set_facecolor(BG)
        ax.tick_params(colors=TEXT, labelsize=8)
        ax.spines[:].set_color(GRID)
        ax.grid(color=GRID, linewidth=0.5, alpha=0.6)

    import matplotlib.pyplot as plt
    plt.setp(ax_price.get_xticklabels(), visible=False)
    return ax_price, ax_vol


def _draw_candlesticks(ax, df: pd.DataFrame) -> None:
    """Draw OHLC candlestick bodies and wicks on *ax*.

    *df* must have columns: open, high, low, close.
    X positions are integer indices 0..len(df)-1.
    """
    from matplotlib.patches import FancyBboxPatch

    body_width = 0.6

    for i in range(len(df)):
        o = float(df["open"].iloc[i])
        h = float(df["high"].iloc[i])
        lo = float(df["low"].iloc[i])
        c = float(df["close"].iloc[i])

        if not (math.isfinite(o) and math.isfinite(h) and
                math.isfinite(lo) and math.isfinite(c)):
            continue

        color = UP_BODY if c >= o else DOWN_BODY
        body_lo = min(o, c)
        body_hi = max(o, c)
        body_h  = max(body_hi - body_lo, 1e-8)  # avoid zero-height box

        # Body rectangle
        rect = FancyBboxPatch(
            (i - body_width / 2, body_lo),
            body_width,
            body_h,
            boxstyle="square,pad=0",
            linewidth=0.3,
            edgecolor=color,
            facecolor=color,
            alpha=0.9,
            zorder=3,
        )
        ax.add_patch(rect)

        # Upper wick: body_hi → high
        if h > body_hi:
            ax.plot(
                [i, i], [body_hi, h],
                color=color, linewidth=0.7, zorder=2,
            )
        # Lower wick: low → body_lo
        if lo < body_lo:
            ax.plot(
                [i, i], [lo, body_lo],
                color=color, linewidth=0.7, zorder=2,
            )


def _draw_volume(ax_vol, df: pd.DataFrame) -> None:
    """Draw volume bars coloured by candle direction."""
    n = len(df)
    colors = [
        UP_BODY if df["close"].iloc[i] >= df["open"].iloc[i] else DOWN_BODY
        for i in range(n)
    ]
    ax_vol.bar(range(n), df["volume"], color=colors, alpha=0.5, width=0.8, zorder=2)
    ax_vol.set_ylabel("Vol", color=TEXT, fontsize=7)
    ax_vol.yaxis.tick_right()


def _set_xaxis_dates(ax_vol, df: pd.DataFrame, step: int = 7) -> None:
    """Place date tick labels every *step* bars on the volume axis."""
    n = len(df)
    tick_idx = list(range(0, n, max(step, 1)))
    ax_vol.set_xticks(tick_idx)
    ax_vol.set_xticklabels(
        [str(df.index[i])[:10] for i in tick_idx],
        rotation=30, ha="right", fontsize=7, color=TEXT,
    )


def _annotate_hline(
    ax,
    price: float,
    label: str,
    color: str,
    linestyle: str = "--",
    alpha: float = 0.85,
) -> None:
    """Draw a horizontal line at *price* with a small text label."""
    if not math.isfinite(price):
        return
    ax.axhline(price, color=color, linewidth=0.8, linestyle=linestyle, alpha=alpha)
    ax.text(
        0.005,
        price,
        f" {label} {price:.2f}",
        transform=ax.get_yaxis_transform(),
        color=color,
        fontsize=7,
        va="bottom",
    )


def _compute_avwap_series(df: pd.DataFrame, anchor_idx: int) -> list[float]:
    """Running AVWAP from *anchor_idx* forward; NaN before the anchor.

    Returns a plain list of floats, length == len(df), suitable for direct
    x-indexed plotting on the display chart.
    """
    n = len(df)
    result = [math.nan] * n
    if anchor_idx < 0 or anchor_idx >= n:
        return result

    subset = df.iloc[anchor_idx:].copy()
    tp = (subset["high"] + subset["low"] + subset["close"]) / 3.0
    if "volume" in subset.columns:
        vol = subset["volume"].clip(lower=0)
    else:
        vol = pd.Series(1.0, index=subset.index)

    cum_tpv = (tp * vol).cumsum()
    cum_vol = vol.cumsum()
    avwap_vals = (cum_tpv / cum_vol.where(cum_vol > 0)).values

    for i, v in enumerate(avwap_vals):
        try:
            fv = float(v)
            result[anchor_idx + i] = fv if math.isfinite(fv) else math.nan
        except (TypeError, ValueError):
            pass
    return result


def _find_ytd_anchor(df: pd.DataFrame) -> int:
    """Return the integer position of the first bar in the current year."""
    if df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return 0
    last_year = df.index[-1].year
    year_mask = df.index.year == last_year
    if not year_mask.any():
        return 0
    return int(year_mask.argmax())


def _find_swing_low_anchor(df: pd.DataFrame, lookback: int = 252) -> int:
    """Return the position of the lowest low in the last *lookback* bars."""
    n = len(df)
    start = max(0, n - lookback)
    rel_idx = int(df["low"].iloc[start:].values.argmin())
    return start + rel_idx


def _find_swing_high_anchor(df: pd.DataFrame, lookback: int = 252) -> int:
    """Return the position of the highest high in the last *lookback* bars."""
    n = len(df)
    start = max(0, n - lookback)
    rel_idx = int(df["high"].iloc[start:].values.argmax())
    return start + rel_idx


def _find_breakout_anchor(df: pd.DataFrame, pivot: float, days_in_state: int) -> int:
    """Approximate breakout-day anchor from *days_in_state*.

    Falls back to the first bar where close >= pivot when days_in_state is 0.
    """
    n = len(df)
    if days_in_state > 0:
        return max(0, n - 1 - days_in_state)
    if math.isfinite(pivot):
        for i in range(n):
            if float(df["close"].iloc[i]) >= pivot:
                return i
    return max(0, n - 20)


def _overlay_avwap_curves(
    ax,
    df_full: pd.DataFrame,
    display_start: int,
    pivot: float = math.nan,
    days_in_state: int = 0,
    avwap_rows: list[dict] | None = None,
) -> None:
    """Draw anchored VWAP *curves* (not horizontal lines) on the price axis.

    Each AVWAP is computed as a running VWAP from its anchor date forward.
    The curve starts at the anchor bar (or the first visible bar if the anchor
    is before the display window) and ends at the last bar.

    Anchors used:
      - YTD: first bar of the current calendar year
      - Swing Low: lowest low in the last 252 bars
      - Swing High: highest high in the last 252 bars
      - Breakout Day: approximated from days_in_state (TRIGGERED/ACCEPTED only)
    """
    if df_full.empty:
        return

    n_full = len(df_full)
    n_disp = n_full - display_start   # number of bars in the display window
    if n_disp <= 0:
        return

    supported_rows = []
    if avwap_rows:
        supported_rows = [
            row for row in avwap_rows
            if bool(row.get("supported")) and row.get("anchor_date")
        ]

    if supported_rows:
        palette = [BLUE, GREEN, RED, AMBER, PURPLE, ORANGE, TEXT]
        linestyles = ["--", "-.", ":", "-", "--", "-.", ":"]
        for idx, row in enumerate(supported_rows[:6]):
            try:
                anchor_date = pd.Timestamp(str(row.get("anchor_date")))
                valid_idx = df_full.index[df_full.index >= anchor_date]
                if valid_idx.empty:
                    continue
                anchor_pos = int(df_full.index.get_loc(valid_idx[0]))
                avwap_full = _compute_avwap_series(df_full, anchor_pos)
                avwap_disp = avwap_full[display_start:]
                xs_plot = [i for i, v in enumerate(avwap_disp) if math.isfinite(v)]
                ys_plot = [avwap_disp[i] for i in xs_plot]
                if len(xs_plot) < 2:
                    continue

                color = palette[idx % len(palette)]
                linestyle = linestyles[idx % len(linestyles)]
                label = str(row.get("anchor", f"AVWAP {idx + 1}"))

                ax.plot(
                    xs_plot,
                    ys_plot,
                    color=color,
                    linewidth=0.9,
                    linestyle=linestyle,
                    alpha=0.80,
                    label=label,
                    zorder=4,
                )
                last_x, last_y = xs_plot[-1], ys_plot[-1]
                ax.text(
                    last_x + 0.5,
                    last_y,
                    f" {label} {last_y:.2f}",
                    color=color,
                    fontsize=7,
                    va="center",
                    alpha=0.85,
                    zorder=5,
                )
            except Exception as exc:
                log.debug("packet AVWAP overlay error for %s: %s", row.get("anchor"), exc)
        return

    avwap_specs = [
        ("ytd",        "YTD AVWAP",   BLUE,  "--"),
        ("swing_low",  "SwgLo AVWAP", GREEN, "-."),
        ("swing_high", "SwgHi AVWAP", RED,   "-."),
    ]
    # Add breakout anchor only when the symbol is in trade
    if days_in_state > 0 and math.isfinite(pivot):
        avwap_specs.append(("breakout", "Bkout AVWAP", AMBER, ":"))

    for kind, label, color, ls in avwap_specs:
        try:
            if kind == "ytd":
                anchor_pos = _find_ytd_anchor(df_full)
            elif kind == "swing_low":
                anchor_pos = _find_swing_low_anchor(df_full)
            elif kind == "swing_high":
                anchor_pos = _find_swing_high_anchor(df_full)
            else:  # breakout
                anchor_pos = _find_breakout_anchor(df_full, pivot, days_in_state)

            # Compute AVWAP on full history
            avwap_full = _compute_avwap_series(df_full, anchor_pos)

            # Slice to display window; x positions are 0..n_disp-1
            avwap_disp = avwap_full[display_start:]
            xs_plot = [i for i, v in enumerate(avwap_disp) if math.isfinite(v)]
            ys_plot = [avwap_disp[i] for i in xs_plot]

            if len(xs_plot) < 2:
                continue

            ax.plot(
                xs_plot, ys_plot,
                color=color, linewidth=0.9, linestyle=ls, alpha=0.80,
                label=label, zorder=4,
            )

            # Endpoint label
            last_x, last_y = xs_plot[-1], ys_plot[-1]
            ax.text(
                last_x + 0.5, last_y,
                f" {label} {last_y:.2f}",
                color=color, fontsize=7, va="center", alpha=0.85,
                zorder=5,
            )

        except Exception as exc:
            log.debug("AVWAP curve error (%s): %s", kind, exc)


# ── Public chart generators ───────────────────────────────────────────────────

def generate_daily_chart(
    provider_symbol: str,
    output_dir: Path,
    *,
    pivot: float = math.nan,
    entry_lo: float = math.nan,
    entry_hi: float = math.nan,
    stop: float = math.nan,
    t1: float = math.nan,
    t2: float = math.nan,
    t3: float = math.nan,
    s1: float = math.nan,
    s2: float = math.nan,
    r1: float = math.nan,
    r2: float = math.nan,
    state: str = "",
    action_label: str = "",
    setup_class: str = "",
    score: float = math.nan,
    failure: float = math.nan,
    days_in_state: int = 0,
    avwap_rows: list[dict] | None = None,
) -> Path | None:
    """Generate a daily OHLC candlestick chart with trade levels.

    Requires 200+ bars in the parquet for SMA200; displays the last
    DAILY_BARS (130) bars.  Returns the Path to the PNG, or None.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not available; skipping chart generation")
        return None

    data_path = _RAW_DAILY_DIR / f"{provider_symbol}.parquet"
    if not data_path.exists():
        return None

    try:
        df_full = pd.read_parquet(data_path)
    except Exception as exc:
        log.warning("Chart: could not read %s: %s", data_path, exc)
        return None

    # Need enough history for SMA200 to be meaningful
    min_required = 20
    if len(df_full) < min_required:
        return None

    # Compute MAs on full history, then slice display window
    df_full = df_full.copy()
    df_full["sma5"]   = _sma(df_full["close"],   5)
    df_full["sma10"]  = _sma(df_full["close"],  10)
    df_full["sma20"]  = _sma(df_full["close"],  20)
    df_full["sma50"]  = _sma(df_full["close"],  50)
    df_full["sma200"] = _sma(df_full["close"], 200)

    n_full = len(df_full)
    display_start = max(0, n_full - DAILY_BARS)
    df = df_full.iloc[display_start:].copy()
    if len(df) < 10:
        return None

    n  = len(df)
    xs = range(n)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 8), facecolor=BG)
    ax, ax_vol = _setup_dark_axes(fig, height_ratios=(4, 1))

    title_parts = [f"{provider_symbol} - Daily"]
    if action_label:
        title_parts.append(action_label)
    ax.set_title("  |  ".join(title_parts), color=TEXT, fontsize=10, loc="left", pad=4)
    ax.set_xlim(-1, n)
    ax.yaxis.tick_right()
    ax_vol.set_xlim(-1, n)
    ax_vol.yaxis.tick_right()

    # Candlesticks
    _draw_candlesticks(ax, df)

    # SMA overlays
    ax.plot(xs, df["sma5"],   color=BLUE,   linewidth=0.7, alpha=0.65, label="SMA5")
    ax.plot(xs, df["sma10"],  color=BLUE,   linewidth=0.8, alpha=0.70, label="SMA10",
            linestyle=(0, (5, 2)))
    ax.plot(xs, df["sma20"],  color=GREEN,  linewidth=0.9, alpha=0.75, label="SMA20")
    ax.plot(xs, df["sma50"],  color=AMBER,  linewidth=1.0, alpha=0.85, label="SMA50")
    ax.plot(xs, df["sma200"], color=PURPLE, linewidth=1.1, alpha=0.90, label="SMA200")

    # AVWAP curves (anchored running VWAP from each anchor date forward)
    _overlay_avwap_curves(
        ax,
        df_full,
        display_start,
        pivot=pivot,
        days_in_state=days_in_state,
        avwap_rows=avwap_rows,
    )

    # Trade level annotations
    _annotate_hline(ax, pivot,    "Pivot",    PURPLE, "--")
    _annotate_hline(ax, entry_lo, "Entry lo", GREEN,  ":")
    _annotate_hline(ax, entry_hi, "Entry hi", GREEN,  ":")
    _annotate_hline(ax, stop,     "Stop",     RED,    "-.")
    _annotate_hline(ax, t1,       "T1",       ORANGE, ":")
    _annotate_hline(ax, t2,       "T2",       ORANGE, ":")
    _annotate_hline(ax, t3,       "T3",       AMBER,  ":")

    # Entry zone fill
    if math.isfinite(entry_lo) and math.isfinite(entry_hi):
        ax.axhspan(entry_lo, entry_hi, alpha=0.08, color=GREEN)

    # Base zone fill — light green band between stop and pivot
    if math.isfinite(stop) and math.isfinite(pivot):
        lo_band = min(stop, pivot)
        hi_band = max(stop, pivot)
        ax.axhspan(lo_band, hi_band, alpha=0.05, color=GREEN)

    # S1/S2 support lines (no text label — visible in card levels grid)
    if math.isfinite(s1):
        ax.axhline(s1, color=GREEN, linewidth=0.5, linestyle=":", alpha=0.4)
    if math.isfinite(s2):
        ax.axhline(s2, color=GREEN, linewidth=0.5, linestyle=":", alpha=0.4)

    # R1/R2 resistance lines
    if math.isfinite(r1):
        ax.axhline(r1, color=RED, linewidth=0.5, linestyle=":", alpha=0.4)
    if math.isfinite(r2):
        ax.axhline(r2, color=RED, linewidth=0.5, linestyle=":", alpha=0.4)

    # Current price horizontal marker
    last_close = float(df["close"].iloc[-1])
    if math.isfinite(last_close):
        ax.axhline(last_close, color="white", linewidth=0.5, alpha=0.4)

    # ATR bar size for arrow placement
    atr_bar = float((df["high"] - df["low"]).tail(10).mean())
    if not math.isfinite(atr_bar) or atr_bar <= 0:
        atr_bar = float(df["high"].iloc[-1] - df["low"].iloc[-1])

    # Breakout bar marker — find first bar where close >= pivot
    if state in {"TRIGGERED", "ACCEPTED"} and math.isfinite(pivot):
        breakout_idx: int | None = None
        for i in range(n):
            if float(df["close"].iloc[i]) >= pivot:
                breakout_idx = i
                break
        if breakout_idx is not None:
            ax.annotate(
                "\u25b2",
                xy=(breakout_idx, float(df["low"].iloc[breakout_idx]) - 0.3 * atr_bar),
                color=GREEN,
                fontsize=9,
                ha="center",
                va="top",
                zorder=5,
            )

    # Info text box — top-right corner
    info_lines = []
    if state:
        info_lines.append(state)
    if math.isfinite(score):
        info_lines.append(f"Score {score:.2f}")
    if math.isfinite(failure):
        info_lines.append(f"Fail {failure:.2f}")
    if days_in_state > 0:
        info_lines.append(f"Day {days_in_state}")
    if info_lines:
        ax.text(
            0.99, 0.98, "\n".join(info_lines),
            transform=ax.transAxes, color=TEXT, fontsize=7,
            ha="right", va="top", alpha=0.85,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": BG, "edgecolor": GRID, "alpha": 0.8},
        )

    # Setup class label — bottom-right corner
    if setup_class:
        ax.text(
            0.99, 0.01, setup_class,
            transform=ax.transAxes, color=AMBER, fontsize=7,
            ha="right", va="bottom", alpha=0.85,
        )

    ax.legend(
        fontsize=7, facecolor=BG, edgecolor=GRID, labelcolor=TEXT, loc="upper left"
    )

    # Volume
    _draw_volume(ax_vol, df)
    _set_xaxis_dates(ax_vol, df, step=7)

    # Save
    out_dir = Path(output_dir) / "charts"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{provider_symbol}_daily.png"

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    log.debug("Chart saved: %s", out_path)
    return out_path


def generate_weekly_chart(
    provider_symbol: str,
    output_dir: Path,
    *,
    pivot: float = math.nan,
    t1: float = math.nan,
    t2: float = math.nan,
    s1: float = math.nan,
    stop: float = math.nan,
    avwap_rows: list[dict] | None = None,
) -> Path | None:
    """Generate a weekly OHLC candlestick chart.

    Tries data/raw/weekly/{sym}.parquet first; falls back to resampling daily.
    Returns the Path to the PNG, or None.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    weekly_path = _RAW_WEEKLY_DIR / f"{provider_symbol}.parquet"
    daily_path  = _RAW_DAILY_DIR  / f"{provider_symbol}.parquet"

    df: pd.DataFrame | None = None

    # Try pre-built weekly file
    if weekly_path.exists():
        with contextlib.suppress(Exception):
            df = pd.read_parquet(weekly_path)

    # Fall back: resample from daily
    if df is None and daily_path.exists():
        try:
            daily = pd.read_parquet(daily_path)
            agg: dict = {
                "close":  ("close",  "last"),
                "open":   ("open",   "first"),
                "high":   ("high",   "max"),
                "low":    ("low",    "min"),
            }
            if "volume" in daily.columns:
                agg["volume"] = ("volume", "sum")
            df = daily.resample("W-FRI").agg(**agg)
            df = df.dropna(subset=["close"])
        except Exception as exc:
            log.warning(
                "Weekly chart: could not build weekly bars for %s: %s",
                provider_symbol, exc,
            )

    if df is None or len(df) < 5:
        return None

    # Ensure open column exists (fallback to close for older weekly files)
    if "open" not in df.columns:
        df = df.copy()
        df["open"] = df["close"]

    # Compute MAs on full history; slice display window
    df_wk_full = df.copy()
    df_wk_full["wma10"] = _sma(df_wk_full["close"], 10)
    df_wk_full["wma30"] = _sma(df_wk_full["close"], 30)

    n_wk_full = len(df_wk_full)
    wk_display_start = max(0, n_wk_full - WEEKLY_BARS)
    df = df_wk_full.iloc[wk_display_start:].copy()

    n  = len(df)
    xs = range(n)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 8), facecolor=BG)
    ax, ax_vol = _setup_dark_axes(fig, height_ratios=(4, 1))

    ax.set_title(f"{provider_symbol} — Weekly", color=TEXT, fontsize=10, loc="left", pad=4)
    ax.set_xlim(-1, n)
    ax.yaxis.tick_right()
    ax_vol.set_xlim(-1, n)
    ax_vol.yaxis.tick_right()

    # Candlesticks
    _draw_candlesticks(ax, df)

    # Weekly MAs
    ax.plot(xs, df["wma10"], color=GREEN, linewidth=0.9, alpha=0.85, label="WMA10")
    ax.plot(xs, df["wma30"], color=AMBER, linewidth=1.0, alpha=0.85, label="WMA30")

    # AVWAP curves: YTD + swing_low on weekly chart
    _overlay_avwap_curves(
        ax,
        df_wk_full,
        wk_display_start,
        pivot=pivot,
        avwap_rows=avwap_rows,
    )

    # Trade levels
    _annotate_hline(ax, pivot, "Pivot", PURPLE, "--")
    _annotate_hline(ax, t1,    "T1",    ORANGE, ":")
    _annotate_hline(ax, t2,    "T2",    ORANGE, ":")

    # Base zone fill between stop and pivot
    if math.isfinite(stop) and math.isfinite(pivot):
        lo_band = min(stop, pivot)
        hi_band = max(stop, pivot)
        ax.axhspan(lo_band, hi_band, alpha=0.05, color=GREEN)

    # Prior swing high/low from the visible bars (exclude last 3)
    if len(df) >= 10:
        swing_hi = float(df["high"].iloc[:-3].max())
        swing_lo = float(df["low"].iloc[:-3].min())
        _annotate_hline(ax, swing_hi, "P.Hi", TEXT, ":", alpha=0.30)
        _annotate_hline(ax, swing_lo, "P.Lo", TEXT, ":", alpha=0.30)

    ax.legend(
        fontsize=7, facecolor=BG, edgecolor=GRID, labelcolor=TEXT, loc="upper left"
    )

    # Volume
    if "volume" in df.columns:
        _draw_volume(ax_vol, df)
    else:
        ax_vol.set_ylabel("Vol", color=TEXT, fontsize=7)
    _set_xaxis_dates(ax_vol, df, step=7)

    # Save
    out_dir = Path(output_dir) / "charts"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{provider_symbol}_weekly.png"

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    log.debug("Chart saved: %s", out_path)
    return out_path


def generate_intraday_chart(
    provider_symbol: str,
    output_dir: Path,
    *,
    pivot: float = math.nan,
) -> Path | None:
    """Generate an intraday 5m OHLC candlestick chart.

    Returns None if data/raw/intraday/{sym}.parquet does not exist
    (caller should show an "intraday unavailable" panel in that case).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    intraday_path = _RAW_INTRADAY_DIR / f"{provider_symbol}.parquet"
    if not intraday_path.exists():
        return None

    try:
        df = pd.read_parquet(intraday_path)
    except Exception:
        return None

    df = df.tail(INTRADAY_BARS).copy()
    if len(df) < 10:
        return None

    # Ensure open column exists
    if "open" not in df.columns:
        df["open"] = df["close"]

    # Session VWAP (cumulative from first bar shown)
    if "volume" in df.columns:
        tp = (
            df.get("high", df["close"])
            + df.get("low", df["close"])
            + df["close"]
        ) / 3
        cum_vol = df["volume"].cumsum()
        df["vwap"] = (tp * df["volume"]).cumsum() / cum_vol.replace(0, pd.NA)

    # Short EMAs
    df["ema5"]  = _ema(df["close"],  5)
    df["ema10"] = _ema(df["close"], 10)

    n  = len(df)
    xs = range(n)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(12, 7), facecolor=BG)
    ax, ax_vol = _setup_dark_axes(fig, height_ratios=(4, 1))

    ax.set_title(
        f"{provider_symbol} — Intraday (5m)", color=TEXT, fontsize=10, loc="left", pad=4
    )
    ax.set_xlim(-1, n)
    ax.yaxis.tick_right()
    ax_vol.set_xlim(-1, n)
    ax_vol.yaxis.tick_right()

    # Candlesticks
    _draw_candlesticks(ax, df)

    # Session VWAP
    if "vwap" in df.columns:
        ax.plot(
            xs, df["vwap"],
            color=PURPLE, linewidth=0.9, linestyle="--", label="VWAP", alpha=0.9,
        )

    # Short EMAs
    ax.plot(xs, df["ema5"],  color=BLUE,  linewidth=0.7, alpha=0.75, label="EMA5")
    ax.plot(xs, df["ema10"], color=GREEN, linewidth=0.8, alpha=0.80, label="EMA10")

    # Pivot
    _annotate_hline(ax, pivot, "Pivot", GREEN, "--")

    ax.legend(
        fontsize=7, facecolor=BG, edgecolor=GRID, labelcolor=TEXT, loc="upper left"
    )

    # Volume
    if "volume" in df.columns:
        _draw_volume(ax_vol, df)
    else:
        ax_vol.set_ylabel("Vol", color=TEXT, fontsize=7)
    _set_xaxis_dates(ax_vol, df, step=7)

    # Save
    out_dir = Path(output_dir) / "charts"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{provider_symbol}_intraday.png"

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    log.debug("Chart saved: %s", out_path)
    return out_path


# ── Packet-level entry point ──────────────────────────────────────────────────

def generate_charts_for_packet(packet: dict, output_dir: Path) -> dict:
    """Generate daily and weekly charts for a packet.

    Parameters
    ----------
    packet     : dict from packet.build_packet()
    output_dir : directory containing this day's snapshot.html

    Returns
    -------
    Updated packet dict with chart_daily and chart_weekly set to POSIX-relative
    paths from output_dir (suitable for <img src="...">), or None.

    Intraday is intentionally not generated in the packet-first v1 workflow.
    Surfaced cards use a compact policy note instead of implying live intraday
    confirmation.
    """
    sym = packet.get("provider_symbol") or packet.get("symbol", "")
    if not sym or sym == "—":
        return packet

    def _rel(p: Path | None) -> str | None:
        if p is None:
            return None
        try:
            return p.relative_to(output_dir).as_posix()
        except ValueError:
            return p.as_posix()

    def _lvl(k: str) -> float:
        v = packet.get(k, "—")
        if isinstance(v, (int, float)):
            try:
                return float(v)
            except (TypeError, ValueError):
                return math.nan
        try:
            return float(str(v).replace(",", "")) if v not in ("—", None, "") else math.nan
        except (ValueError, AttributeError):
            return math.nan

    result = dict(packet)

    with contextlib.suppress(Exception):
        result["chart_daily"] = _rel(
            generate_daily_chart(
                sym, output_dir,
                pivot=_lvl("pivot"),
                entry_lo=_lvl("entry_lo"),
                entry_hi=_lvl("entry_hi"),
                stop=_lvl("stop"),
                t1=_lvl("t1"),
                t2=_lvl("t2"),
                t3=_lvl("t3"),
                s1=_lvl("s1"),
                s2=_lvl("s2"),
                r1=_lvl("r1"),
                r2=_lvl("r2"),
                state=str(packet.get("state", "")),
                action_label=str(packet.get("action_label", "")),
                setup_class=str(packet.get("setup_classification", "")),
                score=_lvl("composite_score"),
                failure=_lvl("failure_risk"),
                days_in_state=int(packet.get("days_in_state", 0) or 0),
                avwap_rows=packet.get("avwap_table"),
            )
        )

    with contextlib.suppress(Exception):
        result["chart_weekly"] = _rel(
            generate_weekly_chart(
                sym, output_dir,
                pivot=_lvl("pivot"),
                t1=_lvl("t1"),
                t2=_lvl("t2"),
                s1=_lvl("s1"),
                stop=_lvl("stop"),
                avwap_rows=packet.get("avwap_table"),
            )
        )

    result["chart_intraday"] = None
    result["intraday_policy"] = "daily_only"
    result["intraday_available"] = False
    result["intraday_used_in_qualification"] = False
    result["intraday_note"] = (
        packet.get("intraday_note")
        or "Intraday confirmation is not part of v1 qualification; surfaced setup truth is daily/weekly only."
    )

    return result
