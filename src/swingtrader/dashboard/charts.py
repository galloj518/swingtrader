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
        l = float(df["low"].iloc[i])
        c = float(df["close"].iloc[i])

        if not (math.isfinite(o) and math.isfinite(h) and
                math.isfinite(l) and math.isfinite(c)):
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
        if l < body_lo:
            ax.plot(
                [i, i], [l, body_lo],
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


def _overlay_avwap(ax, feat_row: pd.Series | None) -> None:
    """Draw AVWAP horizontal reference lines from the features row."""
    avwap_specs = [
        ("ytd_avwap",         "YTD AVWAP",   BLUE,   "--"),
        ("swing_low_avwap",   "SwgLo AVWAP", GREEN,  "-."),
        ("swing_high_avwap",  "SwgHi AVWAP", RED,    "-."),
        ("breakout_day_avwap","Bkout AVWAP", AMBER,  ":"),
    ]
    for key, label, color, ls in avwap_specs:
        val = _safe_f(feat_row, key)
        if math.isfinite(val):
            _annotate_hline(ax, val, label, color, ls, alpha=0.75)


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

    df = df_full.tail(DAILY_BARS).copy()
    if len(df) < 10:
        return None

    n  = len(df)
    xs = range(n)

    # Load feature row for AVWAP overlays
    feat_row = _load_features_row(provider_symbol)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 8), facecolor=BG)
    ax, ax_vol = _setup_dark_axes(fig, height_ratios=(4, 1))

    ax.set_title(f"{provider_symbol} — Daily", color=TEXT, fontsize=10, loc="left", pad=4)
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

    # AVWAP overlays
    _overlay_avwap(ax, feat_row)

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

    df = df.tail(WEEKLY_BARS).copy()

    # Weekly MAs on close
    df["wma10"] = _sma(df["close"], 10)
    df["wma30"] = _sma(df["close"], 30)

    n  = len(df)
    xs = range(n)

    feat_row = _load_features_row(provider_symbol)

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

    # AVWAP overlays (ytd + swing_low only for weekly)
    for key, label, color, ls in [
        ("ytd_avwap",       "YTD AVWAP",   BLUE,  "--"),
        ("swing_low_avwap", "SwgLo AVWAP", GREEN, "-."),
    ]:
        val = _safe_f(feat_row, key)
        if math.isfinite(val):
            _annotate_hline(ax, val, label, color, ls, alpha=0.75)

    # Trade levels
    _annotate_hline(ax, pivot, "Pivot", PURPLE, "--")
    _annotate_hline(ax, t1,    "T1",    ORANGE, ":")
    _annotate_hline(ax, t2,    "T2",    ORANGE, ":")

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
    """Generate all charts for a setup packet; return updated packet with chart paths.

    Parameters
    ----------
    packet     : dict from packet.build_packet()
    output_dir : directory containing this day's snapshot.html

    Returns
    -------
    Updated packet dict with chart_daily, chart_weekly, chart_intraday set to
    POSIX-relative paths from output_dir (suitable for <img src="...">), or None.
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
            )
        )

    with contextlib.suppress(Exception):
        result["chart_weekly"] = _rel(
            generate_weekly_chart(
                sym, output_dir,
                pivot=_lvl("pivot"),
                t1=_lvl("t1"),
                t2=_lvl("t2"),
            )
        )

    with contextlib.suppress(Exception):
        result["chart_intraday"] = _rel(
            generate_intraday_chart(sym, output_dir, pivot=_lvl("pivot"))
        )

    return result
