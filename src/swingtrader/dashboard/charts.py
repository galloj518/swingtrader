"""Static chart generation for top setups.

Generates PNG charts using matplotlib (already a project dependency).
Charts are saved to:
  {output_dir}/charts/{SYMBOL}_weekly.png
  {output_dir}/charts/{SYMBOL}_daily.png
  {output_dir}/charts/{SYMBOL}_intraday.png  (only if 5m data exists)

Design
------
- Dark background to match the site theme (#0d1117).
- Each chart shows: price, key MAs, pivot line, entry zone, stop, T1/T2, volume.
- Deliberately simple: no interactive JS, no 3rd-party charting libs.
- Charts are generated only for symbols present in data/raw/{timeframe}/.
  If data is missing, the chart is skipped and the path is not returned.

All matplotlib imports are local to the functions so the module can be
imported without matplotlib in testing environments where it is unavailable.
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from swingtrader.utils.config import REPO_ROOT
from swingtrader.utils.logging import get_logger

log = get_logger(__name__)

_RAW_DAILY_DIR = REPO_ROOT / "data" / "raw" / "daily"
_RAW_WEEKLY_DIR = REPO_ROOT / "data" / "raw" / "weekly"
_RAW_INTRADAY_DIR = REPO_ROOT / "data" / "raw" / "intraday"

# ── Dark theme colours (match site palette) ───────────────────────────────────
BG = "#0d1117"
GRID = "#21262d"
TEXT = "#c9d1d9"
BLUE = "#58a6ff"
GREEN = "#3fb950"
RED = "#f78166"
AMBER = "#d29922"
PURPLE = "#bc8cff"
ORANGE = "#fb8f44"

# Max bars shown on each timeframe chart
DAILY_BARS = 130
WEEKLY_BARS = 60
INTRADAY_BARS = 78  # ~6.5 hours at 5-min intervals


def _setup_dark_figure(figsize=(12, 7)):
    """Return (fig, axes) with dark theme applied."""
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=figsize, facecolor=BG)
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05, figure=fig)
    ax_price = fig.add_subplot(gs[0])
    ax_vol = fig.add_subplot(gs[1], sharex=ax_price)

    for ax in (ax_price, ax_vol):
        ax.set_facecolor(BG)
        ax.tick_params(colors=TEXT, labelsize=8)
        ax.spines[:].set_color(GRID)
        ax.grid(color=GRID, linewidth=0.5, alpha=0.6)

    plt.setp(ax_price.get_xticklabels(), visible=False)
    return fig, ax_price, ax_vol


def _ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()


def _sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n, min_periods=1).mean()


def _annotate_hline(ax, price: float, label: str, color: str, linestyle: str = "--") -> None:
    if not math.isfinite(price):
        return
    ax.axhline(price, color=color, linewidth=0.8, linestyle=linestyle, alpha=0.85)
    ax.text(
        0.005, price, f" {label} {price:.2f}",
        transform=ax.get_yaxis_transform(),
        color=color, fontsize=7, va="bottom",
    )


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
) -> Path | None:
    """Generate a daily price chart with trade levels.

    Returns the Path to the PNG file, or None if data is unavailable.
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
        df = pd.read_parquet(data_path)
    except Exception as exc:
        log.warning("Chart: could not read %s: %s", data_path, exc)
        return None

    df = df.tail(DAILY_BARS).copy()
    if len(df) < 10:
        return None

    # Compute indicators
    df["ema20"] = _ema(df["close"], 20)
    df["sma50"] = _sma(df["close"], 50)
    xs = range(len(df))

    fig, ax, ax_vol = _setup_dark_figure()
    ax.set_title(f"{provider_symbol} — Daily", color=TEXT, fontsize=10, loc="left", pad=4)

    # Close line + MAs
    ax.plot(xs, df["close"], color=BLUE, linewidth=1.2, label="Close")
    ax.plot(xs, df["ema20"], color=GREEN, linewidth=0.8, linestyle="-", label="EMA20", alpha=0.85)
    ax.plot(xs, df["sma50"], color=AMBER, linewidth=0.8, linestyle="-", label="SMA50", alpha=0.85)

    # Trade levels
    _annotate_hline(ax, pivot, "Pivot", PURPLE, "--")
    _annotate_hline(ax, entry_lo, "Entry lo", GREEN, ":")
    _annotate_hline(ax, entry_hi, "Entry hi", GREEN, ":")
    _annotate_hline(ax, stop, "Stop", RED, "-.")
    _annotate_hline(ax, t1, "T1", ORANGE, ":")
    _annotate_hline(ax, t2, "T2", ORANGE, ":")

    # Entry zone fill
    if math.isfinite(entry_lo) and math.isfinite(entry_hi):
        ax.axhspan(entry_lo, entry_hi, alpha=0.08, color=GREEN)

    ax.legend(fontsize=7, facecolor=BG, edgecolor=GRID, labelcolor=TEXT, loc="upper left")
    ax.yaxis.tick_right()

    # Volume
    ax_vol.bar(xs, df["volume"], color=BLUE, alpha=0.4, width=0.8)
    ax_vol.set_ylabel("Vol", color=TEXT, fontsize=7)
    ax_vol.yaxis.tick_right()

    # X-axis date labels (every ~20 bars)
    step = max(len(df) // 6, 1)
    tick_idx = list(range(0, len(df), step))
    ax_vol.set_xticks(tick_idx)
    ax_vol.set_xticklabels(
        [str(df.index[i])[:10] for i in tick_idx],
        rotation=30, ha="right", fontsize=7, color=TEXT,
    )

    out_dir = Path(output_dir) / "charts"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{provider_symbol}_daily.png"

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
    """Generate a weekly chart."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    # Try weekly-resampled data first; fall back to daily and resample
    weekly_path = _RAW_WEEKLY_DIR / f"{provider_symbol}.parquet"
    daily_path = _RAW_DAILY_DIR / f"{provider_symbol}.parquet"

    import contextlib

    df = None
    if weekly_path.exists():
        with contextlib.suppress(Exception):
            df = pd.read_parquet(weekly_path)

    if df is None and daily_path.exists():
        try:
            daily = pd.read_parquet(daily_path)
            df = daily["close"].resample("W-FRI").last().rename("close").to_frame()
            if "volume" in daily.columns:
                df["volume"] = daily["volume"].resample("W-FRI").sum()
            if "high" in daily.columns:
                df["high"] = daily["high"].resample("W-FRI").max()
            if "low" in daily.columns:
                df["low"] = daily["low"].resample("W-FRI").min()
        except Exception as exc:
            log.warning("Weekly chart: could not build weekly bars for %s: %s", provider_symbol, exc)

    if df is None or len(df) < 5:
        return None

    df = df.tail(WEEKLY_BARS).copy()
    df["ema10"] = _ema(df["close"], 10)
    df["ema30"] = _ema(df["close"], 30)
    xs = range(len(df))

    fig, ax, ax_vol = _setup_dark_figure()
    ax.set_title(f"{provider_symbol} — Weekly", color=TEXT, fontsize=10, loc="left", pad=4)

    ax.plot(xs, df["close"], color=BLUE, linewidth=1.2)
    ax.plot(xs, df["ema10"], color=GREEN, linewidth=0.8, alpha=0.85, label="EMA10w")
    ax.plot(xs, df["ema30"], color=AMBER, linewidth=0.8, alpha=0.85, label="EMA30w")

    _annotate_hline(ax, pivot, "Pivot", PURPLE, "--")
    _annotate_hline(ax, t1, "T1", ORANGE, ":")
    _annotate_hline(ax, t2, "T2", ORANGE, ":")

    ax.legend(fontsize=7, facecolor=BG, edgecolor=GRID, labelcolor=TEXT, loc="upper left")
    ax.yaxis.tick_right()

    if "volume" in df.columns:
        ax_vol.bar(xs, df["volume"], color=BLUE, alpha=0.4, width=0.8)
    ax_vol.set_ylabel("Vol", color=TEXT, fontsize=7)
    ax_vol.yaxis.tick_right()

    step = max(len(df) // 6, 1)
    tick_idx = list(range(0, len(df), step))
    ax_vol.set_xticks(tick_idx)
    ax_vol.set_xticklabels(
        [str(df.index[i])[:10] for i in tick_idx],
        rotation=30, ha="right", fontsize=7, color=TEXT,
    )

    out_dir = Path(output_dir) / "charts"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{provider_symbol}_weekly.png"

    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return out_path


def generate_intraday_chart(
    provider_symbol: str,
    output_dir: Path,
    *,
    pivot: float = math.nan,
) -> Path | None:
    """Generate an intraday 5m chart (only if data exists)."""
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

    # Session VWAP
    if "volume" in df.columns and "close" in df.columns:
        tp = (df.get("high", df["close"]) + df.get("low", df["close"]) + df["close"]) / 3
        cumvol = df["volume"].cumsum()
        df["vwap"] = (tp * df["volume"]).cumsum() / cumvol.replace(0, pd.NA)

    xs = range(len(df))
    fig, ax, ax_vol = _setup_dark_figure()
    ax.set_title(f"{provider_symbol} — Intraday (5m)", color=TEXT, fontsize=10, loc="left", pad=4)

    ax.plot(xs, df["close"], color=BLUE, linewidth=1.0)
    if "vwap" in df.columns:
        ax.plot(xs, df["vwap"], color=PURPLE, linewidth=0.8, linestyle="--", label="VWAP", alpha=0.9)

    _annotate_hline(ax, pivot, "Pivot", GREEN, "--")
    ax.legend(fontsize=7, facecolor=BG, edgecolor=GRID, labelcolor=TEXT, loc="upper left")
    ax.yaxis.tick_right()

    if "volume" in df.columns:
        ax_vol.bar(xs, df["volume"], color=BLUE, alpha=0.4, width=0.8)
    ax_vol.yaxis.tick_right()

    out_dir = Path(output_dir) / "charts"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{provider_symbol}_intraday.png"

    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return out_path


def generate_charts_for_packet(packet: dict, output_dir: Path) -> dict:
    """Generate all charts for a setup packet, returning updated chart paths.

    Parameters
    ----------
    packet     : dict from packet.build_packet()
    output_dir : directory containing this day's snapshot.html

    Returns
    -------
    Updated packet dict with chart_weekly, chart_daily, chart_intraday set
    to relative paths from output_dir (suitable for <img src="...">), or None.
    """
    sym = packet.get("provider_symbol") or packet.get("symbol", "")
    if not sym or sym == "—":
        return packet

    def _rel(p: Path | None) -> str | None:
        if p is None:
            return None
        try:
            # Always use forward slashes so paths work on Linux (GitHub Pages)
            return p.relative_to(output_dir).as_posix()
        except ValueError:
            return p.as_posix()

    def _lvl(k: str) -> float:
        v = packet.get(k, "—")
        try:
            return float(v.replace(",", "")) if v != "—" else math.nan
        except (ValueError, AttributeError):
            return math.nan

    daily_path = generate_daily_chart(
        sym, output_dir,
        pivot=_lvl("pivot"),
        entry_lo=_lvl("entry_lo"),
        entry_hi=_lvl("entry_hi"),
        stop=_lvl("stop"),
        t1=_lvl("t1"),
        t2=_lvl("t2"),
    )
    weekly_path = generate_weekly_chart(
        sym, output_dir,
        pivot=_lvl("pivot"),
        t1=_lvl("t1"),
        t2=_lvl("t2"),
    )
    intraday_path = generate_intraday_chart(sym, output_dir, pivot=_lvl("pivot"))

    result = dict(packet)
    result["chart_daily"] = _rel(daily_path)
    result["chart_weekly"] = _rel(weekly_path)
    result["chart_intraday"] = _rel(intraday_path)
    return result
