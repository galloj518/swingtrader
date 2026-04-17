"""yfinance ingest — the only module that imports :mod:`yfinance`.

Responsibilities:
- Batch-fetch daily OHLCV per :mod:`config.data_sources`
- Normalize column names and index
- Write per-symbol parquet under ``data/raw/daily/{SYMBOL}.parquet``
- Resample to weekly bars on demand (no separate vendor call — avoids vendor quirks)
- Optionally pull intraday 5m bars (forward-only accumulation; yfinance capped ~60d)

Non-goals: features, labels, any analysis. Those live downstream.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

try:  # pragma: no cover - optional at test time
    import yfinance as yf  # type: ignore[import-not-found]
except ImportError:
    yf = None  # type: ignore[assignment]

from swingtrader.utils.config import REPO_ROOT, Config, load_config
from swingtrader.utils.io import append_parquet, read_parquet, write_parquet
from swingtrader.utils.logging import get_logger

log = get_logger(__name__)

EXPECTED_COLS = ["open", "high", "low", "close", "volume"]


@dataclass
class IngestResult:
    """Outcome of fetching a single symbol."""

    symbol: str
    rows: int
    path: Path
    ok: bool
    error: str | None = None


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase columns, enforce DatetimeIndex named ``date``, keep OHLCV."""
    df = df.copy()
    df.columns = [str(c).lower() for c in df.columns]
    keep = [c for c in EXPECTED_COLS if c in df.columns]
    df = df[keep]
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index.name = "date"
    return df.dropna(how="all")


def _extract_symbol(data: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Pull a single symbol's OHLCV from a multi-ticker yfinance download."""
    if isinstance(data.columns, pd.MultiIndex):
        top_level = data.columns.get_level_values(0).unique()
        if symbol in top_level:
            return data[symbol]
        bottom_level = data.columns.get_level_values(-1).unique()
        if symbol in bottom_level:
            return data.xs(symbol, axis=1, level=-1)
        raise KeyError(f"Symbol {symbol} not found in multi-index download")
    return data


def fetch_daily(symbols: list[str], cfg: Config | None = None) -> list[IngestResult]:
    """Fetch daily bars for each symbol and write parquet files.

    Network-bound. Designed to be run from a scheduled workflow; handles partial failures
    per symbol rather than aborting the batch.
    """
    if yf is None:
        raise RuntimeError("yfinance not installed — pip install yfinance")
    cfg = cfg or load_config("data_sources")
    ds = cfg["yfinance"]["daily"]
    storage = cfg["storage"]
    rl = cfg["yfinance"]["rate_limit"]

    out_dir = REPO_ROOT / storage["daily_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[IngestResult] = []

    batch_size = int(rl["batch_size"])
    sleep_s = float(rl["sleep_between_batches_seconds"])

    for i in range(0, len(symbols), batch_size):
        batch = symbols[i : i + batch_size]
        log.info("Fetching daily batch %d..%d (%d symbols)", i, i + len(batch), len(batch))
        try:
            data = yf.download(
                tickers=batch,
                period=ds["period"],
                auto_adjust=ds["auto_adjust"],
                prepost=ds["prepost"],
                progress=False,
                group_by="ticker",
                threads=True,
            )
        except Exception as e:
            log.exception("Batch download failed: %s", e)
            for s in batch:
                results.append(
                    IngestResult(s, 0, out_dir / f"{s}.parquet", ok=False, error=str(e))
                )
            continue

        for sym in batch:
            path = out_dir / f"{sym}.parquet"
            try:
                df = _normalize(_extract_symbol(data, sym))
                if df.empty:
                    raise ValueError("empty after normalize")
                write_parquet(df, path, compression=storage["compression"])
                results.append(IngestResult(sym, len(df), path, ok=True))
            except Exception as e:
                log.warning("Failed to process %s: %s", sym, e)
                results.append(IngestResult(sym, 0, path, ok=False, error=str(e)))

        if i + batch_size < len(symbols):
            time.sleep(sleep_s)

    ok_n = sum(1 for r in results if r.ok)
    log.info("Daily ingest complete: %d ok / %d total", ok_n, len(results))
    return results


def resample_weekly(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Resample daily OHLCV to Friday-ending weekly bars."""
    rules = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    return daily_df.resample("W-FRI").agg(rules).dropna(how="any")


def build_weekly_from_daily(symbols: list[str], cfg: Config | None = None) -> list[IngestResult]:
    """Build weekly parquet files from cached daily parquet — no network call."""
    cfg = cfg or load_config("data_sources")
    storage = cfg["storage"]
    daily_dir = REPO_ROOT / storage["daily_dir"]
    weekly_dir = REPO_ROOT / storage["weekly_dir"]
    weekly_dir.mkdir(parents=True, exist_ok=True)

    results: list[IngestResult] = []
    for sym in symbols:
        src = daily_dir / f"{sym}.parquet"
        dst = weekly_dir / f"{sym}.parquet"
        if not src.exists():
            results.append(IngestResult(sym, 0, dst, ok=False, error="missing daily parquet"))
            continue
        try:
            df = read_parquet(src)
            w = resample_weekly(df)
            write_parquet(w, dst, compression=storage["compression"])
            results.append(IngestResult(sym, len(w), dst, ok=True))
        except Exception as e:
            log.warning("Weekly build failed for %s: %s", sym, e)
            results.append(IngestResult(sym, 0, dst, ok=False, error=str(e)))
    return results


def fetch_intraday_5m(symbols: list[str], cfg: Config | None = None) -> list[IngestResult]:
    """Append recent 5-minute bars to per-symbol intraday parquet files.

    Forward-only accumulator: yfinance caps ~60 days of 5m data, so this function is
    intended to be run frequently (e.g. nightly) to keep rolling history building up
    inside the repo. Historical backfill beyond the yfinance window is not possible.
    """
    if yf is None:
        raise RuntimeError("yfinance not installed")
    cfg = cfg or load_config("data_sources")
    ds = cfg["yfinance"]["intraday"]
    if not ds.get("enabled", False):
        log.info("Intraday ingest disabled in config")
        return []
    storage = cfg["storage"]
    out_dir = REPO_ROOT / storage["intraday_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[IngestResult] = []

    for sym in symbols:
        path = out_dir / f"{sym}.parquet"
        try:
            data = yf.download(
                tickers=sym,
                interval=ds["interval"],
                period=ds["period"],
                prepost=ds["prepost"],
                progress=False,
                auto_adjust=False,
            )
            df = _normalize(data)
            if df.empty:
                raise ValueError("empty intraday fetch")
            append_parquet(df, path, compression=storage["compression"])
            results.append(IngestResult(sym, len(df), path, ok=True))
        except Exception as e:
            log.warning("Intraday fetch failed for %s: %s", sym, e)
            results.append(IngestResult(sym, 0, path, ok=False, error=str(e)))
    return results


def load_daily(symbol: str, cfg: Config | None = None) -> pd.DataFrame:
    """Read a symbol's cached daily bars."""
    cfg = cfg or load_config("data_sources")
    path = REPO_ROOT / cfg["storage"]["daily_dir"] / f"{symbol}.parquet"
    return read_parquet(path)


def load_weekly(symbol: str, cfg: Config | None = None) -> pd.DataFrame:
    """Read a symbol's cached weekly bars."""
    cfg = cfg or load_config("data_sources")
    path = REPO_ROOT / cfg["storage"]["weekly_dir"] / f"{symbol}.parquet"
    return read_parquet(path)
