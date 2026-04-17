"""Universe loader.

Combines configured sources (benchmarks, sector ETFs, S&P 500 / NDX member CSVs, and a
user watchlist) into a deduped, ordered list of symbols. Liquidity filtering is applied
separately after daily bars are available — see :func:`apply_liquidity_filter`.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from swingtrader.utils.config import REPO_ROOT, Config, load_config
from swingtrader.utils.logging import get_logger

log = get_logger(__name__)


def _load_symbols_csv(relpath: str) -> list[str]:
    """Load a symbol CSV (column ``symbol`` if present, else first column)."""
    p = REPO_ROOT / relpath
    if not p.exists():
        log.warning("Symbol CSV missing: %s (run scripts/refresh_universe.py)", p)
        return []
    df = pd.read_csv(p)
    col = "symbol" if "symbol" in df.columns else df.columns[0]
    return [str(s).strip().upper() for s in df[col].dropna().tolist()]


def load_universe(cfg: Config | None = None) -> list[str]:
    """Return the deduplicated list of provider symbols for the configured universe.

    Delegates to :func:`swingtrader.ingest.symbols.resolve_universe` which handles
    the current config schema (active_sources, symbol_aliases, non_equity_symbols).

    Returns provider symbols (e.g. BRK-B not BRK.B) suitable for data fetching.
    """
    from swingtrader.ingest.symbols import resolve_universe
    records = resolve_universe(cfg)
    return [r.provider_symbol for r in records]


def apply_liquidity_filter(
    symbols: list[str],
    daily_dir: Path | str,
    cfg: Config | None = None,
) -> list[str]:
    """Filter symbols by 20-day median dollar volume and price floor.

    Requires daily parquet files at ``<daily_dir>/<SYMBOL>.parquet``. Symbols lacking
    data are kept (they'll be flagged by the quality report) so this filter cannot
    silently erase a freshly added name.
    """
    cfg = cfg or load_config("universe")
    lf = cfg.get("liquidity_filter", {})
    if not lf.get("enabled", False):
        return symbols

    min_dv = float(lf.get("min_dollar_volume_median_20d", 0))
    min_price = float(lf.get("min_price", 0))
    max_symbols = int(lf.get("max_symbols", 10_000))
    daily_dir = Path(daily_dir)

    scored: list[tuple[str, float]] = []
    for sym in symbols:
        path = daily_dir / f"{sym}.parquet"
        if not path.exists():
            scored.append((sym, float("inf")))  # keep; let quality check flag it
            continue
        try:
            df = pd.read_parquet(path)
        except Exception:
            log.warning("Unreadable parquet for %s, keeping in universe", sym)
            scored.append((sym, float("inf")))
            continue
        if df.empty or "close" not in df.columns or "volume" not in df.columns:
            continue
        last = df.tail(20)
        dv = (last["close"] * last["volume"]).median()
        price = float(last["close"].iloc[-1])
        if price < min_price or dv < min_dv:
            continue
        scored.append((sym, float(dv)))

    # Sort by liquidity desc and cap
    scored.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in scored[:max_symbols]]
