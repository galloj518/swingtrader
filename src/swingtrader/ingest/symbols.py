"""Symbol normalization and universe resolution.

Responsibilities:
- Load and merge symbol groups from config/universe.yaml
- Translate user-facing symbols to provider-specific symbols (e.g. BRK.B → BRK-B)
- Tag each symbol with its group memberships (benchmark, etf, portfolio, watchlist)
- Identify non-equity symbols (cash, money-market) that skip breakout scoring
- Write the resolved daily universe artifact to data/universe/

Assumption: symbol_aliases in universe.yaml is the explicit mapping.
  Automatic dot→hyphen substitution is NOT performed to avoid silent mismatches.
  Any new alias must be added explicitly.

Output type: SymbolRecord carries both user_symbol (display) and provider_symbol (fetch).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from swingtrader.utils.config import REPO_ROOT, Config, load_config
from swingtrader.utils.logging import get_logger

if TYPE_CHECKING:
    pass

log = get_logger(__name__)

# Ordered source priority for deduplication.
_SOURCE_ORDER = ["benchmarks", "sector_etfs", "sp500", "nasdaq100", "portfolio_holdings", "custom_watchlist"]


@dataclass
class SymbolRecord:
    """Single resolved symbol with metadata."""

    user_symbol: str                # as written in config (e.g. BRK.B)
    provider_symbol: str            # for data fetch (e.g. BRK-B)
    groups: list[str] = field(default_factory=list)   # [benchmark, portfolio, watchlist, …]
    is_portfolio: bool = False
    is_watchlist: bool = False
    is_benchmark: bool = False
    is_etf: bool = False
    is_non_equity: bool = False     # cash / money-market; skip breakout scoring

    @property
    def display(self) -> str:
        return self.user_symbol

    @property
    def score_eligible(self) -> bool:
        return not self.is_non_equity


def resolve_universe(cfg: Config | None = None) -> list[SymbolRecord]:
    """Return the deduplicated ordered list of SymbolRecords for today's run.

    Sources are merged in priority order. The first occurrence sets the provider_symbol;
    later occurrences only add group tags.
    """
    cfg = cfg or load_config("universe")
    active = cfg.get("active_sources", {})
    sources = cfg.get("sources", {})
    aliases: dict[str, str] = cfg.get("symbol_aliases", {}) or {}
    non_equity: set[str] = set(cfg.get("non_equity_symbols", []) or [])

    # Map: user_symbol → SymbolRecord (ordered by first seen)
    records: dict[str, SymbolRecord] = {}

    def _add(raw_sym: str, group: str) -> None:
        user_sym = str(raw_sym).strip().upper()
        if not user_sym:
            return
        provider_sym = aliases.get(user_sym, user_sym)
        if user_sym not in records:
            records[user_sym] = SymbolRecord(
                user_symbol=user_sym,
                provider_symbol=provider_sym,
                groups=[group],
                is_non_equity=user_sym in non_equity,
            )
        else:
            # Already present — just add the group tag
            if group not in records[user_sym].groups:
                records[user_sym].groups.append(group)
        # Set boolean flags
        rec = records[user_sym]
        if group == "benchmarks":
            rec.is_benchmark = True
        elif group == "sector_etfs":
            rec.is_etf = True
        elif group == "portfolio_holdings":
            rec.is_portfolio = True
        elif group == "custom_watchlist":
            rec.is_watchlist = True

    for source in _SOURCE_ORDER:
        if not active.get(source, True):
            continue
        section = sources.get(source, {})
        if section is None:
            continue

        if source in ("sp500", "nasdaq100"):
            path_rel = section.get("cache_path")
            if path_rel:
                syms = _load_csv(path_rel)
                for s in syms:
                    _add(s, source)
        else:
            for s in (section.get("symbols") or []):
                _add(s, source)

    out = list(records.values())
    log.info(
        "Resolved universe: %d symbols (%d portfolio, %d watchlist, %d non-equity, %d score-eligible)",
        len(out),
        sum(1 for r in out if r.is_portfolio),
        sum(1 for r in out if r.is_watchlist),
        sum(1 for r in out if r.is_non_equity),
        sum(1 for r in out if r.score_eligible),
    )
    return out


def _load_csv(relpath: str) -> list[str]:
    p = REPO_ROOT / relpath
    if not p.exists():
        log.warning("Universe CSV missing: %s", p)
        return []
    df = pd.read_csv(p)
    col = "symbol" if "symbol" in df.columns else df.columns[0]
    return [str(s).strip().upper() for s in df[col].dropna()]


def to_dataframe(records: list[SymbolRecord]) -> pd.DataFrame:
    """Serialise universe records to a DataFrame (for daily snapshot artifact)."""
    rows = []
    for r in records:
        rows.append({
            "user_symbol": r.user_symbol,
            "provider_symbol": r.provider_symbol,
            "groups": ",".join(r.groups),
            "is_portfolio": r.is_portfolio,
            "is_watchlist": r.is_watchlist,
            "is_benchmark": r.is_benchmark,
            "is_etf": r.is_etf,
            "is_non_equity": r.is_non_equity,
            "score_eligible": r.score_eligible,
        })
    return pd.DataFrame(rows)


def write_universe_artifact(records: list[SymbolRecord], as_of: pd.Timestamp | None = None) -> Path:
    """Persist the resolved universe to data/universe/universe.parquet."""
    as_of = as_of or pd.Timestamp.today().normalize()
    df = to_dataframe(records)
    df["as_of"] = as_of
    out_dir = REPO_ROOT / "data" / "universe"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "universe.parquet"
    df.to_parquet(path, index=False)
    return path
