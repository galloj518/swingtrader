"""Tests for symbol normalization and universe resolution."""
from __future__ import annotations

from swingtrader.ingest.symbols import resolve_universe, to_dataframe


def test_resolve_universe_includes_portfolio() -> None:
    records = resolve_universe()
    syms = [r.user_symbol for r in records]
    # From portfolio_holdings in config
    assert "NVDA" in syms
    assert "GOOGL" in syms


def test_resolve_universe_includes_watchlist() -> None:
    records = resolve_universe()
    syms = [r.user_symbol for r in records]
    assert "AAPL" in syms
    assert "MSFT" in syms
    assert "META" in syms


def test_resolve_universe_no_duplicates() -> None:
    records = resolve_universe()
    syms = [r.user_symbol for r in records]
    assert len(syms) == len(set(syms))


def test_brk_b_alias_mapped() -> None:
    records = resolve_universe()
    brk = next((r for r in records if r.user_symbol == "BRK.B"), None)
    assert brk is not None, "BRK.B should be in watchlist"
    assert brk.provider_symbol == "BRK-B"
    assert brk.user_symbol == "BRK.B"


def test_spaxx_tagged_non_equity() -> None:
    records = resolve_universe()
    spaxx = next((r for r in records if r.user_symbol == "SPAXX"), None)
    assert spaxx is not None
    assert spaxx.is_non_equity is True
    assert spaxx.score_eligible is False


def test_portfolio_holdings_tagged_correctly() -> None:
    records = resolve_universe()
    portfolio = [r for r in records if r.is_portfolio]
    syms = [r.user_symbol for r in portfolio]
    assert "GEV" in syms
    assert "ETN" in syms
    assert len(portfolio) >= 5


def test_benchmarks_tagged_correctly() -> None:
    records = resolve_universe()
    benchmarks = [r for r in records if r.is_benchmark]
    syms = [r.user_symbol for r in benchmarks]
    assert "SPY" in syms
    assert "QQQ" in syms


def test_to_dataframe_serialises_correctly() -> None:
    records = resolve_universe()
    df = to_dataframe(records)
    assert "user_symbol" in df.columns
    assert "provider_symbol" in df.columns
    assert "is_portfolio" in df.columns
    assert len(df) == len(records)


def test_score_eligible_excludes_non_equity() -> None:
    records = resolve_universe()
    non_eq = [r for r in records if r.is_non_equity]
    for r in non_eq:
        assert not r.score_eligible
