"""Tests for trade journal schema, persistence, and summary statistics."""
from __future__ import annotations

import pandas as pd
import pytest

from swingtrader.journal.schema import (
    REASON_CONFIRMED,
    REASON_FAILED,
    REASON_OPEN,
    TradeRecord,
    journal_summary,
    load_journal,
    save_journal,
    upsert_record,
)

# ── Fixture ──────────────────────────────────────────────────────────────────

def _make_record(symbol: str = "AAPL", trigger_date: str = "2024-03-01") -> TradeRecord:
    ts = pd.Timestamp(trigger_date)
    return TradeRecord(
        symbol=symbol,
        trigger_date=ts,
        entry_date=ts,
        entry_price=150.0,
        pivot=155.0,
        trigger_pivot=154.5,
        atr_at_trigger=3.0,
        target_price=154.5 + 2.0 * 3.0,  # 160.5
        stop_price=154.5 - 1.0 * 3.0,     # 151.5
    )


# ── TradeRecord ───────────────────────────────────────────────────────────────

def test_trade_record_to_series_has_correct_columns() -> None:
    rec = _make_record()
    s = rec.to_series()
    assert "symbol" in s.index
    assert "trigger_date" in s.index
    assert "composite_score" in s.index
    assert "exit_reason" in s.index


def test_trade_record_default_exit_reason_is_open() -> None:
    rec = _make_record()
    assert rec.exit_reason == REASON_OPEN


def test_trade_record_is_open_true_when_no_exit() -> None:
    rec = _make_record()
    assert rec.is_open


def test_trade_record_is_open_false_when_exit_set() -> None:
    rec = _make_record()
    closed = TradeRecord(**{**rec.__dict__, "exit_date": pd.Timestamp("2024-03-10"),
                            "exit_price": 162.0, "exit_reason": REASON_CONFIRMED})
    assert not closed.is_open


def test_trade_record_is_winner_none_when_open() -> None:
    rec = _make_record()
    assert rec.is_winner is None


def test_trade_record_is_winner_true() -> None:
    rec = _make_record()
    winner = TradeRecord(**{**rec.__dict__, "pnl_pct": 0.05, "exit_reason": REASON_CONFIRMED,
                            "exit_date": pd.Timestamp("2024-03-10"), "exit_price": 157.5})
    assert winner.is_winner is True


def test_trade_record_is_winner_false() -> None:
    rec = _make_record()
    loser = TradeRecord(**{**rec.__dict__, "pnl_pct": -0.03, "exit_reason": REASON_FAILED,
                           "exit_date": pd.Timestamp("2024-03-10"), "exit_price": 145.5})
    assert loser.is_winner is False


# ── Persistence ───────────────────────────────────────────────────────────────

def test_load_journal_returns_empty_df_when_no_file(tmp_path) -> None:
    path = tmp_path / "journal" / "trades.parquet"
    df = load_journal(path)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
    assert "symbol" in df.columns


def test_save_and_load_journal_roundtrip(tmp_path) -> None:
    path = tmp_path / "trades.parquet"
    rec = _make_record()
    row = rec.to_series().to_frame().T.reset_index(drop=True)
    save_journal(row, path)
    loaded = load_journal(path)
    assert len(loaded) == 1
    assert loaded.iloc[0]["symbol"] == "AAPL"


def test_upsert_record_adds_new_record(tmp_path) -> None:
    path = tmp_path / "trades.parquet"
    rec = _make_record("AAPL", "2024-03-01")
    upsert_record(rec, path)
    df = load_journal(path)
    assert len(df) == 1


def test_upsert_record_updates_existing(tmp_path) -> None:
    path = tmp_path / "trades.parquet"
    rec = _make_record("AAPL", "2024-03-01")
    upsert_record(rec, path)

    # Now update with exit info — create an updated record with exit fields
    closed2 = TradeRecord(
        symbol=rec.symbol,
        trigger_date=rec.trigger_date,
        entry_date=rec.entry_date,
        entry_price=rec.entry_price,
        pivot=rec.pivot,
        trigger_pivot=rec.trigger_pivot,
        atr_at_trigger=rec.atr_at_trigger,
        target_price=rec.target_price,
        stop_price=rec.stop_price,
        exit_date=pd.Timestamp("2024-03-10"),
        exit_price=162.0,
        exit_reason=REASON_CONFIRMED,
        pnl_atr=4.0,
        pnl_pct=0.08,
    )
    upsert_record(closed2, path)
    df = load_journal(path)
    # Should still be 1 row (update, not duplicate)
    assert len(df) == 1
    assert df.iloc[0]["exit_reason"] == REASON_CONFIRMED


def test_upsert_multiple_symbols(tmp_path) -> None:
    path = tmp_path / "trades.parquet"
    for sym in ["AAPL", "MSFT", "NVDA"]:
        upsert_record(_make_record(sym, "2024-03-01"), path)
    df = load_journal(path)
    assert len(df) == 3
    assert set(df["symbol"]) == {"AAPL", "MSFT", "NVDA"}


# ── journal_summary ───────────────────────────────────────────────────────────

def test_journal_summary_empty() -> None:
    summ = journal_summary(pd.DataFrame(columns=TradeRecord._COLUMNS))
    assert summ["n_total"] == 0


def test_journal_summary_all_open() -> None:
    recs = [_make_record(f"SYM{i}", f"2024-0{i+1}-01") for i in range(3)]
    rows = pd.concat([r.to_series().to_frame().T for r in recs], ignore_index=True)
    summ = journal_summary(rows)
    assert summ["n_open"] == 3
    assert summ["n_closed"] == 0


def test_journal_summary_win_rate() -> None:
    rows = []
    for i, (pnl, reason) in enumerate([
        (2.0, REASON_CONFIRMED),
        (-0.5, REASON_FAILED),
        (1.5, REASON_CONFIRMED),
    ]):
        r = TradeRecord(
            symbol=f"SYM{i}",
            trigger_date=pd.Timestamp(f"2024-0{i+1}-01"),
            entry_date=pd.Timestamp(f"2024-0{i+1}-01"),
            entry_price=100.0,
            pivot=102.0,
            trigger_pivot=101.5,
            atr_at_trigger=2.0,
            target_price=105.5,
            stop_price=99.5,
            exit_date=pd.Timestamp(f"2024-0{i+1}-15"),
            exit_price=100.0 + pnl,
            exit_reason=reason,
            pnl_atr=pnl,
            pnl_pct=pnl / 100.0,
        )
        rows.append(r.to_series())
    df = pd.concat([s.to_frame().T for s in rows], ignore_index=True)
    summ = journal_summary(df)
    assert summ["n_closed"] == 3
    assert summ["win_rate"] == pytest.approx(2 / 3, abs=1e-4)


def test_journal_summary_profit_factor() -> None:
    """gross_profit / gross_loss."""
    # 2 winners at 3 ATR each, 1 loser at -1 ATR → PF = 6/1 = 6
    rows = []
    data = [(3.0, REASON_CONFIRMED), (3.0, REASON_CONFIRMED), (-1.0, REASON_FAILED)]
    for i, (pnl, reason) in enumerate(data):
        r = TradeRecord(
            symbol=f"SYM{i}",
            trigger_date=pd.Timestamp(f"2024-0{i+1}-01"),
            entry_date=pd.Timestamp(f"2024-0{i+1}-01"),
            entry_price=100.0, pivot=102.0, trigger_pivot=101.5, atr_at_trigger=2.0,
            target_price=105.5, stop_price=99.5,
            exit_date=pd.Timestamp(f"2024-0{i+1}-15"),
            exit_price=100.0 + pnl, exit_reason=reason,
            pnl_atr=pnl, pnl_pct=pnl / 100.0,
        )
        rows.append(r.to_series())
    df = pd.concat([s.to_frame().T for s in rows], ignore_index=True)
    summ = journal_summary(df)
    assert summ["profit_factor"] == pytest.approx(6.0, abs=1e-4)
