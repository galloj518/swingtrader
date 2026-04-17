"""Trade journal schema and persistence.

Each triggered breakout that is tracked in the live run gets a TradeRecord.
Records are created when a symbol first enters TRIGGERED state and updated when
the state resolves (CONFIRMED, FAILED, or STILL_OPEN after the tracking window).

Journal is stored as parquet at data/journal/trades.parquet.
The journal is append-only in normal operation; records are updated (not
duplicated) when a position resolves — the file is rewritten in full.

Performance summary helpers support journal-based analysis of strategy outcomes.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd

from swingtrader.utils.config import REPO_ROOT
from swingtrader.utils.logging import get_logger

log = get_logger(__name__)

_DEFAULT_JOURNAL = REPO_ROOT / "data" / "journal" / "trades.parquet"

# Exit reason codes
REASON_CONFIRMED = "CONFIRMED"
REASON_FAILED = "FAILED"
REASON_TIMEOUT = "TIMEOUT"
REASON_OPEN = "STILL_OPEN"


@dataclass
class TradeRecord:
    """One row of the trade journal.

    Fields labelled '[filled at trigger]' are set when the record is created.
    Fields labelled '[filled at exit]' are set when the trade resolves.
    """

    # Identity
    symbol: str                    # user_symbol (display form, e.g. BRK.B)
    trigger_date: pd.Timestamp     # date state first became TRIGGERED
    entry_date: pd.Timestamp       # same as trigger_date in v1 (market-on-open)

    # Geometry at trigger  [filled at trigger]
    entry_price: float             # close on trigger bar (proxy for entry)
    pivot: float                   # base pivot (max_high of base window)
    trigger_pivot: float           # pivot used for trigger — prior bar's pivot
    atr_at_trigger: float          # ATR(14) on trigger bar

    # Derived levels [filled at trigger]
    target_price: float            # trigger_pivot + 2.0 × ATR
    stop_price: float              # trigger_pivot - 1.0 × ATR

    # Scores at entry [filled at trigger, NaN when models not yet fitted]
    setup_score: float = float("nan")
    trade_score: float = float("nan")
    failure_risk: float = float("nan")
    composite_score: float = float("nan")

    # Exit [filled at exit]
    exit_date: pd.Timestamp | None = None
    exit_price: float | None = None
    exit_reason: str = REASON_OPEN   # CONFIRMED / FAILED / TIMEOUT / STILL_OPEN

    # Outcome [filled at exit]
    pnl_atr: float | None = None    # (exit_price - entry_price) / atr_at_trigger
    pnl_pct: float | None = None    # (exit_price - entry_price) / entry_price

    # Column order preserved across parquet round-trips
    _COLUMNS: ClassVar[list[str]] = [
        "symbol", "trigger_date", "entry_date", "entry_price",
        "pivot", "trigger_pivot", "atr_at_trigger",
        "target_price", "stop_price",
        "setup_score", "trade_score", "failure_risk", "composite_score",
        "exit_date", "exit_price", "exit_reason",
        "pnl_atr", "pnl_pct",
    ]

    def to_series(self) -> pd.Series:
        d = asdict(self)
        d.pop("_COLUMNS", None)
        return pd.Series(d)[self._COLUMNS]

    @property
    def is_open(self) -> bool:
        return self.exit_date is None

    @property
    def is_winner(self) -> bool | None:
        if self.pnl_pct is None:
            return None
        return self.pnl_pct > 0


def _empty_journal() -> pd.DataFrame:
    return pd.DataFrame(columns=TradeRecord._COLUMNS)


def load_journal(path: Path | None = None) -> pd.DataFrame:
    """Load the journal parquet; returns an empty DataFrame if file does not exist."""
    path = Path(path or _DEFAULT_JOURNAL)
    if not path.exists():
        return _empty_journal()
    df = pd.read_parquet(path)
    for col in ("trigger_date", "entry_date", "exit_date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=False)
    return df


def save_journal(df: pd.DataFrame, path: Path | None = None) -> None:
    """Overwrite journal parquet (full rewrite)."""
    path = Path(path or _DEFAULT_JOURNAL)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def upsert_record(record: TradeRecord, path: Path | None = None) -> None:
    """Add a new record or update an existing one (matched by symbol + trigger_date).

    This performs a full read → merge → write cycle; suitable for low-volume
    daily updates (one upsert per new trigger or exit per symbol).
    """
    path = Path(path or _DEFAULT_JOURNAL)
    df = load_journal(path)
    row = record.to_series().to_frame().T.reset_index(drop=True)

    if df.empty:
        save_journal(row, path)
        return

    key = (df["symbol"] == record.symbol) & (
        pd.to_datetime(df["trigger_date"]) == pd.to_datetime(record.trigger_date)
    )
    if key.any():
        df = df[~key]

    df = pd.concat([df, row], ignore_index=True)
    save_journal(df, path)


def journal_summary(df: pd.DataFrame) -> dict:
    """Aggregate performance statistics from the journal.

    Returns
    -------
    dict with keys:
        n_total, n_closed, n_open, n_winners, n_losers,
        win_rate, avg_pnl_pct, avg_pnl_atr, avg_winner_atr, avg_loser_atr,
        profit_factor, expectancy_atr
    """
    if df.empty:
        return {"n_total": 0, "n_closed": 0, "n_open": 0}

    closed = df[df["exit_reason"] != REASON_OPEN].copy()
    n_total = len(df)
    n_closed = len(closed)
    n_open = n_total - n_closed

    if n_closed == 0:
        return {"n_total": n_total, "n_closed": 0, "n_open": n_open}

    pnl_atr = pd.to_numeric(closed["pnl_atr"], errors="coerce").dropna()
    pnl_pct = pd.to_numeric(closed["pnl_pct"], errors="coerce").dropna()
    winners = pnl_atr[pnl_atr > 0]
    losers = pnl_atr[pnl_atr <= 0]

    win_rate = len(winners) / n_closed if n_closed > 0 else float("nan")
    gross_profit = winners.sum() if len(winners) > 0 else 0.0
    gross_loss = abs(losers.sum()) if len(losers) > 0 else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("nan")
    expectancy_atr = pnl_atr.mean() if len(pnl_atr) > 0 else float("nan")

    return {
        "n_total": n_total,
        "n_closed": n_closed,
        "n_open": n_open,
        "n_winners": len(winners),
        "n_losers": len(losers),
        "win_rate": round(win_rate, 4) if np.isfinite(win_rate) else float("nan"),
        "avg_pnl_pct": round(float(pnl_pct.mean()), 6) if len(pnl_pct) > 0 else float("nan"),
        "avg_pnl_atr": round(float(expectancy_atr), 4) if np.isfinite(expectancy_atr) else float("nan"),
        "avg_winner_atr": round(float(winners.mean()), 4) if len(winners) > 0 else float("nan"),
        "avg_loser_atr": round(float(losers.mean()), 4) if len(losers) > 0 else float("nan"),
        "profit_factor": round(float(profit_factor), 4) if np.isfinite(profit_factor) else float("nan"),
        "expectancy_atr": round(float(expectancy_atr), 4) if np.isfinite(expectancy_atr) else float("nan"),
        "by_exit_reason": closed["exit_reason"].value_counts().to_dict(),
    }


def auto_update_open_trades(
    states_dir: Path,
    journal_path: Path | None = None,
    *,
    atr_stop_mult: float = 1.0,
    atr_target_mult: float = 2.0,
) -> int:
    """Scan states parquets and close any open journal records that have resolved.

    Checks each open trade's symbol states for CONFIRMED/FAILED transitions
    after the trigger_date. Updates the journal record in place.

    Returns number of records updated.
    """
    journal_path = Path(journal_path or _DEFAULT_JOURNAL)
    df = load_journal(journal_path)
    if df.empty:
        return 0

    open_trades = df[df["exit_reason"] == REASON_OPEN].copy()
    if open_trades.empty:
        return 0

    updated = 0
    for _, row in open_trades.iterrows():
        sym = str(row["symbol"])
        states_path = states_dir / f"{sym}.parquet"
        if not states_path.exists():
            continue

        try:
            states = pd.read_parquet(states_path)
        except Exception:
            continue

        if "state" not in states.columns or states.empty:
            continue

        trigger_date = pd.to_datetime(row["trigger_date"])
        after = states[states.index > trigger_date]
        if after.empty:
            continue

        # Check for CONFIRMED or FAILED after trigger
        confirmed_rows = after[after["state"] == "CONFIRMED"]
        failed_rows = after[after["state"] == "FAILED"]

        record_kwargs = {
            "symbol": sym,
            "trigger_date": trigger_date,
            "entry_date": pd.to_datetime(row["entry_date"]),
            "entry_price": float(row["entry_price"]),
            "pivot": float(row["pivot"]),
            "trigger_pivot": float(row["trigger_pivot"]),
            "atr_at_trigger": float(row["atr_at_trigger"]),
            "target_price": float(row["target_price"]),
            "stop_price": float(row["stop_price"]),
            "setup_score": float(row.get("setup_score", float("nan"))),
            "trade_score": float(row.get("trade_score", float("nan"))),
            "failure_risk": float(row.get("failure_risk", float("nan"))),
            "composite_score": float(row.get("composite_score", float("nan"))),
        }

        if not confirmed_rows.empty:
            exit_date = confirmed_rows.index[0]
            exit_px = float(states.loc[exit_date, "pivot"]) if "pivot" in states.columns else float("nan")
            record = TradeRecord(
                **record_kwargs,
                exit_date=exit_date,
                exit_price=exit_px,
                exit_reason=REASON_CONFIRMED,
                pnl_atr=(exit_px - record_kwargs["entry_price"]) / record_kwargs["atr_at_trigger"]
                if np.isfinite(exit_px) and record_kwargs["atr_at_trigger"] > 0 else None,
                pnl_pct=(exit_px - record_kwargs["entry_price"]) / record_kwargs["entry_price"]
                if np.isfinite(exit_px) and record_kwargs["entry_price"] > 0 else None,
            )
            upsert_record(record, journal_path)
            updated += 1
        elif not failed_rows.empty:
            exit_date = failed_rows.index[0]
            exit_px = float(states.loc[exit_date, "pivot"]) if "pivot" in states.columns else float("nan")
            record = TradeRecord(
                **record_kwargs,
                exit_date=exit_date,
                exit_price=exit_px,
                exit_reason=REASON_FAILED,
                pnl_atr=(exit_px - record_kwargs["entry_price"]) / record_kwargs["atr_at_trigger"]
                if np.isfinite(exit_px) and record_kwargs["atr_at_trigger"] > 0 else None,
                pnl_pct=(exit_px - record_kwargs["entry_price"]) / record_kwargs["entry_price"]
                if np.isfinite(exit_px) and record_kwargs["entry_price"] > 0 else None,
            )
            upsert_record(record, journal_path)
            updated += 1

    log.info("auto_update_open_trades: %d records updated", updated)
    return updated
