"""Data-quality checks for ingested OHLCV.

Runs cheap structural checks: column presence, missing-bar ratio, longest unexplained
gap, data staleness. Thresholds come from ``config/data_sources.yaml``.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from swingtrader.utils.config import Config, load_config


@dataclass
class QualityReport:
    """Structured result of :func:`check_daily`."""

    symbol: str
    n_rows: int
    missing_cols: list[str]
    missing_pct: float
    max_gap_bars: int
    stale_days: int
    ok: bool
    notes: list[str] = field(default_factory=list)


def _max_run(mask: Iterable[bool]) -> int:
    best = cur = 0
    for v in mask:
        if v:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def check_daily(symbol: str, df: pd.DataFrame, cfg: Config | None = None) -> QualityReport:
    """Run basic quality checks on a daily OHLCV DataFrame indexed by date."""
    cfg = cfg or load_config("data_sources")
    q = cfg["quality"]
    required: list[str] = list(q["required_columns"])
    notes: list[str] = []

    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        notes.append(f"missing columns: {missing_cols}")

    if df.empty:
        notes.append("empty")
        return QualityReport(
            symbol=symbol,
            n_rows=0,
            missing_cols=missing_cols,
            missing_pct=1.0,
            max_gap_bars=0,
            stale_days=10**6,
            ok=False,
            notes=notes,
        )

    idx = pd.DatetimeIndex(df.index).normalize().unique()
    bd_range = pd.bdate_range(start=idx.min(), end=idx.max())
    present = idx.intersection(bd_range)
    missing_pct = 1.0 - (len(present) / len(bd_range)) if len(bd_range) else 0.0

    missing_mask = [d not in present for d in bd_range]
    max_gap = _max_run(missing_mask)

    last_bar = pd.Timestamp(df.index.max()).normalize()
    stale_days = int((pd.Timestamp(date.today()) - last_bar).days)

    ok = (
        not missing_cols
        and missing_pct <= float(q["max_missing_days_pct"])
        and max_gap <= int(q["max_unexplained_gap_bars"])
        and stale_days <= int(q["stale_data_max_age_days"])
    )
    if missing_pct > float(q["max_missing_days_pct"]):
        notes.append(f"missing_pct={missing_pct:.3f} exceeds threshold")
    if max_gap > int(q["max_unexplained_gap_bars"]):
        notes.append(f"max_gap_bars={max_gap} exceeds threshold")
    if stale_days > int(q["stale_data_max_age_days"]):
        notes.append(f"stale_days={stale_days} exceeds threshold")

    return QualityReport(
        symbol=symbol,
        n_rows=len(df),
        missing_cols=missing_cols,
        missing_pct=float(missing_pct),
        max_gap_bars=int(max_gap),
        stale_days=stale_days,
        ok=bool(ok),
        notes=notes,
    )
