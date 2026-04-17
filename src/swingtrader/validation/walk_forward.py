"""Walk-forward cross-validation splitter for panel time-series data.

Splits on calendar date so that all symbols respect the same train/test boundary.
This prevents any within-symbol leakage that row-wise shuffles would introduce.

Schema
------
  train:   [split_start, train_end]  inclusive
  embargo: (train_end, test_start)   exclusive on both ends
  test:    [test_start, test_end]    inclusive

The embargo gap prevents label-horizon leakage: if forward labels cover H bars,
embargo_days must be ≥ H to guarantee no future return leaks into training targets.

For daily bars with horizon_bars=20 the default embargo_days=25 is sufficient.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from swingtrader.utils.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class WalkForwardSplit:
    """One walk-forward fold."""

    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    embargo_end: pd.Timestamp   # test starts the day after this
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    @property
    def n_train_days(self) -> int:
        return (self.train_end - self.train_start).days

    @property
    def n_test_days(self) -> int:
        return (self.test_end - self.test_start).days

    def train_mask(self, dates: pd.DatetimeIndex) -> pd.Series:
        """Boolean Series aligned to dates; True for training rows."""
        return pd.Series(
            (dates >= self.train_start) & (dates <= self.train_end),
            index=dates,
        )

    def test_mask(self, dates: pd.DatetimeIndex) -> pd.Series:
        """Boolean Series aligned to dates; True for test rows."""
        return pd.Series(
            (dates >= self.test_start) & (dates <= self.test_end),
            index=dates,
        )

    def embargo_mask(self, dates: pd.DatetimeIndex) -> pd.Series:
        """Rows in the embargo gap (must never appear in train or test)."""
        return pd.Series(
            (dates > self.train_end) & (dates < self.test_start),
            index=dates,
        )


def make_splits(
    dates: pd.DatetimeIndex,
    *,
    train_window_days: int = 1460,
    test_window_days: int = 90,
    embargo_days: int = 25,
    cfg: dict | None = None,
) -> list[WalkForwardSplit]:
    """Generate walk-forward splits from a sorted DatetimeIndex.

    Parameters
    ----------
    dates:
        Sorted DatetimeIndex of the full dataset (union of all symbol dates).
        May contain duplicates if passed a panel index; they are deduplicated
        before computing boundaries.
    train_window_days:
        Calendar days in each training window.
    test_window_days:
        Calendar days in each test window.
    embargo_days:
        Calendar days between training end and test start (embargo gap).
    cfg:
        Optional pre-loaded scoring config dict; overrides keyword defaults
        when the ``validation`` section is present.

    Returns
    -------
    list of WalkForwardSplit, ordered from earliest to latest test window.
    Returns an empty list when there is insufficient history.
    """
    if cfg is not None:
        val = cfg.get("validation", {})
        train_window_days = int(val.get("train_window_days", train_window_days))
        test_window_days = int(val.get("test_window_days", test_window_days))
        embargo_days = int(val.get("embargo_days", embargo_days))

    unique_dates = dates.unique().sort_values()
    if len(unique_dates) == 0:
        return []

    global_start = unique_dates[0]
    global_end = unique_dates[-1]

    total_needed = train_window_days + embargo_days + test_window_days
    if (global_end - global_start).days < total_needed:
        log.warning(
            "Insufficient history (%d days) for even one walk-forward fold "
            "(need %d). Returning empty split list.",
            (global_end - global_start).days,
            total_needed,
        )
        return []

    splits: list[WalkForwardSplit] = []
    fold = 0

    # First test window ends at global_end; step backward.
    # Generate all non-overlapping test windows.
    test_end = global_end
    while True:
        test_start = test_end - pd.Timedelta(days=test_window_days - 1)
        embargo_end = test_start - pd.Timedelta(days=1)
        train_end = embargo_end - pd.Timedelta(days=embargo_days)
        train_start = train_end - pd.Timedelta(days=train_window_days - 1)

        if train_start < global_start:
            break

        splits.append(
            WalkForwardSplit(
                fold=fold,
                train_start=train_start,
                train_end=train_end,
                embargo_end=embargo_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
        fold += 1
        test_end = test_start - pd.Timedelta(days=1)

    # Return chronologically (earliest fold first)
    splits = list(reversed(splits))
    for i, sp in enumerate(splits):
        object.__setattr__(sp, "fold", i)

    log.info(
        "Generated %d walk-forward folds | train=%dd embargo=%dd test=%dd",
        len(splits),
        train_window_days,
        embargo_days,
        test_window_days,
    )
    return splits


def split_summary(splits: list[WalkForwardSplit]) -> pd.DataFrame:
    """Human-readable summary table of all folds."""
    rows = [
        {
            "fold": sp.fold,
            "train_start": sp.train_start.date(),
            "train_end": sp.train_end.date(),
            "embargo_end": sp.embargo_end.date(),
            "test_start": sp.test_start.date(),
            "test_end": sp.test_end.date(),
            "n_train_days": sp.n_train_days,
            "n_test_days": sp.n_test_days,
        }
        for sp in splits
    ]
    return pd.DataFrame(rows)
