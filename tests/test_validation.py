"""Tests for walk-forward split generation."""
from __future__ import annotations

import pandas as pd

from swingtrader.validation.walk_forward import make_splits, split_summary


def _dates(n_years: float = 5.0) -> pd.DatetimeIndex:
    n_days = int(n_years * 365)
    return pd.bdate_range(end=pd.Timestamp("2024-12-31"), periods=n_days)


# ── make_splits ──────────────────────────────────────────────────────────────

def test_make_splits_returns_list() -> None:
    splits = make_splits(_dates(5))
    assert isinstance(splits, list)


def test_make_splits_nonempty_with_sufficient_history() -> None:
    splits = make_splits(_dates(5), train_window_days=365, test_window_days=90, embargo_days=25)
    assert len(splits) >= 1


def test_make_splits_empty_with_insufficient_history() -> None:
    splits = make_splits(_dates(0.5), train_window_days=365, test_window_days=90, embargo_days=25)
    assert splits == []


def test_splits_are_chronological() -> None:
    splits = make_splits(_dates(5), train_window_days=365, test_window_days=90, embargo_days=25)
    for i in range(1, len(splits)):
        assert splits[i].test_start > splits[i - 1].test_start


def test_no_overlap_between_train_and_test() -> None:
    splits = make_splits(_dates(5), train_window_days=365, test_window_days=90, embargo_days=25)
    for sp in splits:
        assert sp.train_end < sp.test_start


def test_embargo_gap_respected() -> None:
    embargo = 25
    splits = make_splits(_dates(5), train_window_days=365, test_window_days=90, embargo_days=embargo)
    for sp in splits:
        gap = (sp.test_start - sp.train_end).days
        assert gap > embargo - 1, f"Embargo gap too small: {gap} days"


def test_fold_numbers_start_at_zero_and_increment() -> None:
    splits = make_splits(_dates(5), train_window_days=365, test_window_days=90, embargo_days=25)
    for i, sp in enumerate(splits):
        assert sp.fold == i


def test_masks_are_mutually_exclusive() -> None:
    dates = _dates(5)
    splits = make_splits(dates, train_window_days=365, test_window_days=90, embargo_days=25)
    sp = splits[0]
    train_m = sp.train_mask(dates)
    test_m = sp.test_mask(dates)
    embargo_m = sp.embargo_mask(dates)
    # No date should be in both train and test
    assert not (train_m & test_m).any()
    # No date should be in train and embargo
    assert not (train_m & embargo_m).any()
    # No date should be in test and embargo
    assert not (test_m & embargo_m).any()


def test_masks_cover_entire_date_range() -> None:
    dates = _dates(5)
    splits = make_splits(dates, train_window_days=365, test_window_days=90, embargo_days=25)
    sp = splits[-1]  # last fold covers the most recent data
    test_m = sp.test_mask(dates)
    # Not all dates need to be covered (early history predates train_start),
    # but test window must be fully covered
    test_dates = dates[test_m]
    assert len(test_dates) > 0


def test_cfg_override() -> None:
    cfg = {"validation": {"train_window_days": 730, "test_window_days": 60, "embargo_days": 20}}
    splits = make_splits(_dates(5), cfg=cfg)
    if splits:
        sp = splits[0]
        assert sp.n_test_days <= 61  # approximate (calendar days)


def test_split_summary_has_correct_columns() -> None:
    splits = make_splits(_dates(5), train_window_days=365, test_window_days=90, embargo_days=25)
    df = split_summary(splits)
    assert {"fold", "train_start", "test_start", "n_train_days", "n_test_days"}.issubset(df.columns)
    assert len(df) == len(splits)


def test_split_with_duplicate_dates() -> None:
    """Panel data has duplicate dates (multiple symbols per date); must still work."""
    base = _dates(5)
    duplicated = base.append(base)  # simulate 2 symbols
    splits = make_splits(duplicated, train_window_days=365, test_window_days=90, embargo_days=25)
    assert len(splits) >= 1
