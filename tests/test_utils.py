"""Tests for small utility modules."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from swingtrader.utils.io import append_parquet, read_parquet, write_parquet


def test_write_then_read_parquet(tmp_path: Path) -> None:
    df = pd.DataFrame({"a": [1, 2, 3]}, index=pd.date_range("2025-01-01", periods=3))
    path = tmp_path / "sub" / "x.parquet"
    written = write_parquet(df, path)
    assert written == path
    round_trip = read_parquet(path)
    # parquet round-trip drops the DatetimeIndex freq attribute; values are identical
    pd.testing.assert_frame_equal(round_trip, df, check_freq=False)


def test_append_parquet_dedupes_on_index(tmp_path: Path) -> None:
    path = tmp_path / "y.parquet"
    idx = pd.date_range("2025-01-01", periods=3)
    df1 = pd.DataFrame({"a": [1, 2, 3]}, index=idx)
    write_parquet(df1, path)

    # Overlap: last row repeats date at idx[-1] with new value
    df2 = pd.DataFrame({"a": [99, 4]}, index=[idx[-1], idx[-1] + pd.Timedelta(days=1)])
    append_parquet(df2, path)

    out = read_parquet(path)
    assert len(out) == 4
    assert out.loc[idx[-1], "a"] == 99   # last-write-wins on duplicate index
