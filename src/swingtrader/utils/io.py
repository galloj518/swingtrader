"""Parquet I/O helpers for the feature and data stores."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_parquet(path: Path | str) -> pd.DataFrame:
    """Read a parquet file into a DataFrame. Raises ``FileNotFoundError`` if missing."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    return pd.read_parquet(p)


def write_parquet(df: pd.DataFrame, path: Path | str, compression: str = "snappy") -> Path:
    """Write a DataFrame to parquet, creating parent directories as needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, compression=compression, index=True)
    return p


def append_parquet(
    df: pd.DataFrame,
    path: Path | str,
    key_col: str | None = None,
    compression: str = "snappy",
) -> Path:
    """Append to an existing parquet file, deduping on index (or ``key_col``) keeping last.

    If the file does not exist, the given DataFrame is written as-is.
    """
    p = Path(path)
    if p.exists():
        existing = read_parquet(p)
        combined = pd.concat([existing, df])
        if key_col is not None:
            combined = combined.drop_duplicates(subset=[key_col], keep="last")
        else:
            combined = combined[~combined.index.duplicated(keep="last")]
        combined = combined.sort_index()
    else:
        combined = df
    return write_parquet(combined, p, compression=compression)
