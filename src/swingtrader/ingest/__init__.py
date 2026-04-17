"""Data ingest layer.

``yfinance_source`` is the single vendor boundary — everything else in the codebase
consumes the parquet artifacts it writes, so swapping providers is a one-file change.
"""
