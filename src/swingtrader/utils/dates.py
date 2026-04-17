"""Trading-calendar helpers.

v1 uses pandas business days as a good-enough approximation of the US equity calendar.
Holidays are not explicitly modeled; the quality checker tolerates small gaps so
genuine holiday closures do not trigger false alerts.
"""
from __future__ import annotations

from datetime import UTC, date, datetime

import pandas as pd

US_TZ = "America/New_York"


def today_utc() -> datetime:
    """Return the current UTC time as an aware datetime."""
    return datetime.now(UTC)


def to_trading_day(ts: pd.Timestamp | datetime) -> pd.Timestamp:
    """Normalize a timestamp to the most recent US business day (naive, midnight)."""
    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        t = t.tz_convert(US_TZ).tz_localize(None)
    return pd.bdate_range(end=t.normalize(), periods=1)[0]


def trading_days_between(start: date | str, end: date | str) -> pd.DatetimeIndex:
    """Inclusive business-day range between two dates."""
    return pd.bdate_range(start=start, end=end)


def last_completed_trading_day(now: datetime | None = None) -> pd.Timestamp:
    """Last business day strictly before ``now`` (used to gate EOD runs)."""
    now = now or today_utc()
    ts = pd.Timestamp(now).tz_convert(US_TZ).tz_localize(None) if now.tzinfo else pd.Timestamp(now)
    return pd.bdate_range(end=ts.normalize() - pd.Timedelta(days=1), periods=1)[0]
