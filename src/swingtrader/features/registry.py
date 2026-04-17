"""Feature registry.

Every feature is registered with (name, timeframe, lookback_bars, allows_realtime).
The registry is the single source of truth for what features exist and their metadata.

Registration happens at import time when feature modules are loaded.
Call :func:`load_all_feature_modules` once before computing features to ensure all
modules have been imported and their @register decorators executed.

Leakage contract (enforced in tests/test_labels_no_leakage.py):
  A feature registered with lookback_bars=L must produce the same value at bar t
  whether computed on history[:t] or history[:t+K] for any K > 0.
"""
from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

from swingtrader.utils.logging import get_logger

log = get_logger(__name__)

Timeframe = Literal["daily", "weekly", "intraday"]

_FEATURE_MODULES = [
    "swingtrader.features.daily",
    "swingtrader.features.weekly",
    "swingtrader.features.relstrength",
    "swingtrader.features.regime",
    "swingtrader.features.pivot_features",
    "swingtrader.features.intraday",
]


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    timeframe: Timeframe
    lookback_bars: int
    allows_realtime: bool
    fn: Callable[..., pd.Series]


REGISTRY: dict[str, FeatureSpec] = {}


def register(
    name: str,
    *,
    timeframe: Timeframe,
    lookback_bars: int,
    allows_realtime: bool = True,
) -> Callable[[Callable], Callable]:
    """Decorator that registers a feature function in the global REGISTRY."""

    def decorator(fn: Callable) -> Callable:
        if name in REGISTRY:
            log.warning("Feature %r already registered; overwriting", name)
        REGISTRY[name] = FeatureSpec(
            name=name,
            timeframe=timeframe,
            lookback_bars=lookback_bars,
            allows_realtime=allows_realtime,
            fn=fn,
        )
        return fn

    return decorator


def load_all_feature_modules() -> None:
    """Import all feature modules so their @register decorators execute."""
    for mod in _FEATURE_MODULES:
        try:
            importlib.import_module(mod)
        except ImportError as e:
            log.warning("Could not load feature module %s: %s", mod, e)


def compute_features(
    df: pd.DataFrame,
    timeframe: Timeframe,
    extra_kwargs: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Compute all registered features for the given timeframe.

    Returns a wide DataFrame indexed like df. Features that fail are logged and
    left as NaN columns rather than aborting the run.

    ``extra_kwargs`` is forwarded to each feature function (e.g. benchmark_df=spy_df).
    """
    kwargs = extra_kwargs or {}
    results: dict[str, pd.Series] = {}
    for name, spec in REGISTRY.items():
        if spec.timeframe != timeframe:
            continue
        try:
            s = spec.fn(df, **kwargs)
            if not isinstance(s, pd.Series):
                raise TypeError(f"expected pd.Series, got {type(s).__name__}")
            results[name] = s.reindex(df.index)
        except Exception as e:
            log.warning("Feature %r failed: %s", name, e)
            results[name] = pd.Series(dtype=float, index=df.index)
    return pd.DataFrame(results, index=df.index)
