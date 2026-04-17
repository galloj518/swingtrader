"""Smoke tests for YAML config loading."""
from __future__ import annotations

import pytest

from swingtrader.utils.config import load_all_configs, load_config

REQUIRED_CONFIGS = [
    "universe",
    "data_sources",
    "features",
    "avwap_anchors",
    "labels",
    "regimes",
    "scoring",
]


@pytest.mark.parametrize("name", REQUIRED_CONFIGS)
def test_required_config_loads(name: str) -> None:
    cfg = load_config(name)
    assert cfg.data, f"{name}.yaml loaded empty"


def test_all_required_configs_present() -> None:
    configs = load_all_configs()
    for name in REQUIRED_CONFIGS:
        assert name in configs, f"missing config: {name}"


def test_universe_has_benchmarks_and_etfs() -> None:
    cfg = load_config("universe")
    # New schema: active_sources controls enablement; sources contains symbol lists
    active = cfg.get("active_sources", {})
    sources = cfg["sources"]
    assert active.get("benchmarks", True)
    assert "SPY" in sources["benchmarks"]["symbols"]
    assert active.get("sector_etfs", True)
    assert "XLK" in sources["sector_etfs"]["symbols"]


def test_labels_pre_registered_horizons() -> None:
    cfg = load_config("labels")
    fwd = cfg["forward_return"]
    assert fwd["horizons_bars"] == [5, 10, 20]
    assert fwd["normalize_by_atr"] is True
    assert cfg["embargo_bars"] >= fwd["horizons_bars"][-1], (
        "embargo must cover the longest forward horizon to prevent leakage"
    )


def test_scoring_final_rank_not_hand_weighted() -> None:
    """AGENTS.md rule 2 — no hand-picked weights in the scoring config."""
    cfg = load_config("scoring")
    final = cfg["final_rank"]
    # Explicit guard: only 'components_by_state' and 'rank_method' at top-level of final_rank.
    # If someone adds a 'weights' dict with literal numbers, fail the test loudly.
    assert "weights" not in final, (
        "final_rank must not carry hand-picked composite weights; weights are learned OOS"
    )
    assert "components_by_state" in final
    assert "rank_method" in final
