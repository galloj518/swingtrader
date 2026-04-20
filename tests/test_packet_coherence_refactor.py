from __future__ import annotations

import math

import pandas as pd

import swingtrader.dashboard.context as context_mod
from swingtrader.dashboard.action import ACTION_EXTENDED
from swingtrader.dashboard.buckets import BUCKET_BREAKOUT, BUCKET_EXTENDED, BUCKET_PULLBACK
from swingtrader.dashboard.packet import build_lightweight_packet
from swingtrader.dashboard.selector import select_packets
from swingtrader.reports.artifacts import _section_packet
from swingtrader.reports.dashboard import render_dashboard


def _row(**overrides) -> pd.Series:
    base = {
        "user_symbol": "TEST",
        "provider_symbol": "TEST",
        "state": "ARMED",
        "pivot": 100.0,
        "atr14": 2.0,
        "close": 99.5,
        "dist_to_pivot_atr": 0.3,
        "days_in_state": 5,
        "composite_score": 0.62,
        "setup_score": 0.58,
        "trade_score": 0.52,
        "failure_risk": 0.20,
        "percentile_rank": 82.0,
        "base_length": 28,
        "atr_compression_pct": 34.0,
        "volume_dryup": 0.60,
        "close_vs_sma50": 0.03,
        "daily_rs_63": 0.09,
        "ytd_dist_atr": 1.3,
        "swing_low_dist_atr": 2.8,
        "regime_spy_trend": 1.0,
        "is_portfolio": False,
        "is_watchlist": True,
        "is_non_equity": False,
        "groups": "tech",
    }
    base.update(overrides)
    return pd.Series(base)


def _raw_history(start: str, end: str) -> pd.DataFrame:
    idx = pd.bdate_range(start, end)
    close = pd.Series(range(len(idx)), index=idx, dtype="float64") * 0.08 + 80.0
    return pd.DataFrame(
        {
            "open": close - 0.4,
            "high": close + 0.8,
            "low": close - 0.9,
            "close": close,
            "volume": 1_000_000,
        },
        index=idx,
    )


def _state_history() -> pd.Series:
    idx = pd.bdate_range("2026-03-03", "2026-04-17")
    values = ["BASE"] * len(idx)
    for i, date in enumerate(idx):
        if date >= pd.Timestamp("2026-03-20"):
            values[i] = "TRIGGERED"
    return pd.Series(values, index=idx)


def test_selector_rejects_incoherent_breakout_packet() -> None:
    pkt = build_lightweight_packet(_row(user_symbol="BAD", provider_symbol="BAD"))
    assert pkt["bucket"] == BUCKET_BREAKOUT

    pkt["setup_key"] = "reclaim_pullback"
    pkt["setup_classification"] = "Pullback reclaim"
    pkt["trade_plan"] = {
        **pkt["trade_plan"],
        "entry_style": "pullback",
        "best_entry_style": "pullback",
        "actionable_now": False,
    }
    pkt["coherence_ok"] = False
    pkt["coherence_issues"] = ["breakout bucket cannot use pullback trade plan"]

    selections = select_packets([pkt])

    assert selections["breakout"] == []
    assert "breakout bucket cannot use pullback trade plan" in pkt["selector_blockers"]
    assert "not_a_true_breakout_packet" in pkt["selector_blockers"]


def test_extended_candidate_is_demoted_out_of_breakout() -> None:
    pkt = build_lightweight_packet(
        _row(
            user_symbol="LATE",
            provider_symbol="LATE",
            state="TRIGGERED",
            days_in_state=2,
            dist_to_pivot_atr=0.4,
            close=108.0,
            close_vs_sma50=0.15,
            ytd_dist_atr=4.6,
            swing_low_dist_atr=5.4,
        )
    )

    assert pkt["bucket"] == BUCKET_EXTENDED
    assert pkt["setup_key"] == "extended_leader"
    assert pkt["action_label"] == ACTION_EXTENDED
    assert "too_extended_for_fresh_entry" in pkt["demotion_reason"]


def test_constructive_pullback_surfaces_when_breakout_pool_is_empty() -> None:
    blocked_breakout = build_lightweight_packet(_row(user_symbol="BAD", provider_symbol="BAD"))
    blocked_breakout["coherence_ok"] = False
    blocked_breakout["coherence_issues"] = ["forced incoherent breakout for test"]

    pullback = build_lightweight_packet(
        _row(
            user_symbol="PULL",
            provider_symbol="PULL",
            dist_to_pivot_atr=2.2,
            close=95.6,
        )
    )
    assert pullback["bucket"] == BUCKET_PULLBACK

    selections = select_packets([blocked_breakout, pullback], n_breakout=3, n_pullback=3)

    assert selections["breakout"] == []
    assert [pkt["symbol"] for pkt in selections["pullback"]] == ["PULL"]
    assert pullback["promotion_reason"] != ""


def test_packet_sections_expose_coherence_intraday_and_reason_fields() -> None:
    pkt = build_lightweight_packet(_row(user_symbol="WHY", provider_symbol="WHY"))
    sections = _section_packet(pkt, "2026-04-19")

    assert sections["identity"]["promotion_reason"] == pkt["promotion_reason"]
    assert sections["coherence"]["coherence_ok"] is True
    assert sections["coherence"]["packet_complete_for_surface"] is True
    assert sections["intraday"]["policy"] == "daily_only"
    assert sections["intraday"]["available"] is False


def test_option_b_intraday_policy_is_explicit_in_packet_and_card() -> None:
    pkt = build_lightweight_packet(_row(user_symbol="DAILY", provider_symbol="DAILY"))
    html = render_dashboard(pd.DataFrame([_row().to_dict()]), [pkt], pd.Timestamp("2026-04-19"))

    assert pkt["intraday_policy"] == "daily_only"
    assert pkt["intraday_available"] is False
    assert pkt["intraday_used_in_qualification"] is False
    assert "not part of v1 qualification" in html
    assert "daily/weekly only" in html


def test_dashboard_explains_when_breakout_bucket_is_blocked_by_coherence() -> None:
    bad_row = _row(user_symbol="BAD", provider_symbol="BAD")
    pull_row = _row(
        user_symbol="PULL",
        provider_symbol="PULL",
        dist_to_pivot_atr=2.2,
        close=95.6,
    )
    blocked_breakout = build_lightweight_packet(bad_row)
    blocked_breakout["coherence_ok"] = False
    blocked_breakout["coherence_issues"] = ["forced incoherent breakout for dashboard test"]
    pullback = build_lightweight_packet(pull_row)

    selections = select_packets([blocked_breakout, pullback], n_breakout=3, n_pullback=3)
    html = render_dashboard(
        pd.DataFrame([bad_row.to_dict(), pull_row.to_dict()]),
        selections["top"],
        pd.Timestamp("2026-04-19"),
        selections=selections,
    )

    assert "No breakout candidates qualify today." in html
    assert "blocked for packet coherence or completeness" in html
    assert "PULL" in html


def test_build_avwap_table_includes_configured_global_and_symbol_anchors(monkeypatch) -> None:
    raw_df = _raw_history("2020-03-02", "2026-04-17")
    close = float(raw_df["close"].iloc[-1])

    monkeypatch.setattr(context_mod, "_load_raw_daily", lambda symbol: raw_df)
    monkeypatch.setattr(context_mod, "_load_state_history", lambda symbol: _state_history())
    context_mod._load_avwap_config.cache_clear()

    table = context_mod.build_avwap_table("NVDA", close, 2.0)
    rows = {row["anchor"]: row for row in table}

    for label in ("War Start", "Ceasefire", "COVID Low", "COVID High", "Earnings Gap May 2025"):
        assert label in rows
        assert rows[label]["supported"] is True

    assert rows["Earnings Anchor"]["supported"] is False
    assert "not available in v1" in rows["Earnings Anchor"]["unavailable_reason"]


def test_build_avwap_table_marks_predated_anchor_unavailable(monkeypatch) -> None:
    raw_df = _raw_history("2025-01-02", "2026-04-17")
    close = float(raw_df["close"].iloc[-1])

    monkeypatch.setattr(context_mod, "_load_raw_daily", lambda symbol: raw_df)
    monkeypatch.setattr(context_mod, "_load_state_history", lambda symbol: _state_history())
    context_mod._load_avwap_config.cache_clear()

    table = context_mod.build_avwap_table("NVDA", close, 2.0)
    rows = {row["anchor"]: row for row in table}

    assert rows["COVID Low"]["supported"] is False
    assert "predates available history" in rows["COVID Low"]["unavailable_reason"]
    assert rows["War Start"]["supported"] is True
    assert not math.isnan(float(rows["War Start"]["avwap"]))
