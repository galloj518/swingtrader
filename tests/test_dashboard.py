"""Tests for the trader-facing dashboard layer.

Covers:
  - freshness classification
  - levels computation
  - action label assignment
  - narrative generation
  - top setup selection
  - packet assembly
  - dashboard HTML rendering
  - charts (output-path logic only; no matplotlib rendering in CI)
"""
from __future__ import annotations

import math

import pandas as pd
import pytest

from swingtrader.dashboard.action import (
    ACTION_AVOID,
    ACTION_BREAKOUT,
    ACTION_EXTENDED,
    ACTION_NOW,
    ACTION_PULLBACK,
    add_action_column,
    assign_action,
)
from swingtrader.dashboard.freshness import (
    EXT_ATR,
    add_freshness_columns,
    classify_row,
)
from swingtrader.dashboard.levels import compute_levels
from swingtrader.dashboard.narrative import build_narrative
from swingtrader.dashboard.packet import build_packet, build_packets
from swingtrader.dashboard.selector import select_top_setups
from swingtrader.reports.dashboard import render_dashboard, write_dashboard

# ── Fixtures ──────────────────────────────────────────────────────────────────

def _row(state="ARMED", pivot=100.0, atr14=2.0, close=99.0,
         dist_atr=0.5, days=5, score=0.55, failure=0.25,
         is_non_equity=False, base_len=25):
    return pd.Series({
        "user_symbol": "TEST",
        "provider_symbol": "TEST",
        "state": state,
        "pivot": pivot,
        "atr14": atr14,
        "close": close,
        "dist_to_pivot_atr": dist_atr,
        "days_in_state": days,
        "composite_score": score,
        "failure_risk": failure,
        "setup_score": score * 0.9,
        "trade_score": score * 0.8,
        "percentile_rank": 75.0,
        "base_length": base_len,
        "atr_compression_pct": 35.0,
        "volume_dryup": 0.6,
        "close_vs_sma50": 0.02,
        "daily_rs_63": 0.08,
        "ytd_dist_atr": 1.0,
        "swing_low_dist_atr": 2.5,
        "regime_spy_trend": 1.0,
        "is_portfolio": False,
        "is_watchlist": True,
        "is_non_equity": is_non_equity,
        "groups": "watchlist",
        "action_label": "",
        "is_fresh": True,
        "is_extended": False,
    })


def _snapshot(n=10):
    """Build a small snapshot DataFrame with mixed states."""
    states = ["ARMED", "TRIGGERED", "TRIGGERED", "BASE", "BASE",
              "CONFIRMED", "FAILED", "LATE", "NONE", "ARMED"]
    rows = []
    for i, st in enumerate(states[:n]):
        dist = 0.3 + i * 0.4
        days = 3 + i * 2
        score = max(0.1, 0.7 - i * 0.05)
        rows.append({
            "user_symbol": f"SYM{i:02d}",
            "provider_symbol": f"SYM{i:02d}",
            "state": st,
            "pivot": 100.0 + i,
            "atr14": 2.0,
            "close": 100.5 + i,
            "dist_to_pivot_atr": dist,
            "days_in_state": days,
            "composite_score": score,
            "failure_risk": 0.2 + i * 0.03,
            "setup_score": score * 0.85,
            "trade_score": score * 0.9,
            "percentile_rank": 80.0 - i * 5,
            "base_length": 30 + i,
            "atr_compression_pct": 40.0,
            "volume_dryup": 0.5,
            "close_vs_sma50": 0.01,
            "daily_rs_63": 0.05,
            "ytd_dist_atr": 1.0,
            "swing_low_dist_atr": 2.0,
            "regime_spy_trend": 1.0,
            "is_portfolio": i == 0,
            "is_watchlist": True,
            "is_non_equity": False,
            "groups": "tech",
            "skip_reason": "",
        })
    return pd.DataFrame(rows)


# ── Freshness tests ───────────────────────────────────────────────────────────

class TestFreshness:
    def test_fresh_triggered(self):
        r = _row(state="TRIGGERED", days=2, dist_atr=0.5)
        result = classify_row(r)
        assert result["is_fresh"] is True
        assert result["is_actionable"] is True
        assert result["freshness_label"] == "fresh"

    def test_stale_triggered(self):
        r = _row(state="TRIGGERED", days=20, dist_atr=0.5)
        result = classify_row(r)
        assert result["is_fresh"] is False
        assert result["freshness_label"] == "stale"

    def test_extended(self):
        r = _row(state="ARMED", dist_atr=EXT_ATR + 0.5)
        result = classify_row(r)
        assert result["is_extended"] is True
        assert result["is_fresh"] is False

    def test_stale_confirmed(self):
        r = _row(state="CONFIRMED", days=30, dist_atr=2.5)
        result = classify_row(r)
        assert result["is_stale_confirmed"] is True

    def test_late_always_extended(self):
        r = _row(state="LATE", dist_atr=1.0)
        result = classify_row(r)
        assert result["is_extended"] is True

    def test_non_scored_state_not_actionable(self):
        r = _row(state="FAILED")
        result = classify_row(r)
        assert result["is_actionable"] is False

    def test_add_freshness_columns_preserves_shape(self):
        df = _snapshot()
        out = add_freshness_columns(df)
        assert len(out) == len(df)
        assert "is_fresh" in out.columns
        assert "is_actionable" in out.columns
        assert "freshness_label" in out.columns

    def test_add_freshness_empty_df(self):
        out = add_freshness_columns(pd.DataFrame())
        assert out.empty


# ── Levels tests ──────────────────────────────────────────────────────────────

class TestLevels:
    def test_basic_armed_levels(self):
        r = _row(state="ARMED", pivot=100.0, atr14=2.0)
        lvl = compute_levels(r)
        assert math.isfinite(lvl.stop)
        assert lvl.stop == pytest.approx(100.0 - 1.0 * 2.0)   # pivot - STOP_ATR * atr
        assert lvl.t1 == pytest.approx(100.0 + 2.0 * 2.0)      # pivot + CONF_ATR * atr
        assert lvl.entry_lo == pytest.approx(100.0)             # = pivot
        assert lvl.t2 > lvl.t1 > lvl.entry_hi > lvl.entry_lo > lvl.stop

    def test_triggered_support_is_pivot(self):
        r = _row(state="TRIGGERED", pivot=100.0, atr14=2.0)
        lvl = compute_levels(r)
        # In-trade: S1 == pivot
        assert lvl.s1 == pytest.approx(100.0)

    def test_r1_equals_pivot_for_pretrigger(self):
        r = _row(state="BASE", pivot=100.0, atr14=2.0)
        lvl = compute_levels(r)
        assert lvl.r1 == pytest.approx(100.0)

    def test_nan_pivot_returns_nan_levels(self):
        r = _row(pivot=float("nan"))
        lvl = compute_levels(r)
        assert math.isnan(lvl.stop)
        assert math.isnan(lvl.t1)

    def test_risk_reward_positive(self):
        r = _row(state="ARMED", pivot=100.0, atr14=2.0, close=99.0)
        lvl = compute_levels(r)
        assert math.isfinite(lvl.risk_reward_t1)
        assert lvl.risk_reward_t1 > 0

    def test_to_dict_all_floats(self):
        r = _row(state="ARMED", pivot=100.0, atr14=2.0)
        lvl = compute_levels(r)
        d = lvl.to_dict()
        assert isinstance(d, dict)
        assert "pivot" in d


# ── Action label tests ────────────────────────────────────────────────────────

class TestActionLabel:
    def test_triggered_fresh_is_now(self):
        r = _row(state="TRIGGERED", days=2, dist_atr=0.4, score=0.5, failure=0.2)
        r["is_extended"] = False
        assert assign_action(r) == ACTION_NOW

    def test_armed_near_pivot_is_breakout(self):
        r = _row(state="ARMED", dist_atr=1.0, score=0.5)
        r["is_extended"] = False
        assert assign_action(r) == ACTION_BREAKOUT

    def test_armed_far_from_pivot_is_pullback(self):
        r = _row(state="ARMED", dist_atr=2.5, score=0.5)
        r["is_extended"] = False
        assert assign_action(r) == ACTION_PULLBACK

    def test_extended_is_extended_wait(self):
        r = _row(state="ARMED", dist_atr=EXT_ATR + 1.0)
        r["is_extended"] = True
        assert assign_action(r) == ACTION_EXTENDED

    def test_late_is_extended(self):
        r = _row(state="LATE")
        assert assign_action(r) == ACTION_EXTENDED

    def test_low_score_is_avoid(self):
        r = _row(state="ARMED", score=0.05)
        assert assign_action(r) == ACTION_AVOID

    def test_failed_is_avoid(self):
        r = _row(state="FAILED")
        assert assign_action(r) == ACTION_AVOID

    def test_none_state_is_avoid(self):
        r = _row(state="NONE")
        assert assign_action(r) == ACTION_AVOID

    def test_triggered_old_is_pullback(self):
        r = _row(state="TRIGGERED", days=20, dist_atr=0.4, score=0.5)
        r["is_extended"] = False
        assert assign_action(r) == ACTION_PULLBACK

    def test_add_action_column_adds_column(self):
        df = _snapshot()
        df = add_freshness_columns(df)
        out = add_action_column(df)
        assert "action_label" in out.columns
        assert len(out) == len(df)

    def test_all_labels_are_valid_strings(self):
        df = _snapshot()
        df = add_freshness_columns(df)
        out = add_action_column(df)
        valid = {ACTION_NOW, ACTION_BREAKOUT, ACTION_PULLBACK, ACTION_EXTENDED, ACTION_AVOID}
        assert set(out["action_label"]).issubset(valid)


# ── Narrative tests ───────────────────────────────────────────────────────────

class TestNarrative:
    def test_keys_present(self):
        r = _row(state="ARMED", pivot=100.0, atr14=2.0)
        lvl = compute_levels(r)
        result = build_narrative(r, lvl, ACTION_BREAKOUT)
        for key in ("setup", "why", "entry", "risk", "targets", "verdict", "ma_context"):
            assert key in result
            assert isinstance(result[key], str)

    def test_no_exception_with_nan_inputs(self):
        r = _row(pivot=float("nan"), atr14=float("nan"), close=float("nan"))
        r["composite_score"] = float("nan")
        r["failure_risk"] = float("nan")
        lvl = compute_levels(r)
        result = build_narrative(r, lvl, ACTION_AVOID)
        assert isinstance(result["verdict"], str)

    def test_triggered_entry_mentions_pullback(self):
        r = _row(state="TRIGGERED", pivot=100.0, atr14=2.0, close=101.0)
        lvl = compute_levels(r)
        result = build_narrative(r, lvl, ACTION_NOW)
        assert "pullback" in result["entry"].lower() or "entry" in result["entry"].lower()

    def test_verdict_matches_action(self):
        r = _row(state="ARMED", pivot=100.0, atr14=2.0)
        lvl = compute_levels(r)
        result = build_narrative(r, lvl, ACTION_AVOID)
        assert "avoid" in result["verdict"].lower() or "not" in result["verdict"].lower()


# ── Selector tests ────────────────────────────────────────────────────────────

class TestSelector:
    def test_returns_at_most_top_n(self):
        df = _snapshot(10)
        df = add_freshness_columns(df)
        df = add_action_column(df)
        top = select_top_setups(df)
        assert len(top) <= 7

    def test_no_avoid_in_top(self):
        df = _snapshot(10)
        df = add_freshness_columns(df)
        df = add_action_column(df)
        top = select_top_setups(df)
        if not top.empty:
            assert ACTION_AVOID not in top["action_label"].values

    def test_empty_df_returns_empty(self):
        out = select_top_setups(pd.DataFrame())
        assert out.empty

    def test_all_avoid_returns_empty(self):
        df = _snapshot(5)
        df = add_freshness_columns(df)
        df["action_label"] = ACTION_AVOID
        out = select_top_setups(df)
        assert out.empty

    def test_diversity_cap(self):
        """No more than MAX_PER_GROUP (3) from same group."""
        from swingtrader.dashboard.selector import MAX_PER_GROUP
        df = _snapshot(10)
        df["groups"] = "same_group"          # all same group
        df = add_freshness_columns(df)
        df = add_action_column(df)
        top = select_top_setups(df)
        if not top.empty:
            assert len(top) <= MAX_PER_GROUP


# ── Packet tests ──────────────────────────────────────────────────────────────

class TestPacket:
    def test_packet_has_required_keys(self):
        r = _row(state="ARMED", pivot=100.0, atr14=2.0)
        r["action_label"] = ACTION_BREAKOUT
        r["is_fresh"] = True
        r["is_extended"] = False
        pkt = build_packet(r)
        for key in (
            "symbol", "state", "action_label", "pivot", "stop", "t1", "t2", "t3",
            "entry_lo", "entry_hi", "narrative", "risk_reward_t1",
        ):
            assert key in pkt, f"Missing key: {key}"

    def test_narrative_is_dict(self):
        r = _row(state="ARMED")
        r["action_label"] = ACTION_BREAKOUT
        pkt = build_packet(r)
        assert isinstance(pkt["narrative"], dict)
        assert "verdict" in pkt["narrative"]

    def test_build_packets_list(self):
        df = _snapshot(5)
        df = add_freshness_columns(df)
        df = add_action_column(df)
        top = select_top_setups(df)
        pkts = build_packets(top)
        assert isinstance(pkts, list)
        assert len(pkts) == len(top)

    def test_chart_paths_initialise_as_none(self):
        r = _row(state="ARMED")
        r["action_label"] = ACTION_BREAKOUT
        pkt = build_packet(r)
        # Charts are not generated in unit tests — paths are None
        assert pkt["chart_daily"] is None
        assert pkt["chart_weekly"] is None


# ── Dashboard HTML rendering tests ───────────────────────────────────────────

class TestDashboardRender:
    def test_renders_without_error(self):
        df = _snapshot(5)
        df = add_freshness_columns(df)
        df = add_action_column(df)
        top = select_top_setups(df)
        pkts = build_packets(top)
        as_of = pd.Timestamp("2026-01-15")
        html = render_dashboard(df, pkts, as_of)
        assert "<!DOCTYPE html>" in html
        assert "2026-01-15" in html

    def test_empty_snapshot_renders_gracefully(self):
        html = render_dashboard(pd.DataFrame(), [], pd.Timestamp("2026-01-15"))
        assert "<!DOCTYPE html>" in html

    def test_top_cards_in_output(self):
        df = _snapshot(5)
        df = add_freshness_columns(df)
        df = add_action_column(df)
        top = select_top_setups(df)
        pkts = build_packets(top)
        as_of = pd.Timestamp("2026-01-15")
        html = render_dashboard(df, pkts, as_of)
        # At least one symbol should appear if there are non-avoid setups
        if pkts:
            assert pkts[0]["symbol"] in html

    def test_write_dashboard_creates_file(self, tmp_path):
        df = _snapshot(3)
        df = add_freshness_columns(df)
        df = add_action_column(df)
        as_of = pd.Timestamp("2026-01-15")
        top = select_top_setups(df)
        pkts = build_packets(top)
        path = write_dashboard(df, pkts, as_of, tmp_path)
        assert path.exists()
        assert path.name == "dashboard.html"
        content = path.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content

    def test_state_tables_present_in_html(self):
        df = _snapshot(10)
        df = add_freshness_columns(df)
        df = add_action_column(df)
        top = select_top_setups(df)
        pkts = build_packets(top)
        html = render_dashboard(df, pkts, pd.Timestamp("2026-01-15"))
        # ARMED and BASE are in the snapshot — their collapsible sections should appear
        assert "ARMED" in html
        assert "BASE" in html

    def test_portfolio_chips_shown(self):
        df = _snapshot(5)
        df.loc[0, "is_portfolio"] = True
        df = add_freshness_columns(df)
        df = add_action_column(df)
        top = select_top_setups(df)
        pkts = build_packets(top)
        html = render_dashboard(df, pkts, pd.Timestamp("2026-01-15"))
        assert "Portfolio Holdings" in html


# ── Charts path logic test (no matplotlib rendering) ─────────────────────────

class TestChartsPathLogic:
    def test_generate_charts_for_packet_no_data(self, tmp_path):
        """When raw parquet files don't exist, chart paths remain None."""
        from swingtrader.dashboard.charts import generate_charts_for_packet

        pkt = {
            "symbol": "NONEXISTENT",
            "provider_symbol": "NONEXISTENT",
            "pivot": "100.00",
            "entry_lo": "100.00",
            "entry_hi": "100.20",
            "stop": "98.00",
            "t1": "104.00",
            "t2": "107.00",
            "chart_daily": None,
            "chart_weekly": None,
            "chart_intraday": None,
        }
        result = generate_charts_for_packet(pkt, tmp_path)
        assert result["chart_daily"] is None
        assert result["chart_weekly"] is None
        assert result["chart_intraday"] is None

    def test_generate_charts_handles_missing_symbol(self, tmp_path):
        """Empty symbol should not crash."""
        from swingtrader.dashboard.charts import generate_charts_for_packet
        pkt = {"symbol": "", "provider_symbol": ""}
        result = generate_charts_for_packet(pkt, tmp_path)
        assert result is not None
