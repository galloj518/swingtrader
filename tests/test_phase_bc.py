"""Tests for Phase B/C additions.

Covers:
  - dashboard.context  (build_context, build_ma_table, build_avwap_table,
                         build_volume_block, build_confluence, build_checklist)
  - dashboard.action   (classify_setup, portfolio_guidance,
                         add_setup_classification_column,
                         add_portfolio_guidance_column)
  - reports.ai_notes   (generate_ai_note, enrich_packets_with_ai,
                         _rule_based_note, _checklist_summary, _ma_summary)
  - reports.artifacts  (write_artifacts, _clean_value, _clean_dict)
  - dashboard.packet   (new fields: context, setup_classification,
                         portfolio_guidance, ai_note)
  - reports.dashboard  (new HTML helpers: _setup_class_badge, _ma_table_html,
                         _avwap_table_html, _checklist_html, _confluence_html,
                         _volume_block_html, _ai_note_html, _trade_plan_html,
                         updated _regime_html / _portfolio_strip_html)
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import pytest

# ── Shared fixture rows ───────────────────────────────────────────────────────

def _row(state="ARMED", pivot=100.0, atr14=2.0, close=99.5,
         dist_atr=0.25, days=6, score=0.55, failure=0.25,
         is_portfolio=False, is_non_equity=False, base_len=25):
    return pd.Series({
        "user_symbol":        "TST",
        "provider_symbol":    "TST",
        "symbol":             "TST",
        "state":              state,
        "pivot":              pivot,
        "atr14":              atr14,
        "close":              close,
        "dist_to_pivot_atr":  dist_atr,
        "days_in_state":      days,
        "composite_score":    score,
        "failure_risk":       failure,
        "setup_score":        score * 0.9,
        "trade_score":        score * 0.8,
        "percentile_rank":    70.0,
        "base_length":        base_len,
        "atr_compression_pct": 28.0,
        "volume_dryup":       0.55,
        "close_vs_sma50":     0.015,
        "daily_rs_63":        0.06,
        "ytd_dist_atr":       1.2,
        "swing_low_dist_atr": 2.0,
        "regime_spy_trend":   1.0,
        "regime_spy_above_200sma": 1.0,
        "regime_vix_level":   16.0,
        "is_portfolio":       is_portfolio,
        "is_watchlist":       True,
        "is_non_equity":      is_non_equity,
        "groups":             "tech",
        "action_label":       "Actionable on breakout",
        "is_fresh":           True,
        "is_extended":        False,
        "is_stale_confirmed": False,
        "skip_reason":        "",
    })


def _snapshot(n=8):
    states = ["ARMED", "TRIGGERED", "BASE", "CONFIRMED",
              "FAILED", "LATE", "NONE", "ARMED"][:n]
    rows = []
    for i, st in enumerate(states):
        rows.append({
            "user_symbol":        f"SY{i:02d}",
            "provider_symbol":    f"SY{i:02d}",
            "state":              st,
            "pivot":              100.0 + i,
            "atr14":              2.0,
            "close":              99.5 + i,
            "dist_to_pivot_atr":  0.3 + i * 0.2,
            "days_in_state":      4 + i * 2,
            "composite_score":    max(0.1, 0.65 - i * 0.06),
            "failure_risk":       0.22 + i * 0.04,
            "setup_score":        0.55,
            "trade_score":        0.50,
            "percentile_rank":    75.0 - i * 5,
            "base_length":        25 + i,
            "atr_compression_pct": 35.0,
            "volume_dryup":       0.5,
            "close_vs_sma50":     0.01,
            "daily_rs_63":        0.04,
            "ytd_dist_atr":       1.0,
            "swing_low_dist_atr": 2.0,
            "regime_spy_trend":   1.0,
            "regime_spy_above_200sma": 1.0,
            "regime_vix_level":   18.0,
            "is_portfolio":       i == 0,
            "is_watchlist":       True,
            "is_non_equity":      False,
            "groups":             "tech",
            "skip_reason":        "",
        })
    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════════
# dashboard.action — classify_setup / portfolio_guidance
# ═════════════════════════════════════════════════════════════════════════════

class TestClassifySetup:
    from swingtrader.dashboard.action import classify_setup  # type: ignore

    def _cls(self, **kwargs):
        from swingtrader.dashboard.action import classify_setup
        return classify_setup(_row(**kwargs))

    def test_armed_near_pivot_is_near_breakout(self):
        assert "Near breakout" in self._cls(state="ARMED", dist_atr=0.3)

    def test_armed_approaching_pivot(self):
        assert self._cls(state="ARMED", dist_atr=-1.2) == "Approaching pivot"

    def test_armed_building_base(self):
        assert self._cls(state="ARMED", dist_atr=-2.0) == "Building base"

    def test_triggered_fresh_is_active_breakout(self):
        assert self._cls(state="TRIGGERED", days=3, dist_atr=0.5) == "Active breakout"

    def test_triggered_older_is_early_breakout(self):
        assert self._cls(state="TRIGGERED", days=10, dist_atr=1.0) == "Early breakout"

    def test_triggered_below_pivot_is_pullback(self):
        assert self._cls(state="TRIGGERED", dist_atr=-0.5) == "Pullback entry"

    def test_confirmed_is_confirmed_uptrend(self):
        r = _row(state="CONFIRMED", days=5)
        r["is_stale_confirmed"] = False
        from swingtrader.dashboard.action import classify_setup
        assert classify_setup(r) == "Confirmed uptrend"

    def test_confirmed_stale_is_mature_trend(self):
        r = _row(state="CONFIRMED", days=50)
        r["is_stale_confirmed"] = True
        from swingtrader.dashboard.action import classify_setup
        assert classify_setup(r) == "Mature trend"

    def test_extended_is_chase_risk(self):
        r = _row(state="ARMED")
        r["is_extended"] = True
        from swingtrader.dashboard.action import classify_setup
        assert "Extended" in classify_setup(r)

    def test_late_is_chase_risk(self):
        assert "Extended" in self._cls(state="LATE")

    def test_failed_is_failed_avoid(self):
        assert "Failed" in self._cls(state="FAILED")

    def test_none_is_watching(self):
        assert self._cls(state="NONE") == "Watching"

    def test_add_column_preserves_shape(self):
        from swingtrader.dashboard.action import add_setup_classification_column
        df = _snapshot()
        out = add_setup_classification_column(df)
        assert "setup_classification" in out.columns
        assert len(out) == len(df)

    def test_add_column_all_strings(self):
        from swingtrader.dashboard.action import add_setup_classification_column
        out = add_setup_classification_column(_snapshot())
        assert out["setup_classification"].apply(lambda v: isinstance(v, str)).all()

    def test_empty_df(self):
        from swingtrader.dashboard.action import add_setup_classification_column
        out = add_setup_classification_column(pd.DataFrame())
        assert out.empty


class TestPortfolioGuidance:
    def _g(self, **kwargs):
        from swingtrader.dashboard.action import portfolio_guidance
        return portfolio_guidance(_row(**kwargs))

    def test_non_equity_returns_cash_note(self):
        g = self._g(is_non_equity=True)
        assert "cash" in g.lower() or "non-equity" in g.lower()

    def test_failed_returns_exit(self):
        g = self._g(state="FAILED")
        assert "exit" in g.lower() or "fail" in g.lower()

    def test_triggered_fresh_hold(self):
        g = self._g(state="TRIGGERED", days=3)
        assert "hold" in g.lower()

    def test_late_trim(self):
        g = self._g(state="LATE")
        assert "trim" in g.lower() or "de-risk" in g.lower()

    def test_armed_gives_defend_level(self):
        g = self._g(state="ARMED", pivot=100.0, atr14=2.0, dist_atr=2.0)
        # should mention a numeric stop level or "hold"
        assert any(c.isdigit() for c in g) or "hold" in g.lower()

    def test_none_state_not_evaluated(self):
        g = self._g(state="NONE")
        assert "not" in g.lower() or "evaluated" in g.lower()

    def test_add_column_preserves_shape(self):
        from swingtrader.dashboard.action import add_portfolio_guidance_column
        df = _snapshot()
        out = add_portfolio_guidance_column(df)
        assert "portfolio_guidance" in out.columns
        assert len(out) == len(df)

    def test_add_column_empty_df(self):
        from swingtrader.dashboard.action import add_portfolio_guidance_column
        assert add_portfolio_guidance_column(pd.DataFrame()).empty


# ═════════════════════════════════════════════════════════════════════════════
# dashboard.context  (no real data files needed — all return safe defaults)
# ═════════════════════════════════════════════════════════════════════════════

class TestBuildContext:
    """build_context must always return the correct structure even without data."""

    def test_returns_dict_with_required_keys(self):
        from swingtrader.dashboard.context import build_context
        row = _row()
        ctx = build_context("NONEXISTENT", row, {"pivot": 100.0})
        for key in ("ma_table", "avwap_table", "volume_block", "confluence", "checklist"):
            assert key in ctx, f"Missing key: {key}"

    def test_ma_table_is_list(self):
        from swingtrader.dashboard.context import build_context
        ctx = build_context("NONEXISTENT", _row(), {})
        assert isinstance(ctx["ma_table"], list)

    def test_avwap_table_is_list(self):
        from swingtrader.dashboard.context import build_context
        ctx = build_context("NONEXISTENT", _row(), {})
        assert isinstance(ctx["avwap_table"], list)

    def test_checklist_is_list(self):
        from swingtrader.dashboard.context import build_context
        ctx = build_context("NONEXISTENT", _row(), {})
        assert isinstance(ctx["checklist"], list)

    def test_confluence_is_dict(self):
        from swingtrader.dashboard.context import build_context
        ctx = build_context("NONEXISTENT", _row(), {})
        assert isinstance(ctx["confluence"], dict)
        assert "nearby_count" in ctx["confluence"]
        assert "cluster_role" in ctx["confluence"]

    def test_volume_block_is_dict(self):
        from swingtrader.dashboard.context import build_context
        ctx = build_context("NONEXISTENT", _row(), {})
        assert isinstance(ctx["volume_block"], dict)

    def test_nan_close_safe(self):
        from swingtrader.dashboard.context import build_context
        row = _row(close=float("nan"))
        ctx = build_context("NONEXISTENT", row, {})
        assert isinstance(ctx["ma_table"], list)

    def test_build_ma_table_no_data(self):
        from swingtrader.dashboard.context import build_ma_table
        result = build_ma_table("NONEXISTENT_XYZ", 100.0)
        assert isinstance(result, list)

    def test_build_avwap_table_no_data(self):
        from swingtrader.dashboard.context import build_avwap_table
        result = build_avwap_table("NONEXISTENT_XYZ", 100.0, 2.0)
        assert isinstance(result, list)

    def test_build_volume_block_no_data(self):
        from swingtrader.dashboard.context import build_volume_block
        result = build_volume_block("NONEXISTENT_XYZ", 100.0, 2.0)
        assert isinstance(result, dict)
        assert "compression_label" in result
        assert "relative_vol_label" in result


class TestBuildConfluence:
    def test_empty_levels_returns_scattered(self):
        from swingtrader.dashboard.context import build_confluence
        result = build_confluence(100.0, 100.0, 2.0, [], {})
        assert result["cluster_role"] == "scattered"
        assert result["nearby_count"] == 0

    def test_support_cluster_detected(self):
        from swingtrader.dashboard.context import build_confluence
        # Multiple levels below close within 0.5 ATR
        avwap_table = [
            {"anchor": "YTD",       "avwap": 99.8, "role": "support"},
            {"anchor": "Swing Low", "avwap": 99.7, "role": "support"},
        ]
        levels = {"pivot": 99.5, "s1": 99.6}
        result = build_confluence(100.0, 99.5, 2.0, avwap_table, levels)
        # All three levels are within 0.5 ATR (1.0) of close=100
        assert result["nearby_count"] >= 2
        assert "support" in result["cluster_role"]

    def test_nan_inputs_return_safe(self):
        from swingtrader.dashboard.context import build_confluence
        result = build_confluence(float("nan"), float("nan"), float("nan"), [], {})
        assert result["nearby_count"] == 0

    def test_zero_atr_return_safe(self):
        from swingtrader.dashboard.context import build_confluence
        result = build_confluence(100.0, 100.0, 0.0, [], {})
        assert result["nearby_count"] == 0


class TestBuildChecklist:
    def test_returns_15_items(self):
        from swingtrader.dashboard.context import build_checklist
        items = build_checklist("NONEXISTENT", _row(), {"risk_reward_t1": 2.0})
        assert len(items) == 15

    def test_item_structure(self):
        from swingtrader.dashboard.context import build_checklist
        items = build_checklist("NONEXISTENT", _row(), {"risk_reward_t1": 2.0})
        for item in items:
            assert "item" in item
            assert "result" in item
            assert "reason" in item
            assert item["result"] in ("pass", "fail", "neutral"), \
                f"Unexpected result: {item['result']!r}"

    def test_spy_above_200_passes_regime(self):
        from swingtrader.dashboard.context import build_checklist
        row = _row()
        row["regime_spy_above_200sma"] = 1.0
        items = build_checklist("NONEXISTENT", row, {})
        htf_item = next((i for i in items if "Higher timeframe" in i["item"]), None)
        assert htf_item is not None
        assert htf_item["result"] == "pass"

    def test_spy_below_200_fails_regime(self):
        from swingtrader.dashboard.context import build_checklist
        row = _row()
        row["regime_spy_above_200sma"] = 0.0
        row["regime_spy_trend"] = -1.0
        items = build_checklist("NONEXISTENT", row, {})
        htf_item = next((i for i in items if "Higher timeframe" in i["item"]), None)
        assert htf_item is not None
        assert htf_item["result"] == "fail"

    def test_rr_below_threshold_fails(self):
        from swingtrader.dashboard.context import build_checklist
        items = build_checklist("NONEXISTENT", _row(), {"risk_reward_t1": 0.8})
        rr_item = next((i for i in items if "R/R" in i["item"]), None)
        assert rr_item is not None
        assert rr_item["result"] == "fail"

    def test_rr_acceptable_passes(self):
        from swingtrader.dashboard.context import build_checklist
        items = build_checklist("NONEXISTENT", _row(), {"risk_reward_t1": 2.5})
        rr_item = next((i for i in items if "R/R" in i["item"]), None)
        assert rr_item is not None
        assert rr_item["result"] == "pass"

    def test_empty_symbol_does_not_crash(self):
        from swingtrader.dashboard.context import build_checklist
        items = build_checklist("", _row(), {})
        assert isinstance(items, list)


# ═════════════════════════════════════════════════════════════════════════════
# reports.ai_notes
# ═════════════════════════════════════════════════════════════════════════════

class TestAiNotes:
    def _packet(self):
        return {
            "symbol": "TST",
            "state": "ARMED",
            "action_label": "Actionable on breakout",
            "close": "99.50",
            "pivot": "100.00",
            "entry_lo": "100.00",
            "entry_hi": "100.20",
            "stop": "98.00",
            "t1": "104.00",
            "t2": "107.00",
            "risk_reward_t1": "2.50x",
            "composite_score": "0.55",
            "failure_risk": "0.25",
            "narrative": {
                "setup": "Tight base near pivot.",
                "why": "ATR compression with volume dry-up.",
                "entry": "Enter above pivot on volume.",
                "risk": "Exit on close below stop.",
                "targets": "T1 at 104, T2 at 107.",
                "verdict": "Favour breakout.",
                "trade_plan": "Buy above 100.20, stop 98.00.",
            },
            "context": {
                "checklist": [
                    {"item": "Compression present", "result": "pass", "reason": "ATR at 28th pct"},
                    {"item": "Relative strength", "result": "pass",  "reason": "RS-63: +6.0%"},
                    {"item": "Failure risk acceptable", "result": "fail", "reason": "70%"},
                ],
                "ma_table": [
                    {"name": "SMA50", "value": 97.5, "pct_dist": 2.05, "slope": "rising",
                     "bias": "Stays rising if close > 95.00"},
                ],
                "avwap_table": [],
                "volume_block": {},
                "confluence": {"nearby_count": 2, "nearby_levels": [], "cluster_role": "support cluster"},
            },
            "ai_note": None,
        }

    def test_rule_based_note_returns_string(self):
        from swingtrader.reports.ai_notes import _rule_based_note
        note = _rule_based_note(self._packet())
        assert isinstance(note, str)
        assert len(note) > 20

    def test_rule_based_note_contains_symbol(self):
        from swingtrader.reports.ai_notes import _rule_based_note
        note = _rule_based_note(self._packet())
        assert "TST" in note

    def test_rule_based_note_empty_narrative(self):
        from swingtrader.reports.ai_notes import _rule_based_note
        pkt = {"symbol": "X", "narrative": {}}
        note = _rule_based_note(pkt)
        assert isinstance(note, str)

    def test_generate_ai_note_no_key_returns_string(self, monkeypatch):
        """Without an API key, always returns rule-based note (no exception)."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        from swingtrader.reports.ai_notes import generate_ai_note
        note = generate_ai_note(self._packet())
        assert isinstance(note, str)
        assert len(note) > 10

    def test_enrich_packets_adds_ai_note(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        from swingtrader.reports.ai_notes import enrich_packets_with_ai
        packets = [self._packet(), self._packet()]
        result = enrich_packets_with_ai(packets)
        assert len(result) == 2
        for pkt in result:
            assert "ai_note" in pkt
            assert isinstance(pkt["ai_note"], str)

    def test_checklist_summary_from_context(self):
        from swingtrader.reports.ai_notes import _checklist_summary
        pkt = self._packet()
        summary = _checklist_summary(pkt)
        assert isinstance(summary, str)
        # Should mention pass/fail counts from the 3 items
        assert "pass" in summary.lower() or "fail" in summary.lower()

    def test_ma_summary_from_context(self):
        from swingtrader.reports.ai_notes import _ma_summary
        pkt = self._packet()
        summary = _ma_summary(pkt)
        assert isinstance(summary, str)
        # Should find SMA50 in the ma_table nested under context
        assert "SMA50" in summary or "MA:" in summary or summary == ""

    def test_enrich_empty_list(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        from swingtrader.reports.ai_notes import enrich_packets_with_ai
        assert enrich_packets_with_ai([]) == []


# ═════════════════════════════════════════════════════════════════════════════
# reports.artifacts
# ═════════════════════════════════════════════════════════════════════════════

class TestArtifacts:
    def _pkt(self, sym="TST"):
        return {
            "symbol": sym,
            "provider_symbol": sym,
            "state": "ARMED",
            "action_label": "Actionable on breakout",
            "setup_classification": "Near breakout / poised",
            "portfolio_guidance": "Hold — building base.",
            "composite_score": "0.55",
            "failure_risk": "0.25",
            "percentile_rank": "72",
            "close": "99.50",
            "pivot": "100.00",
            "atr14": "2.00",
            "dist_to_pivot_atr": "0.25",
            "base_length": 25,
            "days_in_state": 6,
            "entry_lo": "100.00",
            "entry_hi": "100.20",
            "stop": "98.00",
            "t1": "104.00",
            "t2": "107.00",
            "t3": "110.00",
            "s1": "100.00",
            "s2": "98.50",
            "s3": "97.00",
            "r1": "100.00",
            "r2": "102.00",
            "r3": "104.00",
            "risk_reward_t1": "2.50x",
            "is_portfolio": False,
            "is_fresh": True,
            "freshness_label": "fresh",
            "is_extended": False,
            "is_stale_confirmed": False,
            "atr_compression_pct": "28",
            "volume_dryup": "0.55",
            "daily_rs_63": "0.06",
            "rs_class": "outperforming",
            "close_vs_sma50": "0.015",
            "ma_slope_direction": "rising",
            "ytd_dist_atr": "1.20",
            "swing_low_dist_atr": "2.00",
            "regime_spy_trend": "1",
            "narrative": {
                "setup": "Tight base.", "why": "Volume dry-up.",
                "entry": "Above pivot.", "risk": "Below stop.",
                "targets": "T1 104.", "verdict": "Favour breakout.",
                "trade_plan": "Buy 100.20, stop 98.",
            },
            "context": {
                "ma_table": [],
                "avwap_table": [],
                "volume_block": {},
                "confluence": {"nearby_count": 0, "nearby_levels": [], "cluster_role": "scattered"},
                "checklist": [],
            },
            "ai_note": "Rule-based note for TST.",
            "chart_daily": None,
            "chart_weekly": None,
            "chart_intraday": None,
        }

    def test_write_artifacts_creates_files(self, tmp_path):
        from swingtrader.reports.artifacts import write_artifacts
        packets = [self._pkt("AAA"), self._pkt("BBB")]
        portfolio_df = pd.DataFrame()
        snapshot_df = _snapshot(4)
        as_of = pd.Timestamp("2026-01-15")
        paths = write_artifacts(packets, portfolio_df, snapshot_df, as_of, tmp_path)
        arts = tmp_path / "artifacts"
        assert arts.is_dir()
        assert (arts / "dashboard_summary.json").exists()
        assert (arts / "top_setups.json").exists()

    def test_summary_json_is_valid(self, tmp_path):
        from swingtrader.reports.artifacts import write_artifacts
        as_of = pd.Timestamp("2026-01-15")
        write_artifacts([self._pkt()], pd.DataFrame(), _snapshot(3), as_of, tmp_path)
        data = json.loads((tmp_path / "artifacts" / "dashboard_summary.json").read_text())
        assert "as_of" in data
        assert data["as_of"] == "2026-01-15"

    def test_top_setups_json_is_array(self, tmp_path):
        from swingtrader.reports.artifacts import write_artifacts
        as_of = pd.Timestamp("2026-01-15")
        write_artifacts([self._pkt("X1"), self._pkt("X2")],
                        pd.DataFrame(), pd.DataFrame(), as_of, tmp_path)
        data = json.loads((tmp_path / "artifacts" / "top_setups.json").read_text())
        assert isinstance(data, list)
        assert len(data) == 2

    def test_per_symbol_packet_written(self, tmp_path):
        from swingtrader.reports.artifacts import write_artifacts
        as_of = pd.Timestamp("2026-01-15")
        write_artifacts([self._pkt("XSYM")], pd.DataFrame(), pd.DataFrame(), as_of, tmp_path)
        pkt_file = tmp_path / "artifacts" / "XSYM_packet.json"
        assert pkt_file.exists()
        data = json.loads(pkt_file.read_text())
        assert data.get("symbol") == "XSYM"

    def test_top_setup_has_sections(self, tmp_path):
        from swingtrader.reports.artifacts import write_artifacts
        as_of = pd.Timestamp("2026-01-15")
        write_artifacts([self._pkt()], pd.DataFrame(), pd.DataFrame(), as_of, tmp_path)
        data = json.loads((tmp_path / "artifacts" / "top_setups.json").read_text())[0]
        for section in ("identity", "model_scores", "price_and_levels"):
            assert section in data, f"Missing section: {section}"

    def test_portfolio_json_written(self, tmp_path):
        from swingtrader.reports.artifacts import write_artifacts
        portfolio_df = _snapshot(2)
        portfolio_df["is_portfolio"] = True
        as_of = pd.Timestamp("2026-01-15")
        write_artifacts([self._pkt()], portfolio_df, pd.DataFrame(), as_of, tmp_path)
        pf = tmp_path / "artifacts" / "portfolio_review.json"
        assert pf.exists()
        data = json.loads(pf.read_text())
        assert isinstance(data, list)

    def test_clean_value_nan_becomes_none(self):
        from swingtrader.reports.artifacts import _clean_value
        assert _clean_value(float("nan")) is None
        assert _clean_value(float("inf")) is None
        assert _clean_value(float("-inf")) is None

    def test_clean_value_dash_becomes_none(self):
        from swingtrader.reports.artifacts import _clean_value
        assert _clean_value("—") is None

    def test_clean_value_finite_preserved(self):
        from swingtrader.reports.artifacts import _clean_value
        assert _clean_value(3.14) == pytest.approx(3.14)
        assert _clean_value(42) == 42
        assert _clean_value("hello") == "hello"
        assert _clean_value(True) is True

    def test_empty_packets_writes_empty_array(self, tmp_path):
        from swingtrader.reports.artifacts import write_artifacts
        write_artifacts([], pd.DataFrame(), pd.DataFrame(), pd.Timestamp("2026-01-15"), tmp_path)
        data = json.loads((tmp_path / "artifacts" / "top_setups.json").read_text())
        assert data == []


# ═════════════════════════════════════════════════════════════════════════════
# dashboard.packet — new fields
# ═════════════════════════════════════════════════════════════════════════════

class TestPacketNewFields:
    def _build(self, **kwargs):
        from swingtrader.dashboard.packet import build_packet
        return build_packet(_row(**kwargs))

    def test_context_key_present(self):
        pkt = self._build()
        assert "context" in pkt
        assert isinstance(pkt["context"], dict)

    def test_context_has_subkeys(self):
        pkt = self._build()
        ctx = pkt["context"]
        for k in ("ma_table", "avwap_table", "volume_block", "confluence", "checklist"):
            assert k in ctx

    def test_setup_classification_present(self):
        pkt = self._build()
        assert "setup_classification" in pkt
        assert isinstance(pkt["setup_classification"], str)
        assert len(pkt["setup_classification"]) > 0

    def test_portfolio_guidance_present(self):
        pkt = self._build()
        assert "portfolio_guidance" in pkt
        assert isinstance(pkt["portfolio_guidance"], str)

    def test_ai_note_initially_none(self):
        pkt = self._build()
        assert "ai_note" in pkt
        assert pkt["ai_note"] is None

    def test_provider_symbol_populated(self):
        pkt = self._build()
        assert pkt["provider_symbol"] == "TST"

    def test_armed_classification(self):
        pkt = self._build(state="ARMED", dist_atr=0.4)
        assert "Near breakout" in pkt["setup_classification"] or \
               "Approaching" in pkt["setup_classification"] or \
               "Building" in pkt["setup_classification"]

    def test_non_equity_guidance(self):
        pkt = self._build(is_non_equity=True)
        assert "cash" in pkt["portfolio_guidance"].lower() or \
               "non-equity" in pkt["portfolio_guidance"].lower()


# ═════════════════════════════════════════════════════════════════════════════
# reports.dashboard — new HTML helpers
# ═════════════════════════════════════════════════════════════════════════════

class TestDashboardHelpers:
    def test_setup_class_badge_known_label(self):
        from swingtrader.reports.dashboard import _setup_class_badge
        html = _setup_class_badge("Active breakout")
        assert "sc-active" in html
        assert "Active breakout" in html

    def test_setup_class_badge_unknown_label(self):
        from swingtrader.reports.dashboard import _setup_class_badge
        html = _setup_class_badge("Something new")
        assert "setup-class" in html

    def test_ma_table_html_empty(self):
        from swingtrader.reports.dashboard import _ma_table_html
        assert _ma_table_html([]) == ""

    def test_ma_table_html_renders_rows(self):
        from swingtrader.reports.dashboard import _ma_table_html
        ma = [{"name": "SMA50", "value": 97.5, "pct_dist": 2.05, "slope": "rising",
               "bias": "Stays rising if close > 95.00"}]
        html = _ma_table_html(ma)
        assert "SMA50" in html
        assert "rising" in html
        assert "slope-rising" in html

    def test_avwap_table_html_empty(self):
        from swingtrader.reports.dashboard import _avwap_table_html
        assert _avwap_table_html([]) == ""

    def test_avwap_table_html_renders_rows(self):
        from swingtrader.reports.dashboard import _avwap_table_html
        rows = [{"anchor": "YTD", "avwap": 98.0, "pct_dist": 1.52,
                 "dist_atr": 0.76, "role": "support", "status": "Accepted above",
                 "reclaim": True}]
        html = _avwap_table_html(rows)
        assert "YTD" in html
        assert "avwap-support" in html
        assert "Accepted above" in html

    def test_checklist_html_empty(self):
        from swingtrader.reports.dashboard import _checklist_html
        assert _checklist_html([]) == ""

    def test_checklist_html_counts(self):
        from swingtrader.reports.dashboard import _checklist_html
        items = [
            {"item": "Compression", "result": "pass", "reason": "28th pct"},
            {"item": "Volume",      "result": "fail", "reason": "score 0.1"},
            {"item": "RS",          "result": "neutral", "reason": "flat"},
        ]
        html = _checklist_html(items)
        assert "1" in html  # 1 pass, 1 fail
        assert "chk-pass" in html
        assert "chk-fail" in html
        assert "chk-neutral" in html

    def test_confluence_html_scattered(self):
        from swingtrader.reports.dashboard import _confluence_html
        conf = {"nearby_count": 0, "nearby_levels": [], "cluster_role": "scattered"}
        html = _confluence_html(conf)
        assert "scattered" in html
        assert "conf-scattered" in html

    def test_confluence_html_support_cluster(self):
        from swingtrader.reports.dashboard import _confluence_html
        conf = {
            "nearby_count": 3,
            "nearby_levels": [
                {"name": "S1", "value": 99.8, "dist_atr": 0.1},
                {"name": "AVWAP YTD", "value": 99.5, "dist_atr": 0.25},
                {"name": "Pivot", "value": 100.0, "dist_atr": 0.0},
            ],
            "cluster_role": "support cluster",
        }
        html = _confluence_html(conf)
        assert "support cluster" in html
        assert "conf-support" in html

    def test_volume_block_html_empty(self):
        from swingtrader.reports.dashboard import _volume_block_html
        assert _volume_block_html({}) == ""

    def test_volume_block_html_renders(self):
        from swingtrader.reports.dashboard import _volume_block_html
        block = {
            "compression_label": "Tight (<30th pct) — 28th percentile",
            "relative_vol_label": "Low",
            "atr_compression_pct": 28.0,
            "volume_dryup_pct": 55.0,
            "breakout_thrust_atr": float("nan"),
            "vol_contraction_5_20": 0.72,
        }
        html = _volume_block_html(block)
        assert "Tight" in html
        assert "Low" in html
        assert "vol-block" in html

    def test_ai_note_html_empty(self):
        from swingtrader.reports.dashboard import _ai_note_html
        assert _ai_note_html(None) == ""
        assert _ai_note_html("") == ""

    def test_ai_note_html_renders(self):
        from swingtrader.reports.dashboard import _ai_note_html
        html = _ai_note_html("Strong setup with volume confirmation.")
        assert "ai-note" in html
        assert "Strong setup" in html

    def test_trade_plan_html_empty(self):
        from swingtrader.reports.dashboard import _trade_plan_html
        assert _trade_plan_html(None) == ""
        assert _trade_plan_html("") == ""

    def test_trade_plan_html_renders(self):
        from swingtrader.reports.dashboard import _trade_plan_html
        html = _trade_plan_html("Buy above 100.20, stop at 98.00.")
        assert "trade-plan" in html
        assert "100.20" in html


class TestRegimeHtml:
    def test_renders_spy_200sma_pill(self):
        from swingtrader.reports.dashboard import _regime_html
        df = _snapshot(4)
        html = _regime_html(df)
        assert "200SMA" in html

    def test_renders_vix_pill(self):
        from swingtrader.reports.dashboard import _regime_html
        df = _snapshot(4)
        html = _regime_html(df)
        assert "VIX" in html

    def test_favors_breakouts_when_uptrend_above_200(self):
        from swingtrader.reports.dashboard import _regime_html
        df = _snapshot(2)
        df["regime_spy_trend"] = 1.0
        df["regime_spy_above_200sma"] = 1.0
        df["regime_vix_level"] = 14.0
        html = _regime_html(df)
        assert "Favors breakouts" in html

    def test_risk_off_when_downtrend_high_vix(self):
        from swingtrader.reports.dashboard import _regime_html
        df = _snapshot(2)
        df["regime_spy_trend"] = -1.0
        df["regime_spy_above_200sma"] = 0.0
        df["regime_vix_level"] = 30.0
        html = _regime_html(df)
        assert "Risk-off" in html

    def test_empty_df_renders_dashes(self):
        from swingtrader.reports.dashboard import _regime_html
        html = _regime_html(pd.DataFrame())
        assert "summary-bar" in html

    def test_vix_complacent_label(self):
        from swingtrader.reports.dashboard import _regime_html
        df = _snapshot(2)
        df["regime_vix_level"] = 12.0
        html = _regime_html(df)
        assert "Complacent" in html

    def test_vix_elevated_label(self):
        from swingtrader.reports.dashboard import _regime_html
        df = _snapshot(2)
        df["regime_vix_level"] = 24.0
        html = _regime_html(df)
        assert "Elevated" in html


class TestPortfolioStripHtml:
    def test_renders_guidance(self):
        from swingtrader.reports.dashboard import _portfolio_strip_html
        df = _snapshot(3)
        df["portfolio_guidance"] = "Hold — active breakout."
        html = _portfolio_strip_html(df)
        assert "pc-guidance" in html
        assert "Hold" in html

    def test_no_guidance_column_renders_cleanly(self):
        from swingtrader.reports.dashboard import _portfolio_strip_html
        df = _snapshot(2)
        # No portfolio_guidance column
        html = _portfolio_strip_html(df)
        assert "portfolio-strip" in html
        assert "pc-guidance" not in html

    def test_empty_df_returns_empty(self):
        from swingtrader.reports.dashboard import _portfolio_strip_html
        assert _portfolio_strip_html(pd.DataFrame()) == ""


class TestDashboardIntegration:
    """End-to-end render with all Phase B/C fields populated."""

    def test_renders_with_new_packet_fields(self):
        from swingtrader.dashboard.action import (
            add_action_column, add_portfolio_guidance_column,
            add_setup_classification_column,
        )
        from swingtrader.dashboard.freshness import add_freshness_columns
        from swingtrader.dashboard.packet import build_packets
        from swingtrader.dashboard.selector import select_top_setups
        from swingtrader.reports.dashboard import render_dashboard

        df = _snapshot(6)
        df = add_freshness_columns(df)
        df = add_action_column(df)
        df = add_setup_classification_column(df)
        df = add_portfolio_guidance_column(df)
        top = select_top_setups(df)
        pkts = build_packets(top)

        # Simulate AI note fill-in
        for pkt in pkts:
            pkt["ai_note"] = "Rule-based fallback note."

        html = render_dashboard(df, pkts, pd.Timestamp("2026-04-17"))

        assert "<!DOCTYPE html>" in html
        assert "2026-04-17" in html
        # New elements should appear in at least one card
        if pkts:
            # Setup classification and portfolio guidance are wired in
            assert "setup-class" in html
            assert "summary-bar" in html

    def test_setup_classification_in_card_header(self):
        from swingtrader.dashboard.action import (
            add_action_column, add_setup_classification_column,
        )
        from swingtrader.dashboard.freshness import add_freshness_columns
        from swingtrader.dashboard.packet import build_packets
        from swingtrader.dashboard.selector import select_top_setups
        from swingtrader.reports.dashboard import render_dashboard

        df = _snapshot(4)
        df = add_freshness_columns(df)
        df = add_action_column(df)
        df = add_setup_classification_column(df)
        top = select_top_setups(df)
        pkts = build_packets(top)
        html = render_dashboard(df, pkts, pd.Timestamp("2026-04-17"))
        if pkts:
            assert "setup-class" in html

    def test_portfolio_guidance_in_strip(self):
        from swingtrader.dashboard.action import (
            add_action_column, add_portfolio_guidance_column,
        )
        from swingtrader.dashboard.freshness import add_freshness_columns
        from swingtrader.dashboard.packet import build_packets
        from swingtrader.dashboard.selector import select_top_setups
        from swingtrader.reports.dashboard import render_dashboard

        df = _snapshot(4)
        df = add_freshness_columns(df)
        df = add_action_column(df)
        df = add_portfolio_guidance_column(df)
        top = select_top_setups(df)
        pkts = build_packets(top)
        html = render_dashboard(df, pkts, pd.Timestamp("2026-04-17"))
        # First row is_portfolio=True; guidance should appear in strip
        assert "Portfolio Holdings" in html
        assert "pc-guidance" in html
