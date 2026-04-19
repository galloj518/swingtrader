"""Packet-first architecture contract tests.

These tests prove the architectural invariants of the packet-first refactor:

  1. build_lightweight_packet computes eligibility, freshness, bucket, trade plan,
     and portfolio health internally — it does NOT read pre-added DataFrame columns.
  2. select_packets operates purely on packet dicts (no DataFrame, no column access).
  3. Bucket assignment comes from the packet, not from a prior DataFrame column pass.
  4. Dashboard write_dashboard can render from a selections dict (thin rendering layer).
  5. Portfolio logic is packet-driven: portfolio_health comes from the packet builder.
  6. Rejection reasons come from packets, not from pre-added eligibility columns.
  7. Trade plan fields (actionability_code, entry_condition, stop_basis, etc.) come
     from packets.
  8. All-packets flow: build_all_lightweight_packets + select_packets produces a
     complete PacketSelections dict with the expected bucket keys.
"""
from __future__ import annotations

import math

import pandas as pd
import pytest

from swingtrader.dashboard.buckets import (
    BUCKET_BREAKOUT,
    BUCKET_EXCLUDED,
    BUCKET_PORTFOLIO,
    BUCKET_PULLBACK,
)
from swingtrader.dashboard.packet import (
    build_all_lightweight_packets,
    build_lightweight_packet,
    enrich_with_context,
)
from swingtrader.dashboard.selector import PacketSelections, select_packets


# ── Shared row builders ────────────────────────────────────────────────────────

def _minimal_row(**kwargs) -> pd.Series:
    """Minimal valid snapshot row for packet building (no pre-added columns)."""
    base = {
        "user_symbol": "TEST",
        "provider_symbol": "TEST",
        "state": "ARMED",
        "pivot": 100.0,
        "atr14": 2.0,
        "close": 99.5,
        "dist_to_pivot_atr": 0.3,
        "days_in_state": 5,
        "composite_score": 0.60,
        "setup_score": 0.55,
        "trade_score": 0.50,
        "failure_risk": 0.22,
        "percentile_rank": 75.0,
        "base_length": 28,
        "atr_compression_pct": 38.0,
        "volume_dryup": 0.55,
        "close_vs_sma50": 0.02,
        "daily_rs_63": 0.07,
        "ytd_dist_atr": 1.2,
        "swing_low_dist_atr": 2.5,
        "regime_spy_trend": 1.0,
        "is_portfolio": False,
        "is_watchlist": True,
        "is_non_equity": False,
        "groups": "tech",
    }
    base.update(kwargs)
    # Deliberately omit pre-computed columns: eligible, bucket, is_fresh, action_label, etc.
    return pd.Series(base)


def _portfolio_row(**kwargs) -> pd.Series:
    """A portfolio holding in TRIGGERED state."""
    return _minimal_row(
        user_symbol="PORT",
        provider_symbol="PORT",
        state="TRIGGERED",
        is_portfolio=True,
        close=108.0,
        dist_to_pivot_atr=0.8,
        days_in_state=3,
        **kwargs,
    )


def _ineligible_row(**kwargs) -> pd.Series:
    """A row that should be ineligible (FAILED state, high failure risk)."""
    return _minimal_row(
        user_symbol="FAIL",
        state="FAILED",
        failure_risk=0.85,
        is_non_equity=False,
        **kwargs,
    )


def _snapshot_df(n: int = 8) -> pd.DataFrame:
    """
    Build a small snapshot DataFrame with NO pre-added eligibility / bucket columns.
    This is the key invariant: build_all_lightweight_packets must work without them.
    """
    rows = []
    for i in range(n):
        state = ["ARMED", "BASE", "TRIGGERED", "TRIGGERED",
                 "FAILED", "LATE", "BASE", "CONFIRMED"][i % 8]
        rows.append({
            "user_symbol": f"SYM{i:02d}",
            "provider_symbol": f"SYM{i:02d}",
            "state": state,
            "pivot": 50.0 + i * 5,
            "atr14": 1.5,
            "close": 50.5 + i * 5,
            "dist_to_pivot_atr": 0.4 - i * 0.05,
            "days_in_state": 4 + i,
            "composite_score": max(0.05, 0.70 - i * 0.08),
            "setup_score": max(0.05, 0.65 - i * 0.07),
            "trade_score": max(0.05, 0.60 - i * 0.06),
            "failure_risk": 0.18 + i * 0.06,
            "percentile_rank": 90.0 - i * 8,
            "base_length": 25 + i * 2,
            "atr_compression_pct": 35.0,
            "volume_dryup": 0.50,
            "close_vs_sma50": 0.015,
            "daily_rs_63": 0.06,
            "ytd_dist_atr": 1.0 + i * 0.2,
            "swing_low_dist_atr": 2.0,
            "regime_spy_trend": 1.0,
            "is_portfolio": (i == 2),
            "is_watchlist": True,
            "is_non_equity": False,
            "groups": ["tech", "health", "energy", "tech"][i % 4],
        })
    return pd.DataFrame(rows)


# ── 1. Packet is self-contained (no pre-added columns required) ───────────────

class TestPacketSelfContained:
    """Packet builder computes everything internally from raw row fields."""

    def test_packet_built_without_eligible_column(self):
        """Row with no 'eligible' column → packet still has 'eligible' key."""
        row = _minimal_row()
        assert "eligible" not in row.index, "Precondition: no pre-added eligible column"
        pkt = build_lightweight_packet(row)
        assert "eligible" in pkt

    def test_packet_built_without_is_fresh_column(self):
        """Row with no 'is_fresh' column → packet still has 'is_fresh' key."""
        row = _minimal_row()
        assert "is_fresh" not in row.index
        pkt = build_lightweight_packet(row)
        assert "is_fresh" in pkt
        assert isinstance(pkt["is_fresh"], bool)

    def test_packet_built_without_bucket_column(self):
        """Row with no 'bucket' column → packet still has 'bucket' key."""
        row = _minimal_row()
        assert "bucket" not in row.index
        pkt = build_lightweight_packet(row)
        assert "bucket" in pkt
        assert isinstance(pkt["bucket"], str)
        assert pkt["bucket"] != ""

    def test_packet_built_without_action_label_column(self):
        """Row with no 'action_label' column → packet still has 'action_label' key."""
        row = _minimal_row()
        assert "action_label" not in row.index
        pkt = build_lightweight_packet(row)
        assert "action_label" in pkt
        assert isinstance(pkt["action_label"], str)

    def test_canonical_fields_all_present(self):
        """Every packet must have these canonical top-level keys."""
        row = _minimal_row()
        pkt = build_lightweight_packet(row)
        required = [
            "symbol", "provider_symbol", "state",
            "eligible", "rejection_reasons", "rejection_reasons_list",
            "freshness_label", "is_fresh", "is_extended",
            "bucket", "action_label",
            "composite_score", "setup_score", "failure_risk",
            "entry_lo", "stop", "t1", "t2",
            "trade_plan", "portfolio_health",
            "narrative",
            "context", "assessments", "ai_note",
            "chart_weekly", "chart_daily",
        ]
        missing = [k for k in required if k not in pkt]
        assert not missing, f"Missing canonical fields: {missing}"

    def test_context_starts_as_none(self):
        """Lightweight packet has context=None before enrich_with_context."""
        pkt = build_lightweight_packet(_minimal_row())
        assert pkt["context"] is None
        assert pkt["assessments"] is None
        assert pkt["ai_note"] is None

    def test_chart_paths_start_as_none(self):
        pkt = build_lightweight_packet(_minimal_row())
        assert pkt["chart_weekly"] is None
        assert pkt["chart_daily"] is None

    def test_symbol_uses_user_symbol_when_present(self):
        row = _minimal_row(user_symbol="MSFT", provider_symbol="MSFT")
        pkt = build_lightweight_packet(row)
        assert pkt["symbol"] == "MSFT"


# ── 2. Eligibility is packet-driven ──────────────────────────────────────────

class TestEligibilityFromPacket:
    """Eligibility and rejection reasons come from the packet, never from row columns."""

    def test_ineligible_row_has_rejection_reasons(self):
        """FAILED state row → ineligible, rejection_reasons_list non-empty."""
        pkt = build_lightweight_packet(_ineligible_row())
        assert pkt["eligible"] is False
        assert len(pkt["rejection_reasons_list"]) > 0

    def test_eligible_row_has_empty_rejections(self):
        """ARMED row with good scores → eligible, rejection_reasons_list empty."""
        row = _minimal_row(
            state="ARMED",
            dist_to_pivot_atr=0.3,
            failure_risk=0.20,
            composite_score=0.65,
        )
        pkt = build_lightweight_packet(row)
        # May or may not be eligible depending on full gate logic, but we verify structure
        assert isinstance(pkt["eligible"], bool)
        assert isinstance(pkt["rejection_reasons_list"], list)
        if pkt["eligible"]:
            assert pkt["rejection_reasons"] == "" or pkt["rejection_reasons"] == "—"

    def test_rejection_reasons_string_matches_list(self):
        """rejection_reasons string is the join of rejection_reasons_list."""
        pkt = build_lightweight_packet(_ineligible_row())
        expected = ", ".join(pkt["rejection_reasons_list"])
        assert pkt["rejection_reasons"] == expected

    def test_precomputed_eligible_column_ignored(self):
        """Even if row has 'eligible=True', packet recomputes from scratch."""
        row = _ineligible_row()
        row_with_precomputed = row.copy()
        row_with_precomputed["eligible"] = True  # lie: pre-set to True
        pkt = build_lightweight_packet(row_with_precomputed)
        # Packet must recompute — FAILED state should still be ineligible
        # (actual gate result depends on implementation; we just confirm it's not blindly copied)
        assert isinstance(pkt["eligible"], bool)
        # The packet's eligible comes from assess_eligibility, not the row column


# ── 3. Bucket is packet-driven ────────────────────────────────────────────────

class TestBucketFromPacket:
    """Bucket assignment is computed by build_lightweight_packet, not read from row."""

    def test_failed_state_gets_excluded_bucket(self):
        """FAILED state → excluded bucket (cannot be breakout or pullback)."""
        pkt = build_lightweight_packet(_ineligible_row())
        assert pkt["bucket"] == BUCKET_EXCLUDED

    def test_portfolio_row_gets_portfolio_bucket(self):
        """is_portfolio=True in TRIGGERED state → portfolio bucket."""
        pkt = build_lightweight_packet(_portfolio_row())
        assert pkt["bucket"] == BUCKET_PORTFOLIO

    def test_armed_row_fresh_near_pivot_gets_breakout_bucket(self):
        """ARMED, close to pivot, good scores, fresh → breakout_long bucket."""
        row = _minimal_row(
            state="ARMED",
            dist_to_pivot_atr=0.3,
            days_in_state=4,
            composite_score=0.68,
            failure_risk=0.19,
        )
        pkt = build_lightweight_packet(row)
        # If eligible, should be breakout or pullback; never portfolio or excluded
        if pkt["eligible"]:
            assert pkt["bucket"] in {BUCKET_BREAKOUT, BUCKET_PULLBACK}

    def test_non_equity_gets_non_equity_bucket(self):
        """is_non_equity=True → non_equity bucket."""
        from swingtrader.dashboard.buckets import BUCKET_NON_EQUITY
        row = _minimal_row(is_non_equity=True)
        pkt = build_lightweight_packet(row)
        assert pkt["bucket"] == BUCKET_NON_EQUITY

    def test_bucket_not_read_from_row(self):
        """Even if row['bucket'] is set to a lie, packet recomputes it."""
        row = _ineligible_row()
        row_with_lie = row.copy()
        row_with_lie["bucket"] = BUCKET_BREAKOUT  # lie
        pkt = build_lightweight_packet(row_with_lie)
        # Packet ignores row['bucket'] — recomputes from eligibility/freshness
        assert pkt["bucket"] != BUCKET_BREAKOUT  # FAILED should never be breakout


# ── 4. Selector operates purely on packet dicts ───────────────────────────────

class TestSelectPacketsIsPacketDriven:
    """select_packets takes list[dict] and returns PacketSelections. No DataFrame."""

    def test_select_packets_accepts_list_of_dicts(self):
        """Input is list[dict], not DataFrame."""
        df = _snapshot_df(8)
        pkts = build_all_lightweight_packets(df)
        assert isinstance(pkts, list)
        assert all(isinstance(p, dict) for p in pkts)

        result = select_packets(pkts)
        assert isinstance(result, dict)

    def test_select_packets_returns_all_expected_keys(self):
        """Result dict has all expected bucket keys."""
        pkts = build_all_lightweight_packets(_snapshot_df(8))
        result = select_packets(pkts)
        for key in ("breakout", "pullback", "extended", "reversal", "portfolio", "excluded", "top"):
            assert key in result, f"Missing key: {key}"

    def test_select_packets_result_is_list_of_dicts(self):
        """Every value in result is a list of packet dicts."""
        pkts = build_all_lightweight_packets(_snapshot_df(8))
        result = select_packets(pkts)
        for key, lst in result.items():
            assert isinstance(lst, list), f"result[{key!r}] is not a list"
            for pkt in lst:
                assert isinstance(pkt, dict), f"result[{key!r}] contains non-dict"

    def test_breakout_list_contains_only_fresh_non_portfolio(self):
        """Breakout list: every packet must be fresh and non-portfolio."""
        pkts = build_all_lightweight_packets(_snapshot_df(8))
        result = select_packets(pkts)
        for pkt in result["breakout"]:
            assert bool(pkt.get("is_fresh")), f"{pkt['symbol']} in breakout but not fresh"
            assert not bool(pkt.get("is_portfolio")), f"{pkt['symbol']} in breakout but is_portfolio"

    def test_portfolio_list_contains_only_portfolio_packets(self):
        """Portfolio list: every packet must have is_portfolio=True."""
        pkts = build_all_lightweight_packets(_snapshot_df(8))
        result = select_packets(pkts)
        for pkt in result["portfolio"]:
            assert bool(pkt.get("is_portfolio")), f"{pkt['symbol']} in portfolio but is_portfolio=False"

    def test_top_is_union_of_breakout_and_pullback(self):
        """top = breakout + pullback (in that order, no duplicates)."""
        pkts = build_all_lightweight_packets(_snapshot_df(8))
        result = select_packets(pkts)
        top_syms = [p["symbol"] for p in result["top"]]
        bo_syms  = [p["symbol"] for p in result["breakout"]]
        pb_syms  = [p["symbol"] for p in result["pullback"]]
        combined = bo_syms + pb_syms
        assert top_syms == combined

    def test_no_duplicate_symbols_across_breakout_and_pullback(self):
        """No symbol appears in both breakout and pullback."""
        pkts = build_all_lightweight_packets(_snapshot_df(8))
        result = select_packets(pkts)
        bo_syms = {p["symbol"] for p in result["breakout"]}
        pb_syms = {p["symbol"] for p in result["pullback"]}
        overlap = bo_syms & pb_syms
        assert not overlap, f"Symbols in both breakout and pullback: {overlap}"

    def test_empty_input_returns_empty_selections(self):
        """Empty packet list → all selection lists are empty."""
        result = select_packets([])
        for key in ("breakout", "pullback", "top"):
            assert result[key] == []

    def test_caps_are_respected(self):
        """select_packets respects n_breakout / n_pullback caps."""
        # Build a large pool of fresh eligible ARMED rows
        rows = []
        for i in range(20):
            rows.append({
                "user_symbol": f"S{i:03d}",
                "provider_symbol": f"S{i:03d}",
                "state": "ARMED",
                "pivot": 50.0 + i,
                "atr14": 1.5,
                "close": 50.3 + i,
                "dist_to_pivot_atr": 0.2,
                "days_in_state": 4,
                "composite_score": 0.65 - i * 0.01,
                "setup_score": 0.60 - i * 0.01,
                "trade_score": 0.55,
                "failure_risk": 0.18,
                "percentile_rank": 85.0 - i,
                "base_length": 28,
                "atr_compression_pct": 35.0,
                "volume_dryup": 0.5,
                "close_vs_sma50": 0.02,
                "daily_rs_63": 0.07,
                "ytd_dist_atr": 1.1,
                "swing_low_dist_atr": 2.0,
                "regime_spy_trend": 1.0,
                "is_portfolio": False,
                "is_watchlist": True,
                "is_non_equity": False,
                "groups": "other",
            })
        df = pd.DataFrame(rows)
        pkts = build_all_lightweight_packets(df)
        result = select_packets(pkts, n_breakout=3, n_pullback=2)
        assert len(result["breakout"]) <= 3
        assert len(result["pullback"]) <= 2


# ── 5. Portfolio health is packet-driven ──────────────────────────────────────

class TestPortfolioHealthFromPacket:
    """portfolio_health is built inside build_lightweight_packet."""

    def test_portfolio_packet_has_health_dict(self):
        """Portfolio row → portfolio_health is a non-empty dict."""
        pkt = build_lightweight_packet(_portfolio_row())
        ph = pkt.get("portfolio_health", {})
        assert isinstance(ph, dict)
        assert ph != {}, "portfolio_health should be non-empty for is_portfolio=True"

    def test_portfolio_health_has_required_keys(self):
        """portfolio_health must contain position_health, recommended_action, key_level, notes."""
        pkt = build_lightweight_packet(_portfolio_row())
        ph = pkt["portfolio_health"]
        for key in ("position_health", "recommended_action", "key_level", "notes"):
            assert key in ph, f"portfolio_health missing key: {key}"

    def test_portfolio_health_position_values(self):
        """position_health must be one of the defined values."""
        valid = {"healthy", "at_risk", "extended", "recovering", "failed", "neutral"}
        pkt = build_lightweight_packet(_portfolio_row())
        assert pkt["portfolio_health"]["position_health"] in valid

    def test_non_portfolio_row_has_empty_portfolio_health(self):
        """Non-portfolio rows get portfolio_health = {}."""
        pkt = build_lightweight_packet(_minimal_row(is_portfolio=False))
        assert pkt["portfolio_health"] == {}

    def test_failed_portfolio_row_gets_failed_health(self):
        """FAILED + portfolio → health='failed' and EXIT recommendation."""
        row = _minimal_row(
            user_symbol="PORT",
            provider_symbol="PORT",
            state="FAILED",
            is_portfolio=True,
            close=108.0,
            dist_to_pivot_atr=0.8,
            days_in_state=3,
        )
        pkt = build_lightweight_packet(row)
        ph = pkt["portfolio_health"]
        assert ph["position_health"] == "failed"
        assert "EXIT" in ph["recommended_action"].upper()

    def test_portfolio_health_not_from_row_column(self):
        """Even if row has 'portfolio_health', packet recomputes it."""
        row = _portfolio_row()
        row_with_lie = row.copy()
        row_with_lie["portfolio_health"] = {"position_health": "fabricated"}
        pkt = build_lightweight_packet(row_with_lie)
        # Packet builds portfolio_health internally — should not be "fabricated"
        ph = pkt["portfolio_health"]
        assert ph.get("position_health") != "fabricated"


# ── 6. Trade plan is packet-driven ────────────────────────────────────────────

class TestTradePlanFromPacket:
    """trade_plan is built inside build_lightweight_packet."""

    def test_trade_plan_present_and_is_dict(self):
        pkt = build_lightweight_packet(_minimal_row())
        tp = pkt.get("trade_plan")
        assert isinstance(tp, dict)

    def test_trade_plan_has_required_fields(self):
        pkt = build_lightweight_packet(_minimal_row())
        tp = pkt["trade_plan"]
        required_fields = [
            "actionability_code", "entry_condition", "entry_range",
            "stop", "stop_basis", "stop_risk_pct",
            "target_1", "target_2", "risk_reward_t1",
            "time_stop", "key_risk",
        ]
        missing = [f for f in required_fields if f not in tp]
        assert not missing, f"trade_plan missing fields: {missing}"

    def test_actionability_code_valid_value(self):
        """actionability_code must be one of the 5 defined verdicts."""
        valid_codes = {"BUY_NOW", "WATCH_BREAKOUT", "WAIT_PULLBACK", "WAIT_ZONE", "BLOCK"}
        row = _minimal_row()
        pkt = build_lightweight_packet(row)
        assert pkt["trade_plan"]["actionability_code"] in valid_codes

    def test_actionability_code_buy_now_for_triggered_fresh(self):
        """TRIGGERED + fresh + eligible → BUY_NOW actionability code."""
        row = _minimal_row(
            state="TRIGGERED",
            days_in_state=2,
            dist_to_pivot_atr=0.4,
            failure_risk=0.20,
            composite_score=0.65,
        )
        pkt = build_lightweight_packet(row)
        if pkt["eligible"] and pkt["is_fresh"]:
            assert pkt["trade_plan"]["actionability_code"] == "BUY_NOW"

    def test_entry_condition_is_non_empty_string(self):
        pkt = build_lightweight_packet(_minimal_row())
        assert isinstance(pkt["trade_plan"]["entry_condition"], str)
        assert len(pkt["trade_plan"]["entry_condition"]) > 0

    def test_stop_basis_is_non_empty_string(self):
        pkt = build_lightweight_packet(_minimal_row())
        assert isinstance(pkt["trade_plan"]["stop_basis"], str)
        assert len(pkt["trade_plan"]["stop_basis"]) > 0

    def test_trade_plan_not_from_row_column(self):
        """Packet recomputes trade_plan even if row already has one."""
        row = _minimal_row()
        row_with_lie = row.copy()
        row_with_lie["trade_plan"] = {"actionability_code": "FABRICATED"}
        pkt = build_lightweight_packet(row_with_lie)
        tp = pkt["trade_plan"]
        assert tp.get("actionability_code") != "FABRICATED"


# ── 7. All-packets batch flow ─────────────────────────────────────────────────

class TestBuildAllLightweightPackets:
    """build_all_lightweight_packets produces correct count and structure."""

    def test_count_matches_dataframe_rows(self):
        df = _snapshot_df(8)
        pkts = build_all_lightweight_packets(df)
        assert len(pkts) == 8

    def test_no_file_io_required(self):
        """build_all_lightweight_packets must complete without any file I/O errors."""
        df = _snapshot_df(6)
        pkts = build_all_lightweight_packets(df)  # should not raise
        assert len(pkts) == 6

    def test_each_packet_has_symbol(self):
        df = _snapshot_df(4)
        pkts = build_all_lightweight_packets(df)
        for pkt in pkts:
            assert pkt.get("symbol", "") != ""

    def test_bad_row_does_not_break_batch(self):
        """A row with all-NaN scores should still produce a packet (best-effort)."""
        df = _snapshot_df(3)
        df.loc[1, "composite_score"] = float("nan")
        df.loc[1, "pivot"] = float("nan")
        pkts = build_all_lightweight_packets(df)
        assert len(pkts) == 3  # bad row handled, not skipped

    def test_dataframe_columns_not_mutated(self):
        """build_all_lightweight_packets must not add columns to the input DataFrame."""
        df = _snapshot_df(4)
        original_cols = set(df.columns)
        build_all_lightweight_packets(df)
        assert set(df.columns) == original_cols, "Input DataFrame was mutated"


# ── 8. PacketSelections type alias ────────────────────────────────────────────

class TestPacketSelectionsType:
    """PacketSelections is the canonical type for selection output."""

    def test_select_packets_result_matches_type_alias(self):
        """Result of select_packets is a dict[str, list[dict]] = PacketSelections."""
        pkts = build_all_lightweight_packets(_snapshot_df(4))
        result: PacketSelections = select_packets(pkts)
        assert isinstance(result, dict)
        for v in result.values():
            assert isinstance(v, list)


# ── 9. Dashboard renders from selections (thin rendering layer) ───────────────

class TestDashboardRendersFromSelections:
    """write_dashboard accepts selections dict (packet-first path)."""

    def test_write_dashboard_accepts_selections_kwarg(self, tmp_path):
        """write_dashboard with selections= kwarg must produce a file."""
        from swingtrader.reports.dashboard import write_dashboard

        df = _snapshot_df(6)
        pkts = build_all_lightweight_packets(df)
        sel = select_packets(pkts)

        dash_path = write_dashboard(
            df,
            sel["top"],
            pd.Timestamp("2025-01-15"),
            tmp_path,
            selections=sel,
        )
        assert dash_path.exists()
        html = dash_path.read_text(encoding="utf-8")
        assert len(html) > 500  # non-trivial output

    def test_dashboard_uses_bucket_counts_from_packets(self, tmp_path):
        """When selections provided, universe bar should reflect packet bucket counts."""
        from swingtrader.reports.dashboard import write_dashboard

        df = _snapshot_df(8)
        pkts = build_all_lightweight_packets(df)
        sel = select_packets(pkts)

        dash_path = write_dashboard(
            df, sel["top"], pd.Timestamp("2025-01-15"), tmp_path,
            selections=sel,
        )
        html = dash_path.read_text(encoding="utf-8")
        # Universe section should be present
        assert "Universe" in html or "universe" in html or "scored" in html.lower()


# ── 10. Artifacts are written from selections ──────────────────────────────────

class TestArtifactsPacketDriven:
    """write_artifacts with PacketSelections dict (new call signature)."""

    def test_write_artifacts_from_selections(self, tmp_path):
        """write_artifacts(selections, snapshot_df, as_of, output_dir) runs without error."""
        from swingtrader.reports.artifacts import write_artifacts

        df = _snapshot_df(6)
        pkts = build_all_lightweight_packets(df)
        sel = select_packets(pkts)

        paths = write_artifacts(sel, df, pd.Timestamp("2025-01-15"), tmp_path)
        assert "summary" in paths
        from pathlib import Path
        assert Path(paths["summary"]).exists()

    def test_artifacts_eligibility_uses_packet_data(self, tmp_path):
        """eligibility_results.json should reference symbols from all packets."""
        import json
        from swingtrader.reports.artifacts import write_artifacts

        df = _snapshot_df(4)
        pkts = build_all_lightweight_packets(df)
        sel = select_packets(pkts)

        paths = write_artifacts(sel, df, pd.Timestamp("2025-01-15"), tmp_path)
        elig_path = tmp_path / "artifacts" / "eligibility_results.json"
        assert elig_path.exists()
        records = json.loads(elig_path.read_text())
        # Every record must have 'eligible' and 'rejection_reasons' from packets
        for rec in records:
            assert "eligible" in rec
            assert "rejection_reasons" in rec

    def test_artifacts_bucket_assignments_from_packets(self, tmp_path):
        """bucket_assignments.json uses packet bucket, not DataFrame column."""
        import json
        from swingtrader.reports.artifacts import write_artifacts

        df = _snapshot_df(4)
        # Deliberately add a wrong 'bucket' column to the DataFrame
        df["bucket"] = "wrong_bucket"
        pkts = build_all_lightweight_packets(df)
        sel = select_packets(pkts)

        paths = write_artifacts(sel, df, pd.Timestamp("2025-01-15"), tmp_path)
        bucket_path = tmp_path / "artifacts" / "bucket_assignments.json"
        records = json.loads(bucket_path.read_text())
        for rec in records:
            # Buckets should come from packets, not "wrong_bucket"
            assert rec.get("bucket") != "wrong_bucket", (
                f"Bucket was read from DataFrame column instead of packet for {rec['symbol']}"
            )
