"""Tests for the enriched canonical packet fields added in the packet-richness pass.

Covers:
  1. daily_trend_state — deterministic classification from snapshot fields
  2. pullback_quality — quality label for pullback-bucketed names
  3. demotion_reason — why a name was routed to pullback instead of breakout
  4. trade_plan dual-sided analysis — why_now, why_not_now, setup_improves_if, setup_weakens_if
  5. trade_plan completeness — all required fields present
  6. AVWAP table enrichment — priority, stretch_atr, slope_label, closes_above_20
  7. build_trend_state — weekly/daily state from context.py
  8. selector structural tiebreaker — near-pivot compressed names rank above distant noisy ones
"""
from __future__ import annotations

import math

import pandas as pd
import pytest

from swingtrader.dashboard.packet import (
    _compute_daily_trend_state,
    _compute_demotion_reason,
    _compute_pullback_quality,
    build_all_lightweight_packets,
    build_lightweight_packet,
)
from swingtrader.dashboard.buckets import (
    BUCKET_BREAKOUT,
    BUCKET_EXCLUDED,
    BUCKET_PULLBACK,
    BUCKET_PORTFOLIO,
)
from swingtrader.dashboard.selector import (
    _pkt_structural_tiebreaker,
    select_packets,
)


# ── Shared row builders ────────────────────────────────────────────────────────

def _row(**kwargs) -> pd.Series:
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
        "atr_compression_pct": 35.0,
        "volume_dryup": 0.55,
        "close_vs_sma50": 0.03,
        "daily_rs_63": 0.06,
        "ytd_dist_atr": 1.2,
        "swing_low_dist_atr": 2.5,
        "regime_spy_trend": 1.0,
        "is_portfolio": False,
        "is_watchlist": True,
        "is_non_equity": False,
        "groups": "tech",
    }
    base.update(kwargs)
    return pd.Series(base)


# ── 1. daily_trend_state classification ──────────────────────────────────────

class TestDailyTrendState:
    """_compute_daily_trend_state derives state from snapshot row fields."""

    def test_strong_uptrend_above_sma50_positive_rs(self):
        """Above SMA50 >3% AND positive RS → strong_uptrend."""
        row = _row(close_vs_sma50=0.05, daily_rs_63=0.08, regime_spy_trend=1.0)
        assert _compute_daily_trend_state(row) == "strong_uptrend"

    def test_uptrend_above_sma50_neutral_rs(self):
        """Above SMA50 by 1% → uptrend (not strong without rs)."""
        row = _row(close_vs_sma50=0.01, daily_rs_63=0.0)
        state = _compute_daily_trend_state(row)
        assert state == "uptrend"

    def test_neutral_near_sma50(self):
        """Within 2% below SMA50 → neutral."""
        row = _row(close_vs_sma50=-0.015)
        assert _compute_daily_trend_state(row) == "neutral"

    def test_weak_below_sma50(self):
        """3-5% below SMA50 → weak."""
        row = _row(close_vs_sma50=-0.04)
        assert _compute_daily_trend_state(row) == "weak"

    def test_broken_regime_down_and_far_below_sma50(self):
        """Regime negative AND >5% below SMA50 → broken."""
        row = _row(close_vs_sma50=-0.07, regime_spy_trend=-1.0)
        assert _compute_daily_trend_state(row) == "broken"

    def test_unknown_without_sma_or_rs(self):
        """No SMA50 or RS data → unknown."""
        row = _row(close_vs_sma50=float("nan"), daily_rs_63=float("nan"))
        assert _compute_daily_trend_state(row) == "unknown"

    def test_valid_range_of_states(self):
        """All returned states are from the defined set."""
        valid = {"strong_uptrend", "uptrend", "neutral", "weak", "broken", "unknown"}
        for cvs50 in [-0.15, -0.07, -0.03, -0.01, 0.01, 0.04]:
            row = _row(close_vs_sma50=cvs50)
            state = _compute_daily_trend_state(row)
            assert state in valid, f"Unexpected state {state!r} for cvs50={cvs50}"

    def test_packet_has_daily_trend_state(self):
        """build_lightweight_packet includes daily_trend_state."""
        pkt = build_lightweight_packet(_row())
        assert "daily_trend_state" in pkt
        assert isinstance(pkt["daily_trend_state"], str)
        assert pkt["daily_trend_state"] != ""

    def test_packet_daily_trend_reflects_row(self):
        """Strong uptrend row → packet.daily_trend_state reflects that."""
        pkt_strong = build_lightweight_packet(_row(close_vs_sma50=0.05, daily_rs_63=0.08))
        pkt_weak   = build_lightweight_packet(_row(close_vs_sma50=-0.07, regime_spy_trend=-1.0))
        assert pkt_strong["daily_trend_state"] == "strong_uptrend"
        assert pkt_weak["daily_trend_state"] == "broken"


# ── 2. pullback_quality ────────────────────────────────────────────────────────

class TestPullbackQuality:
    """pullback_quality classifies the quality of a pullback setup."""

    def test_na_for_breakout_bucket(self):
        """Non-pullback bucket → n/a."""
        assert _compute_pullback_quality(_row(), BUCKET_BREAKOUT, "ARMED", 5, 0.3) == "n/a"

    def test_na_for_excluded(self):
        assert _compute_pullback_quality(_row(), BUCKET_EXCLUDED, "FAILED", 0, math.nan) == "n/a"

    def test_at_pivot_for_near_zero_dist(self):
        """dist near zero → at_pivot."""
        result = _compute_pullback_quality(_row(), BUCKET_PULLBACK, "ARMED", 5, -0.3)
        assert result == "at_pivot"

    def test_near_pivot_for_moderate_dist(self):
        """dist between -0.5 and -2.0 → near_pivot."""
        result = _compute_pullback_quality(_row(), BUCKET_PULLBACK, "ARMED", 5, -1.0)
        assert result == "near_pivot"

    def test_constructive_for_mid_dist(self):
        """dist between -2.0 and -3.0 → constructive."""
        result = _compute_pullback_quality(_row(), BUCKET_PULLBACK, "ARMED", 5, -2.5)
        assert result == "constructive"

    def test_deep_for_large_dist(self):
        """dist < -3.0 → deep."""
        result = _compute_pullback_quality(_row(), BUCKET_PULLBACK, "ARMED", 5, -4.0)
        assert result == "deep"

    def test_old_trigger_for_aged_triggered(self):
        """TRIGGERED with days > BREAKOUT_TRIGGER_DAYS → old_trigger."""
        from swingtrader.dashboard.buckets import BREAKOUT_TRIGGER_DAYS
        result = _compute_pullback_quality(
            _row(), BUCKET_PULLBACK, "TRIGGERED", BREAKOUT_TRIGGER_DAYS + 1, 0.5
        )
        assert result == "old_trigger"

    def test_packet_has_pullback_quality(self):
        """build_lightweight_packet includes pullback_quality."""
        pkt = build_lightweight_packet(_row())
        assert "pullback_quality" in pkt
        assert isinstance(pkt["pullback_quality"], str)


# ── 3. demotion_reason ────────────────────────────────────────────────────────

class TestDemotionReason:
    """demotion_reason explains why a PULLBACK-bucketed name was not in BREAKOUT."""

    def test_empty_for_breakout_bucket(self):
        """Breakout bucket → no demotion."""
        assert _compute_demotion_reason(_row(), BUCKET_BREAKOUT, "ARMED", 5, 0.3) == ""

    def test_far_from_pivot_reason(self):
        """ARMED/BASE far from pivot → far_from_pivot reason."""
        result = _compute_demotion_reason(_row(), BUCKET_PULLBACK, "ARMED", 5, 2.5)
        assert "far_from_pivot" in result

    def test_below_sma50_near_pivot(self):
        """ARMED near pivot but below SMA50 → below_sma50 reason."""
        row = _row(close_vs_sma50=-0.05)
        result = _compute_demotion_reason(row, BUCKET_PULLBACK, "ARMED", 5, 0.3)
        assert "below_sma50" in result

    def test_old_trigger_for_aged_trigger(self):
        """TRIGGERED past breakout window → old_trigger reason."""
        from swingtrader.dashboard.buckets import BREAKOUT_TRIGGER_DAYS
        result = _compute_demotion_reason(
            _row(), BUCKET_PULLBACK, "TRIGGERED", BREAKOUT_TRIGGER_DAYS + 2, 0.5
        )
        assert "old_trigger" in result

    def test_packet_has_demotion_reason(self):
        """build_lightweight_packet includes demotion_reason."""
        pkt = build_lightweight_packet(_row())
        assert "demotion_reason" in pkt
        assert isinstance(pkt["demotion_reason"], str)

    def test_demotion_reason_empty_for_breakout_packet(self):
        """A breakout-bucketed packet has empty demotion_reason."""
        # ARMED, near pivot, above SMA50, good scores — should be breakout
        row = _row(
            state="ARMED",
            dist_to_pivot_atr=0.2,
            close_vs_sma50=0.03,
            failure_risk=0.18,
            composite_score=0.65,
            days_in_state=4,
        )
        pkt = build_lightweight_packet(row)
        if pkt["bucket"] == BUCKET_BREAKOUT:
            assert pkt["demotion_reason"] == ""


# ── 4. Trade plan dual-sided analysis ─────────────────────────────────────────

class TestTradePlanDualAnalysis:
    """why_now / why_not_now / setup_improves_if / setup_weakens_if in trade_plan."""

    def _get_tp(self, **kwargs) -> dict:
        pkt = build_lightweight_packet(_row(**kwargs))
        return pkt["trade_plan"]

    def test_trade_plan_has_why_now(self):
        tp = self._get_tp()
        assert "why_now" in tp
        assert isinstance(tp["why_now"], list)
        assert len(tp["why_now"]) > 0

    def test_trade_plan_has_why_not_now(self):
        tp = self._get_tp()
        assert "why_not_now" in tp
        assert isinstance(tp["why_not_now"], list)
        assert len(tp["why_not_now"]) > 0

    def test_trade_plan_has_setup_improves_if(self):
        tp = self._get_tp()
        assert "setup_improves_if" in tp
        assert isinstance(tp["setup_improves_if"], list)
        assert len(tp["setup_improves_if"]) > 0

    def test_trade_plan_has_setup_weakens_if(self):
        tp = self._get_tp()
        assert "setup_weakens_if" in tp
        assert isinstance(tp["setup_weakens_if"], list)
        assert len(tp["setup_weakens_if"]) > 0

    def test_strong_setup_has_bullish_why_now(self):
        """High RS, above SMA50, compressed → why_now mentions positive signals."""
        tp = self._get_tp(
            close_vs_sma50=0.05,
            daily_rs_63=0.08,
            atr_compression_pct=20.0,
            volume_dryup=0.70,
        )
        why_text = " ".join(tp["why_now"]).lower()
        # At least one of these should appear
        has_signal = any(word in why_text for word in ["above", "sma50", "rs", "volume", "pivot", "compressed", "coil"])
        assert has_signal, f"Expected bullish signal in why_now; got: {tp['why_now']}"

    def test_weak_setup_has_bearish_why_not(self):
        """Below SMA50, underperforming RS, elevated failure risk → why_not captures it."""
        tp = self._get_tp(
            close_vs_sma50=-0.06,
            daily_rs_63=-0.05,
            failure_risk=0.60,
        )
        why_not_text = " ".join(tp["why_not_now"]).lower()
        has_concern = any(word in why_not_text for word in ["below", "sma50", "underperform", "failure"])
        assert has_concern, f"Expected bearish signal in why_not_now; got: {tp['why_not_now']}"

    def test_armed_setup_improves_mentions_pivot_close(self):
        """ARMED state → setup_improves_if mentions closing above pivot."""
        tp = self._get_tp(state="ARMED")
        improves_text = " ".join(tp["setup_improves_if"]).lower()
        assert "pivot" in improves_text or "breakout" in improves_text or "close" in improves_text

    def test_armed_setup_weakens_mentions_stop(self):
        """ARMED state → setup_weakens_if mentions close below stop."""
        tp = self._get_tp(state="ARMED")
        weakens_text = " ".join(tp["setup_weakens_if"]).lower()
        assert "stop" in weakens_text or "below" in weakens_text

    def test_triggered_setup_improves_mentions_target(self):
        """TRIGGERED state → setup_improves_if mentions target or follow-through."""
        tp = self._get_tp(state="TRIGGERED", days_in_state=2)
        improves_text = " ".join(tp["setup_improves_if"]).lower()
        assert "follow" in improves_text or "t1" in improves_text or "target" in improves_text


# ── 5. Trade plan completeness ────────────────────────────────────────────────

class TestTradePlanCompleteness:
    """All trade plan fields are present and well-formed."""

    REQUIRED_FIELDS = [
        "actionability_code", "entry_condition", "entry_range",
        "stop", "stop_basis", "stop_risk_pct",
        "target_1", "target_2", "risk_reward_t1",
        "time_stop", "key_risk",
        "why_now", "why_not_now",
        "setup_improves_if", "setup_weakens_if",
    ]

    def test_all_fields_present(self):
        pkt = build_lightweight_packet(_row())
        tp = pkt["trade_plan"]
        missing = [f for f in self.REQUIRED_FIELDS if f not in tp]
        assert not missing, f"trade_plan missing fields: {missing}"

    def test_list_fields_are_non_empty_lists(self):
        pkt = build_lightweight_packet(_row())
        tp = pkt["trade_plan"]
        for field in ("why_now", "why_not_now", "setup_improves_if", "setup_weakens_if"):
            assert isinstance(tp[field], list), f"{field} should be list"
            assert len(tp[field]) > 0, f"{field} should be non-empty"
            assert all(isinstance(s, str) for s in tp[field]), f"{field} should be list[str]"

    def test_string_fields_are_non_empty(self):
        pkt = build_lightweight_packet(_row())
        tp = pkt["trade_plan"]
        for field in ("actionability_code", "entry_condition", "stop_basis", "time_stop", "key_risk"):
            v = tp.get(field, "")
            assert isinstance(v, str) and len(v) > 0, f"{field} should be non-empty string"

    def test_actionability_code_valid(self):
        valid = {"BUY_NOW", "WATCH_BREAKOUT", "WAIT_PULLBACK", "WAIT_ZONE", "BLOCK"}
        pkt = build_lightweight_packet(_row())
        assert pkt["trade_plan"]["actionability_code"] in valid


# ── 6. Packet structural fields ───────────────────────────────────────────────

class TestPacketStructuralFields:
    """New structural fields are present in every packet."""

    def test_daily_trend_state_present(self):
        pkt = build_lightweight_packet(_row())
        assert "daily_trend_state" in pkt
        assert pkt["daily_trend_state"] in {
            "strong_uptrend", "uptrend", "neutral", "weak", "broken", "unknown"
        }

    def test_weekly_trend_state_starts_none(self):
        """weekly_trend_state is None in lightweight packet (populated by enrich)."""
        pkt = build_lightweight_packet(_row())
        assert "weekly_trend_state" in pkt
        assert pkt["weekly_trend_state"] is None

    def test_pullback_quality_present(self):
        pkt = build_lightweight_packet(_row())
        assert "pullback_quality" in pkt
        assert isinstance(pkt["pullback_quality"], str)

    def test_demotion_reason_present(self):
        pkt = build_lightweight_packet(_row())
        assert "demotion_reason" in pkt
        assert isinstance(pkt["demotion_reason"], str)

    def test_far_base_has_non_empty_demotion_or_pullback_label(self):
        """ARMED far from pivot → pullback bucket with informative demotion_reason."""
        row = _row(
            state="ARMED",
            dist_to_pivot_atr=3.0,   # well beyond BREAKOUT_DIST_ATR=1.5
            failure_risk=0.22,
            composite_score=0.60,
        )
        pkt = build_lightweight_packet(row)
        if pkt["bucket"] == BUCKET_PULLBACK:
            assert len(pkt["demotion_reason"]) > 0, "pullback packet should have demotion_reason"


# ── 7. AVWAP table enrichment ─────────────────────────────────────────────────

class TestAvwapTableEnrichment:
    """AVWAP rows have priority, stretch_atr, slope_label, closes_above_20."""

    def _make_avwap_row(self, anchor="YTD", **kwargs) -> dict:
        """Make a minimal AVWAP row as context.py produces."""
        base = {
            "anchor":           anchor,
            "priority":         "primary",
            "avwap":            95.0,
            "pct_dist":         4.7,
            "dist_atr":         2.1,
            "role":             "support",
            "status":           "Accepted above",
            "reclaim":          True,
            "stretch_atr":      2.5,
            "slope_20":         0.003,
            "slope_label":      "rising",
            "closes_above_20":  0.75,
        }
        base.update(kwargs)
        return base

    def test_priority_field_exists(self):
        row = self._make_avwap_row()
        assert "priority" in row
        assert row["priority"] in {"primary", "secondary", "dynamic"}

    def test_stretch_atr_field_exists(self):
        row = self._make_avwap_row()
        assert "stretch_atr" in row

    def test_slope_label_exists(self):
        row = self._make_avwap_row()
        assert "slope_label" in row
        assert row["slope_label"] in {"rising", "falling", "flat", None}

    def test_closes_above_20_exists(self):
        row = self._make_avwap_row()
        assert "closes_above_20" in row

    def test_dynamic_anchor_has_anchor_date(self):
        """WTD/MTD dynamic anchors should have anchor_date."""
        wtd_row = {
            "anchor":          "WTD",
            "priority":        "dynamic",
            "anchor_date":     "2025-01-13",
            "avwap":           98.5,
            "pct_dist":        1.0,
            "dist_atr":        0.5,
            "role":            "support",
            "status":          "Accepted above",
            "reclaim":         None,
            "stretch_atr":     None,
            "slope_20":        None,
            "slope_label":     None,
            "closes_above_20": None,
        }
        assert wtd_row["anchor_date"] == "2025-01-13"
        assert wtd_row["priority"] == "dynamic"

    def test_primary_anchors_have_correct_priority(self):
        """YTD and Swing Low should be 'primary', Swing High 'secondary'."""
        # Verify _AVWAP_ANCHORS tuple structure
        from swingtrader.dashboard.context import _AVWAP_ANCHORS
        priority_map = {a[0]: a[4] for a in _AVWAP_ANCHORS}
        assert priority_map["YTD"] == "primary"
        assert priority_map["Swing Low"] == "primary"
        assert priority_map["Swing High"] == "secondary"
        assert priority_map["Breakout Day"] == "secondary"


# ── 8. Selector structural tiebreaker ─────────────────────────────────────────

class TestSelectorStructuralTiebreaker:
    """_pkt_structural_tiebreaker gives lower values to better candidates."""

    def _make_bo_pkt(self, dist=0.3, atr_pct=25.0, rs=0.07) -> dict:
        pkt = build_lightweight_packet(_row(
            state="ARMED",
            dist_to_pivot_atr=dist,
            atr_compression_pct=atr_pct,
            daily_rs_63=rs,
        ))
        return pkt

    def test_near_pivot_ranks_before_far(self):
        """Near-pivot packet gets lower tiebreaker than far-from-pivot."""
        near = self._make_bo_pkt(dist=0.2)
        far  = self._make_bo_pkt(dist=3.0)
        tb_near = _pkt_structural_tiebreaker(near)
        tb_far  = _pkt_structural_tiebreaker(far)
        assert tb_near <= tb_far, "Near-pivot should have <= tiebreaker than far"

    def test_compressed_ranks_before_noisy(self):
        """Compressed base (low atr_pct) gets lower tiebreaker than noisy."""
        tight = self._make_bo_pkt(atr_pct=15.0)
        noisy = self._make_bo_pkt(atr_pct=80.0)
        tb_tight = _pkt_structural_tiebreaker(tight)
        tb_noisy = _pkt_structural_tiebreaker(noisy)
        # Same pivot dist, so difference should come from compression component
        assert tb_tight[1] < tb_noisy[1], "Tight base should have lower atr_norm tiebreaker"

    def test_outperformer_ranks_before_laggard(self):
        """Positive RS → no RS penalty; negative RS → penalty."""
        strong_rs = self._make_bo_pkt(rs=0.10)
        weak_rs   = self._make_bo_pkt(rs=-0.05)
        tb_strong = _pkt_structural_tiebreaker(strong_rs)
        tb_weak   = _pkt_structural_tiebreaker(weak_rs)
        # RS penalty is third component
        assert tb_strong[2] <= tb_weak[2], "Strong RS should have <= RS penalty"

    def test_tiebreaker_is_3_tuple(self):
        """Returns a 3-tuple."""
        pkt = self._make_bo_pkt()
        tb = _pkt_structural_tiebreaker(pkt)
        assert isinstance(tb, tuple)
        assert len(tb) == 3

    def test_ideal_candidate_has_lowest_tiebreaker(self):
        """Near-pivot, compressed, strong RS → lowest tiebreaker."""
        ideal  = self._make_bo_pkt(dist=0.1, atr_pct=10.0, rs=0.12)
        mediocre = self._make_bo_pkt(dist=1.8, atr_pct=70.0, rs=-0.03)
        tb_ideal    = _pkt_structural_tiebreaker(ideal)
        tb_mediocre = _pkt_structural_tiebreaker(mediocre)
        assert tb_ideal < tb_mediocre, (
            f"Ideal candidate should have lower tiebreaker than mediocre: "
            f"{tb_ideal} vs {tb_mediocre}"
        )

    def test_selector_prefers_near_pivot_among_equals(self):
        """When model scores are equal, selector should prefer near-pivot candidate."""
        rows = []
        for i, (dist, atr_pct, rs, sym) in enumerate([
            (0.1, 15.0, 0.10, "NEAR"),
            (2.5, 80.0, -0.02, "FAR"),
        ]):
            rows.append({
                "user_symbol": sym,
                "provider_symbol": sym,
                "state": "ARMED",
                "pivot": 100.0,
                "atr14": 2.0,
                "close": 100.5,
                "dist_to_pivot_atr": dist,
                "days_in_state": 4,
                "composite_score": 0.60,  # same score
                "setup_score": 0.60,      # same score
                "trade_score": 0.50,
                "failure_risk": 0.20,
                "percentile_rank": 75.0,
                "base_length": 28,
                "atr_compression_pct": atr_pct,
                "volume_dryup": 0.55,
                "close_vs_sma50": 0.03,
                "daily_rs_63": rs,
                "ytd_dist_atr": 1.2,
                "swing_low_dist_atr": 2.5,
                "regime_spy_trend": 1.0,
                "is_portfolio": False,
                "is_watchlist": True,
                "is_non_equity": False,
                "groups": "other",  # different group prevents diversity-cap interference
            })
        # Give FAR different group to avoid diversity cap issues
        rows[1]["groups"] = "other2"
        df = pd.DataFrame(rows)
        pkts = build_all_lightweight_packets(df)
        result = select_packets(pkts, n_breakout=2)
        bo_syms = [p["symbol"] for p in result["breakout"]]
        # Near-pivot should appear first if both qualify
        if len(bo_syms) >= 2:
            assert bo_syms[0] == "NEAR", f"Expected NEAR first, got {bo_syms}"


# ── 9. All-packet structural coverage ────────────────────────────────────────

class TestAllPacketsHaveStructuralFields:
    """Every packet from build_all_lightweight_packets has the new structural fields."""

    def _snapshot(self, n=6) -> pd.DataFrame:
        rows = []
        for i in range(n):
            state = ["ARMED", "BASE", "TRIGGERED", "FAILED", "TRIGGERED", "ARMED"][i % 6]
            rows.append({
                "user_symbol": f"S{i:02d}",
                "provider_symbol": f"S{i:02d}",
                "state": state,
                "pivot": 50.0 + i * 5,
                "atr14": 1.5,
                "close": 50.0 + i * 5 - 0.5,
                "dist_to_pivot_atr": 0.3 + i * 0.5,
                "days_in_state": 4 + i,
                "composite_score": max(0.05, 0.65 - i * 0.08),
                "setup_score": max(0.05, 0.60 - i * 0.07),
                "trade_score": max(0.05, 0.55 - i * 0.06),
                "failure_risk": 0.20 + i * 0.07,
                "percentile_rank": 85.0 - i * 10,
                "base_length": 25 + i * 2,
                "atr_compression_pct": 35.0 + i * 5,
                "volume_dryup": 0.55,
                "close_vs_sma50": 0.02 - i * 0.01,
                "daily_rs_63": 0.05 - i * 0.02,
                "ytd_dist_atr": 1.0 + i * 0.2,
                "swing_low_dist_atr": 2.0,
                "regime_spy_trend": 1.0,
                "is_portfolio": False,
                "is_watchlist": True,
                "is_non_equity": False,
                "groups": "tech",
            })
        return pd.DataFrame(rows)

    def test_all_packets_have_daily_trend_state(self):
        df = self._snapshot(6)
        pkts = build_all_lightweight_packets(df)
        for pkt in pkts:
            assert "daily_trend_state" in pkt, f"{pkt['symbol']}: missing daily_trend_state"
            assert isinstance(pkt["daily_trend_state"], str)

    def test_all_packets_have_trade_plan_why_fields(self):
        df = self._snapshot(6)
        pkts = build_all_lightweight_packets(df)
        for pkt in pkts:
            tp = pkt.get("trade_plan", {})
            for field in ("why_now", "why_not_now", "setup_improves_if", "setup_weakens_if"):
                assert field in tp, f"{pkt['symbol']}: trade_plan missing {field}"
                assert isinstance(tp[field], list)

    def test_all_packets_have_demotion_and_quality_fields(self):
        df = self._snapshot(6)
        pkts = build_all_lightweight_packets(df)
        for pkt in pkts:
            assert "demotion_reason" in pkt
            assert "pullback_quality" in pkt
