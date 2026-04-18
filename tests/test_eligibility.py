"""Tests for eligibility gates, bucket assignment, and selector changes.

Covers:
  - assess_eligibility() gate firing
  - add_eligibility_columns() DataFrame enrichment
  - assign_bucket() bucket routing
  - select_breakout_candidates() / select_pullback_candidates() / select_portfolio_holdings()
  - select_top_setups() combined logic with eligibility
  - Regime-adjusted action labels (assign_action with regime_spy_trend)
  - Portfolio holdings excluded from fresh entry lists
  - Stale names excluded from breakout bucket
"""
from __future__ import annotations

import contextlib
import math

import pandas as pd
import pytest

from swingtrader.dashboard.action import (
    ACTION_AVOID,
    ACTION_BREAKOUT,
    ACTION_NOW,
    ACTION_PULLBACK,
    assign_action,
)
from swingtrader.dashboard.buckets import (
    BUCKET_BREAKOUT,
    BUCKET_EXCLUDED,
    BUCKET_EXTENDED,
    BUCKET_NON_EQUITY,
    BUCKET_PORTFOLIO,
    BUCKET_PULLBACK,
    add_bucket_column,
    assign_bucket,
    bucket_counts,
)
from swingtrader.dashboard.eligibility import (
    BROKEN_TREND_SMA200,
    FAILURE_RISK_HARD,
    GATE_BROKEN_TREND,
    GATE_HIGH_FAILURE_RISK,
    GATE_INVALID_STATE,
    GATE_LOW_SCORE,
    GATE_NON_EQUITY,
    GATE_POOR_RS,
    GATE_THIN_BASE,
    GATE_WEAK_REGIME_POOR_RS,
    POOR_RS_THRESHOLD,
    SCORE_FLOOR,
    WARN_AGING_BASE,
    WARN_BELOW_SMA50,
    add_eligibility_columns,
    assess_eligibility,
)
from swingtrader.dashboard.selector import (
    select_breakout_candidates,
    select_portfolio_holdings,
    select_pullback_candidates,
    select_top_setups,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _row(**kwargs) -> pd.Series:
    """Build a minimal valid snapshot row that passes all gates by default."""
    defaults = {
        "symbol": "TST",
        "user_symbol": "TST",
        "provider_symbol": "TST",
        "state": "ARMED",
        "composite_score": 0.45,
        "setup_score": 0.50,
        "trade_score": math.nan,
        "failure_risk": 0.25,
        "dist_to_pivot_atr": 0.5,    # near pivot → breakout bucket
        "days_in_state": 5,
        "base_length": 12,
        "is_portfolio": False,
        "is_non_equity": False,
        "is_fresh": True,
        "is_extended": False,
        "is_actionable": True,
        "action_label": ACTION_BREAKOUT,
        "eligible": True,
        "rejection_reasons": "",
        "eligibility_warnings": "",
        "bucket": BUCKET_BREAKOUT,
        "close_vs_sma200": 0.05,      # above 200 SMA
        "close_vs_sma50": 0.03,       # above 50 SMA
        "daily_rs_63": 0.08,          # outperforming SPY
        "regime_spy_trend": 1.0,      # uptrend
        "groups": "Technology",
        "is_watchlist": True,
        "percentile_rank": 75.0,
    }
    defaults.update(kwargs)
    return pd.Series(defaults)


def _df(*rows) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


# ===========================================================================
# Eligibility gate tests
# ===========================================================================

class TestEligibilityGates:

    def test_valid_row_is_eligible(self):
        result = assess_eligibility(_row())
        assert result["eligible"] is True
        assert result["rejection_reasons"] == []

    def test_non_equity_rejected(self):
        result = assess_eligibility(_row(is_non_equity=True))
        assert result["eligible"] is False
        assert GATE_NON_EQUITY in result["rejection_reasons"]

    def test_non_equity_short_circuits_other_gates(self):
        """Non-equity rejection returns early without checking other gates."""
        result = assess_eligibility(_row(
            is_non_equity=True,
            composite_score=0.01,   # would also trigger LOW_SCORE
            failure_risk=0.99,       # would also trigger HIGH_FAILURE
        ))
        assert result["rejection_reasons"] == [GATE_NON_EQUITY]

    def test_invalid_state_rejected(self):
        result = assess_eligibility(_row(state="NONE"))
        assert result["eligible"] is False
        assert GATE_INVALID_STATE in result["rejection_reasons"]

    def test_failed_state_rejected(self):
        result = assess_eligibility(_row(state="FAILED"))
        assert result["eligible"] is False
        assert GATE_INVALID_STATE in result["rejection_reasons"]

    def test_confirmed_state_rejected(self):
        """CONFIRMED is not a scored state — should be rejected."""
        result = assess_eligibility(_row(state="CONFIRMED"))
        assert result["eligible"] is False
        assert GATE_INVALID_STATE in result["rejection_reasons"]

    def test_broken_trend_rejected(self):
        """Price significantly below 200 SMA → broken trend."""
        result = assess_eligibility(_row(close_vs_sma200=BROKEN_TREND_SMA200 - 0.02))
        assert result["eligible"] is False
        assert GATE_BROKEN_TREND in result["rejection_reasons"]

    def test_slightly_below_200sma_passes(self):
        """Just below threshold should NOT trigger the gate."""
        result = assess_eligibility(_row(close_vs_sma200=BROKEN_TREND_SMA200 + 0.01))
        assert GATE_BROKEN_TREND not in result["rejection_reasons"]

    def test_missing_close_vs_sma200_passes_gate(self):
        """If field is missing/NaN, gate should not fire (graceful)."""
        result = assess_eligibility(_row(close_vs_sma200=math.nan))
        assert GATE_BROKEN_TREND not in result["rejection_reasons"]

    def test_poor_rs_rejected(self):
        result = assess_eligibility(_row(daily_rs_63=POOR_RS_THRESHOLD - 0.02))
        assert result["eligible"] is False
        assert GATE_POOR_RS in result["rejection_reasons"]

    def test_marginally_acceptable_rs_passes(self):
        result = assess_eligibility(_row(daily_rs_63=POOR_RS_THRESHOLD + 0.01))
        assert GATE_POOR_RS not in result["rejection_reasons"]

    def test_weak_regime_and_poor_rs_rejected(self):
        """Market downtrend + underperforming stock = rejected."""
        result = assess_eligibility(_row(regime_spy_trend=-1.0, daily_rs_63=-0.02))
        assert result["eligible"] is False
        assert GATE_WEAK_REGIME_POOR_RS in result["rejection_reasons"]

    def test_weak_regime_but_strong_rs_passes(self):
        """Market downtrend but stock outperforming = should not fire weak-regime gate."""
        result = assess_eligibility(_row(regime_spy_trend=-1.0, daily_rs_63=0.05))
        assert GATE_WEAK_REGIME_POOR_RS not in result["rejection_reasons"]

    def test_high_failure_risk_rejected(self):
        result = assess_eligibility(_row(failure_risk=FAILURE_RISK_HARD + 0.05))
        assert result["eligible"] is False
        assert GATE_HIGH_FAILURE_RISK in result["rejection_reasons"]

    def test_failure_risk_at_ceiling_passes(self):
        result = assess_eligibility(_row(failure_risk=FAILURE_RISK_HARD - 0.01))
        assert GATE_HIGH_FAILURE_RISK not in result["rejection_reasons"]

    def test_low_score_rejected(self):
        result = assess_eligibility(_row(composite_score=SCORE_FLOOR - 0.02))
        assert result["eligible"] is False
        assert GATE_LOW_SCORE in result["rejection_reasons"]

    def test_missing_score_does_not_reject(self):
        """NaN composite_score should not trigger LOW_SCORE gate."""
        result = assess_eligibility(_row(composite_score=math.nan))
        assert GATE_LOW_SCORE not in result["rejection_reasons"]

    def test_thin_base_rejected(self):
        result = assess_eligibility(_row(state="BASE", base_length=3))
        assert result["eligible"] is False
        assert GATE_THIN_BASE in result["rejection_reasons"]

    def test_thin_base_for_armed_rejected(self):
        result = assess_eligibility(_row(state="ARMED", base_length=2))
        assert GATE_THIN_BASE in result["rejection_reasons"]

    def test_adequate_base_passes_thin_gate(self):
        result = assess_eligibility(_row(state="BASE", base_length=10))
        assert GATE_THIN_BASE not in result["rejection_reasons"]

    def test_triggered_state_not_subject_to_thin_base(self):
        """Thin base gate only applies to BASE/ARMED states."""
        result = assess_eligibility(_row(state="TRIGGERED", base_length=2))
        assert GATE_THIN_BASE not in result["rejection_reasons"]

    def test_multiple_rejections_accumulated(self):
        """Multiple gates can fire simultaneously."""
        result = assess_eligibility(_row(
            composite_score=0.01,   # LOW_SCORE
            failure_risk=0.90,      # HIGH_FAILURE_RISK
            daily_rs_63=-0.20,      # POOR_RS
        ))
        assert result["eligible"] is False
        assert len(result["rejection_reasons"]) >= 2

    def test_warning_below_sma50(self):
        result = assess_eligibility(_row(close_vs_sma50=-0.04))
        assert result["eligible"] is True          # warning, not rejection
        assert WARN_BELOW_SMA50 in result["warnings"]

    def test_warning_aging_base(self):
        result = assess_eligibility(_row(state="BASE", days_in_state=50))
        assert result["eligible"] is True
        assert WARN_AGING_BASE in result["warnings"]

    def test_warnings_do_not_affect_eligibility(self):
        result = assess_eligibility(_row(
            close_vs_sma50=-0.05,    # WARN_BELOW_SMA50
            state="BASE",
            days_in_state=50,        # WARN_AGING_BASE
        ))
        assert result["eligible"] is True


class TestAddEligibilityColumns:

    def test_adds_required_columns(self):
        df = _df(_row(symbol="A"), _row(symbol="B"))
        result = add_eligibility_columns(df)
        assert "eligible" in result.columns
        assert "rejection_reasons" in result.columns
        assert "eligibility_warnings" in result.columns

    def test_valid_rows_are_eligible(self):
        df = _df(_row(symbol="A"), _row(symbol="B"))
        result = add_eligibility_columns(df)
        assert result["eligible"].all()

    def test_ineligible_row_marked(self):
        df = _df(_row(symbol="A"), _row(symbol="CASH", is_non_equity=True))
        result = add_eligibility_columns(df)
        assert result[result["symbol"] == "A"]["eligible"].iloc[0] == True  # noqa: E712
        assert result[result["symbol"] == "CASH"]["eligible"].iloc[0] == False  # noqa: E712

    def test_empty_df_handled(self):
        result = add_eligibility_columns(pd.DataFrame())
        assert result.empty

    def test_rejection_reasons_populated(self):
        df = _df(_row(symbol="WEAK", composite_score=0.01, failure_risk=0.90))
        result = add_eligibility_columns(df)
        reasons = result["rejection_reasons"].iloc[0]
        assert len(reasons) > 0


# ===========================================================================
# Bucket assignment tests
# ===========================================================================

class TestBucketAssignment:

    def test_non_equity_gets_non_equity_bucket(self):
        assert assign_bucket(_row(is_non_equity=True)) == BUCKET_NON_EQUITY

    def test_portfolio_gets_portfolio_bucket(self):
        assert assign_bucket(_row(is_portfolio=True)) == BUCKET_PORTFOLIO

    def test_ineligible_gets_excluded_bucket(self):
        row = _row(eligible=False, state="ARMED")
        assert assign_bucket(row) == BUCKET_EXCLUDED

    def test_extended_gets_extended_bucket(self):
        row = _row(is_extended=True, eligible=True)
        assert assign_bucket(row) == BUCKET_EXTENDED

    def test_late_state_gets_extended_bucket(self):
        row = _row(state="LATE", eligible=True)
        assert assign_bucket(row) == BUCKET_EXTENDED

    def test_armed_near_pivot_fresh_gets_breakout_bucket(self):
        row = _row(state="ARMED", dist_to_pivot_atr=0.8, is_fresh=True, eligible=True)
        assert assign_bucket(row) == BUCKET_BREAKOUT

    def test_base_near_pivot_fresh_gets_breakout_bucket(self):
        row = _row(state="BASE", dist_to_pivot_atr=1.2, is_fresh=True, eligible=True)
        assert assign_bucket(row) == BUCKET_BREAKOUT

    def test_armed_far_from_pivot_gets_pullback_bucket(self):
        row = _row(state="ARMED", dist_to_pivot_atr=-3.0, is_fresh=True, eligible=True)
        assert assign_bucket(row) == BUCKET_PULLBACK

    def test_base_stale_gets_pullback_bucket(self):
        row = _row(state="BASE", dist_to_pivot_atr=0.5, is_fresh=False, eligible=True)
        assert assign_bucket(row) == BUCKET_PULLBACK

    def test_triggered_early_gets_breakout_bucket(self):
        """Fresh TRIGGERED within BREAKOUT_TRIGGER_DAYS → breakout."""
        row = _row(state="TRIGGERED", days_in_state=3, is_fresh=True, eligible=True,
                   action_label=ACTION_NOW)
        assert assign_bucket(row) == BUCKET_BREAKOUT

    def test_triggered_late_gets_pullback_bucket(self):
        """TRIGGERED beyond breakout window → pullback."""
        row = _row(state="TRIGGERED", days_in_state=12, is_fresh=True, eligible=True,
                   action_label=ACTION_PULLBACK)
        assert assign_bucket(row) == BUCKET_PULLBACK

    def test_portfolio_overrides_state_logic(self):
        """Portfolio holdings always go to PORTFOLIO bucket regardless of state."""
        for state in ("ARMED", "TRIGGERED", "CONFIRMED", "BASE"):
            row = _row(is_portfolio=True, state=state, eligible=True)
            assert assign_bucket(row) == BUCKET_PORTFOLIO, \
                f"State {state} should be PORTFOLIO but got {assign_bucket(row)}"


class TestAddBucketColumn:

    def test_adds_bucket_column(self):
        df = _df(_row(symbol="A"), _row(symbol="B"))
        result = add_bucket_column(df)
        assert "bucket" in result.columns

    def test_empty_df_handled(self):
        result = add_bucket_column(pd.DataFrame())
        assert result.empty

    def test_bucket_counts_correct(self):
        df = _df(
            _row(symbol="A", state="ARMED", is_fresh=True, eligible=True, dist_to_pivot_atr=0.5),
            _row(symbol="B", is_portfolio=True, state="TRIGGERED"),
            _row(symbol="C", is_non_equity=True),
            _row(symbol="D", eligible=False, state="ARMED"),
        )
        df = add_bucket_column(df)
        counts = bucket_counts(df)
        assert counts["breakout_long"] >= 1
        assert counts["portfolio_hold"] >= 1
        assert counts["non_equity"] >= 1
        assert counts["excluded"] >= 1


# ===========================================================================
# Selector tests
# ===========================================================================

def _rich_df(n_breakout=3, n_pullback=2, n_portfolio=1, n_excluded=1) -> pd.DataFrame:
    """Build a comprehensive DataFrame with multiple buckets."""
    rows = []
    for i in range(n_breakout):
        rows.append(_row(
            symbol=f"BO{i:02d}",
            bucket=BUCKET_BREAKOUT,
            eligible=True,
            is_portfolio=False,
            is_fresh=True,
            action_label=ACTION_BREAKOUT,
            composite_score=0.5 + i * 0.02,
            percentile_rank=70 + i * 5,
        ))
    for i in range(n_pullback):
        rows.append(_row(
            symbol=f"PB{i:02d}",
            bucket=BUCKET_PULLBACK,
            eligible=True,
            is_portfolio=False,
            is_fresh=True,
            action_label=ACTION_PULLBACK,
            composite_score=0.35 + i * 0.02,
            percentile_rank=50 + i * 5,
        ))
    for i in range(n_portfolio):
        rows.append(_row(
            symbol=f"PORT{i:02d}",
            bucket=BUCKET_PORTFOLIO,
            eligible=True,
            is_portfolio=True,
            action_label=ACTION_NOW,
            composite_score=0.60,
        ))
    for i in range(n_excluded):
        rows.append(_row(
            symbol=f"EXCL{i:02d}",
            bucket=BUCKET_EXCLUDED,
            eligible=False,
            is_portfolio=False,
            action_label=ACTION_AVOID,
            rejection_reasons=GATE_LOW_SCORE,
        ))
    return pd.DataFrame(rows)


class TestSelectBreakoutCandidates:

    def test_returns_only_breakout_bucket(self):
        df = _rich_df()
        result = select_breakout_candidates(df)
        assert all(result["bucket"] == BUCKET_BREAKOUT)

    def test_excludes_portfolio(self):
        df = _rich_df()
        result = select_breakout_candidates(df)
        if "is_portfolio" in result.columns:
            assert not result["is_portfolio"].any()

    def test_excludes_ineligible(self):
        df = _rich_df()
        result = select_breakout_candidates(df)
        if "eligible" in result.columns:
            assert result["eligible"].all()

    def test_returns_at_most_n(self):
        df = _rich_df(n_breakout=10)
        result = select_breakout_candidates(df, n=5)
        assert len(result) <= 5

    def test_empty_df_returns_empty(self):
        result = select_breakout_candidates(pd.DataFrame())
        assert result.empty

    def test_ranked_by_composite_score(self):
        df = _rich_df(n_breakout=3)
        result = select_breakout_candidates(df)
        scores = result["composite_score"].tolist()
        # Should be sorted descending
        for i in range(len(scores) - 1):
            with contextlib.suppress(TypeError, ValueError):
                assert float(scores[i]) >= float(scores[i + 1]) - 0.001

    def test_no_breakout_bucket_returns_empty(self):
        df = _rich_df(n_breakout=0, n_pullback=3)
        result = select_breakout_candidates(df)
        assert result.empty


class TestSelectPullbackCandidates:

    def test_returns_only_pullback_bucket(self):
        df = _rich_df()
        result = select_pullback_candidates(df)
        assert all(result["bucket"] == BUCKET_PULLBACK)

    def test_excludes_portfolio(self):
        df = _rich_df()
        result = select_pullback_candidates(df)
        if "is_portfolio" in result.columns:
            assert not result["is_portfolio"].any()

    def test_returns_at_most_n(self):
        df = _rich_df(n_pullback=10)
        result = select_pullback_candidates(df, n=3)
        assert len(result) <= 3

    def test_empty_df_returns_empty(self):
        result = select_pullback_candidates(pd.DataFrame())
        assert result.empty


class TestSelectPortfolioHoldings:

    def test_returns_only_portfolio(self):
        df = _rich_df()
        result = select_portfolio_holdings(df)
        assert all(result["bucket"] == BUCKET_PORTFOLIO)

    def test_includes_all_portfolio_names(self):
        df = _rich_df(n_portfolio=3)
        result = select_portfolio_holdings(df)
        assert len(result) == 3

    def test_empty_df_returns_empty(self):
        result = select_portfolio_holdings(pd.DataFrame())
        assert result.empty

    def test_portfolio_not_in_breakout_list(self):
        """Portfolio symbols must never appear in breakout candidates."""
        df = _rich_df(n_portfolio=2, n_breakout=3)
        breakout = select_breakout_candidates(df)
        portfolio_syms = set(df[df["bucket"] == BUCKET_PORTFOLIO]["symbol"].tolist())
        breakout_syms = set(breakout["symbol"].tolist()) if "symbol" in breakout.columns else set()
        assert portfolio_syms.isdisjoint(breakout_syms), \
            f"Portfolio symbols {portfolio_syms} appeared in breakout list {breakout_syms}"


class TestSelectTopSetups:

    def test_returns_breakout_first(self):
        df = _rich_df(n_breakout=3, n_pullback=2)
        result = select_top_setups(df)
        if "bucket" in result.columns:
            buckets = result["bucket"].tolist()
            # All breakout should appear before pullback
            saw_pullback = False
            for b in buckets:
                if b == BUCKET_PULLBACK:
                    saw_pullback = True
                if b == BUCKET_BREAKOUT and saw_pullback:
                    pytest.fail("Breakout appeared after pullback in combined list")

    def test_excludes_portfolio(self):
        df = _rich_df(n_portfolio=2, n_breakout=3)
        result = select_top_setups(df)
        if "is_portfolio" in result.columns:
            assert not result["is_portfolio"].any()

    def test_excludes_ineligible(self):
        df = _rich_df(n_excluded=3, n_breakout=2)
        result = select_top_setups(df)
        if "eligible" in result.columns:
            assert result["eligible"].all()

    def test_at_most_top_n(self):
        df = _rich_df(n_breakout=8, n_pullback=5)
        result = select_top_setups(df)
        assert len(result) <= 7

    def test_no_candidates_returns_empty(self):
        df = _rich_df(n_breakout=0, n_pullback=0, n_portfolio=2, n_excluded=3)
        result = select_top_setups(df)
        assert result.empty

    def test_legacy_fallback_when_no_bucket_column(self):
        """Without bucket column, select_top_setups falls back gracefully."""
        rows = [_row(symbol=f"S{i}", action_label=ACTION_BREAKOUT) for i in range(5)]
        df = pd.DataFrame(rows)
        # Ensure no bucket column
        if "bucket" in df.columns:
            df = df.drop(columns=["bucket"])
        result = select_top_setups(df)
        assert isinstance(result, pd.DataFrame)


# ===========================================================================
# Regime-adjusted action labels
# ===========================================================================

class TestRegimeAdjustedActions:

    def test_uptrend_normal_thresholds(self):
        """In an uptrend, normal scoring thresholds apply."""
        row = _row(
            state="ARMED",
            composite_score=0.22,  # just above MIN_SCORE=0.20
            dist_to_pivot_atr=0.5,
            regime_spy_trend=1.0,
        )
        label = assign_action(row)
        assert label != ACTION_AVOID

    def test_downtrend_raises_threshold(self):
        """In a downtrend, the score threshold is raised — marginal names pushed to AVOID."""
        row = _row(
            state="ARMED",
            composite_score=0.22,  # passes 0.20 but fails 0.20 + 0.08 = 0.28
            dist_to_pivot_atr=0.5,
            regime_spy_trend=-1.0,
        )
        label = assign_action(row)
        assert label == ACTION_AVOID

    def test_downtrend_strong_name_still_passes(self):
        """Strong names should still get action labels even in downtrends."""
        row = _row(
            state="ARMED",
            composite_score=0.50,  # well above threshold even with penalty
            dist_to_pivot_atr=0.5,
            regime_spy_trend=-1.0,
        )
        label = assign_action(row)
        assert label != ACTION_AVOID

    def test_non_equity_returns_portfolio_action(self):
        row = _row(is_non_equity=True, state="ARMED")
        from swingtrader.dashboard.action import ACTION_PORTFOLIO
        label = assign_action(row)
        assert label == ACTION_PORTFOLIO

    def test_neutral_regime_uses_normal_thresholds(self):
        """Neutral regime (0) should not raise thresholds."""
        row = _row(
            state="ARMED",
            composite_score=0.22,
            dist_to_pivot_atr=0.5,
            regime_spy_trend=0.0,
        )
        label = assign_action(row)
        assert label != ACTION_AVOID


# ===========================================================================
# Integration: eligibility + bucket pipeline
# ===========================================================================

class TestEligibilityBucketPipeline:

    def _make_snapshot(self) -> pd.DataFrame:
        rows = [
            # Good breakout candidate
            _row(symbol="GOOD_BO", state="ARMED", composite_score=0.50, failure_risk=0.20,
                 daily_rs_63=0.10, close_vs_sma200=0.05, base_length=15,
                 regime_spy_trend=1.0, dist_to_pivot_atr=0.8, is_fresh=True),
            # Good triggered (early)
            _row(symbol="GOOD_TRG", state="TRIGGERED", composite_score=0.55, failure_risk=0.18,
                 daily_rs_63=0.12, close_vs_sma200=0.08, base_length=10,
                 regime_spy_trend=1.0, days_in_state=3, is_fresh=True,
                 action_label=ACTION_NOW),
            # Portfolio holding (should NOT be in breakout list)
            _row(symbol="PORT", state="TRIGGERED", is_portfolio=True, composite_score=0.65,
                 failure_risk=0.15),
            # Downtrend + poor RS → excluded
            _row(symbol="DOWNTREND", state="ARMED", regime_spy_trend=-1.0, daily_rs_63=-0.05,
                 composite_score=0.40, base_length=8),
            # Broken structure → excluded
            _row(symbol="BROKEN", state="BASE", close_vs_sma200=-0.15, composite_score=0.35,
                 base_length=7),
            # Thin base → excluded
            _row(symbol="THIN", state="ARMED", base_length=3, composite_score=0.42),
            # Non-equity (SPAXX-like)
            _row(symbol="SPAXX", is_non_equity=True, state="NONE"),
        ]
        df = pd.DataFrame(rows)
        # Add freshness/action if missing
        if "action_label" not in df.columns:
            df["action_label"] = ACTION_BREAKOUT
        return df

    def test_full_pipeline_eligible_count(self):
        from swingtrader.dashboard.buckets import add_bucket_column
        from swingtrader.dashboard.eligibility import add_eligibility_columns
        df = self._make_snapshot()
        df = add_eligibility_columns(df)
        df = add_bucket_column(df)
        # Good candidates should be eligible
        assert df[df["symbol"] == "GOOD_BO"]["eligible"].iloc[0] == True  # noqa: E712
        assert df[df["symbol"] == "GOOD_TRG"]["eligible"].iloc[0] == True  # noqa: E712

    def test_downtrend_poor_rs_excluded(self):
        from swingtrader.dashboard.buckets import add_bucket_column
        from swingtrader.dashboard.eligibility import add_eligibility_columns
        df = self._make_snapshot()
        df = add_eligibility_columns(df)
        df = add_bucket_column(df)
        assert df[df["symbol"] == "DOWNTREND"]["eligible"].iloc[0] == False  # noqa: E712
        assert df[df["symbol"] == "DOWNTREND"]["bucket"].iloc[0] == BUCKET_EXCLUDED

    def test_broken_trend_excluded(self):
        from swingtrader.dashboard.buckets import add_bucket_column
        from swingtrader.dashboard.eligibility import add_eligibility_columns
        df = self._make_snapshot()
        df = add_eligibility_columns(df)
        df = add_bucket_column(df)
        broken_row = df[df["symbol"] == "BROKEN"]
        assert broken_row["eligible"].iloc[0] == False  # noqa: E712

    def test_portfolio_not_in_breakout(self):
        from swingtrader.dashboard.buckets import add_bucket_column
        from swingtrader.dashboard.eligibility import add_eligibility_columns
        df = self._make_snapshot()
        df = add_eligibility_columns(df)
        df = add_bucket_column(df)
        breakout = select_breakout_candidates(df)
        syms = breakout["symbol"].tolist() if "symbol" in breakout.columns else []
        assert "PORT" not in syms, "Portfolio holding should not appear in breakout list"

    def test_non_equity_not_in_breakout(self):
        from swingtrader.dashboard.buckets import add_bucket_column
        from swingtrader.dashboard.eligibility import add_eligibility_columns
        df = self._make_snapshot()
        df = add_eligibility_columns(df)
        df = add_bucket_column(df)
        breakout = select_breakout_candidates(df)
        syms = breakout["symbol"].tolist() if "symbol" in breakout.columns else []
        assert "SPAXX" not in syms

    def test_top_setups_excludes_bad_names(self):
        from swingtrader.dashboard.buckets import add_bucket_column
        from swingtrader.dashboard.eligibility import add_eligibility_columns
        df = self._make_snapshot()
        df = add_eligibility_columns(df)
        df = add_bucket_column(df)
        top = select_top_setups(df)
        syms = top["symbol"].tolist() if "symbol" in top.columns else []
        assert "DOWNTREND" not in syms
        assert "BROKEN" not in syms
        assert "THIN" not in syms
        assert "SPAXX" not in syms
        assert "PORT" not in syms

    def test_good_candidates_in_top_setups(self):
        from swingtrader.dashboard.buckets import add_bucket_column
        from swingtrader.dashboard.eligibility import add_eligibility_columns
        df = self._make_snapshot()
        df = add_eligibility_columns(df)
        df = add_bucket_column(df)
        top = select_top_setups(df)
        syms = top["symbol"].tolist() if "symbol" in top.columns else []
        # Good candidates should appear
        assert "GOOD_BO" in syms or "GOOD_TRG" in syms
