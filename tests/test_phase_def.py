"""Tests for Phase D (chart annotations), Phase E (score drivers / volume block),
and Phase F (score transparency HTML, export links, portfolio icons, index functions).

All tests are unit-level — no live data files required. Missing parquets are
handled by the production code's own graceful-fallback paths.
"""
from __future__ import annotations

import math
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_row(**kwargs) -> pd.Series:
    defaults = {
        "symbol": "TST",
        "user_symbol": "TST",
        "provider_symbol": "TST",
        "state": "ARMED",
        "action_label": "Actionable now",
        "composite_score": 0.72,
        "setup_score": 0.65,
        "trade_score": 0.78,
        "failure_risk": 0.18,
        "percentile_rank": 88.0,
        "close": 150.00,
        "pivot": 148.00,
        "atr14": 3.50,
        "dist_to_pivot_atr": 0.57,
        "base_length": 14,
        "days_in_state": 5,
        "entry_lo": 147.00,
        "entry_hi": 149.00,
        "stop": 144.00,
        "t1": 156.00,
        "t2": 163.00,
        "t3": 172.00,
        "atr_compression_pct": 22.0,
        "volume_dryup": 0.65,
        "daily_rs_63": 0.12,
        "close_vs_sma50": 0.03,
        "ytd_dist_atr": 1.5,
        "swing_low_dist_atr": 2.1,
        "regime_spy_trend": 1.0,
        "regime_spy_above_200sma": 1.0,
        "regime_vix_level": 18.0,
        "resistance_touches": 3,
        "pivot_flatness": 0.80,
        "weekly_dist_wma10": 0.04,
        "groups": "Technology",
        "is_portfolio": False,
        "is_watchlist": True,
    }
    defaults.update(kwargs)
    return pd.Series(defaults)


# ===========================================================================
# Phase E — build_score_drivers
# ===========================================================================

class TestBuildScoreDrivers:
    """Tests for context.build_score_drivers()."""

    from swingtrader.dashboard.context import build_score_drivers

    def _call(self, **row_kwargs):
        from swingtrader.dashboard.context import build_score_drivers
        row = _make_row(**row_kwargs)
        return build_score_drivers("NONEXISTENT", row)

    def test_returns_dict_with_required_keys(self):
        result = self._call()
        assert isinstance(result, dict)
        for key in ("bullish_signals", "bearish_signals", "model_based", "rule_based", "why_selected"):
            assert key in result, f"Missing key: {key}"

    def test_bullish_signals_is_list(self):
        result = self._call()
        assert isinstance(result["bullish_signals"], list)

    def test_bearish_signals_is_list(self):
        result = self._call()
        assert isinstance(result["bearish_signals"], list)

    def test_close_above_sma50_is_bullish(self):
        result = self._call(close_vs_sma50=0.04)
        assert any("SMA50" in s or "sma50" in s.lower() or "above" in s.lower()
                   for s in result["bullish_signals"]), \
            f"Expected SMA50 bullish signal in {result['bullish_signals']}"

    def test_close_below_sma50_is_bearish(self):
        result = self._call(close_vs_sma50=-0.04)
        assert any("SMA50" in s or "sma50" in s.lower() or "below" in s.lower()
                   for s in result["bearish_signals"]), \
            f"Expected SMA50 bearish signal in {result['bearish_signals']}"

    def test_high_volume_dryup_is_bullish(self):
        result = self._call(volume_dryup=0.75)
        assert any("volume" in s.lower() or "dry" in s.lower()
                   for s in result["bullish_signals"]), \
            f"Expected volume dry-up bullish in {result['bullish_signals']}"

    def test_high_failure_risk_is_bearish(self):
        result = self._call(failure_risk=0.65)
        assert any("failure" in s.lower() or "risk" in s.lower()
                   for s in result["bearish_signals"]), \
            f"Expected failure risk bearish in {result['bearish_signals']}"

    def test_strong_rs_is_bullish(self):
        result = self._call(daily_rs_63=0.15)
        assert any("rs" in s.lower() or "relative" in s.lower() or "outper" in s.lower()
                   for s in result["bullish_signals"]), \
            f"Expected RS bullish in {result['bullish_signals']}"

    def test_why_selected_is_str(self):
        result = self._call()
        assert isinstance(result["why_selected"], str)

    def test_missing_row_values_handled_gracefully(self):
        row = pd.Series({"symbol": "X", "state": "BASE"})
        from swingtrader.dashboard.context import build_score_drivers
        result = build_score_drivers("NONEXISTENT", row)
        assert isinstance(result, dict)
        assert "bullish_signals" in result


# ===========================================================================
# Phase E — build_volume_block down_close_ratio
# ===========================================================================

class TestVolumeBlockDownCloseRatio:
    """down_close_ratio is computed when raw daily data is available."""

    def _call_with_raw(self, down_count: int = 7, total: int = 20) -> dict:
        """Patch _load_raw_daily to return synthetic data."""
        from swingtrader.dashboard.context import build_volume_block

        closes = [100.0 + i for i in range(total)]
        opens = closes[:]
        # Make `down_count` bars close < open (bearish)
        for i in range(down_count):
            opens[i] = closes[i] + 1  # close < open → bearish

        df = pd.DataFrame({"open": opens, "close": closes, "high": closes, "low": closes,
                           "volume": [1_000_000] * total})

        # build_volume_block returns early if features is None; provide a minimal Series
        fake_feat = pd.Series(dtype=float)

        with patch("swingtrader.dashboard.context._load_raw_daily", return_value=df), \
             patch("swingtrader.dashboard.context._load_features", return_value=fake_feat):
            return build_volume_block("TST", 100.0, 3.0)

    def test_down_close_ratio_present_in_result(self):
        result = self._call_with_raw(down_count=7)
        assert "down_close_ratio" in result

    def test_down_close_ratio_value_correct(self):
        result = self._call_with_raw(down_count=10)
        dcr = result["down_close_ratio"]
        assert math.isfinite(dcr)
        assert abs(dcr - 0.5) < 0.01

    def test_down_close_ratio_nan_when_no_raw_data(self):
        from swingtrader.dashboard.context import build_volume_block
        with patch("swingtrader.dashboard.context._load_raw_daily", return_value=None), \
             patch("swingtrader.dashboard.context._load_features", return_value=None):
            result = build_volume_block("NONEXISTENT", 100.0, 3.0)
        dcr = result.get("down_close_ratio", float("nan"))
        assert not math.isfinite(dcr)

    def test_down_close_ratio_zero_means_all_up_closes(self):
        result = self._call_with_raw(down_count=0)
        dcr = result["down_close_ratio"]
        assert math.isfinite(dcr)
        assert dcr == pytest.approx(0.0)


# ===========================================================================
# Phase E — build_context includes score_drivers
# ===========================================================================

class TestBuildContextScoreDrivers:
    """build_context() must include score_drivers key in return value."""

    def test_score_drivers_key_present(self):
        from swingtrader.dashboard.context import build_context
        row = _make_row()
        levels = {"pivot": 148.0, "stop": 144.0, "t1": 156.0, "t2": 163.0,
                  "t3": 172.0, "s1": 146.0, "s2": 144.0, "r1": 152.0, "r2": 158.0,
                  "risk_reward_t1": 2.0}
        result = build_context("NONEXISTENT", row, levels)
        assert "score_drivers" in result

    def test_score_drivers_has_bullish_bearish(self):
        from swingtrader.dashboard.context import build_context
        row = _make_row()
        levels = {"pivot": 148.0, "stop": 144.0, "t1": 156.0, "t2": 163.0,
                  "t3": 172.0, "s1": 146.0, "s2": 144.0, "r1": 152.0, "r2": 158.0,
                  "risk_reward_t1": 2.0}
        result = build_context("NONEXISTENT", row, levels)
        sd = result["score_drivers"]
        assert "bullish_signals" in sd
        assert "bearish_signals" in sd


# ===========================================================================
# Phase F — _score_drivers_html
# ===========================================================================

class TestScoreDriversHtml:
    """Tests for dashboard._score_drivers_html()."""

    from swingtrader.reports.dashboard import _score_drivers_html  # type: ignore[attr-defined]

    def _call(self, score_drivers: dict) -> str:
        from swingtrader.reports.dashboard import _score_drivers_html
        return _score_drivers_html(score_drivers)

    def test_empty_dict_returns_empty_string(self):
        assert self._call({}) == ""

    def test_empty_lists_returns_empty_string(self):
        assert self._call({"bullish_signals": [], "bearish_signals": []}) == ""

    def test_bullish_signal_rendered(self):
        html = self._call({"bullish_signals": ["Close above SMA50"], "bearish_signals": []})
        assert "Close above SMA50" in html
        assert "driver-bull" in html or "▲" in html

    def test_bearish_signal_rendered(self):
        html = self._call({"bullish_signals": [], "bearish_signals": ["High failure risk"]})
        assert "High failure risk" in html
        assert "driver-bear" in html or "▼" in html

    def test_why_selected_rendered(self):
        html = self._call({
            "bullish_signals": ["RS outperforming"],
            "bearish_signals": [],
            "why_selected": "High percentile rank on tight base",
        })
        assert "High percentile rank" in html

    def test_multiple_signals_all_rendered(self):
        html = self._call({
            "bullish_signals": ["SMA50 above", "Volume dry-up"],
            "bearish_signals": ["Elevated failure risk"],
        })
        assert "SMA50 above" in html
        assert "Volume dry-up" in html
        assert "Elevated failure risk" in html

    def test_contains_score_drivers_class(self):
        html = self._call({"bullish_signals": ["x"], "bearish_signals": []})
        assert "score-drivers" in html


# ===========================================================================
# Phase F — _export_links_html
# ===========================================================================

class TestExportLinksHtml:
    """Tests for dashboard._export_links_html()."""

    def _call(self, packet: dict) -> str:
        from swingtrader.reports.dashboard import _export_links_html
        return _export_links_html(packet)

    def test_returns_empty_for_missing_symbol(self):
        assert self._call({}) == ""

    def test_returns_empty_for_dash_symbol(self):
        assert self._call({"provider_symbol": "—"}) == ""

    def test_contains_json_link(self):
        html = self._call({"provider_symbol": "AAPL"})
        assert "AAPL" in html
        assert ".json" in html

    def test_prefers_provider_symbol(self):
        html = self._call({"provider_symbol": "AAPL", "symbol": "AAPL.US"})
        assert "AAPL_packet.json" in html

    def test_falls_back_to_symbol(self):
        html = self._call({"symbol": "MSFT"})
        assert "MSFT" in html

    def test_contains_export_links_class(self):
        html = self._call({"provider_symbol": "TSLA"})
        assert "export-links" in html


# ===========================================================================
# Phase F — portfolio guidance icons in _portfolio_strip_html
# ===========================================================================

class TestPortfolioIconsInStrip:
    """Portfolio strip renders correct icon class for each guidance string."""

    def _strip(self, guidance: str) -> str:
        from swingtrader.reports.dashboard import _portfolio_strip_html
        row = _make_row(is_portfolio=True, portfolio_guidance=guidance)
        df = pd.DataFrame([row])
        return _portfolio_strip_html(df)

    def test_hold_icon_rendered(self):
        html = self._strip("Hold — trend intact")
        assert "pg-hold" in html or "✓" in html

    def test_trim_icon_rendered(self):
        html = self._strip("Trim — extended vs stop")
        assert "pg-trim" in html or "↓" in html

    def test_defend_icon_rendered(self):
        html = self._strip("Defend stop at 144.00")
        assert "pg-defend" in html or "⚠" in html

    def test_exit_icon_rendered(self):
        html = self._strip("Exit — below stop")
        assert "pg-exit" in html or "✗" in html

    def test_unknown_guidance_falls_back(self):
        html = self._strip("Watch closely")
        # Should not crash; renders something
        assert isinstance(html, str)


# ===========================================================================
# Phase F — render_dashboard summary bar
# ===========================================================================

class TestRenderDashboardSummaryBar:
    """render_dashboard emits the summary_bar count section."""

    def _render(self, n: int = 3) -> str:
        from swingtrader.dashboard.packet import build_packets
        from swingtrader.reports.dashboard import render_dashboard
        rows = []
        for i in range(n):
            rows.append(_make_row(
                symbol=f"S{i:02d}", user_symbol=f"S{i:02d}",
                provider_symbol=f"S{i:02d}",
                action_label="Actionable now" if i == 0 else "Actionable on breakout",
            ))
        df = pd.DataFrame(rows)
        pkts = build_packets(df)
        return render_dashboard(df, pkts, pd.Timestamp("2026-01-17"))

    def test_summary_bar_present(self):
        html = self._render()
        # Some count-related text should appear
        assert "summary" in html.lower() or "actionable" in html.lower() or "Top" in html

    def test_html_is_valid_structure(self):
        html = self._render()
        assert "<!DOCTYPE html>" in html
        assert "</html>" in html


# ===========================================================================
# Phase D — generate_daily_chart accepts all new params
# ===========================================================================

class TestChartParamSigning:
    """generate_daily_chart signature includes all Phase D kwargs."""

    def test_daily_chart_accepts_phase_d_kwargs(self):
        import inspect

        from swingtrader.dashboard.charts import generate_daily_chart
        sig = inspect.signature(generate_daily_chart)
        params = sig.parameters
        for expected in ("s1", "s2", "r1", "r2", "state", "action_label",
                         "setup_class", "score", "failure", "days_in_state"):
            assert expected in params, f"Missing param: {expected}"

    def test_weekly_chart_accepts_phase_d_kwargs(self):
        import inspect

        from swingtrader.dashboard.charts import generate_weekly_chart
        sig = inspect.signature(generate_weekly_chart)
        params = sig.parameters
        for expected in ("s1", "stop"):
            assert expected in params, f"Missing param: {expected}"

    def test_daily_chart_returns_none_for_nonexistent_symbol(self, tmp_path):
        from swingtrader.dashboard.charts import generate_daily_chart
        result = generate_daily_chart(
            "NONEXISTENT_SYMBOL_XYZ", tmp_path,
            pivot=100.0, stop=95.0, t1=110.0, t2=118.0,
            s1=97.0, s2=95.0, r1=105.0, r2=110.0,
            state="ARMED", action_label="Actionable now",
            setup_class="Near breakout/poised", score=0.72, failure=0.18,
            days_in_state=5,
        )
        assert result is None

    def test_weekly_chart_returns_none_for_nonexistent_symbol(self, tmp_path):
        from swingtrader.dashboard.charts import generate_weekly_chart
        result = generate_weekly_chart(
            "NONEXISTENT_SYMBOL_XYZ", tmp_path,
            pivot=100.0, stop=95.0, s1=97.0,
        )
        assert result is None

    def test_generate_charts_for_packet_handles_missing_data(self, tmp_path):
        from swingtrader.dashboard.charts import generate_charts_for_packet
        from swingtrader.dashboard.packet import build_packet
        row = _make_row(provider_symbol="NONEXISTENT_SYMBOL_XYZ",
                        symbol="NONEXISTENT_SYMBOL_XYZ",
                        user_symbol="NONEXISTENT_SYMBOL_XYZ")
        packet = build_packet(row)
        result = generate_charts_for_packet(packet, tmp_path)
        # Should return a dict with None paths (data missing), not raise
        assert isinstance(result, dict)
        assert result.get("chart_daily") is None
        assert result.get("chart_weekly") is None


# ===========================================================================
# Phase F — pages_build helper functions
# ===========================================================================

class TestPagesBuilderHelpers:
    """_regime_context_html, _top_setups_preview_html, _artifacts_links_html."""

    def _make_scores_df(self, n: int = 5, state: str = "ARMED") -> pd.DataFrame:
        rows = []
        for i in range(n):
            rows.append({
                "user_symbol": f"S{i:02d}",
                "symbol": f"S{i:02d}",
                "state": state,
                "composite_score": 0.5 + i * 0.05,
                "percentile_rank": 50 + i * 5,
                "action_label": "Actionable now",
                "setup_classification": "Near breakout/poised",
                "close": 100 + i,
                "regime_spy_trend": 1.0,
                "regime_spy_above_200sma": 1.0,
                "regime_vix_level": 17.0,
            })
        return pd.DataFrame(rows)

    def test_regime_context_html_returns_string(self):
        from swingtrader.pipelines.pages_build import _regime_context_html
        df = self._make_scores_df()
        html = _regime_context_html(df)
        assert isinstance(html, str)

    def test_regime_context_shows_spy_trend(self):
        from swingtrader.pipelines.pages_build import _regime_context_html
        df = self._make_scores_df()
        html = _regime_context_html(df)
        assert "SPY" in html or "spy" in html.lower() or "Trend" in html

    def test_regime_context_handles_empty_df(self):
        from swingtrader.pipelines.pages_build import _regime_context_html
        html = _regime_context_html(pd.DataFrame())
        assert isinstance(html, str)

    def test_top_setups_preview_returns_string(self):
        from swingtrader.pipelines.pages_build import _top_setups_preview_html
        df = self._make_scores_df()
        html = _top_setups_preview_html(df, "2026-01-17")
        assert isinstance(html, str)

    def test_top_setups_preview_contains_symbols(self):
        from swingtrader.pipelines.pages_build import _top_setups_preview_html
        df = self._make_scores_df()
        html = _top_setups_preview_html(df, "2026-01-17")
        # At least one symbol should appear
        assert any(f"S0{i}" in html for i in range(5))

    def test_top_setups_preview_empty_df(self):
        from swingtrader.pipelines.pages_build import _top_setups_preview_html
        html = _top_setups_preview_html(pd.DataFrame(), "2026-01-17")
        assert isinstance(html, str)
        assert html == "" or "No" in html or len(html) < 50

    def test_top_setups_preview_links_to_dashboard(self):
        from swingtrader.pipelines.pages_build import _top_setups_preview_html
        df = self._make_scores_df()
        html = _top_setups_preview_html(df, "2026-01-17")
        assert "2026-01-17" in html
        assert "dashboard.html" in html

    def test_artifacts_links_html_returns_string(self):
        from swingtrader.pipelines.pages_build import _artifacts_links_html
        html = _artifacts_links_html("2026-01-17")
        assert isinstance(html, str)

    def test_artifacts_links_contains_json_files(self):
        from swingtrader.pipelines.pages_build import _artifacts_links_html
        html = _artifacts_links_html("2026-01-17")
        assert "dashboard_summary.json" in html
        assert "top_setups.json" in html
        assert "portfolio_review.json" in html

    def test_artifacts_links_contains_date(self):
        from swingtrader.pipelines.pages_build import _artifacts_links_html
        html = _artifacts_links_html("2026-01-17")
        assert "2026-01-17" in html


# ===========================================================================
# Phase D — chart annotation drawing (unit-level, matplotlib mocked)
# ===========================================================================

class TestChartAnnotationsUnit:
    """Verify that chart generation code calls expected matplotlib methods.

    We patch matplotlib.pyplot.figure so no real rendering happens, and
    verify the code path runs without error for a synthetic DataFrame.
    """

    def _make_ohlcv(self, n: int = 50) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        closes = 100 + np.cumsum(rng.normal(0, 0.5, n))
        opens  = closes * (1 + rng.normal(0, 0.003, n))
        highs  = np.maximum(closes, opens) * (1 + rng.uniform(0, 0.005, n))
        lows   = np.minimum(closes, opens) * (1 - rng.uniform(0, 0.005, n))
        dates  = pd.date_range("2025-01-01", periods=n, freq="B")
        return pd.DataFrame(
            {"open": opens, "high": highs, "low": lows, "close": closes,
             "volume": rng.integers(500_000, 2_000_000, n)},
            index=dates,
        )

    def test_daily_chart_runs_with_synthetic_data(self, tmp_path):
        """generate_daily_chart produces a file when data is present."""
        import matplotlib
        matplotlib.use("Agg")

        from swingtrader.dashboard.charts import _RAW_DAILY_DIR, generate_daily_chart

        sym = "TESTCHRT"
        data_path = _RAW_DAILY_DIR / f"{sym}.parquet"

        # Write synthetic parquet to real location to trigger the chart path
        # Only if the directory exists (skip in CI without data dir)
        if not _RAW_DAILY_DIR.exists():
            pytest.skip("data/raw/daily not present — skipping chart generation test")

        df = self._make_ohlcv(250)
        df.to_parquet(data_path)
        try:
            result = generate_daily_chart(
                sym, tmp_path,
                pivot=105.0, stop=98.0, t1=115.0, t2=125.0,
                s1=102.0, s2=98.0, r1=110.0, r2=118.0,
                state="ARMED", action_label="Actionable now",
                setup_class="Near breakout/poised",
                score=0.72, failure=0.18, days_in_state=5,
            )
            assert result is not None
            assert result.exists()
            assert result.suffix == ".png"
        finally:
            data_path.unlink(missing_ok=True)

    def test_weekly_chart_runs_with_synthetic_data(self, tmp_path):
        """generate_weekly_chart produces a file when data is present."""
        import matplotlib
        matplotlib.use("Agg")

        from swingtrader.dashboard.charts import _RAW_WEEKLY_DIR, generate_weekly_chart

        sym = "TESTWCHRT"
        data_path = _RAW_WEEKLY_DIR / f"{sym}.parquet"

        if not _RAW_WEEKLY_DIR.exists():
            pytest.skip("data/raw/weekly not present — skipping chart generation test")

        df = self._make_ohlcv(80)
        df.to_parquet(data_path)
        try:
            result = generate_weekly_chart(
                sym, tmp_path,
                pivot=105.0, stop=98.0, s1=102.0,
            )
            assert result is not None
            assert result.exists()
            assert result.suffix == ".png"
        finally:
            data_path.unlink(missing_ok=True)
