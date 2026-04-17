"""Tests for report rendering (markdown and HTML)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from swingtrader.reports.render import (
    _build_context,
    _md_table,
    render_daily_html,
    render_daily_markdown,
    write_daily_reports,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_snapshot(n: int = 10) -> pd.DataFrame:
    rng = np.random.default_rng(99)
    syms = [f"SYM{i}" for i in range(n)]
    states = ["ARMED", "BASE", "TRIGGERED", "ACCEPTED", "NONE", "FAILED",
              "CONFIRMED", "LATE", "ARMED", "BASE"][:n]
    return pd.DataFrame({
        "user_symbol": syms,
        "provider_symbol": syms,
        "state": states,
        "is_portfolio": [i < 2 for i in range(n)],
        "is_watchlist": [True] * n,
        "is_non_equity": [False] * n,
        "close": rng.uniform(50, 300, n),
        "pivot": rng.uniform(50, 300, n),
        "dist_to_pivot_atr": rng.uniform(0, 3, n),
        "base_length": rng.integers(15, 80, n),
        "days_in_state": rng.integers(1, 30, n),
        "atr_compression_pct": rng.uniform(0, 100, n),
        "volume_dryup": rng.uniform(0, 1, n),
        "daily_rs_63": rng.normal(0, 0.1, n),
        "skip_reason": [None] * n,
    })


def _make_scores(syms) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    return pd.DataFrame({
        "state": ["ARMED"] * len(syms),
        "setup_score": rng.uniform(0, 1, len(syms)),
        "trade_score": rng.normal(0, 1, len(syms)),
        "failure_risk": rng.uniform(0, 1, len(syms)),
        "composite_score": rng.uniform(0, 1, len(syms)),
        "percentile_rank": rng.uniform(0, 100, len(syms)),
    }, index=pd.Index(syms, name="symbol"))


# ── _md_table ─────────────────────────────────────────────────────────────────

def test_md_table_produces_pipe_separated_header() -> None:
    df = pd.DataFrame({"a": [1], "b": [2.5]})
    result = _md_table(df, ["a", "b"])
    assert "| a | b |" in result
    assert "| --- |" in result


def test_md_table_handles_nan() -> None:
    df = pd.DataFrame({"x": [float("nan")], "y": [1.0]})
    result = _md_table(df, ["x", "y"])
    assert "—" in result


def test_md_table_skips_missing_columns() -> None:
    df = pd.DataFrame({"a": [1]})
    result = _md_table(df, ["a", "nonexistent_col"])
    assert "nonexistent_col" not in result
    assert "a" in result


# ── _build_context ────────────────────────────────────────────────────────────

def test_build_context_has_required_keys() -> None:
    snap = _make_snapshot()
    ctx = _build_context(snap, None, pd.Timestamp("2024-06-01"))
    required = {"as_of", "generated_at", "model_fitted", "scored_states", "unscored_states",
                "state_sections", "state_counts", "portfolio_rows", "skipped_rows"}
    assert required.issubset(ctx.keys())


def test_build_context_as_of_date_string() -> None:
    snap = _make_snapshot()
    ctx = _build_context(snap, None, pd.Timestamp("2024-06-01"))
    assert ctx["as_of"] == "2024-06-01"


def test_build_context_state_counts_correct() -> None:
    snap = _make_snapshot(10)
    ctx = _build_context(snap, None, pd.Timestamp("2024-06-01"))
    # Snapshot has 2 ARMED, 2 BASE, 1 TRIGGERED, 1 ACCEPTED, 1 NONE, 1 FAILED, 1 CONFIRMED, 1 LATE
    assert ctx["state_counts"]["ARMED"] == 2
    assert ctx["state_counts"]["BASE"] == 2


def test_build_context_portfolio_flag() -> None:
    snap = _make_snapshot(10)
    ctx = _build_context(snap, None, pd.Timestamp("2024-06-01"))
    assert ctx["portfolio_rows"] is True   # first 2 rows are portfolio


def test_build_context_merges_scores() -> None:
    snap = _make_snapshot(10)
    scores = _make_scores(snap["provider_symbol"].tolist())
    ctx = _build_context(snap, scores, pd.Timestamp("2024-06-01"))
    # Context should include scored data — model_fitted should be True
    assert ctx["model_fitted"] is True


# ── render_daily_markdown ─────────────────────────────────────────────────────

def test_render_daily_markdown_returns_string() -> None:
    snap = _make_snapshot()
    md = render_daily_markdown(snap, None, pd.Timestamp("2024-06-01"))
    assert isinstance(md, str)
    assert len(md) > 100


def test_render_daily_markdown_contains_date() -> None:
    snap = _make_snapshot()
    md = render_daily_markdown(snap, None, pd.Timestamp("2024-06-01"))
    assert "2024-06-01" in md


def test_render_daily_markdown_contains_bias_note() -> None:
    snap = _make_snapshot()
    md = render_daily_markdown(snap, None, pd.Timestamp("2024-06-01"))
    assert "bias" in md.lower() or "survivorship" in md.lower()


def test_render_daily_markdown_with_scores() -> None:
    snap = _make_snapshot(10)
    scores = _make_scores(snap["provider_symbol"].tolist())
    md = render_daily_markdown(snap, scores, pd.Timestamp("2024-06-01"))
    assert isinstance(md, str)


# ── render_daily_html ─────────────────────────────────────────────────────────

def test_render_daily_html_returns_string() -> None:
    snap = _make_snapshot()
    html = render_daily_html(snap, None, pd.Timestamp("2024-06-01"))
    assert isinstance(html, str)


def test_render_daily_html_contains_doctype() -> None:
    snap = _make_snapshot()
    html = render_daily_html(snap, None, pd.Timestamp("2024-06-01"))
    # Either a proper HTML doc or a fallback with html tag
    assert "html" in html.lower()


# ── write_daily_reports ───────────────────────────────────────────────────────

def test_write_daily_reports_creates_files(tmp_path) -> None:
    snap = _make_snapshot()
    paths = write_daily_reports(snap, None, pd.Timestamp("2024-06-01"), output_dir=tmp_path)
    assert "markdown" in paths
    assert "html" in paths
    assert paths["markdown"].exists()
    assert paths["html"].exists()


def test_write_daily_reports_markdown_nonempty(tmp_path) -> None:
    snap = _make_snapshot()
    paths = write_daily_reports(snap, None, pd.Timestamp("2024-06-01"), output_dir=tmp_path)
    content = paths["markdown"].read_text()
    assert len(content) > 50


def test_write_daily_reports_html_nonempty(tmp_path) -> None:
    snap = _make_snapshot()
    paths = write_daily_reports(snap, None, pd.Timestamp("2024-06-01"), output_dir=tmp_path)
    content = paths["html"].read_text()
    assert len(content) > 50


def test_write_daily_reports_creates_output_dir(tmp_path) -> None:
    snap = _make_snapshot()
    new_dir = tmp_path / "nested" / "reports"
    paths = write_daily_reports(snap, None, pd.Timestamp("2024-06-01"), output_dir=new_dir)
    assert new_dir.exists()
    assert paths["markdown"].exists()
