"""Daily evaluation pipeline.

Entry point for the scheduled GitHub Actions workflow and local runs.
Processes every symbol in the configured universe end-to-end:

  1. Resolve universe (watchlist + portfolio + optionally broader files)
  2. Ingest / update daily bars from yfinance (skips non-equity symbols)
  3. Build weekly bars by resampling
  4. Load benchmark data (SPY for RS features, ^VIX for regime features)
  5. Compute daily feature DataFrame per symbol
  6. Compute weekly feature DataFrame per symbol
  7. Compute AVWAP features per symbol
  8. Detect bases + compute pivot features per symbol
  9. Run breakout state machine per symbol
 10. Generate training labels per symbol (historical bars only)
 11. Write per-symbol artifacts to data/{features,states,labels}/
 12. Build daily snapshot: one row per symbol with state + key features
 13. Write snapshot to data/snapshots/YYYY-MM-DD.parquet
 14. Write markdown summary to reports/daily/YYYY-MM-DD/snapshot.md

Run:
  python -m swingtrader.pipelines.daily_run            # uses today's date
  python -m swingtrader.pipelines.daily_run 2025-01-15 # specific date (backfill)
"""
from __future__ import annotations

import argparse
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from swingtrader.avwap.features import compute_avwap_features
from swingtrader.bases.base_detect import detect_bases
from swingtrader.features.pivot_features import compute_pivot_features
from swingtrader.features.primitives import atr_wilder
from swingtrader.features.registry import compute_features, load_all_feature_modules
from swingtrader.ingest.quality import check_daily
from swingtrader.ingest.symbols import SymbolRecord, resolve_universe, write_universe_artifact
from swingtrader.ingest.yfinance_source import (
    fetch_daily,
    load_daily,
    resample_weekly,
)
from swingtrader.labels.generators import compute_all_labels
from swingtrader.states.machine import compute_states
from swingtrader.utils.config import REPO_ROOT, load_config
from swingtrader.utils.io import write_parquet
from swingtrader.utils.logging import get_logger

log = get_logger(__name__)

# Minimum bars required to enter feature computation
_MIN_DAILY_BARS = 50
_MIN_WEEKLY_BARS = 10


@dataclass
class SymbolResult:
    """Outcome of processing one symbol in the daily run."""

    symbol: SymbolRecord
    ok: bool
    state: str = "NONE"
    pivot: float = float("nan")
    atr14: float = float("nan")
    close: float = float("nan")
    base_length: int = 0
    dist_to_pivot_atr: float = float("nan")
    days_in_state: int = 0
    skipped: bool = False
    skip_reason: str = ""
    error: str = ""
    # Selected features for the snapshot (populated after feature pass)
    snapshot_row: dict = field(default_factory=dict)


class DailyRunner:
    """Orchestrates the full daily evaluation pipeline."""

    def __init__(self, as_of: pd.Timestamp | None = None, ingest: bool = True):
        self.as_of = as_of or pd.Timestamp.today().normalize()
        self.ingest = ingest  # set False for offline/replay mode
        load_all_feature_modules()

        self.cfg_data = load_config("data_sources")
        self.cfg_score = load_config("scoring")
        self.cfg_labels = load_config("labels")

        self.data_dir = REPO_ROOT / "data"
        self.daily_dir = self.data_dir / "raw" / "daily"
        self.weekly_dir = self.data_dir / "raw" / "weekly"
        self.features_dir = self.data_dir / "features"
        self.states_dir = self.data_dir / "states"
        self.labels_dir = self.data_dir / "labels"
        self.snapshot_dir = self.data_dir / "snapshots"

        for d in [self.features_dir, self.states_dir, self.labels_dir, self.snapshot_dir]:
            d.mkdir(parents=True, exist_ok=True)

    # ── Public entry point ────────────────────────────────────────────────────

    def run(self) -> dict:
        """Execute the full pipeline. Returns a summary dict."""
        log.info("=== Daily run: %s ===", self.as_of.date())

        universe = resolve_universe()
        write_universe_artifact(universe, self.as_of)

        # Load shared benchmark / regime data once
        spy_df, vix_df, spy_weekly = self._load_benchmarks(universe)

        results: list[SymbolResult] = []
        for rec in universe:
            result = self._process_symbol(rec, spy_df=spy_df, vix_df=vix_df, spy_weekly=spy_weekly)
            results.append(result)

        snapshot = self._build_snapshot(results)
        snap_path = self._write_snapshot(snapshot)
        report_path = self._write_markdown_report(snapshot)

        ok_count = sum(1 for r in results if r.ok)
        skip_count = sum(1 for r in results if r.skipped)
        err_count = sum(1 for r in results if not r.ok and not r.skipped)

        summary = {
            "as_of": str(self.as_of.date()),
            "total": len(results),
            "ok": ok_count,
            "skipped": skip_count,
            "errors": err_count,
            "snapshot_path": str(snap_path),
            "report_path": str(report_path),
        }
        log.info(
            "Run complete: %d ok, %d skipped, %d errors of %d symbols",
            ok_count, skip_count, err_count, len(results),
        )
        return summary

    # ── Per-symbol processing ─────────────────────────────────────────────────

    def _process_symbol(
        self,
        rec: SymbolRecord,
        *,
        spy_df: pd.DataFrame,
        vix_df: pd.DataFrame,
        spy_weekly: pd.DataFrame,
    ) -> SymbolResult:
        result = SymbolResult(symbol=rec, ok=False)

        # Non-equity symbols: tag and skip scoring
        if rec.is_non_equity:
            result.skipped = True
            result.skip_reason = "non_equity"
            result.ok = True
            log.info("  %s: skipped (non-equity / cash)", rec.user_symbol)
            return result

        try:
            # Step 1 — Ingest daily bars
            if self.ingest:
                fetch_results = fetch_daily([rec.provider_symbol], self.cfg_data)
                if not any(fr.ok for fr in fetch_results):
                    result.skip_reason = "ingest_failed"
                    result.skipped = True
                    result.ok = True
                    log.warning("  %s: ingest failed, skipping", rec.user_symbol)
                    return result

            # Step 2 — Load bars
            try:
                df = load_daily(rec.provider_symbol, self.cfg_data)
            except FileNotFoundError:
                result.skip_reason = "no_data_file"
                result.skipped = True
                result.ok = True
                log.warning("  %s: no daily parquet found, skipping", rec.user_symbol)
                return result

            # Step 3 — Quality check
            qr = check_daily(rec.user_symbol, df)
            if not qr.ok and qr.n_rows < _MIN_DAILY_BARS:
                result.skip_reason = f"quality_fail:{','.join(qr.notes)}"
                result.skipped = True
                result.ok = True
                log.warning("  %s: insufficient data (%d rows), skipping", rec.user_symbol, qr.n_rows)
                return result

            # Step 4 — Build weekly bars
            w_df = resample_weekly(df)

            # Step 5 — Daily features
            kwargs = {"benchmark_df": spy_df, "vix_df": vix_df}
            feat_daily = compute_features(df, "daily", extra_kwargs=kwargs)

            # Step 6 — Weekly features
            if len(w_df) >= _MIN_WEEKLY_BARS:
                spy_wk = spy_weekly if not spy_weekly.empty else None
                feat_weekly = compute_features(w_df, "weekly", extra_kwargs={"benchmark_df": spy_wk})
                # Reindex weekly features back to daily index
                feat_weekly = feat_weekly.reindex(df.index, method="ffill")
            else:
                feat_weekly = pd.DataFrame(index=df.index)

            # Step 7 — Base detection
            bases = detect_bases(df, cfg=self.cfg_score)

            # Step 8 — Pivot features
            atr = atr_wilder(df, 14)
            feat_pivot = compute_pivot_features(df, bases, atr=atr)

            # Step 9 — State machine
            states = compute_states(df, bases, cfg=self.cfg_score)

            # Step 10 — AVWAP features (uses state history)
            state_series = states["state"]
            feat_avwap = compute_avwap_features(df, state_history=state_series)

            # Step 11 — Labels (historical only)
            labels_df = compute_all_labels(df, states, cfg=self.cfg_labels)

            # Step 12 — Combine feature DataFrame
            feat_all = pd.concat(
                [feat_daily, feat_weekly, feat_pivot, feat_avwap],
                axis=1,
            )
            # Drop duplicate column names if any
            feat_all = feat_all.loc[:, ~feat_all.columns.duplicated()]

            # Step 13 — Persist artifacts
            write_parquet(feat_all, self.features_dir / f"{rec.provider_symbol}.parquet")
            write_parquet(states, self.states_dir / f"{rec.provider_symbol}.parquet")
            write_parquet(labels_df, self.labels_dir / f"{rec.provider_symbol}.parquet")

            # Step 14 — Fill result for snapshot
            last_state = states["state"].iloc[-1] if not states.empty else "NONE"
            last_pivot = float(states["pivot"].iloc[-1]) if not states.empty else np.nan
            last_close = float(df["close"].iloc[-1]) if not df.empty else np.nan
            last_atr = float(atr.iloc[-1]) if not atr.empty else np.nan
            last_base_len = int(bases["base_length"].iloc[-1]) if not bases.empty else 0
            last_days = int(states["days_in_state"].iloc[-1]) if not states.empty else 0
            dist = float(feat_pivot["dist_to_pivot_atr"].iloc[-1]) if "dist_to_pivot_atr" in feat_pivot.columns else np.nan

            result.state = last_state
            result.pivot = last_pivot
            result.close = last_close
            result.atr14 = last_atr
            result.base_length = last_base_len
            result.dist_to_pivot_atr = dist
            result.days_in_state = last_days
            result.ok = True

            # Key features for snapshot
            def _last(s: pd.Series) -> float:
                v = s.dropna()
                return float(v.iloc[-1]) if not v.empty else np.nan

            result.snapshot_row = {
                "atr_compression_pct": _last(feat_daily.get("atr_compression_pct", pd.Series(dtype=float))),
                "volume_dryup": _last(feat_daily.get("volume_dryup", pd.Series(dtype=float))),
                "close_vs_sma50": _last(feat_daily.get("close_vs_sma50", pd.Series(dtype=float))),
                "daily_rs_63": _last(feat_daily.get("daily_rs_63", pd.Series(dtype=float))),
                "regime_spy_trend": _last(feat_daily.get("regime_spy_trend", pd.Series(dtype=float))),
                "ytd_dist_atr": _last(feat_avwap.get("ytd_dist_atr", pd.Series(dtype=float))),
                "swing_low_dist_atr": _last(feat_avwap.get("swing_low_dist_atr", pd.Series(dtype=float))),
            }

            log.info("  %s: state=%s pivot=%.2f atr=%.2f", rec.user_symbol, last_state, last_pivot or 0, last_atr or 0)

        except Exception:
            result.error = traceback.format_exc()
            log.error("  %s: unexpected error:\n%s", rec.user_symbol, result.error)

        return result

    # ── Benchmarks ────────────────────────────────────────────────────────────

    def _load_benchmarks(
        self, universe: list[SymbolRecord]
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        spy_df = pd.DataFrame()
        vix_df = pd.DataFrame()
        spy_weekly = pd.DataFrame()

        # Ingest SPY + VIX if needed
        benchmark_syms = ["SPY", "^VIX"]
        if self.ingest:
            fetch_daily(benchmark_syms, self.cfg_data)

        try:
            spy_df = load_daily("SPY", self.cfg_data)
            spy_weekly = resample_weekly(spy_df)
        except FileNotFoundError:
            log.warning("SPY daily bars not found; regime/RS features will be NaN")

        try:
            vix_df = load_daily("^VIX", self.cfg_data)
        except FileNotFoundError:
            log.warning("^VIX bars not found; VIX regime feature will be NaN")

        return spy_df, vix_df, spy_weekly

    # ── Snapshot / reporting ──────────────────────────────────────────────────

    def _build_snapshot(self, results: list[SymbolResult]) -> pd.DataFrame:
        rows = []
        for r in results:
            rec = r.symbol
            base = {
                "user_symbol": rec.user_symbol,
                "provider_symbol": rec.provider_symbol,
                "groups": ",".join(rec.groups),
                "is_portfolio": rec.is_portfolio,
                "is_watchlist": rec.is_watchlist,
                "is_non_equity": rec.is_non_equity,
                "score_eligible": rec.score_eligible,
                "state": r.state if not r.skipped else "SKIPPED",
                "skip_reason": r.skip_reason,
                "close": r.close,
                "pivot": r.pivot,
                "atr14": r.atr14,
                "dist_to_pivot_atr": r.dist_to_pivot_atr,
                "base_length": r.base_length,
                "days_in_state": r.days_in_state,
                "ok": r.ok,
                "error": r.error[:200] if r.error else "",
            }
            base.update(r.snapshot_row)
            rows.append(base)
        return pd.DataFrame(rows)

    def _write_snapshot(self, snapshot: pd.DataFrame) -> Path:
        path = self.snapshot_dir / f"{self.as_of.date()}.parquet"
        write_parquet(snapshot, path)
        # Also overwrite latest.parquet
        write_parquet(snapshot, self.snapshot_dir / "latest.parquet")
        return path

    def _write_markdown_report(self, snapshot: pd.DataFrame) -> Path:
        out_dir = REPO_ROOT / "reports" / "daily" / str(self.as_of.date())
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "snapshot.md"

        lines = [
            f"# Daily Snapshot — {self.as_of.date()}",
            "",
            "> **Bias note:** Universe membership uses monthly snapshots, not true point-in-time.",
            "> Survivorship bias is present. All backtest metrics must be interpreted accordingly.",
            "",
        ]

        # Portfolio section
        portfolio = snapshot[snapshot["is_portfolio"] == True]  # noqa: E712
        if not portfolio.empty:
            lines += ["## Portfolio Holdings", ""]
            lines.append(_md_table(portfolio))
            lines.append("")

        # Watchlist by state
        watchlist = snapshot[snapshot["is_watchlist"] == True]  # noqa: E712
        for state_name in ["ARMED", "BASE", "TRIGGERED", "ACCEPTED", "CONFIRMED", "LATE", "FAILED", "NONE"]:
            subset = watchlist[watchlist["state"] == state_name]
            if not subset.empty:
                lines += [f"## Watchlist — {state_name}", ""]
                lines.append(_md_table(subset))
                lines.append("")

        # Skipped / errors
        skipped = snapshot[snapshot["state"] == "SKIPPED"]
        if not skipped.empty:
            lines += ["## Skipped", ""]
            lines.append(_md_table(skipped[["user_symbol", "skip_reason", "is_non_equity"]]))
            lines.append("")

        path.write_text("\n".join(lines), encoding="utf-8")
        return path


def _md_table(df: pd.DataFrame, max_rows: int = 100) -> str:
    cols = [c for c in df.columns if c not in ("error", "groups", "provider_symbol")]
    sub = df[cols].head(max_rows)
    # Header
    header = "| " + " | ".join(str(c) for c in sub.columns) + " |"
    sep = "| " + " | ".join("---" for _ in sub.columns) + " |"
    rows = []
    for _, row in sub.iterrows():
        def _fmt(v) -> str:
            if isinstance(v, float):
                return "—" if np.isnan(v) else f"{v:.3f}"
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return "—"
            return str(v)
        rows.append("| " + " | ".join(_fmt(v) for v in row) + " |")
    return "\n".join([header, sep] + rows)


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="swingtrader daily evaluation pipeline")
    parser.add_argument("date", nargs="?", default=None, help="As-of date YYYY-MM-DD (default: today)")
    parser.add_argument("--no-ingest", action="store_true", help="Skip data fetch (offline mode)")
    args = parser.parse_args()

    as_of = pd.Timestamp(args.date) if args.date else pd.Timestamp.today().normalize()
    runner = DailyRunner(as_of=as_of, ingest=not args.no_ingest)
    summary = runner.run()
    print("\n=== Run summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
