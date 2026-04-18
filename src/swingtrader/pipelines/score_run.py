"""Scoring pipeline — Phase 4/5 daily entry point.

This runs after daily_run.py has written features / states / labels.
Steps:
  1. Load (or train) ModelBundle
  2. Score every symbol on its latest bar
  3. Compute percentile ranks within each state group
  4. Merge scores into the daily snapshot
  5. Add freshness + action-label columns
  6. Select top 5-7 actionable setups; build analysis packets; generate charts
  7. Update open trade journal records
  8. Render trader-facing dashboard (dashboard.html) + research report (snapshot.html/md)
  9. Write data/scores/YYYY-MM-DD.parquet
 10. Append score history; rebuild Pages index

CLI:
  python -m swingtrader.pipelines.score_run                  # score only
  python -m swingtrader.pipelines.score_run --train          # (re)train then score
  python -m swingtrader.pipelines.score_run 2025-03-14       # specific date
  python -m swingtrader.pipelines.score_run --train --skip-oos  # fast retrain
"""
from __future__ import annotations

import argparse
import traceback
from pathlib import Path

import pandas as pd

from swingtrader.dashboard.action import (
    add_action_column,
    add_portfolio_guidance_column,
    add_setup_classification_column,
)
from swingtrader.dashboard.buckets import add_bucket_column, bucket_counts
from swingtrader.dashboard.charts import generate_charts_for_packet
from swingtrader.dashboard.eligibility import add_eligibility_columns
from swingtrader.dashboard.freshness import add_freshness_columns
from swingtrader.dashboard.packet import build_packets
from swingtrader.dashboard.selector import (
    select_breakout_candidates,
    select_portfolio_holdings,
    select_pullback_candidates,
    select_top_setups,
)
from swingtrader.journal.schema import auto_update_open_trades
from swingtrader.models.estimators import ModelBundle
from swingtrader.models.train import train_pipeline
from swingtrader.pipelines.pages_build import build_index
from swingtrader.reports.ai_notes import enrich_packets_with_ai
from swingtrader.reports.artifacts import write_artifacts
from swingtrader.reports.dashboard import write_dashboard
from swingtrader.reports.render import write_daily_reports
from swingtrader.scoring.generator import score_all_symbols
from swingtrader.scoring.history import append_daily_scores
from swingtrader.scoring.ranking import build_ranked_snapshot, rank_within_state
from swingtrader.utils.config import REPO_ROOT, load_config
from swingtrader.utils.io import write_parquet
from swingtrader.utils.logging import get_logger

log = get_logger(__name__)

_FEATURES_DIR = REPO_ROOT / "data" / "features"
_STATES_DIR = REPO_ROOT / "data" / "states"
_LABELS_DIR = REPO_ROOT / "data" / "labels"
_SNAPSHOT_DIR = REPO_ROOT / "data" / "snapshots"
_SCORES_DIR = REPO_ROOT / "data" / "scores"
_MODELS_DIR = REPO_ROOT / "models"
_JOURNAL_PATH = REPO_ROOT / "data" / "journal" / "trades.parquet"


class ScoreRunner:
    """Orchestrates Phase 4: model loading/training + scoring + reporting."""

    def __init__(
        self,
        as_of: pd.Timestamp | None = None,
        *,
        train: bool = False,
        skip_oos: bool = False,
    ) -> None:
        self.as_of = as_of or pd.Timestamp.today().normalize()
        self.train = train
        self.skip_oos = skip_oos
        self.cfg = load_config("scoring")
        self._scores_dir = Path(self.cfg.get("output", {}).get("scores_dir", str(_SCORES_DIR)))
        self._models_dir = Path(self.cfg.get("output", {}).get("models_dir", str(_MODELS_DIR)))

    # ── Public entry point ────────────────────────────────────────────────

    def run(self) -> dict:
        summary: dict = {"as_of": str(self.as_of.date()), "ok": False}

        try:
            # Step 1 — (Optional) retrain models
            oos_metrics: dict = {}
            if self.train:
                log.info("Training models (as_of=%s)…", self.as_of.date())
                train_summary = train_pipeline(
                    as_of=self.as_of,
                    models_dir=self._models_dir,
                    skip_oos=self.skip_oos,
                )
                oos_metrics = train_summary.get("oos_metrics", {})
                summary["train"] = train_summary
                log.info("Training complete: ok=%s", train_summary.get("ok"))

            # Step 2 — Load models
            bundle = _load_bundle(self._models_dir)
            summary["models_fitted"] = bundle.is_fitted

            # Step 3 — Score all symbols
            log.info("Scoring all symbols…")
            scores_df = score_all_symbols(
                features_dir=_FEATURES_DIR,
                states_dir=_STATES_DIR,
                bundle=bundle,
            )
            summary["n_scored"] = len(scores_df)

            # Step 4 — Percentile ranks
            scored_with_ranks = rank_within_state(scores_df)

            # Step 5 — Merge into snapshot
            snapshot_df = _load_latest_snapshot()
            if snapshot_df is not None and not snapshot_df.empty:
                ranked_snapshot = build_ranked_snapshot(snapshot_df, scored_with_ranks)
            else:
                log.warning("No snapshot found — reports will be scores-only.")
                ranked_snapshot = scored_with_ranks.reset_index()

            # Step 6 — Add freshness + action labels + setup classification + portfolio guidance
            ranked_snapshot = add_freshness_columns(ranked_snapshot)
            ranked_snapshot = add_action_column(ranked_snapshot)
            ranked_snapshot = add_setup_classification_column(ranked_snapshot)
            ranked_snapshot = add_portfolio_guidance_column(ranked_snapshot)

            # Step 6b — Hard eligibility gates + setup bucket assignment
            ranked_snapshot = add_eligibility_columns(ranked_snapshot)
            ranked_snapshot = add_bucket_column(ranked_snapshot)

            buckets = bucket_counts(ranked_snapshot)
            log.info(
                "Bucket counts — breakout: %d  pullback: %d  portfolio: %d  "
                "extended: %d  excluded: %d",
                buckets.get("breakout_long", 0),
                buckets.get("pullback_long", 0),
                buckets.get("portfolio_hold", 0),
                buckets.get("extended_leader", 0),
                buckets.get("excluded", 0),
            )
            summary["bucket_counts"] = buckets

            # Step 7 — Persist scores
            scores_path = self._scores_dir / f"{self.as_of.date()}.parquet"
            scores_path.parent.mkdir(parents=True, exist_ok=True)
            write_parquet(scored_with_ranks.reset_index(), scores_path)
            write_parquet(scored_with_ranks.reset_index(), self._scores_dir / "latest.parquet")
            summary["scores_path"] = str(scores_path)

            # Step 8 — Update journal open trades
            try:
                n_updated = auto_update_open_trades(_STATES_DIR, _JOURNAL_PATH)
                summary["journal_updated"] = n_updated
            except Exception as exc:
                log.warning("Journal update error: %s", exc)
                summary["journal_updated"] = 0

            # Step 9 — Build top setup packets + generate charts + render dashboard
            report_dir = REPO_ROOT / "docs" / "reports" / "daily" / str(self.as_of.date())
            report_dir.mkdir(parents=True, exist_ok=True)

            # Select breakout + pullback candidates separately, then combine
            breakout_df = select_breakout_candidates(ranked_snapshot)
            pullback_df = select_pullback_candidates(ranked_snapshot)

            # Combine for packet building: breakout first, then pullback fill
            # (select_top_setups does the same merge but we track counts here)
            top_df = select_top_setups(ranked_snapshot)
            packets = build_packets(top_df)

            summary["n_top_setups"] = len(packets)
            summary["n_breakout"] = len(breakout_df)
            summary["n_pullback"] = len(pullback_df)

            # Generate charts (best-effort; fails silently per symbol)
            packets_with_charts: list[dict] = []
            for pkt in packets:
                try:
                    pkt = generate_charts_for_packet(pkt, report_dir)
                except Exception as exc:
                    log.warning("Chart error for %s: %s", pkt.get("symbol"), exc)
                packets_with_charts.append(pkt)

            # AI analysis notes (best-effort; falls back to rule-based if no key)
            try:
                packets_with_charts = enrich_packets_with_ai(packets_with_charts)
            except Exception as exc:
                log.warning("AI enrichment error: %s", exc)

            # Trader dashboard (primary output)
            try:
                dash_path = write_dashboard(
                    ranked_snapshot,
                    packets_with_charts,
                    self.as_of,
                    report_dir,
                    oos_metrics=oos_metrics or None,
                )
                summary["dashboard"] = str(dash_path)
            except Exception as exc:
                log.warning("Dashboard render error: %s", exc)

            # Machine-readable JSON artifacts
            try:
                portfolio_df = select_portfolio_holdings(ranked_snapshot)
                artifact_paths = write_artifacts(
                    packets_with_charts,
                    portfolio_df,
                    ranked_snapshot,
                    self.as_of,
                    report_dir,
                    breakout_df=breakout_df,
                    pullback_df=pullback_df,
                )
                summary["artifacts_summary"] = str(artifact_paths.get("summary", ""))
            except Exception as exc:
                log.warning("Artifacts write error: %s", exc)

            # Research snapshot (legacy raw tables; kept for completeness)
            try:
                report_paths = write_daily_reports(
                    ranked_snapshot,
                    scored_with_ranks,
                    self.as_of,
                    output_dir=report_dir,
                    oos_metrics=oos_metrics or None,
                )
                summary["report_md"] = str(report_paths.get("markdown", ""))
                summary["report_html"] = str(report_paths.get("html", ""))
            except Exception as exc:
                log.warning("Snapshot report render error: %s", exc)

            # Step 10 — Append to score history
            try:
                append_daily_scores(scored_with_ranks, self.as_of)
            except Exception as exc:
                log.warning("score_history append error: %s", exc)

            # Step 11 — Rebuild GitHub Pages index
            try:
                build_index(as_of=self.as_of)
            except Exception as exc:
                log.warning("pages_build error: %s", exc)

            summary["ok"] = True
            log.info(
                "score_run complete: %d symbols scored, %d top setups",
                summary["n_scored"], summary["n_top_setups"],
            )

        except Exception:
            summary["error"] = traceback.format_exc()
            log.error("score_run failed:\n%s", summary["error"])

        return summary


# ── Helpers ──────────────────────────────────────────────────────────────────

def _load_bundle(models_dir: Path) -> ModelBundle:
    """Load ModelBundle; return an empty unfitted bundle on any error."""
    try:
        return ModelBundle.load(models_dir)
    except Exception as exc:
        log.warning("Could not load models from %s (%s) — scores will be NaN.", models_dir, exc)
        return ModelBundle()


def _load_latest_snapshot() -> pd.DataFrame | None:
    """Load the most recent snapshot parquet, or None if not found."""
    latest = _SNAPSHOT_DIR / "latest.parquet"
    if latest.exists():
        try:
            return pd.read_parquet(latest)
        except Exception as exc:
            log.warning("Could not read snapshot: %s", exc)
    # Try finding any recent snapshot by date
    snaps = sorted(_SNAPSHOT_DIR.glob("*.parquet"))
    snaps = [p for p in snaps if p.name != "latest.parquet"]
    if snaps:
        try:
            return pd.read_parquet(snaps[-1])
        except Exception:
            pass
    return None


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="swingtrader Phase 4 scoring pipeline")
    parser.add_argument("date", nargs="?", default=None, help="As-of date YYYY-MM-DD")
    parser.add_argument("--train", action="store_true", help="Retrain models before scoring")
    parser.add_argument("--skip-oos", action="store_true", help="Skip OOS evaluation when training")
    args = parser.parse_args()

    as_of = pd.Timestamp(args.date) if args.date else pd.Timestamp.today().normalize()
    runner = ScoreRunner(as_of=as_of, train=args.train, skip_oos=args.skip_oos)
    summary = runner.run()

    print("\n=== Score run summary ===")
    for k, v in summary.items():
        if k not in ("error", "train"):
            print(f"  {k}: {v}")
    if "error" in summary:
        print(f"  ERROR:\n{summary['error']}")


if __name__ == "__main__":
    main()
