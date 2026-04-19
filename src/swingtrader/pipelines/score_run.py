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

from swingtrader.dashboard.charts import generate_charts_for_packet
from swingtrader.dashboard.packet import (
    build_all_lightweight_packets,
    enrich_with_context,
)
from swingtrader.dashboard.selector import select_packets
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

            # Step 6 — Build lightweight packets for ALL symbols (packet-first)
            # Eligibility, freshness, action, bucket are all computed internally
            # by build_lightweight_packet.  No separate DataFrame column passes needed.
            log.info("Building lightweight packets for all symbols…")
            all_packets = build_all_lightweight_packets(ranked_snapshot)
            summary["n_all_packets"] = len(all_packets)

            # Bucket counts from packets (canonical)
            from collections import Counter
            bucket_counter = Counter(p.get("bucket", "excluded") for p in all_packets)
            buckets = dict(bucket_counter)
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

            # Step 9 — Select from packets → enrich top-N → charts → artifacts → dashboard
            report_dir = REPO_ROOT / "docs" / "reports" / "daily" / str(self.as_of.date())
            report_dir.mkdir(parents=True, exist_ok=True)

            # Select using packet-first selector
            selections = select_packets(all_packets)
            top_packets = selections["top"]   # breakout + pullback combined

            summary["n_top_setups"] = len(top_packets)
            summary["n_breakout"] = len(selections["breakout"])
            summary["n_pullback"] = len(selections["pullback"])

            # Build row lookup for enrich_with_context (needs raw snapshot row for file paths)
            sym_col = "user_symbol" if "user_symbol" in ranked_snapshot.columns else "symbol"
            row_by_sym = {
                str(row.get(sym_col, row.get("symbol", ""))): row
                for _, row in ranked_snapshot.iterrows()
            }

            # Enrich only the selected top-N packets with MA table / AVWAP / assessments
            enriched: list[dict] = []
            for pkt in top_packets:
                sym = pkt.get("symbol", "")
                row = row_by_sym.get(sym)
                if row is not None:
                    try:
                        pkt = enrich_with_context(pkt, row)
                    except Exception as exc:
                        log.warning("Context enrichment error for %s: %s", sym, exc)
                enriched.append(pkt)

            # Rebuild selections with enriched packets
            enriched_by_sym = {p.get("symbol"): p for p in enriched}
            selections["breakout"] = [enriched_by_sym.get(p["symbol"], p) for p in selections["breakout"]]
            selections["pullback"] = [enriched_by_sym.get(p["symbol"], p) for p in selections["pullback"]]
            selections["top"] = enriched

            # Generate charts (best-effort; fails silently per symbol)
            charted: list[dict] = []
            for pkt in enriched:
                try:
                    pkt = generate_charts_for_packet(pkt, report_dir)
                except Exception as exc:
                    log.warning("Chart error for %s: %s", pkt.get("symbol"), exc)
                charted.append(pkt)
            selections["top"] = charted
            charted_by_sym = {p.get("symbol"): p for p in charted}
            selections["breakout"] = [charted_by_sym.get(p["symbol"], p) for p in selections["breakout"]]
            selections["pullback"] = [charted_by_sym.get(p["symbol"], p) for p in selections["pullback"]]

            # AI analysis notes (best-effort)
            try:
                selections["top"] = enrich_packets_with_ai(selections["top"])
            except Exception as exc:
                log.warning("AI enrichment error: %s", exc)

            # Trader dashboard — thin rendering from packet selections
            try:
                dash_path = write_dashboard(
                    ranked_snapshot,          # kept for regime columns only
                    selections["top"],
                    self.as_of,
                    report_dir,
                    selections=selections,    # packet-first: extended/reversal/portfolio from packets
                    oos_metrics=oos_metrics or None,
                )
                summary["dashboard"] = str(dash_path)
            except Exception as exc:
                log.warning("Dashboard render error: %s", exc)

            # Machine-readable JSON artifacts — fully packet-driven
            try:
                artifact_paths = write_artifacts(
                    selections,
                    ranked_snapshot,
                    self.as_of,
                    report_dir,
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
