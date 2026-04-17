"""Model training orchestration for Phase 4.

Walk-forward OOS evaluation followed by a final production fit on all data
up to the embargo boundary.

Workflow
--------
1. build_training_dataset()
     Pools all per-symbol feature + label + state parquets into one DataFrame.
     Rows are (date, symbol); the date is in the index.

2. walk_forward_evaluate()
     Iterates over walk-forward splits.  For each fold:
       a. Fit all three models on the training slice.
       b. Predict on the test slice.
       c. Collect calibration_report() for classifiers, MAE/correlation for regressor.
     Returns per-fold OOS metrics dict.

3. fit_production_models()
     Fits the final models on the most recent train_window_days worth of data
     (excluding embargo), saves to models_dir, returns ModelBundle.

4. train_pipeline()   (single entry point)
     Calls 1→2→3, logs a summary, returns summary dict.

Anti-leakage check: build_training_dataset() asserts that no LABEL_COLUMNS
appear in the features parquets. If they do, it raises ValueError loudly.
"""
from __future__ import annotations

import json
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from swingtrader.models.calibration import aggregate_oos_reports, calibration_report
from swingtrader.models.estimators import (
    _SETUP_STATES,
    _TRADE_STATES,
    LABEL_COLUMNS,
    ModelBundle,
    _prep_xy,
    feature_cols,
    fit_failure_risk,
    fit_setup_score,
    fit_trade_score,
    predict_failure_risk,
    predict_setup_score,
    predict_trade_score,
)
from swingtrader.utils.config import REPO_ROOT, load_config
from swingtrader.utils.logging import get_logger
from swingtrader.validation.walk_forward import make_splits

log = get_logger(__name__)

_FEATURES_DIR = REPO_ROOT / "data" / "features"
_LABELS_DIR = REPO_ROOT / "data" / "labels"
_STATES_DIR = REPO_ROOT / "data" / "states"
_MODELS_DIR = REPO_ROOT / "models"


# ── Dataset builder ──────────────────────────────────────────────────────────

def build_training_dataset(
    features_dir: Path | None = None,
    labels_dir: Path | None = None,
    states_dir: Path | None = None,
    *,
    min_rows_per_symbol: int = 50,
) -> pd.DataFrame:
    """Pool all per-symbol parquets into one panel DataFrame.

    Returns a DataFrame with:
      - DatetimeIndex (the bar date)
      - column 'symbol'  (string)
      - all feature columns
      - state column 'state'
      - all label columns

    Raises ValueError if any feature parquet contains label columns.
    """
    features_dir = Path(features_dir or _FEATURES_DIR)
    labels_dir = Path(labels_dir or _LABELS_DIR)
    states_dir = Path(states_dir or _STATES_DIR)

    feature_files = sorted(features_dir.glob("*.parquet"))
    if not feature_files:
        raise FileNotFoundError(f"No feature parquets found in {features_dir}")

    frames: list[pd.DataFrame] = []

    for feat_path in feature_files:
        sym = feat_path.stem
        labels_path = labels_dir / feat_path.name
        states_path = states_dir / feat_path.name

        if not labels_path.exists() or not states_path.exists():
            log.debug("Skipping %s — missing labels or states parquet", sym)
            continue

        try:
            feat_df = pd.read_parquet(feat_path)
            lbl_df = pd.read_parquet(labels_path)
            st_df = pd.read_parquet(states_path)
        except Exception as exc:
            log.warning("Skipping %s — read error: %s", sym, exc)
            continue

        # Anti-leakage check
        leaked = LABEL_COLUMNS & set(feat_df.columns)
        if leaked:
            raise ValueError(
                f"Label columns found in features parquet for {sym}: {leaked}. "
                "This is a leakage violation. Do not write labels to the features file."
            )

        # Align all DataFrames on the date index
        combined = feat_df.join(
            lbl_df[list(LABEL_COLUMNS & set(lbl_df.columns))],
            how="left",
        ).join(
            st_df[["state"]],
            how="left",
        )
        combined.insert(0, "symbol", sym)

        if len(combined) < min_rows_per_symbol:
            log.debug("Skipping %s — only %d rows (< %d)", sym, len(combined), min_rows_per_symbol)
            continue

        frames.append(combined)

    if not frames:
        raise ValueError(
            "Training dataset is empty — no symbols with sufficient feature + label history."
        )

    dataset = pd.concat(frames, axis=0)
    dataset.index = pd.to_datetime(dataset.index)
    dataset.sort_index(inplace=True)

    log.info(
        "Training dataset: %d rows x %d columns across %d symbols",
        len(dataset),
        len(dataset.columns),
        dataset["symbol"].nunique(),
    )
    return dataset


# ── Walk-forward OOS evaluation ──────────────────────────────────────────────

def walk_forward_evaluate(
    dataset: pd.DataFrame,
    cfg: dict | None = None,
) -> dict:
    """Run walk-forward OOS evaluation. Returns dict of aggregated metrics."""
    cfg = cfg or load_config("scoring")
    splits = make_splits(dataset.index, cfg=cfg)

    if not splits:
        log.warning("No walk-forward splits generated — dataset may be too short.")
        return {}

    setup_reports: list[dict] = []
    failure_reports: list[dict] = []
    trade_maes: list[float] = []
    trade_cors: list[float] = []

    for sp in splits:
        train_mask = sp.train_mask(dataset.index).to_numpy()
        test_mask = sp.test_mask(dataset.index).to_numpy()

        train_df = dataset[train_mask]
        test_df = dataset[test_mask]

        if len(train_df) < int(cfg.get("validation", {}).get("min_train_samples", 500)):
            log.debug("Fold %d: insufficient training rows (%d) — skipping", sp.fold, len(train_df))
            continue

        # ── Setup score fold ─────────────────────────────────────────────
        try:
            m_setup = fit_setup_score(train_df, cfg)
            x_test, y_test, _ = _prep_xy(test_df, "triggered_breakout", state_filter=_SETUP_STATES)
            if len(y_test) > 10:
                y_prob = predict_setup_score(m_setup, x_test)
                rep = calibration_report(y_test, y_prob, label=f"setup_score_fold{sp.fold}")
                rep["fold"] = sp.fold
                setup_reports.append(rep)
        except Exception as exc:
            log.warning("Fold %d setup_score error: %s", sp.fold, exc)

        # ── Failure risk fold ────────────────────────────────────────────
        try:
            m_fail = fit_failure_risk(train_df, cfg)
            x_test, y_test, _ = _prep_xy(test_df, "failed_within_10", state_filter=_TRADE_STATES)
            if len(y_test) > 10:
                y_prob = predict_failure_risk(m_fail, x_test)
                rep = calibration_report(y_test, y_prob, label=f"failure_risk_fold{sp.fold}")
                rep["fold"] = sp.fold
                failure_reports.append(rep)
        except Exception as exc:
            log.warning("Fold %d failure_risk error: %s", sp.fold, exc)

        # ── Trade score fold ─────────────────────────────────────────────
        try:
            target = "fwd_ret_h20" if "fwd_ret_h20" in train_df.columns else "fwd_ret_h10"
            m_trade = fit_trade_score(train_df, cfg)
            x_test, y_test, _ = _prep_xy(test_df, target, state_filter=_TRADE_STATES)
            if len(y_test) > 10:
                y_pred = predict_trade_score(m_trade, x_test)
                valid = np.isfinite(y_test) & np.isfinite(y_pred)
                if valid.sum() > 5:
                    trade_maes.append(float(np.mean(np.abs(y_test[valid] - y_pred[valid]))))
                    trade_cors.append(float(np.corrcoef(y_test[valid], y_pred[valid])[0, 1]))
        except Exception as exc:
            log.warning("Fold %d trade_score error: %s", sp.fold, exc)

    oos_metrics = {
        "n_folds": len(splits),
        "setup_score": aggregate_oos_reports(setup_reports),
        "failure_risk": aggregate_oos_reports(failure_reports),
        "trade_score": {
            "n_folds": len(trade_maes),
            "mae_mean": float(np.mean(trade_maes)) if trade_maes else float("nan"),
            "mae_std": float(np.std(trade_maes)) if len(trade_maes) > 1 else float("nan"),
            "correlation_mean": float(np.mean(trade_cors)) if trade_cors else float("nan"),
            "correlation_std": float(np.std(trade_cors)) if len(trade_cors) > 1 else float("nan"),
        },
    }

    log.info(
        "OOS evaluation: %d folds | setup Brier=%.4f | failure Brier=%.4f | trade MAE=%.4f",
        len(splits),
        oos_metrics["setup_score"].get("brier_score_mean", float("nan")),
        oos_metrics["failure_risk"].get("brier_score_mean", float("nan")),
        oos_metrics["trade_score"].get("mae_mean", float("nan")),
    )
    return oos_metrics


# ── Production model fit ─────────────────────────────────────────────────────

def fit_production_models(
    dataset: pd.DataFrame,
    cfg: dict | None = None,
    *,
    models_dir: Path | None = None,
    as_of: pd.Timestamp | None = None,
) -> ModelBundle:
    """Fit production models on the most recent training window and save to disk.

    Uses the most recent [train_window_days] of data (ending at the last bar
    that is at least embargo_days before as_of) so no future label leaks in.
    """
    cfg = cfg or load_config("scoring")
    models_dir = Path(models_dir or _MODELS_DIR)
    as_of = as_of or pd.Timestamp.today().normalize()

    val_cfg = cfg.get("validation", {})
    train_window_days = int(val_cfg.get("train_window_days", 1460))
    embargo_days = int(val_cfg.get("embargo_days", 25))

    cutoff = as_of - pd.Timedelta(days=embargo_days)
    train_start = cutoff - pd.Timedelta(days=train_window_days)
    train_df = dataset[(dataset.index >= train_start) & (dataset.index <= cutoff)]

    log.info(
        "Production fit: %d rows, %s → %s",
        len(train_df),
        train_start.date(),
        cutoff.date(),
    )

    feat_names = feature_cols(train_df)
    bundle = ModelBundle(
        fit_date=str(as_of.date()),
        feature_names=feat_names,
    )

    try:
        bundle.setup_score = fit_setup_score(train_df, cfg)
    except Exception as exc:
        log.warning("Production setup_score fit failed: %s", exc)

    try:
        bundle.trade_score = fit_trade_score(train_df, cfg)
    except Exception as exc:
        log.warning("Production trade_score fit failed: %s", exc)

    try:
        bundle.failure_risk = fit_failure_risk(train_df, cfg)
    except Exception as exc:
        log.warning("Production failure_risk fit failed: %s", exc)

    bundle.save(models_dir)
    return bundle


# ── Single entry point ───────────────────────────────────────────────────────

def train_pipeline(
    features_dir: Path | None = None,
    labels_dir: Path | None = None,
    states_dir: Path | None = None,
    *,
    cfg: dict | None = None,
    as_of: pd.Timestamp | None = None,
    models_dir: Path | None = None,
    skip_oos: bool = False,
) -> dict:
    """Full Phase 4 training pipeline.

    Returns a summary dict with OOS metrics and artifact paths.
    """
    cfg = cfg or load_config("scoring")
    as_of = as_of or pd.Timestamp.today().normalize()
    models_dir = Path(models_dir or _MODELS_DIR)

    summary: dict = {"as_of": str(as_of.date()), "ok": False}

    try:
        dataset = build_training_dataset(features_dir, labels_dir, states_dir)
        summary["n_rows"] = len(dataset)
        summary["n_symbols"] = int(dataset["symbol"].nunique())
        summary["date_range"] = f"{dataset.index.min().date()} → {dataset.index.max().date()}"

        if not skip_oos:
            oos = walk_forward_evaluate(dataset, cfg)
            summary["oos_metrics"] = oos
        else:
            summary["oos_metrics"] = {}

        bundle = fit_production_models(dataset, cfg, models_dir=models_dir, as_of=as_of)
        bundle.oos_metrics = summary["oos_metrics"]
        bundle.save(models_dir)  # re-save with oos metrics embedded

        summary["models_dir"] = str(models_dir)
        summary["fit_date"] = bundle.fit_date
        summary["ok"] = True

    except Exception:
        summary["error"] = traceback.format_exc()
        log.error("train_pipeline failed:\n%s", summary["error"])

    return summary


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Fit swingtrader scoring models (Phase 4)")
    parser.add_argument("date", nargs="?", default=None, help="As-of date YYYY-MM-DD")
    parser.add_argument("--skip-oos", action="store_true", help="Skip walk-forward OOS (faster)")
    args = parser.parse_args()

    as_of = pd.Timestamp(args.date) if args.date else pd.Timestamp.today().normalize()
    summary = train_pipeline(as_of=as_of, skip_oos=args.skip_oos)

    print("\n=== Training summary ===")
    for k, v in summary.items():
        if k != "oos_metrics":
            print(f"  {k}: {v}")
    if summary.get("oos_metrics"):
        print("  oos_metrics:", json.dumps(summary["oos_metrics"], indent=4, default=str))


if __name__ == "__main__":
    main()
