# Phase 4: Scoring Models

Phase 4 adds calibrated machine-learning scores to every symbol in the universe.
Features and labels from Phase 3 are consumed as inputs; no new data sources are
introduced.

## Component map

```
validation/walk_forward.py   WalkForwardSplit, make_splits()
models/calibration.py        brier_score, ece, reliability_diagram, calibration_report
models/estimators.py         feature_cols, fit_setup_score, fit_trade_score, fit_failure_risk
                             ModelBundle.save() / .load()
models/train.py              build_training_dataset, walk_forward_evaluate, fit_production_models
                             train_pipeline()  (CLI entry point)
scoring/generator.py         score_features_row, score_all_symbols, load_models
scoring/ranking.py           rank_within_state, top_n_per_state, build_ranked_snapshot
journal/schema.py            TradeRecord, upsert_record, journal_summary, auto_update_open_trades
reports/render.py            render_daily_markdown, render_daily_html, write_daily_reports
reports/templates/           snapshot.md.j2, snapshot.html.j2
pipelines/score_run.py       ScoreRunner.run()  (CLI entry point)
```

## Three models

| model | target | estimator | calibration | training subset |
|---|---|---|---|---|
| `setup_score` | `triggered_breakout` (binary) | LogisticRegression (L2) | isotonic (5-fold CV) | rows where state ∈ {BASE, ARMED} |
| `trade_score` | `fwd_ret_h20` (continuous) | Ridge | none | rows where state ∈ {TRIGGERED, ACCEPTED} |
| `failure_risk` | `failed_within_10` (binary) | LogisticRegression (L2) | isotonic (5-fold CV) | rows where state ∈ {TRIGGERED, ACCEPTED} |

All models use a `SimpleImputer(median) → StandardScaler → estimator` pipeline,
so missing features (e.g. RS features when SPY is unavailable) are handled
without special-casing.

## Anti-leakage design

`LABEL_COLUMNS` in `models/estimators.py` is the exhaustive list of columns
written by `labels/generators.py`. `feature_cols(df)` excludes all of them
plus state-machine metadata, so they can never accidentally enter the model
matrix. `build_training_dataset()` asserts this at dataset construction time and
raises `ValueError` loudly if any label column appears in a features parquet.

## Walk-forward validation

```
train_window_days: 1460   (~4 years)
test_window_days:   90    (~1 quarter)
embargo_days:       25    (covers the longest forward label horizon of 20 bars)
```

`make_splits(dates, cfg=cfg)` generates non-overlapping folds by stepping
backward from the most recent date:

```
fold 0:  train [T₀, T₀+4y]   | embargo 25d | test [+25d, +25d+90d]
fold 1:  train [T₀-90d, T₀+4y-90d] | …
…
```

All symbols share the same calendar split boundary. This prevents the subtle
within-symbol lookahead that row-wise k-fold would introduce (future label of
symbol A could appear in the training set if rows are shuffled).

OOS metrics reported per fold:
- Classifiers: Brier score, Brier skill score, ECE
- Regressor: MAE, Pearson correlation

## Composite score formula

No hand-picked weights. The composite is the joint probability under conditional
independence:

```
BASE / ARMED:       composite = setup_score × (1 − failure_risk)
TRIGGERED / ACCEPTED: composite = σ(trade_score) × (1 − failure_risk)
```

`σ` is the logistic function mapping the unbounded Ridge output onto (0, 1).
All other states receive `NaN` composite; they are not in the entry-finding
domain.

## Ranking

Within each state group, `rank_within_state()` computes the percentile rank
(0–100) of `composite_score`. Cross-state ranking is not performed — a symbol
in ARMED state is not directly compared to one in CONFIRMED state, because the
models predict different things for each.

`top_n_per_state(ranked_df, n=20)` returns the highest-scoring candidates per
state, sorted descending by `percentile_rank`.

## Journal

`TradeRecord` tracks one triggered breakout from entry to exit. Fields:

| field | when set |
|---|---|
| symbol, trigger_date, entry_* | at trigger |
| pivot, trigger_pivot, atr_at_trigger, target_price, stop_price | at trigger |
| setup_score, trade_score, failure_risk, composite_score | at trigger (NaN if models not fitted) |
| exit_date, exit_price, exit_reason, pnl_atr, pnl_pct | at exit (CONFIRMED / FAILED / TIMEOUT) |

`auto_update_open_trades(states_dir, journal_path)` scans every open journal
record's states parquet for CONFIRMED or FAILED transitions after the trigger
date and closes them automatically.

`journal_summary(df)` computes win rate, avg ATR P&L, profit factor, and
expectancy. All stats exclude STILL_OPEN records.

## Reports

Phase 4 replaces the plain Phase 3 markdown with scored, ranked tables in both
Markdown and HTML. Jinja2 templates in `reports/templates/` control layout.

The HTML report uses a dark GitHub-style theme with colour-coded state and score
columns:
- composite ≥ 0.6 → green
- composite ≥ 0.3 → amber
- composite < 0.3 → red

`write_daily_reports(snapshot_df, scores_df, as_of, output_dir)` writes both
files to `reports/daily/YYYY-MM-DD/snapshot.{md,html}`.

## Daily workflow integration

The GitHub Actions daily workflow now runs two steps:

```yaml
- run: uv run python -m swingtrader.pipelines.daily_run    # Phase 3
- run: uv run python -m swingtrader.pipelines.score_run    # Phase 4
```

Models are retrained every Monday (or on manual dispatch):

```yaml
TRAIN_FLAG="--train --skip-oos"   # weekly retrain, skip slow OOS
```

Full OOS evaluation is done separately (manual dispatch or local run):
```bash
python -m swingtrader.pipelines.score_run --train   # includes OOS
```

## Artifact layout (additions)

```
data/
  scores/
    YYYY-MM-DD.parquet   — scored + ranked DataFrame for the day
    latest.parquet

data/
  journal/
    trades.parquet        — cumulative trade journal

models/
  setup_score.joblib
  trade_score.joblib
  failure_risk.joblib
  meta.json               — fit_date, oos_metrics, feature_names

reports/
  daily/
    YYYY-MM-DD/
      snapshot.md         — scored markdown report (replaces Phase 3 version)
      snapshot.html       — styled HTML report (new in Phase 4)

reports/
  templates/
    snapshot.md.j2
    snapshot.html.j2
```

## CLI

```bash
# Score only (uses cached models)
python -m swingtrader.pipelines.score_run

# Retrain then score
python -m swingtrader.pipelines.score_run --train

# Fast retrain (no OOS evaluation)
python -m swingtrader.pipelines.score_run --train --skip-oos

# Train models standalone
python -m swingtrader.models.train
python -m swingtrader.models.train --skip-oos

# Backfill specific date
python -m swingtrader.pipelines.score_run 2025-03-14
```

## Test coverage

82 new tests (163 total) across 5 new test files:

| file | key tests |
|---|---|
| `test_validation.py` | chronological folds, no train/test overlap, embargo gap, mask exclusivity |
| `test_models.py` | feature_cols excludes labels, fit/predict shapes, None model → NaN, ModelBundle save/load, Brier/ECE/reliability |
| `test_scoring.py` | composite = setup × (1−failure) formula verified, NONE/CONFIRMED → NaN, unfitted bundle → NaN, rank range 0–100 |
| `test_journal.py` | TradeRecord round-trip, upsert dedup, win_rate, profit_factor |
| `test_reports.py` | md/html non-empty, bias note present, file creation, subdirectory creation |
