# Architecture

Design doc for the swingtrader research tool. Mirrors the plan agreed at project
kickoff; future edits here supersede the original plan when they conflict.

## 1. Philosophy

- **Scientific, not discretionary.** Every score is the output of a fitted, calibrated
  model trained on a pre-registered label. No arbitrary point weights. See
  [AGENTS.md](../AGENTS.md) rule 2.
- **Interpretable by default.** Regularized logistic / ridge / quantile regression;
  gradient boosting is allowed only if it materially outperforms on walk-forward OOS and
  survives a regime-stratified check.
- **Honest about limits.** yfinance data is retail grade; intraday history is short;
  universe membership is approximately point-in-time. Every report footers these
  caveats.

## 2. Data flow

```
  GitHub Actions schedule
           │
           ▼
   ingest/ (yfinance_source.py)     ──► data/raw/daily/{SYM}.parquet
     resample_weekly                ──► data/raw/weekly/{SYM}.parquet
     intraday 5m accumulator        ──► data/raw/intraday/{SYM}.parquet (gitignored)
           │
           ▼
   ingest/quality.py                ──► data/reports/quality/latest.json
           │
           ▼
   features/ (+ avwap/, bases/)     ──► data/features/{SYM}.parquet
           │
           ▼
   states/machine.py                ──► data/states/{SYM}.parquet
           │
           ▼
   labels/ (setup, trigger,         ──► data/labels/{SYM}.parquet
            followthrough, failure)
           │
           ▼
   models/ (fit: setup, trade,      ──► models/{model_name}.pkl + sidecar JSON
            failure)
           │
           ▼
   scoring/ (compose + decompose)   ──► data/scores/YYYY-MM-DD.parquet
                                        data/scores/latest.parquet
           │
           ▼
   reports/ (daily.py, site.py)     ──► reports/daily/YYYY-MM-DD/*.md
                                        reports/site/index.html (Pages-deployable)
```

## 3. Data model

| Asset | Format | Partition | Notes |
|---|---|---|---|
| Daily bars | parquet (snappy) | per symbol | Committed to repo. |
| Weekly bars | parquet | per symbol | Resampled locally from daily. |
| Intraday 5m | parquet | per symbol | Gitignored; forward-only accumulator. |
| Features | parquet | per symbol | Wide table indexed by date. |
| Labels | parquet | per symbol | One col per pre-registered label. |
| States | parquet | per symbol | Date-indexed state + pivot snapshot. |
| Scores | parquet | per day | `latest.parquet` tracks today's rank. |
| Journal | CSV | single file | Human-editable trade journal. |

Parquet-per-symbol keeps git diffs bounded and lets us parallelize fit/score per
symbol without contention.

## 4. Breakout state machine (summary)

States are driven by objective thresholds parameterized in `config/scoring.yaml`
and `config/labels.yaml`:

| State | Entry rule |
|---|---|
| `BASE` | ≥B weeks consolidating; range, ATR%, and distance-to-pivot below thresholds |
| `ARMED` | In BASE with recent range AND volume contraction, within ε·ATR of pivot |
| `TRIGGERED` | Close > pivot + τ·ATR (daily) OR 5m RVOL spike through pivot |
| `ACCEPTED` | N consecutive closes above pivot; no violation below pivot − σ·ATR |
| `CONFIRMED` | First-passage: +M·ATR reached before −σ·ATR |
| `FAILED` | Close back below pivot after TRIGGERED, or hit stop |
| `LATE` | Extension from 20 EMA or AVWAP > λ·ATR |
| `EXHAUSTED` | Post-CONFIRMED at M2·ATR with momentum divergence |

## 5. Scoring

| Score | Target | Model |
|---|---|---|
| SetupScore | P(trigger within W bars AND RVOL≥ρ on trigger) | L2 logistic + isotonic calibration |
| TradeScore | E[forward return / ATR over H bars given trigger] | Ridge (baseline) or quantile τ=0.5 |
| FailureRisk | P(fail within K bars given trigger) | L2 logistic + isotonic calibration |
| FinalRank | Learned composite, per state | Percentile rank within state |

Composite weights in FinalRank are **not hand-picked** — they are derived by OOS
regression of component scores against realized forward return. Per-symbol score
decomposition shows signed feature contributions `β_i · (x_i − x̄_i)`.

## 6. Validation

- Walk-forward: 4y train / 90d test / 25d embargo.
- Regime-stratified reporting: trend (SPY MA cross), volatility (VIX buckets),
  breadth (pct above 50-DMA across universe).
- Calibration: reliability plots + Brier score for every probability output.
- AVWAP incremental-value test: `with_avwap` vs `no_avwap` feature sets on identical
  folds; report ΔAUC / ΔR².
- Leakage: unit test rebuilds features on a truncated history and compares values at
  time `t` to the full-history computation at `t`.

## 7. Universe

- S&P 500 + Nasdaq 100 membership loaded from CSVs (refreshed monthly, snapshot
  under `config/universe_history/`), plus liquid sector ETFs, benchmarks, and a
  user watchlist.
- Liquidity filter (20d median $-volume ≥ $10M, price ≥ $5) applied after daily
  ingest, capped at 750 symbols for CI-friendly runtimes.
- v1 has approximate-not-true point-in-time universe membership; the survivorship
  bias is surfaced in every decile report footer.

## 8. Phase plan

- **Phase 2 (this commit)** — repo skeleton, configs, utils, yfinance ingest,
  quality checks, CI tests. **Complete.**
- **Phase 3** — feature registry, ~30 daily+weekly features, AVWAP anchors and
  features, base detection + pivot logic, state machine, label generators,
  leakage tests.
- **Phase 4** — fit models, calibration, composite ranking, daily ranking report,
  journal schema + comparison report, static site generation.
- **Phase 5** — walk-forward harness, decile/regime reports, AVWAP incremental
  test, weekly retraining workflow, polished docs.
- **Phase 6 (deferred)** — intraday-conditioned models once ≥1y of 5m history
  has accumulated.

## 9. Known limitations

- Intraday-conditioned model training is not feasible in v1. Intraday is a
  forward-only confirmation overlay.
- Earnings calendar via yfinance is unreliable; v1 uses it as a soft blackout,
  not as a feature.
- Universe membership is monthly-snapshot point-in-time — not perfect.
- All vendor-adjusted prices can silently misrepresent symbol changes; the
  quality checker flags large gaps, but careful users should keep a changelog
  of tickers they actively follow.
