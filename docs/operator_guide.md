# Operator Guide

Personal swing-trading research tool — daily operating reference.

---

## Configuration

### Watchlist
`config/universe.yaml` → `sources.custom_watchlist.symbols`

Add or remove tickers here. Changes take effect on the next daily run. No code change needed.

### Portfolio holdings
`config/universe.yaml` → `sources.portfolio_holdings.symbols`

These always appear in every report (portfolio section). `SPAXX` and any money-market/cash instruments must also be listed in `non_equity_symbols` — they are tagged and skipped during breakout scoring.

### Non-equity / cash instruments
`config/universe.yaml` → `non_equity_symbols`

Any symbol listed here is ingested but skipped for feature/state/label/score computation. It appears in the report as `SKIPPED (non_equity)`. SPAXX is the canonical example.

### Symbol aliases (dot→hyphen etc.)
`config/universe.yaml` → `symbol_aliases`

Format: `USER_SYMBOL: PROVIDER_SYMBOL`. BRK.B is pre-aliased to BRK-B for yfinance. Add new aliases here; do **not** rely on automatic substitution.

### State machine thresholds
`config/scoring.yaml` → `state_machine`

All pivot/ATR/bar thresholds for BASE, ARMED, TRIGGERED, ACCEPTED, CONFIRMED, FAILED, LATE transitions. Changing a threshold invalidates cached state histories — re-run the daily pipeline after any change.

### Data source settings
`config/data_sources.yaml` — yfinance parameters, rate limits, storage paths, quality thresholds.

### Label definitions (pre-registered)
`config/labels.yaml` — horizon windows, ATR thresholds, embargo bars. Do not add ad-hoc horizons without adding them here first (AGENTS.md rule 4).

---

## Data and artifact locations

| Artifact | Path |
|---|---|
| Raw daily bars | `data/raw/daily/{SYMBOL}.parquet` |
| Raw weekly bars | `data/raw/weekly/{SYMBOL}.parquet` |
| Intraday 5m bars | `data/raw/intraday/{SYMBOL}.parquet` |
| Features (per symbol) | `data/features/{SYMBOL}.parquet` |
| State history (per symbol) | `data/states/{SYMBOL}.parquet` |
| Training labels (per symbol) | `data/labels/{SYMBOL}.parquet` |
| Daily snapshot | `data/snapshots/YYYY-MM-DD.parquet` and `latest.parquet` |
| Daily scores | `data/scores/YYYY-MM-DD.parquet` and `latest.parquet` |
| Score history (rolling) | `data/score_history/history.parquet` |
| Universe artifact | `data/universe/universe.parquet` |
| Journal (open/closed trades) | `data/journal/trades.parquet` |
| Fitted models | `models/` (joblib + meta.json) |
| Markdown snapshot report | `reports/daily/YYYY-MM-DD/snapshot.md` |
| HTML snapshot report | `docs/reports/daily/YYYY-MM-DD/snapshot.html` |
| GitHub Pages index | `docs/index.html` |

---

## GitHub Actions jobs

### `ci.yml` — Continuous integration
Runs on every PR and every push to `main`.
- `ruff check src tests` — lint
- `pytest -q` — full test suite

Nothing is committed. Failure blocks merge.

### `daily.yml` — Daily pipeline (scheduled)
Scheduled: **22:00 UTC Monday–Friday** (~30 min after 17:30 ET close).
Can be triggered manually via workflow_dispatch with an optional `as_of_date` override.

Steps in order:
1. `python -m swingtrader.pipelines.daily_run` — ingest → features → states → labels → snapshot
2. `python -m swingtrader.pipelines.score_run [--train --skip-oos]` — score → rank → reports → history → Pages index
3. `git commit + push` — stages and commits all updated artifacts under `data/`, `models/`, `reports/daily/`, `docs/`

Model retraining schedule: `--train --skip-oos` is passed on Mondays and on manual `workflow_dispatch` runs. Full OOS evaluation runs only on Mondays.

### `pages.yml` — GitHub Pages deployment
Triggers on pushes to `main` that change `docs/**` or `reports/daily/**`.
Deploys the `docs/` directory to GitHub Pages.

Pages URL is set in GitHub repo Settings → Pages → Source: `main` branch, `docs/` folder.

---

## Running locally each day

### Full pipeline (ingest + score)

```bash
# Activate venv first:
source .venv/Scripts/activate   # Windows bash
# or: source .venv/bin/activate  (macOS/Linux)

# Full run (today's date):
python -m swingtrader.pipelines.daily_run
python -m swingtrader.pipelines.score_run

# Specific date:
python -m swingtrader.pipelines.daily_run 2025-03-14
python -m swingtrader.pipelines.score_run 2025-03-14

# Retrain models then score:
python -m swingtrader.pipelines.score_run --train --skip-oos

# Full OOS validation + retrain (slow, weekly):
python -m swingtrader.pipelines.score_run --train
```

### Score only (skip ingest, reuse yesterday's bars)

```bash
python -m swingtrader.pipelines.score_run
```

This is useful if ingest already ran and you only want to re-score or re-render reports.

### Rebuild GitHub Pages index only

```bash
python -m swingtrader.pipelines.pages_build
```

### Run tests

```bash
pytest -q
# or with uv:
uv run pytest -q
```

---

## What to inspect each day

1. **`reports/daily/YYYY-MM-DD/snapshot.md`** — Markdown snapshot, readable directly in GitHub. Check the portfolio section first, then ARMED/TRIGGERED watchlist names.

2. **`docs/reports/daily/YYYY-MM-DD/snapshot.html`** — Same data, HTML with score color-coding. Open locally or view via GitHub Pages.

3. **`data/scores/latest.parquet`** — Raw scored DataFrame. Load with `pd.read_parquet(...)` for custom filtering.

4. **Log output** — `daily_run` prints a summary line: `N ok, N skipped, N errors of N symbols`. Any non-zero error count needs investigation.

5. **GitHub Actions run** — Check the daily job at `github.com/{owner}/{repo}/actions`. A failed run means no new data was committed.

---

## Handling ticker failures

### Ingest failure (yfinance returns no data)
Symptom: log line `{SYMBOL}: ingest failed, skipping`. The symbol appears as `SKIPPED (ingest_failed)` in the snapshot.

Actions:
1. Check the yfinance status for that symbol: `python -c "import yfinance as yf; print(yf.download('SYMBOL', period='5d'))"`.
2. If the ticker was renamed or delisted, update `config/universe.yaml` (watchlist or portfolio_holdings).
3. If it's a transient yfinance outage, the next daily run will retry automatically.

### Symbol normalization failure (dot/hyphen)
Symptom: yfinance returns an empty DataFrame for a symbol that exists.

Check:
- Is it listed in `config/universe.yaml` → `symbol_aliases`?
- BRK.B → BRK-B is already mapped. Other `.`-containing symbols need explicit aliases.
- Add the alias under `symbol_aliases` in universe.yaml. No code change needed.

### Insufficient data (quality filter)
Symptom: `{SYMBOL}: insufficient data (N rows), skipping`.

The symbol has fewer rows than the quality threshold in `config/data_sources.yaml` → `quality`. This typically means a recently-listed stock. Either lower `max_missing_days_pct` (affects all symbols) or leave it — the symbol will start passing once it has enough history.

### Missing parquet file (no_data_file)
Symptom: `{SYMBOL}: no daily parquet found, skipping`.

The ingest step succeeded but the file was not written. Usually indicates a permissions or disk issue. Check `data/raw/daily/` for the file. Rerunning `daily_run` with `--no-ingest` skipped will not help — the file needs to be present.

### Non-equity / cash symbol (expected)
Symptom: `{SYMBOL}: skipped (non-equity / cash)`.

This is correct behavior for SPAXX and any symbol in `non_equity_symbols`. No action needed.

---

## Score-history idempotency

`append_daily_scores` is idempotent: it drops any existing rows for `as_of` before appending. Re-running `score_run` for the same date produces the same `data/score_history/history.parquet` regardless of how many times it is called.

---

## Known limitations and failure modes

### Survivorship bias
Universe membership uses monthly config snapshots, not true point-in-time history. Delisted tickers are absent from training data. Every report footer flags this. Do not interpret backtest win rates as deployment-ready statistics.

### Model cold start
On first run, no fitted model exists in `models/`. `score_run` falls back gracefully: all scores are `NaN`, the report renders with blank score columns, and the pipeline still completes. Run `score_run --train --skip-oos` to fit initial models (requires at least a few symbols with training data).

### Intraday features require separate data fetch
5m intraday bars (for `intraday_rvol`, `intraday_vwap_dist_pct`, etc.) must be fetched and stored in `data/raw/intraday/{SYMBOL}.parquet` before they will appear in scores. The daily pipeline does not auto-fetch intraday bars by default (yfinance 5m window is ~60 days; a separate manual or scheduled fetch is needed). Intraday feature columns will be `NaN` if the file is missing.

### yfinance rate limits
Batch size is 50 symbols with a 2-second sleep between batches (configurable in `data_sources.yaml`). Large universes take several minutes. If yfinance returns HTTP 429, the backoff is 5 seconds per retry (3 retries max). Persistent failures usually resolve within 1–2 hours.

### VIX bars
`^VIX` is fetched as a regular symbol. If yfinance returns empty data for `^VIX`, the VIX regime feature will be `NaN` for all symbols. A warning is logged. Everything else continues normally.

### Walk-forward OOS requires sufficient history
Full OOS (`--train` without `--skip-oos`) requires at least `min_train_samples: 2000` rows per fold (from `config/scoring.yaml`). Early in the project, before enough symbols have multi-year history, folds may be skipped. The pipeline logs a warning per skipped fold and continues.

### GitHub Pages deployment lag
Pages deploys on push to `main`. The daily cron commits and pushes at ~22:00 UTC. Pages deployment typically completes within 1–3 minutes after the push. If the Pages job fails, re-trigger `pages.yml` manually via workflow_dispatch.

### Windows-specific path separators
The repo uses `pathlib.Path` throughout and is developed on Windows. All paths use forward slashes in config YAMLs. No known issues on Linux (GitHub Actions uses ubuntu-latest).
