# Release Checklist

Use this before the first production GitHub Actions run or after any significant change.

---

## Repository setup (one-time)

- [ ] **GitHub Pages enabled** — repo Settings → Pages → Source: `main` branch, `docs/` folder
- [ ] **`pages` environment exists** — Settings → Environments → `github-pages` (auto-created on first Pages deploy)
- [ ] **`contents: write` permission** — daily.yml requires this to commit artifacts; confirm it is set in the workflow or repo-level Actions permissions
- [ ] **Branch protection** — if main is protected, ensure the `swingtrader-bot` git user is allowed to push directly, or disable protection for bot commits

## Configuration review

- [ ] `config/universe.yaml` — `portfolio_holdings.symbols` matches current holdings
- [ ] `config/universe.yaml` — `custom_watchlist.symbols` is current
- [ ] `config/universe.yaml` — `non_equity_symbols` includes all cash/money-market positions (e.g., SPAXX)
- [ ] `config/universe.yaml` — `symbol_aliases` has entries for any `.`-containing symbols (BRK.B → BRK-B is pre-configured)
- [ ] `config/universe.yaml` — `active_sources.sp500` and `active_sources.nasdaq100` are `false` unless the CSV files exist under `data/reference/`
- [ ] `config/scoring.yaml` — state machine thresholds reviewed; if changed, plan a full re-run
- [ ] `config/labels.yaml` — label horizons and embargo bars are as intended

## Local smoke test (run before first Actions run)

```bash
# 1. Install
uv sync --all-extras   # or: pip install -e ".[dev]"

# 2. Tests pass
pytest -q
# Expected: 215 passed

# 3. Lint clean
ruff check src tests

# 4. Daily run (will fetch data from yfinance — needs network)
python -m swingtrader.pipelines.daily_run

# 5. Score run (cold start — models not yet fitted, scores will be NaN)
python -m swingtrader.pipelines.score_run

# 6. Train models and re-score
python -m swingtrader.pipelines.score_run --train --skip-oos

# 7. Verify outputs exist
ls data/snapshots/
ls data/scores/
ls data/score_history/
ls docs/reports/daily/
ls reports/daily/

# 8. Check Pages index renders
python -m swingtrader.pipelines.pages_build
# Open docs/index.html in a browser
```

## Per-run verification (daily)

- [ ] GitHub Actions `daily` job completed green
- [ ] `data/scores/latest.parquet` timestamp is today
- [ ] `reports/daily/YYYY-MM-DD/snapshot.md` exists and is readable
- [ ] `docs/reports/daily/YYYY-MM-DD/snapshot.html` exists
- [ ] No unexpected errors in the Actions log (`N errors` should be 0 or explained)
- [ ] GitHub Pages updated (check the Pages URL after deploy completes)

## After adding a new ticker

- [ ] Add to `config/universe.yaml` under `custom_watchlist` or `portfolio_holdings`
- [ ] If the ticker uses a non-standard symbol (dot, special char), add an alias under `symbol_aliases`
- [ ] If it is a cash/non-equity instrument, add it to `non_equity_symbols`
- [ ] Run `python -m swingtrader.pipelines.daily_run` locally to verify it ingests cleanly

## After removing a ticker

- [ ] Remove from `config/universe.yaml`
- [ ] Optionally delete its parquet files from `data/raw/daily/`, `data/features/`, `data/states/`, `data/labels/` — not required, but keeps storage tidy
- [ ] Historical training data already committed is unaffected

## After changing a scoring threshold

- [ ] Update `config/scoring.yaml`
- [ ] Re-run `daily_run` to recompute states (cached states reflect old thresholds)
- [ ] Re-run `score_run --train --skip-oos` to refit models on new state definitions
- [ ] Compare new `data/scores/latest.parquet` to previous version — spot-check ARMED/TRIGGERED lists

## After changing a label definition

- [ ] Update `config/labels.yaml`
- [ ] Re-run `daily_run` to recompute labels
- [ ] Re-run `score_run --train` (full OOS, not `--skip-oos`) — label change requires fresh validation
- [ ] Document the change and re-validation result in `docs/`

## Model cold-start checklist

On first ever run, or after deleting `models/`:

- [ ] At least one full daily_run completed so `data/features/` and `data/states/` are populated
- [ ] Run `python -m swingtrader.pipelines.score_run --train --skip-oos`
- [ ] Check log for `Training complete: ok=True`
- [ ] `models/meta.json` exists and shows `is_fitted: true`
- [ ] Re-run `score_run` (no `--train`) to verify scores load from the new models

## Known issues / not-yet-resolved

- Intraday 5m data requires a separate fetch process; not automated in the daily cron.
  Features `intraday_rvol`, `intraday_vwap_dist_pct`, etc. will be `NaN` until parquets exist.
- Walk-forward OOS (`--train` without `--skip-oos`) requires substantial history (≥2000 samples
  per fold). Early project runs will skip most folds and log warnings.
- Survivorship bias is structural. All model and backtest metrics carry a caveat in the report footer.
