# AGENTS.md — guardrails for future work on swingtrader

These rules apply to any automated agent or human editor working in this repo. They encode
the research philosophy this project is built on.

## Hard constraints

1. **Analysis-only.** Do not add broker integrations, order management, execution routing,
   rebalancing, or any connection to a brokerage / trading API. If a module starts to look
   like a trade executor, stop and raise the concern.

2. **No arbitrary scoring weights.** Never compose a score from hand-picked point values
   (e.g. `trend=25, volume=10, setup=20`). Every score must be a calibrated output of a
   model trained against a measurable, pre-registered label, with out-of-sample validation.
   The exception: a temporary scaffold score clearly labelled `TODO: replace with fitted
   model` — and that must be removed before a phase is called complete.

3. **No lookahead leakage.** Every feature must declare its lookback window and must be
   computable from information available at time `t`. Add a test in
   `tests/test_labels_no_leakage.py` (or extend an existing one) for any new feature that
   could plausibly leak.

4. **Pre-registered horizons.** Label horizons, ATR thresholds, and forward-return
   windows live in `config/labels.yaml`. Do not report results for ad-hoc horizons without
   adding them there first. Multiple-testing discipline is what makes the backtest credible.

5. **Low-cost, GitHub-native.** No paid data vendors in v1. No databases. No cloud infra
   beyond GitHub-hosted runners and Pages. Parquet / CSV / YAML / JSON only.

## Architectural rules

6. **Single vendor boundary.** Only `src/swingtrader/ingest/yfinance_source.py` imports
   `yfinance`. Swapping data providers later should be a one-file change.

7. **Feature registry.** Every feature is declared in `src/swingtrader/features/registry.py`
   with `(callable, timeframe, lookback, allows_realtime)`. Features added without a
   registry entry will not be picked up by the pipeline.

8. **Configs over code.** Parameters (thresholds, windows, horizons, regime bucket edges)
   belong in YAML under `config/`. Changing a threshold should not require a code edit.

9. **Interpretable models first.** Default to L2-logistic / ridge / quantile regression.
   Tree ensembles or gradient boosting are allowed only if they materially outperform the
   baseline on walk-forward OOS metrics and the gain survives a regime-stratified check.

10. **Universe discipline.** v1 operates on the curated liquid universe only (see
    `config/universe.yaml`). Do not add full-market scanners.

## Reporting rules

11. **Every decile/regime report must footer the biases it does not control for.**
    Survivorship bias, point-in-time membership approximation, and data-quality caveats
    must be named in the output, not hidden.

12. **Score decomposition is mandatory.** Any per-symbol score shown to the user must
    carry a decomposition listing the top positive and negative feature contributions.

## Process rules

13. **Phase completion.** A phase is not complete until: (a) code runs, (b) tests pass,
    (c) docs are updated, (d) the commit (or summary) names what was added and what is
    intentionally deferred.

14. **Assumptions are documented.** If you make a judgment call instead of asking, write
    it into the relevant doc or config comment. Silent assumptions rot.
