# Phase 3: Analysis Engine

Phase 3 implements the full per-symbol analysis stack that runs nightly:
ingest → features → base detection → state machine → labels → pipeline runner.
No model fitting happens here; Phase 4 will train calibrated scoring models against
the labels produced here.

## Component map

```
ingest/symbols.py          resolve_universe() → list[SymbolRecord]
ingest/yfinance_source.py  fetch_daily / resample_weekly / load_daily / load_weekly
features/primitives.py     true_range, atr_wilder, ema, sma, rolling_zscore, linear_slope
features/registry.py       @register decorator, REGISTRY, compute_features()
features/daily.py          16 daily features (ATR, volume, trend, proximity)
features/weekly.py         8 weekly features (trend, RS-proxy, compression)
features/relstrength.py    4 RS features (benchmark_df kwarg; NaN when absent)
features/regime.py         4 regime features (SPY/VIX; NaN when absent)
avwap/anchors.py           ytd / swing_low / swing_high / breakout_day anchors
avwap/calc.py              compute_avwap(), compute_avwap_std()
avwap/features.py          compute_avwap_features() → 25 columns
bases/base_detect.py       detect_bases(), resistance_touches()
features/pivot_features.py compute_pivot_features() → 7 columns
states/machine.py          compute_states() → state + metadata columns
labels/generators.py       compute_all_labels() → 8 label columns
pipelines/daily_run.py     DailyRunner.run() — orchestrates everything above
```

## Universe and symbol handling

`config/universe.yaml` controls which symbol groups are active:

```yaml
active_sources:
  benchmarks: true
  sector_etfs: true
  sp500: false          # CSV not required — silently skipped when absent
  nasdaq100: false
  portfolio_holdings: true
  custom_watchlist: true
```

`resolve_universe()` in `ingest/symbols.py` returns `list[SymbolRecord]`.
Each record carries:

| field | purpose |
|---|---|
| `user_symbol` | display form (`BRK.B`) |
| `provider_symbol` | fetch form (`BRK-B`) via `symbol_aliases` |
| `is_non_equity` | True for SPAXX etc. — skipped in feature/state/label computation |
| `is_portfolio` | included in portfolio section of report regardless of state |
| `score_eligible` | `not is_non_equity` |

## Feature registry

Feature functions are decorated with `@register`:

```python
@register("atr_14", timeframe="daily", lookback_bars=14, allows_realtime=True)
def _atr_14(df: pd.DataFrame, **_) -> pd.Series:
    return atr_wilder(df, 14)
```

`compute_features(df, timeframe, extra_kwargs=None)` calls every registered function
for the given timeframe and assembles a DataFrame. Extra kwargs (`benchmark_df`,
`vix_df`) are forwarded; functions that don't accept them receive nothing.

AVWAP features are **not** in the registry — they depend on state history and
produce a variable number of columns. The pipeline calls `compute_avwap_features()`
directly.

## Base detection

`detect_bases(df)` scans every bar `t` backward from `L=1` to `L=max_days`,
tracking `running_max_high` and `running_min_low`. The longest `L ≥ min_days`
where `(max_high - min_low) / max_high ≤ max_depth_pct` is recorded.

Complexity: O(n × max_days). Acceptable for daily bars with max_days=120.

The pivot = `max_high` in the accepted window — pure geometry, no lore.

**Important**: `pivot_arr[t]` includes bar `t`'s high, so `close[t] ≤ pivot_arr[t]`
always. The state machine therefore compares close against `pivot_arr[t-1]`
(yesterday's established resistance). See state machine section.

## Breakout state machine

States: `NONE → BASE → ARMED → TRIGGERED → ACCEPTED → CONFIRMED`
                                        `↘ FAILED`
                    `↘ LATE / EXHAUSTED`

All thresholds come from `config/scoring.yaml → state_machine.*`.

**Trigger correctness**: to avoid a trivial always-false trigger condition,
the machine uses `prior_p = pivot_arr[t-1]` as the resistance level:

```python
prior_p = pivot_arr[t - 1] if t > 0 else p
if close > prior_p + breach_atr * atr:
    new_state = TRIGGERED
    ms.trigger_pivot = prior_p   # stored for subsequent stop/target checks
```

The output DataFrame (`compute_states()`) includes:
`state, pivot, trigger_pivot, trigger_atr, trigger_date, consecutive_above,
days_in_state, state_changed`

## Labels

`compute_all_labels(df, states)` produces 8 columns:

| column | type | description |
|---|---|---|
| `setup_candidate` | 0/1 | BASE or ARMED and within 15% of 52w high |
| `triggered_breakout` | 0/1 | first bar of TRIGGERED state |
| `accepted_breakout` | 0/1 | first bar of ACCEPTED state |
| `failed_within_10` | 0/1/NaN | FAILED occurs in next 10 bars |
| `followthrough_confirmed_20` | 0/1/NaN | CONFIRMED occurs in next 20 bars |
| `fwd_ret_h5` | float/NaN | log(close+5 / close) / ATR |
| `fwd_ret_h10` | float/NaN | log(close+10 / close) / ATR |
| `fwd_ret_h20` | float/NaN | log(close+20 / close) / ATR |

Forward-looking labels are NaN for the last `horizon` bars. They are written
to a **separate parquet file** (`data/labels/{SYM}.parquet`) and must never
be loaded at inference time.

## Leakage guarantee

`test_state_machine.py::test_state_at_t_independent_of_future_bars` verifies that
`compute_states(df.iloc[:t+1])` produces the same state at index `t` as
`compute_states(df)`. Same property holds for features (all primitives use
rolling windows on past data only) and base detection (window is `[t-L+1, t]`).

## Daily pipeline

`DailyRunner.run()` (in `pipelines/daily_run.py`) orchestrates:

1. `resolve_universe()` → write universe artifact to `data/universe/`
2. Fetch benchmarks (SPY, ^VIX) once; resample weekly SPY
3. For each score-eligible symbol:
   - Ingest + quality check
   - Daily features → `data/features/{SYM}.parquet`
   - Weekly features appended to same file
   - AVWAP features appended
   - Base detect → state machine → `data/states/{SYM}.parquet`
   - Labels → `data/labels/{SYM}.parquet`
4. Build snapshot dict → `data/snapshots/YYYY-MM-DD.parquet`
5. Write `reports/daily/YYYY-MM-DD/snapshot.md`

Non-equity symbols (SPAXX) are included in portfolio reporting but skipped
in steps 3-5.

CLI:

```bash
python -m swingtrader.pipelines.daily_run          # use today's date, ingest on
python -m swingtrader.pipelines.daily_run 2025-03-14 --no-ingest
```

## AVWAP anchors

| anchor | definition |
|---|---|
| `ytd` | first trading bar of the current calendar year |
| `swing_low` | lowest daily close in the trailing 252 bars |
| `swing_high` | highest daily close in the trailing 252 bars |
| `breakout_day` | first bar where `state == TRIGGERED` (NaN if not yet triggered) |

For each anchor, `compute_avwap_features()` produces:
`{anchor}_avwap, {anchor}_dist_atr, {anchor}_stretch_atr,
{anchor}_slope_20, {anchor}_closes_above_20, {anchor}_reclaim_flag`

Plus `avwap_confluence_count` = number of AVWAPs within 0.5 ATR of close.

## Artifact layout

```
data/
  universe/          universe_YYYY-MM-DD.parquet
  daily/             {SYM}.parquet    (OHLCV, written by ingest)
  weekly/            {SYM}.parquet    (resampled from daily)
  features/          {SYM}.parquet    (daily + weekly + AVWAP features)
  states/            {SYM}.parquet    (state machine output)
  labels/            {SYM}.parquet    (forward-looking labels — inference NEVER reads this)
  snapshots/         YYYY-MM-DD.parquet

reports/
  daily/
    YYYY-MM-DD/
      snapshot.md
```

## Configuration knobs

All thresholds live in YAML; no magic numbers in code.

| file | section | key parameters |
|---|---|---|
| `scoring.yaml` | `state_machine.base` | `min_days=15, max_days=120, max_depth_pct=0.25` |
| `scoring.yaml` | `state_machine.armed` | `max_dist_to_pivot_atr=2.0, atr_contraction_pct=70` |
| `scoring.yaml` | `state_machine.triggered` | `pivot_breach_atr=0.10` |
| `scoring.yaml` | `state_machine.accepted` | `consecutive_closes=3` |
| `scoring.yaml` | `state_machine.confirmed` | `atr_gain_target=2.0, atr_stop=1.0` |
| `labels.yaml` | `setup_candidate` | `max_pct_from_high=0.15` |
| `labels.yaml` | `failed_breakout` | `horizon_bars=10` |
| `labels.yaml` | `followthrough_confirmed` | `horizon_bars=20` |
| `labels.yaml` | `forward_return` | `horizons_bars=[5,10,20], normalize_by_atr=true` |
| `avwap_anchors.yaml` | `anchors` | per-anchor enabled flag |

## Test coverage

81 tests across 9 test files, 0 ruff errors. Key tests:

- `test_features.py` — all 28 registered features produce correct shapes, no lookahead
- `test_avwap.py` — anchor selection, AVWAP monotone weights, stretch sign
- `test_base_detect.py` — pivot = max_high in window, depth ≤ threshold, min_days enforced
- `test_state_machine.py` — full transition ladder + no-lookahead property test
- `test_labels.py` — tail NaN for forward labels, leakage guard (features file must not contain label columns)
- `test_symbols.py` — BRK.B→BRK-B alias, SPAXX non-equity, portfolio tagging, dedup
- `test_pipeline_smoke.py` — end-to-end on synthetic data; parquet round-trip
