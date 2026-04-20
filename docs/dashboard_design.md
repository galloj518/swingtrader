# Dashboard Design Reference

This document explains how the trader-facing dashboard works — how each piece
is computed, what is model-driven vs rule-based, and where the thresholds live
so you can tune them without hunting through source files.

---

## 1. Overview — Three Output Layers

Every `score_run` execution writes three output files into
`docs/reports/daily/{YYYY-MM-DD}/`:

| File | Purpose | Audience |
|---|---|---|
| `dashboard.html` | Actionable daily decision-support view | Trader, daily review |
| `snapshot.html` | Full research dump (all symbols, all columns) | Backtesting, model review |
| `snapshot.md` | Markdown mirror of snapshot.html | Grep / scripting |

`docs/index.html` (GitHub Pages root) links to the latest `dashboard.html`
with a prominent call-to-action and chips for today's actionable names.

---

## 2. Pipeline Order

The following steps run inside `score_run.ScoreRun.run()` after the model
scoring step:

```
1-5  Fetch data -> compute features -> label states -> score -> rank
6    build_all_lightweight_packets(ranked_snapshot)   # packet.py
7    select_packets(all_packets)                      # selector.py
8    enrich selected packets with context             # packet.py/context.py
9    generate_charts_for_packet(pkt, dir)             # charts.py
10   write_daily_reports(snapshot.html / .md)         # existing
11   write_dashboard(..., selections=...)             # reports/dashboard.py
12   build_index()                                    # pages_build.py
```

As of 2026-04-19, the packet is the source of truth. The dashboard no longer
reconstructs setup meaning late in rendering; it presents the packet produced by
`build_lightweight_packet()` and the selector only surfaces packets that are
coherent and complete enough for a trader-facing card.

---

## 3. Freshness Classification (`dashboard/freshness.py`)

Before any action label is assigned, each row gets five derived boolean
columns. This prevents stale confirmed names and over-extended names from
contaminating the actionable list.

### Thresholds

| Constant | Value | Meaning |
|---|---|---|
| `EXT_ATR` | 3.0 | Close more than 3× ATR above the pivot → extended |
| `FRESH_MAX_DAYS["TRIGGERED"]` | 10 | Triggers older than 10 bars are stale |
| `FRESH_MAX_DAYS["ACCEPTED"]` | 15 | |
| `FRESH_MAX_DAYS["ARMED"]` | 30 | |
| `FRESH_MAX_DAYS["BASE"]` | 60 | |
| `STALE_CONFIRMED_DAYS` | 20 | CONFIRMED names older than 20 bars are excluded |

### Output columns

| Column | Type | Rule |
|---|---|---|
| `is_extended` | bool | `state ∈ {LATE, EXHAUSTED}` OR `dist_to_pivot_atr > EXT_ATR` |
| `is_stale_confirmed` | bool | `state == CONFIRMED AND days_in_state > 20` |
| `is_fresh` | bool | In scored state AND not extended AND `days_in_state ≤ FRESH_MAX_DAYS[state]` |
| `is_actionable` | bool | `is_fresh AND state ∈ {BASE, ARMED, TRIGGERED, ACCEPTED}` |
| `freshness_label` | str | `fresh / stale / extended / stale-confirmed / not-scored` |

---

## 4. Action Labels (`dashboard/action.py`)

Each symbol receives exactly one of five labels. Rules are evaluated in
priority order; the first match wins.

### Labels

| Label | Meaning |
|---|---|
| `Actionable now` | In trade (TRIGGERED/ACCEPTED), fresh, score ≥ 0.30 |
| `Actionable on breakout` | ARMED or BASE, near pivot (≤ 1.5 ATR away) |
| `Actionable on pullback` | Setup exists but price is too far from pivot, or trigger is old |
| `Extended, wait` | Price has run (> 3 ATR above pivot), LATE, or EXHAUSTED |
| `Avoid / low quality` | Score < 0.20, failure_risk > 0.70 with weak score, or non-scored state |

### Priority order

```python
1. state ∈ {LATE, EXHAUSTED}         → Extended, wait
2. state ∉ SCORED_STATES             → Avoid
   score < MIN_SCORE (0.20)          → Avoid
   high failure_risk + weak score    → Avoid
3. is_extended (price > EXT_ATR)     → Extended, wait
4. state ∈ {TRIGGERED, ACCEPTED}
     fresh AND score ≥ 0.30          → Actionable now
     else                            → Actionable on pullback
5. state ∈ {ARMED, BASE}
     dist ≤ ARM_DIST_ATR (1.5)       → Actionable on breakout
     else                            → Actionable on pullback
6. fallthrough                       → Avoid
```

**Why LATE/EXHAUSTED is checked first:** LATE is not in `SCORED_STATES`, so
without the early-exit check it would fall through to the `state ∉ SCORED_STATES`
→ AVOID rule. The intent is "extended, mature trade" not "bad setup", so the
early check ensures the correct label.

### Thresholds (all in `action.py`)

| Constant | Value |
|---|---|
| `MIN_SCORE` | 0.20 |
| `MAX_FAILURE_RISK` | 0.70 |
| `ARM_DIST_ATR` | 1.5 |
| `NOW_MIN_SCORE` | 0.30 |
| `FRESH_TRIGGER_DAYS` | 8 |

---

## 5. Trade Levels (`dashboard/levels.py`)

All levels are derived from two inputs: `pivot` (the base's resistance high,
from `base_detect`) and `atr14` (14-bar ATR from the feature pipeline).
No levels are hand-picked; the formulas mirror the constants in
`config/scoring.yaml` and `config/labels.yaml`.

### Formulas

```
entry_lo  = pivot
entry_hi  = pivot + 0.10 × atr    (BREACH_ATR = 0.10, from labels.yaml)
stop      = pivot - 1.0  × atr    (STOP_ATR   = 1.0,  from scoring.yaml)
t1        = pivot + 2.0  × atr    (CONF_ATR   = 2.0,  from scoring.yaml)
t2        = pivot + 3.5  × atr
t3        = pivot + 5.0  × atr
```

### Support / resistance ladders (state-dependent)

| Level | Pre-trigger (BASE/ARMED) | In-trade (TRIGGERED/ACCEPTED) |
|---|---|---|
| S1 | close − 0.5 ATR (capped above stop) | pivot (now support) |
| S2 | stop | stop |
| S3 | pivot − 2.0 ATR | pivot − 1.5 ATR |
| R1 | pivot (immediate resistance) | T1 |
| R2 | T1 | T2 |
| R3 | T2 | T3 |

### Risk / reward

```
entry_mid    = (entry_lo + entry_hi) / 2
risk         = entry_mid − stop
reward       = t1 − entry_mid
risk_reward_t1 = reward / risk          (≈ 1.95× by construction)
entry_stop_atr = risk / atr             (≈ 1.05× by construction)
```

If `pivot` or `atr14` is missing (NaN), all levels are returned as NaN.

---

## 6. Top Setup Selection (`dashboard/selector.py`)

`select_top_setups(df)` returns at most `TOP_N = 7` rows.

### Filter steps

1. Exclude rows where `action_label == "Avoid / low quality"`.
2. Exclude rows where `is_non_equity == True`.
3. Sort by `(_tier ASC, composite_score DESC)`:
   - Tier 1: Actionable now
   - Tier 2: Actionable on breakout
   - Tier 3: Actionable on pullback
   - Tier 4: Extended, wait
4. **Diversity cap:** skip a row if its `group` tag already has
   `MAX_PER_GROUP = 3` representatives in the selected list. The group tag is
   the first value in the `groups` column (comma-separated sector/ETF cluster
   labels assigned during universe construction).
5. Fill until `TOP_N` or candidates exhausted.

### Floor behaviour

If fewer than `MIN_TOP = 3` fresh actionable setups exist (e.g. early in a
model cold-start when scores are unavailable), the filter is relaxed and
non-extended scored-state symbols fill the remaining slots.

---

## 7. Trade Narrative (`dashboard/narrative.py`)

`build_narrative(row, levels, action_label)` returns a dict with eight
plain-text keys:

| Key | Content |
|---|---|
| `setup` | One-sentence description of the base/trigger state |
| `why` | RS context + composite score + failure risk |
| `entry` | Entry idea based on action label (breakout zone, pullback target, etc.) |
| `risk` | Invalidation level in price terms |
| `targets` | T1/T2/T3 with R/R to T1 |
| `ma_context` | Close position vs SMA50 (ATR distance, slope inference) |
| `avwap_context` | YTD AVWAP distance in ATR units (blank if unavailable) |
| `verdict` | One-line action verdict matching the action label |

All text is **fully deterministic and rule-based** — no LLM, no external API
call. The same inputs always produce the same output.

---

## 8. Chart Generation (`dashboard/charts.py`)

Three PNG charts are generated per symbol, saved to
`docs/reports/daily/{YYYY-MM-DD}/charts/`:

| Chart | Source data | Indicators | Trade levels |
|---|---|---|---|
| `{SYM}_daily.png` | `data/raw/daily/{SYM}.parquet` | EMA20, SMA50, packet AVWAP overlays | Pivot, entry zone, stop, T1, T2 |
| `{SYM}_weekly.png` | `data/raw/weekly/` or daily resampled | EMA10w, EMA30w | Pivot, T1, T2 |

Intraday is no longer a primary dashboard chart in v1. The packet now carries
`intraday_policy = "daily_only"` and `intraday_used_in_qualification = false`,
so cards show a compact policy note instead of implying that an intraday
confirmation chart participated in qualification.

### Design choices

- Dark background `#0d1117` — matches the GitHub Pages site theme.
- Matplotlib only — no JS, no external charting libraries.
- Trade levels are `axhline` annotations with text labels via
  `ax.get_yaxis_transform()` so labels track the price axis on resize.
- Volume is a separate subplot sharing the x-axis (height ratio 3:1).
- All matplotlib imports are **local** (inside each function), so `charts.py`
  can be imported in test environments without matplotlib installed.

---

## 9. AI-Review Packet (`dashboard/packet.py`)

`build_packet(row)` assembles a complete, serialisable dict for each top
setup. It includes:

- All raw numeric fields from the scored snapshot (close, pivot, atr14,
  composite_score, failure_risk, dist_to_pivot_atr, etc.)
- All computed `TradeLevels` fields as formatted strings
- The full `narrative` dict
- `action_label`, freshness flags, `ma_slope` direction string, `rs_class`
  (outperforming / tracking / lagging)
- `chart_daily`, `chart_weekly`, `chart_intraday` — relative paths to PNGs
  (or `None` if charts were not generated)

The packet is embedded as a collapsible JSON block in each setup card under
the label **"AI-review packet"**. You can copy the JSON and paste it into a
chat with any LLM for a second opinion without having to explain the context
manually.

---

## 10. Dashboard HTML (`reports/dashboard.py`)

`render_dashboard(snapshot_df, packets, as_of, *, oos_metrics)` produces a
single self-contained HTML file with inline CSS (dark GitHub theme).

### Sections

| Section | Content |
|---|---|
| **Regime summary bar** | SPY trend (up/down/neutral from `regime_spy_trend`) |
| **Portfolio strip** | Chips for symbols in `config/portfolio.yaml`; shows current state + action label for each; greyed-out if not in the current snapshot |
| **Top setup cards** | One card per packet from `select_top_setups`; shows chart images, trade levels table, narrative, AI-review packet |
| **Full state tables** | Collapsible table per state (ARMED, BASE, TRIGGERED, ACCEPTED, CONFIRMED, FAILED, LATE/EXHAUSTED); all symbols in snapshot |

### Chart embedding

Chart images are referenced via relative paths from the output directory
(e.g. `charts/AAPL_daily.png`). Images render in the browser as long as the
`charts/` folder is in the same directory as `dashboard.html`. GitHub Pages
serves the whole `docs/` tree, so relative paths work correctly in production.

---

## 11. What Is Heuristic vs Model-Based

| Component | Type | Notes |
|---|---|---|
| `composite_score` | **Model** (calibrated) | Logistic / gradient-boosted classifier, fitted on historical state transitions; output is a calibrated probability |
| `failure_risk` | **Model** (calibrated) | Failure classifier; probability that the setup fails within the labeling window |
| `setup_score`, `trade_score` | **Model** | Sub-scores from the scoring pipeline; see `config/scoring.yaml` for feature definitions |
| Action label thresholds (`MIN_SCORE`, `ARM_DIST_ATR`, etc.) | **Heuristic** | Fixed constants; chosen to reflect reasonable entry discipline, not fitted to data |
| Freshness thresholds (`FRESH_MAX_DAYS`, `EXT_ATR`) | **Heuristic** | Chosen conservatively; can be adjusted per trading style |
| Trade levels (entry/stop/T1/T2/T3) | **Rule-based** | ATR multiples from `config/scoring.yaml`; internally consistent but not optimised for maximum expected value |
| Diversity cap (`MAX_PER_GROUP`) | **Heuristic** | Prevents sector clustering; not model-derived |
| Narrative text | **Rule-based** | Fully deterministic from feature values; no model inference |
| MA slope direction | **Derived** | Estimated from `close_vs_sma50`; directional only, not magnitude-calibrated |

**Bottom line:** the scores are model outputs with calibrated probabilities.
Everything else that turns a score into a dashboard card — freshness windows,
action label rules, levels, narrative, selection caps — is explicit rule-based
logic with documented constants. Changing a threshold is a one-line edit in
the relevant module.

---

## 12. Tuning Guide

### I want more/fewer symbols in the top list
Edit `selector.TOP_N` (max list length) and `selector.MIN_TOP` (minimum floor).

### The "extended" cutoff feels too aggressive
Lower `freshness.EXT_ATR` (default 3.0) to flag extensions earlier, or raise
it to tolerate wider moves.

### ARMED names are appearing too early before the pivot
Reduce `action.ARM_DIST_ATR` (default 1.5 ATR) to require setups to be closer
to the pivot before earning `Actionable on breakout`.

### I want "Actionable now" only for high-conviction setups
Raise `action.NOW_MIN_SCORE` (default 0.30).

### Stale CONFIRMED names keep showing up
Lower `freshness.STALE_CONFIRMED_DAYS` (default 20).

### Stop / target spacing feels too tight or loose
Edit the ATR multipliers in `dashboard/levels.py`:
`BREACH_ATR`, `STOP_ATR`, `CONF_ATR`, `T2_ATR`, `T3_ATR`.
These are in sync with `config/scoring.yaml` — keep them consistent.
