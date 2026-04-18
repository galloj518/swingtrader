# Selector Architecture — Eligibility Gates, Setup Buckets, and Portfolio Logic

*Last updated: 2026-04-18. Covers the redesign introduced in the institutional swing-trade workflow pass.*

---

## Overview

The selection pipeline has three sequential stages before a symbol ever appears in the daily dashboard:

```
Universe (all scored symbols)
    │
    ▼
[Stage 1] Hard Eligibility Gates
    │  Reject symbols with structural defects
    ▼
[Stage 2] Bucket Assignment
    │  Route eligible symbols into mutually exclusive setup types
    ▼
[Stage 3] Bucket-Aware Selection
       Pick top names per bucket; portfolio logic fully separated
```

The old pipeline had a single flat ranked list with soft action labels. The redesign enforces hard gates before any ranking occurs, then keeps fresh-entry candidates and portfolio holdings in completely separate code paths.

---

## Stage 1 — Hard Eligibility Gates

**Module:** `src/swingtrader/dashboard/eligibility.py`  
**Entry point:** `assess_eligibility(row)` → `add_eligibility_columns(df)`

Each gate fires independently. A symbol is **ineligible** if *any* gate fires. All fired gate names are stored in `rejection_reasons` for transparency.

| Gate constant | Condition | Rationale |
|---|---|---|
| `GATE_NON_EQUITY` | `is_non_equity=True` | SPAXX, ETFs, indices — informational only, never entry candidates |
| `GATE_INVALID_STATE` | `state` not in `{BASE, ARMED, TRIGGERED, ACCEPTED}` | Only scored states are eligible; LATE/EXHAUSTED/FAILED etc. excluded |
| `GATE_BROKEN_TREND` | `close_vs_sma200 < -0.08` | Price more than 8% below 200 SMA — structural downtrend, avoid long entries |
| `GATE_POOR_RS` | `daily_rs_63 < -0.10` | Lagging SPY by more than 10% over 63 days — systematic underperformer |
| `GATE_WEAK_REGIME_POOR_RS` | `regime_spy_trend < 0` AND `daily_rs_63 < 0` | Market in downtrend AND stock underperforming — both conditions required |
| `GATE_HIGH_FAILURE_RISK` | `failure_risk > 0.65` | Model estimates >65% chance of setup failure — hard stop |
| `GATE_LOW_SCORE` | `composite_score < 0.18` | Score too low to be decision-relevant |
| `GATE_THIN_BASE` | state in `{BASE, ARMED}` AND `base_length < 5` | Base too short to be reliable |

### Warnings (non-disqualifying)
These appear in `eligibility_warnings` and surface in the dashboard as cautions:

| Warning | Condition |
|---|---|
| `WARN_BELOW_SMA50` | `close_vs_sma50 < 0` |
| `WARN_NEUTRAL_REGIME` | `regime_spy_trend` in `[-0.2, 0)` |
| `WARN_AGING_BASE` | state `BASE` or `ARMED` AND `base_length > 40` |
| `WARN_ELEVATED_FAILURE` | `failure_risk > 0.45` (below hard gate) |

### NaN handling
All gates use explicit `math.isfinite()` checks. A missing or NaN field does **not** trigger the gate — the gate is skipped gracefully. This prevents data gaps from incorrectly excluding symbols.

### Non-equity short-circuit
Gate 1 (`GATE_NON_EQUITY`) returns immediately — no other gates are evaluated. This prevents non-equity informational symbols from accumulating spurious rejection labels.

---

## Stage 2 — Bucket Assignment

**Module:** `src/swingtrader/dashboard/buckets.py`  
**Entry point:** `assign_bucket(row)` → `add_bucket_column(df)`

Buckets are mutually exclusive; priority order matters:

```
1. is_non_equity=True          → NON_EQUITY      (informational, never selected)
2. is_portfolio=True           → PORTFOLIO_HOLD  (always, overrides eligibility)
3. not eligible                → EXCLUDED
4. is_extended / LATE state    → EXTENDED_LEADER
5. state not in scored states  → EXCLUDED
6. BASE/ARMED near pivot       → BREAKOUT_LONG   (dist_to_pivot_atr ≤ 1.5 AND is_fresh)
7. BASE/ARMED far from pivot   → PULLBACK_LONG
8. TRIGGERED/ACCEPTED fresh    → BREAKOUT_LONG   (days_in_state ≤ 7)
9. TRIGGERED/ACCEPTED old      → PULLBACK_LONG
10. fallthrough                → EXCLUDED
```

### Key design decisions

**Portfolio overrides eligibility (rule 2).**  
A held position might be below SMA200 or have poor RS — you still need to see it in the portfolio section to manage the trade. Portfolio status is never suppressed.

**Freshness is baked into breakout routing (rule 6, rule 8).**  
Only fresh names go to `BREAKOUT_LONG`. Stale ARMED setups (past the freshness window but still viable) fall to `PULLBACK_LONG`. This means `select_breakout_candidates()` gets pre-filtered input.

**Extended leaders are informational.**  
`EXTENDED_LEADER` bucket is visible in the dashboard's extended section but never appears in breakout or pullback selection. These names are working — let them work; don't chase.

---

## Stage 3 — Bucket-Aware Selection

**Module:** `src/swingtrader/dashboard/selector.py`

### `select_breakout_candidates(df, n=7)`
- Source: `BREAKOUT_LONG` bucket only
- Additional filters: `is_fresh=True`, `is_portfolio=False`, not `ACTION_AVOID`
- Sort: action tier (NOW→BREAKOUT→PULLBACK→EXTENDED→AVOID) then `composite_score` descending
- Diversity cap: at most `MAX_PER_GROUP=3` symbols from the same `groups` tag
- Returns: up to 7 rows — the primary "best setups today" list

### `select_pullback_candidates(df, n=5)`
- Source: `PULLBACK_LONG` bucket only
- Additional filters: `is_fresh=True`, `is_portfolio=False`, not `ACTION_AVOID`
- Sort: same tier + score sort, diversity-capped
- Returns: up to 5 rows — add-on or re-entry setups

### `select_portfolio_holdings(df)`
- Source: `PORTFOLIO_HOLD` bucket (or `is_portfolio=True` fallback)
- No freshness filter — portfolio names are always shown regardless of state
- Sort: state priority (`TRIGGERED > ACCEPTED > CONFIRMED > ARMED > BASE`) then score
- Returns: all portfolio names — used for position guidance, not entry decisions

### `select_top_setups(df, n=7)` (backward compatible)
- Calls `select_breakout_candidates()` first
- Fills remaining slots from `PULLBACK_LONG` (deduped against breakout list)
- If `bucket` column is absent, falls back to `_legacy_select()` (original behavior)
- This function is the primary interface for the dashboard card section

---

## Score Architecture — What Each Score Means

The scores are **model outputs**, never hand-assigned weights.

| Score | Source | Meaning |
|---|---|---|
| `setup_score` | Calibrated classifier | Probability that this base/setup pattern resolves upward |
| `trade_score` | Trade-outcome model | Probability that an active trade (TRIGGERED/ACCEPTED) achieves target |
| `failure_risk` | Failure classifier | Probability of setup failure (stop-out or failed breakout) |
| `composite_score` | Derived | `setup_score × (1−failure_risk)` for BASE/ARMED; `sigmoid(trade_score) × (1−failure_risk)` for TRIGGERED/ACCEPTED |
| `percentile_rank` | Computed per state group | Percentile of composite_score within same-state symbols (0–100) |

The `composite_score` is what drives all ranking. The separate `setup_score` and `trade_score` are exposed in score drivers for transparency but don't have separate thresholds — `composite_score` is the single decision number.

---

## Regime-Adjusted Action Thresholds

**Module:** `src/swingtrader/dashboard/action.py`

In a market downtrend (`regime_spy_trend < 0`), the `assign_action()` function raises the score floor before assigning `ACTION_AVOID`:

```
Normal:     min_score = 0.20,  now_min_score = 0.30
Downtrend:  min_score = 0.28,  now_min_score = 0.35
```

This means marginal names that would be "neutral" in an uptrend get labeled `ACTION_AVOID` (and excluded from the top list) when the market is under pressure. Combined with `GATE_WEAK_REGIME_POOR_RS`, weak names get double-filtered in downtrends: once at the eligibility gate, again at action assignment.

---

## Why Names Are Rejected — Debuggability

Every rejected symbol has its reason stored in `rejection_reasons` (a comma-separated string in the snapshot). The dashboard exposes this in:
1. The **excluded section** (collapsible table below the main cards) — shows all scored-state symbols that were excluded, with their rejection reasons and composite scores
2. `data/reports/daily/YYYY-MM-DD/eligibility_results.json` — full machine-readable table

This makes the filter logic auditable: if a name you expected to see is missing, you can look up exactly which gate fired.

---

## What Remains Heuristic vs. Model-Based

| Decision | Basis |
|---|---|
| `GATE_BROKEN_TREND` threshold (`-0.08`) | Heuristic — structural threshold, not model-fitted |
| `GATE_POOR_RS` threshold (`-0.10`) | Heuristic |
| `GATE_HIGH_FAILURE_RISK` threshold (`0.65`) | Heuristic (hard cap; model provides the underlying risk score) |
| `GATE_THIN_BASE` length (`5`) | Heuristic |
| `BREAKOUT_DIST_ATR` threshold (`1.5`) | Heuristic — bucket routing, not scoring |
| `BREAKOUT_TRIGGER_DAYS` window (`7`) | Heuristic |
| `MIN_FRESH_BREAKOUT` (`2`) | Heuristic |
| `DOWNTREND_SCORE_PENALTY` (`0.08`) | Heuristic |
| All `composite_score`, `setup_score`, `failure_risk` values | **Model-based** (calibrated classifiers) |

Heuristic thresholds define structural eligibility — they encode domain knowledge about what constitutes a viable setup. The model scores rank and differentiate *within* the eligible pool. The separation is intentional: models should not be expected to learn "below SMA200 is uninvestable" from a noisy training set.

---

## Artifact Outputs

The pipeline writes per-bucket JSON files alongside the existing research snapshot:

| File | Contents |
|---|---|
| `breakout_top_setups.json` | Selected breakout candidates with full packet data |
| `pullback_top_setups.json` | Selected pullback candidates |
| `extended_leaders.json` | All extended-leader bucket symbols |
| `eligibility_results.json` | All scored symbols with eligible/rejection_reasons |
| `bucket_assignments.json` | All symbols with bucket/action_label/is_fresh columns |
| `top_setups.json` | Combined top list (retained for backward compatibility) |
| `dashboard_summary.json` | Bucket counts, n_eligible, n_excluded, artifact paths |

---

## Test Coverage

`tests/test_eligibility.py` — 78 tests across:
- `TestEligibilityGates` (21): each gate fires/passes, NaN handling, multi-gate accumulation, warnings
- `TestAddEligibilityColumns` (5): DataFrame enrichment
- `TestBucketAssignment` (14): each bucket routing case, portfolio override, extended routing
- `TestAddBucketColumn` (3): DataFrame enrichment and counts
- `TestSelectBreakoutCandidates` (7): filtering, ranking, portfolio exclusion
- `TestSelectPullbackCandidates` (4): filtering and portfolio exclusion
- `TestSelectPortfolioHoldings` (5): mutual exclusivity with entry lists
- `TestSelectTopSetups` (6): combined logic, legacy fallback
- `TestRegimeAdjustedActions` (5): downtrend threshold penalties
- `TestEligibilityBucketPipeline` (7): end-to-end with diverse symbol set
