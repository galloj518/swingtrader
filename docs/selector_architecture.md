# Selector Architecture — Eligibility Gates, Setup Buckets, and Portfolio Logic

*Last updated: 2026-04-19. Covers the packet-first architectural refactor, packet richness improvements, and the original gate/bucket redesign.*

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

---

## Packet-First Architecture

The system was refactored from *research-first* (DataFrame column passes before packet
building) to *packet-first* (packet is built before selection; selector reads packets).

### Why it matters

Old model: columns → selection → packets (for top-N only).  
New model: packets (for all) → selection → enrich (top-N only).

In the old model the packet could not be the source of truth — it only read pre-computed
column values. In the new model, `build_lightweight_packet` calls all analysis functions
internally and the packet is the unit of analysis.

### Two-tier packet model

**Tier 1 — `build_lightweight_packet(row)`**  
No file I/O. Built for *every* symbol (200+). Calls internally:

```
assess_eligibility(row)   → eligible, rejection_reasons
classify_row(row)         → is_fresh, is_extended, freshness_label
assign_action(enriched)   → action_label
assign_bucket(enriched)   → bucket
compute_levels(row)       → TradeLevels
_build_trade_plan(...)    → trade_plan dict
_build_portfolio_health() → portfolio_health dict
build_narrative(...)      → narrative dict
```

The row does **not** need pre-added `eligible`, `bucket`, `is_fresh` columns.

**Tier 2 — `enrich_with_context(pkt, row)`**  
File I/O. Called only for top-N selected packets. Adds MA table, AVWAP map,
assessments, checklist. Populates `pkt["context"]`.

### Pipeline flow

```
scored_snapshot_df
    │
    ▼  build_all_lightweight_packets(df)   [no file I/O]
    │
    ▼  select_packets(all_packets)          [pure dict ops]
    │
    ├─→ enrich_with_context() for top-N    [file I/O]
    ├─→ write_artifacts(selections, ...)   [JSON, packet-driven]
    └─→ write_dashboard(..., selections=.) [thin rendering only]
```

### Packet-first selector

`select_packets(all_packets: list[dict]) -> PacketSelections`

Input is `list[dict]` from `build_all_lightweight_packets`.  
Output is `PacketSelections = dict[str, list[dict]]` with keys:
`breakout`, `pullback`, `extended`, `reversal`, `portfolio`, `excluded`, `top`.

No DataFrame access. Partitions by `pkt["bucket"]` (pre-computed in packet builder).

The DataFrame-based legacy functions (`select_breakout_candidates`,
`select_pullback_candidates`, `select_top_setups`) are preserved for backward
compatibility but are not called in the main pipeline.

### Old-repo adaptations (swing_engine)

| Concept | Origin | Location |
|---|---|---|
| Trade plan fields | `swing_engine/checklist.py evaluate_actionability()` | `packet.py _build_trade_plan()` |
| Actionability verdicts | swing_engine labelling | `BUY_NOW / WATCH_BREAKOUT / WAIT_PULLBACK / WAIT_ZONE / BLOCK` |
| Portfolio health | `swing_engine/packets.py portfolio_guidance()` | `packet.py _build_portfolio_health()` |
| MA `need_tomorrow` / `tomorrow_bias` | `swing_engine/features.py extract_ma_state()` | `context.py build_ma_table()` |
| Assessment suite (6 functions) | `swing_engine/features.py assess_*` | `assessments.py` |
| WTD/MTD AVWAP anchors | `swing_engine/features.py get_dynamic_anchor_dates()` | `context.py build_avwap_table()` |

**Intentionally NOT ported:** calibrated gating (`_check_weekly_gate`), AI narrative
templates, broker/OMS execution fields.

### Dashboard as thin rendering layer

`write_dashboard` reads packet fields and renders HTML. It does not:
- Reassign buckets, eligibility, or trade plan fields.
- Call analysis functions (eligibility, freshness, action, bucket).

When `selections=` is provided, the universe bar, portfolio strip, extended section,
and reversal section are all driven by packet data from the selections dict.

---

## Packet Richness — Decision-Quality Fields

*Added 2026-04-19.*

Beyond the minimal fields needed for selection, each packet now carries enough
information for a trader to evaluate and act without leaving the dashboard.

### Structural classification fields (Tier 1)

These are computed inside `build_lightweight_packet` from snapshot columns alone —
no file I/O required.

| Field | Type | Description |
|---|---|---|
| `daily_trend_state` | str | `"strong_uptrend"` \| `"uptrend"` \| `"neutral"` \| `"weak"` \| `"broken"` \| `"unknown"` — derived from `close_vs_sma50`, `daily_rs_63`, `regime_spy_trend` |
| `weekly_trend_state` | str \| None | Populated by `enrich_with_context()` (Tier 2); `None` until then |
| `pullback_quality` | str | For `PULLBACK_LONG` bucket: `"at_pivot"` \| `"near_pivot"` \| `"constructive"` \| `"deep"` \| `"old_trigger"` \| `"far_base"` \| `"n/a"` |
| `demotion_reason` | str | Why a BASE/ARMED name went to PULLBACK instead of BREAKOUT: `"far_from_pivot"` \| `"below_sma50"` \| `"old_trigger"` \| `"stale_base"` \| `"not_fresh"` \| `""` |

**`daily_trend_state` derivation logic:**

```
broken:        regime_spy_trend < 0 AND close_vs_sma50 < -0.05
weak:          close_vs_sma50 < 0 (but not broken)
neutral:       close_vs_sma50 in [0, 0.03] AND rs63 <= 0
strong_uptrend: close_vs_sma50 > 0.03 AND rs63 > 0
uptrend:       everything else above SMA50 with positive RS
```

**`pullback_quality` derivation logic (dist_to_pivot_atr based):**

```
old_trigger:  TRIGGERED/ACCEPTED with days_in_state > BREAKOUT_TRIGGER_DAYS (7)
at_pivot:     dist_to_pivot_atr >= -0.5 (within half an ATR)
near_pivot:   dist_to_pivot_atr >= -2.0
constructive: dist_to_pivot_atr >= -3.0
deep:         dist_to_pivot_atr < -3.0
far_base:     BASE/ARMED with dist_to_pivot_atr > 1.5 ATR above pivot (not near entry)
```

### Dual-sided trade analysis (Tier 1)

`trade_plan` now includes four diagnostic lists produced deterministically from
snapshot fields. These are not narratives — they are structured signal lists
derived from hard conditions, not model scores or prose templates.

| Field | Contents |
|---|---|
| `why_now` | Reasons supporting entry today: fresh breakout days, near-pivot distance, above SMA50, positive RS, ATR compressed, volume dry-up, YTD AVWAP acceptance, swing-low buffer, low failure risk |
| `why_not_now` | Structural cautions against immediate entry: below SMA50, underperforming RS, elevated ATR, active volume, below YTD AVWAP, high failure risk, elevated failure risk |
| `setup_improves_if` | Forward-looking positive scenarios: "price holds above pivot tomorrow", "RS turns positive", "volume contracts further", "regime improves" |
| `setup_weakens_if` | Forward-looking risk scenarios: "price undercuts SMA50", "volume surges on down day", "RS deteriorates further", "market downtrend deepens" |

These four lists surface in `artifacts/top_setups.json` → `trade_plan` section and
in per-symbol `{SYM}_packet.json`. They are also available to the dashboard renderer.

### Weekly trend state (Tier 2)

`enrich_with_context()` calls `build_trend_state(provider_symbol, snapshot_row)` which
reads the features parquet (`features/{sym}.parquet`) for weekly columns
(`weekly_dist_wma10`, `weekly_trend_slope_26`, `weekly_rs_26`, `weekly_dist_wma40`)
and returns a `trend_state` dict:

```
{
  "daily_trend_state":  "...",    # re-derived from richer features (more precise)
  "weekly_trend_state": "...",    # from weekly features
  "daily_detail":       {...},    # raw input values used
  "weekly_detail":      {...},    # raw input values used
  "trend_summary":      "..."     # single-sentence human summary
}
```

`pkt["weekly_trend_state"]` is populated from `trend_state["weekly_trend_state"]`
at enrich time. Before enrichment it is `None`.

### AVWAP table enrichment (Tier 2)

Each row in `pkt["avwap_table"]` now carries additional context fields:

| Field | Description |
|---|---|
| `priority` | `"primary"` (YTD, Swing Low), `"secondary"` (Swing High, Breakout Day), `"dynamic"` (WTD/MTD) |
| `stretch_atr` | Distance from close to this AVWAP in ATR units (positive = above AVWAP) |
| `slope_20` | 20-day slope of the AVWAP line (raw) |
| `slope_label` | `"rising"` \| `"flat"` \| `"falling"` |
| `closes_above_20` | Count of last 20 closes above this AVWAP (0–20) |
| `anchor_date` | ISO date string for WTD/MTD dynamic anchors only |

The distinction between `"primary"` and `"secondary"` anchors is intentional:
primary anchors (YTD and Swing Low) are the most structurally significant;
secondary anchors provide context but are not the primary decision reference.

### Structural tiebreaker in selector (selector-internal only)

`select_packets` uses a structural tiebreaker for breakout candidates when model scores
are similar. It is an **ordinal sort key** (not stored in packets, not a score):

```python
_pkt_structural_tiebreaker(pkt) → (not_near_pivot: int, atr_norm: float, rs_penalty: float)
```

- `not_near_pivot`: 0 if `|dist_to_pivot_atr| ≤ 0.5`, else 1
- `atr_norm`: `atr_compression_pct / 100.0` (lower = tighter base = preferred)
- `rs_penalty`: 0.0 if `daily_rs_63 > 0`, else 0.1

This is applied after model score sorting — it only separates names that have
essentially identical composite scores. It encodes structural preference
(near the entry zone, tight base, outperforming) without inventing a new score.

---

## Test Coverage

`tests/test_packet_first.py` — 50 tests proving the packet-first architecture contract:
- `TestPacketSelfContained` (8): packet built without pre-added columns
- `TestEligibilityFromPacket` (4): eligibility/rejection reasons from packet
- `TestBucketFromPacket` (5): bucket from packet, not row column
- `TestSelectPacketsIsPacketDriven` (9): selector takes list[dict], no DataFrame
- `TestPortfolioHealthFromPacket` (6): portfolio_health from packet builder
- `TestTradePlanFromPacket` (7): trade_plan fields from packet builder
- `TestBuildAllLightweightPackets` (5): batch builder, no file I/O, no mutation
- `TestPacketSelectionsType` (1): type alias contract
- `TestDashboardRendersFromSelections` (2): dashboard renders from selections dict
- `TestArtifactsPacketDriven` (3): artifacts written from packet selections

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

`tests/test_packet_richness.py` — 56 tests covering packet richness fields:
- `TestDailyTrendState` (10): `_compute_daily_trend_state` all states, edge cases, NaN handling
- `TestPullbackQuality` (8): all pullback quality classifications, non-pullback passthrough
- `TestDemotionReason` (6): demotion reason strings for BASE/ARMED and TRIGGERED/ACCEPTED
- `TestTradePlanDualAnalysis` (7): `why_now`/`why_not_now`/`setup_improves_if`/`setup_weakens_if` presence and content
- `TestTradePlanCompleteness` (4): all 16 trade_plan fields present in top packets
- `TestPacketStructuralFields` (5): `daily_trend_state`/`weekly_trend_state`/`pullback_quality`/`demotion_reason` in packets
- `TestAvwapTableEnrichment` (6): `priority`/`stretch_atr`/`slope_label`/`closes_above_20`/`anchor_date` in AVWAP rows
- `TestSelectorStructuralTiebreaker` (6): tiebreaker logic; selector prefers near-pivot over far-from-pivot at equal score
- `TestAllPacketsHaveStructuralFields` (3): batch `build_all_lightweight_packets` includes all new fields
