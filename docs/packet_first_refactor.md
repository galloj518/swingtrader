# Packet-First Refactor Notes

Last updated: 2026-04-19

This note documents the packet-coherence refactor that made the packet the
source of truth for selector routing, dashboard rendering, artifacts, and
trader-facing trade plans.

## Old-repo logic adapted

| Old repo file | Current file | Logic adapted | Intentionally not ported |
|---|---|---|---|
| `swing_engine/packets.py` | `src/swingtrader/dashboard/packet.py` | Canonical trade-plan completeness, portfolio-hold separation, promotion/demotion reasoning, trader-facing verdict blocks | Legacy architecture shape and any execution/sizing assumptions |
| `swing_engine/features.py` | `src/swingtrader/dashboard/context.py` | Dynamic AVWAP anchors, configured macro/symbol anchors, MA tomorrow / need-flat style context | Old feature sprawl that bypassed the current registry/config flow |
| `swing_engine/checklist.py` | `src/swingtrader/dashboard/packet.py` | Deterministic actionability structure: breakout watch vs pullback wait vs buy-now semantics | Checklist items that implied broker actions |
| `swing_engine/dashboard.py` | `src/swingtrader/reports/dashboard.py` | Full trade-packet cards, AVWAP/MA/checklist richness, thin presentation over packet truth | Late-stage "repair" logic that contradicted packet state |

## Packet coherence rules

The packet now enforces agreement across:

- `bucket`
- `setup_classification`
- `action_label`
- `trade_plan.entry_style`
- `trade_plan.entry_condition`
- `trade_plan.why_now`
- `trade_plan.why_not_now`
- `final_verdict`
- `extension_state`

The packet records:

- `coherence_ok`
- `coherence_issues`
- `packet_complete_for_surface`
- `packet_completeness_issues`

Top breakout and pullback sections only accept packets where both flags are
true. A packet that says "reclaim pivot" cannot surface as `breakout_long`, and
a packet that says `actionable_now` must also have `trade_plan.actionable_now`.

## Selector promotion and demotion rules

The selector now uses packet truth rather than reconstructing meaning from
DataFrame columns.

- `breakout_long` is reserved for `fresh_breakout` and `breakout_watch`
  packets with breakout entry style.
- `pullback_long` is reserved for `reclaim_pullback`,
  `constructive_pullback`, and `aged_breakout_pullback`.
- `extended_leader` is the sink for healthy but too-late names.
- `portfolio_hold` is fully separate from fresh-entry routing.
- `excluded` keeps explicit rejection reasons from hard eligibility gates.

Every packet persists:

- `promotion_reason`
- `demotion_reason`
- `rejection_reasons`
- `route_reason`
- `selector_blockers`
- `surfaced_in_top`
- `surface_section`
- `not_surfaced_reason`

This makes it answerable, after the fact, why a symbol surfaced, why it did
not surface, and why it was treated as breakout vs pullback vs extended.

## Anchor framework

`config/avwap_anchors.yaml` is the source of truth for discretionary AVWAP
context. `src/swingtrader/dashboard/context.py` loads that config and emits a
single packet/dashboard/artifact table that includes:

- anchor name
- anchor date
- AVWAP value
- distance from price in percent and ATR
- support/resistance role
- status
- priority
- significance
- `supported`
- `unavailable_reason`

Implemented anchor families:

- YTD
- MTD
- WTD
- swing low
- swing high
- breakout day
- configurable global event anchors
- war start
- ceasefire
- COVID low
- COVID high
- configured per-symbol event anchors
- earnings anchor placeholder when event data is unavailable

Unsupported anchors are not silently dropped. They stay in the packet with
`supported = false` and an honest reason.

## Intraday product decision

v1 now takes explicit Option B:

- intraday confirmation is not part of primary qualification
- packets carry `intraday_policy = "daily_only"`
- packets carry `intraday_available = false`
- packets carry `intraday_used_in_qualification = false`
- dashboard cards render a compact policy note instead of implying a missing
  intraday confirmation chart

This matches the current data reality and avoids pretending that intraday
confirmation participated in ranking when it did not.

## Deferred

The following remain intentionally deferred:

- real event-driven earnings anchors beyond configured placeholders
- per-symbol news/event anchors from a reliable event feed
- true intraday-conditioned qualification and chart confirmation
- any broker, execution, order-management, or sizing workflow
