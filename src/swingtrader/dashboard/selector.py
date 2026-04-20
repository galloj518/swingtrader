"""Top setup selection logic — bucket-aware and packet-first.

Packet-first API (preferred)
-----------------------------
select_packets(all_packets, ...)
    Accepts a list of lightweight packets (from build_all_lightweight_packets).
    Returns a PacketSelections dict keyed by bucket name, each value being a
    ranked list of packet dicts.  No DataFrame access — pure dict operations.

DataFrame API (backward compatibility)
---------------------------------------
select_breakout_candidates / select_pullback_candidates / select_top_setups
    Legacy functions that operate on DataFrame rows.  Kept for callers that
    have not yet migrated to the packet-first flow.

Implements three separate selection functions:

  select_breakout_candidates(df, n)
      Primary long candidates.  Eligible, fresh, non-portfolio symbols in the
      BREAKOUT_LONG bucket, ranked by composite_score descending.
      This is the main "best swing setups today" list.

  select_pullback_candidates(df, n)
      Secondary candidates.  Eligible, non-portfolio symbols in the
      PULLBACK_LONG bucket, ranked by composite_score descending.
      These are add-on / pullback re-entry setups.

  select_portfolio_holdings(df)
      All portfolio holdings (PORTFOLIO_HOLD bucket).
      Not ranked by entry quality — used for position guidance.

  select_top_setups(df, n)
      Combined shortlist for backward compatibility and the main dashboard
      card section.  Selects BREAKOUT_LONG first, then fills remaining
      slots from PULLBACK_LONG if needed.
      Only eligible, fresh, non-portfolio names are included.

Design notes
------------
- All selection functions require eligibility and bucket columns (from
  add_eligibility_columns and add_bucket_column).
- Diversity cap: at most MAX_PER_GROUP symbols from the same sector group
  in the combined shortlist to avoid sector clustering.
- Freshness preference: if the breakout bucket has < MIN_FRESH_BREAKOUT
  fresh names, the combined list will be short rather than filled with
  stale/pullback names.  Quality over quantity.
"""
from __future__ import annotations

import math

import pandas as pd

from swingtrader.dashboard.action import (
    ACTION_AVOID,
    ACTION_BREAKOUT,
    ACTION_EXTENDED,
    ACTION_NOW,
    ACTION_PULLBACK,
)
from swingtrader.dashboard.buckets import (
    BUCKET_BREAKOUT,
    BUCKET_EXCLUDED,
    BUCKET_EXTENDED,
    BUCKET_NON_EQUITY,
    BUCKET_PORTFOLIO,
    BUCKET_PULLBACK,
    BUCKET_REVERSAL,
)

# Type alias for the structured selection result
PacketSelections = dict[str, list[dict]]

# ── Configuration ─────────────────────────────────────────────────────────────

# Maximum total setups in the combined top list (breakout + pullback fill).
# The dashboard enforces exactly TOP_N cards when enough qualify.
TOP_N: int = 5

# Maximum breakout candidates shown in their own section
TOP_N_BREAKOUT: int = 5

# Maximum pullback candidates shown in their own section
TOP_N_PULLBACK: int = 5

# Minimum number of breakout-bucket names required before pulling from pullback
# (prevents the top list from being dominated by inferior setups)
MIN_FRESH_BREAKOUT: int = 2

# Max symbols from the same sector/group tag (diversity cap)
MAX_PER_GROUP: int = 3

# Tier mapping within breakout bucket (lower = higher priority)
_BREAKOUT_TIER: dict[str, int] = {
    ACTION_NOW: 1,
    ACTION_BREAKOUT: 2,
    ACTION_PULLBACK: 3,
    ACTION_EXTENDED: 4,
    ACTION_AVOID: 5,
}


def _safe_score(row: pd.Series) -> float:
    v = row.get("composite_score", math.nan)
    try:
        f = float(v)
        return f if math.isfinite(f) else -1.0
    except (TypeError, ValueError):
        return -1.0


def _breakout_ranking_score(row: pd.Series) -> float:
    """Score for ranking within the breakout bucket.

    For BASE/ARMED (pre-trigger), use setup_score: it is the model's dedicated
    assessment of base / setup quality independent of failure-risk scaling.
    For TRIGGERED/ACCEPTED (already triggered), composite_score is appropriate
    since it uses the trade_score model output designed for in-trade assessment.

    Falls back to composite_score if setup_score is unavailable.
    """
    state = str(row.get("state", ""))
    if state in {"BASE", "ARMED"}:
        v = row.get("setup_score", math.nan)
        try:
            f = float(v)
            if math.isfinite(f):
                return f
        except (TypeError, ValueError):
            pass
    return _safe_score(row)


def _primary_group(row: pd.Series) -> str:
    """Return the first group tag for diversity checking."""
    g = str(row.get("groups", ""))
    return g.split(",")[0].strip() if g else "other"


def _sort_and_cap(
    candidates: pd.DataFrame,
    n: int,
    diversity: bool = True,
    score_fn=None,
) -> pd.DataFrame:
    """Sort by (tier, score desc), apply group diversity cap, return top-n.

    Parameters
    ----------
    score_fn : optional callable(row) → float for bucket-specific ranking.
               Defaults to _safe_score (composite_score) when not provided.
    """
    if candidates.empty:
        return candidates.head(0)

    _score_fn = score_fn if score_fn is not None else _safe_score

    df = candidates.copy()
    df["_tier"] = df["action_label"].map(lambda x: _BREAKOUT_TIER.get(x, 9)) \
        if "action_label" in df.columns else 9
    df["_score"] = df.apply(_score_fn, axis=1)
    df["_group"] = df.apply(_primary_group, axis=1)
    df = df.sort_values(["_tier", "_score"], ascending=[True, False])

    if not diversity:
        result = df.head(n)
        return result.drop(columns=["_tier", "_score", "_group"], errors="ignore")

    selected: list[pd.Series] = []
    group_counts: dict[str, int] = {}

    for _, row in df.iterrows():
        if len(selected) >= n:
            break
        g = row["_group"]
        if group_counts.get(g, 0) >= MAX_PER_GROUP:
            continue
        selected.append(row)
        group_counts[g] = group_counts.get(g, 0) + 1

    if not selected:
        return df.head(0).drop(columns=["_tier", "_score", "_group"], errors="ignore")

    result = pd.DataFrame(selected)
    return result.drop(columns=["_tier", "_score", "_group"], errors="ignore").head(n)


def _base_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Apply common pre-filters: remove non-equity, portfolio, and avoid."""
    if df.empty:
        return df

    mask = pd.Series(True, index=df.index)

    # Remove non-equity (always)
    if "is_non_equity" in df.columns:
        mask &= ~df["is_non_equity"].astype(bool)

    # Remove ACTION_AVOID
    if "action_label" in df.columns:
        mask &= df["action_label"] != ACTION_AVOID

    return df[mask].copy()


# ── Primary selection functions ───────────────────────────────────────────────

def select_breakout_candidates(df: pd.DataFrame, n: int = TOP_N_BREAKOUT) -> pd.DataFrame:
    """Select top breakout-bucket candidates for the primary long list.

    Returns only eligible, fresh, non-portfolio symbols in BREAKOUT_LONG.
    Ranked by action tier then composite_score. Diversity-capped.

    Parameters
    ----------
    df : snapshot DataFrame with eligibility, bucket, freshness, and action columns.
    n  : max symbols to return.

    Returns
    -------
    DataFrame of up to n rows, ordered by (tier, composite_score desc).
    """
    if df.empty or "bucket" not in df.columns:
        return df.head(0)

    # Eligible breakout-bucket names only (eligibility already encoded in bucket)
    candidates = df[df["bucket"] == BUCKET_BREAKOUT].copy()

    # Require is_fresh: breakout candidates must be actionably fresh
    if "is_fresh" in candidates.columns:
        candidates = candidates[candidates["is_fresh"].astype(bool)]

    # Remove portfolio (double-check even though bucket should handle this)
    if "is_portfolio" in candidates.columns:
        candidates = candidates[~candidates["is_portfolio"].astype(bool)]

    candidates = _base_filter(candidates)

    # Breakout bucket ranks by setup_score for BASE/ARMED (model's dedicated base-quality
    # assessment), falling back to composite_score for TRIGGERED/ACCEPTED.
    return _sort_and_cap(candidates, n, diversity=True, score_fn=_breakout_ranking_score)


def select_pullback_candidates(df: pd.DataFrame, n: int = TOP_N_PULLBACK) -> pd.DataFrame:
    """Select top pullback-bucket candidates for the secondary list.

    Returns eligible, non-portfolio symbols in PULLBACK_LONG.
    Ranked by composite_score. Diversity-capped.

    Parameters
    ----------
    df : snapshot DataFrame with eligibility, bucket, freshness, and action columns.
    n  : max symbols to return.

    Returns
    -------
    DataFrame of up to n rows.
    """
    if df.empty or "bucket" not in df.columns:
        return df.head(0)

    candidates = df[df["bucket"] == BUCKET_PULLBACK].copy()

    # Require is_fresh for pullback candidates too (no stale names)
    if "is_fresh" in candidates.columns:
        candidates = candidates[candidates["is_fresh"].astype(bool)]

    if "is_portfolio" in candidates.columns:
        candidates = candidates[~candidates["is_portfolio"].astype(bool)]

    candidates = _base_filter(candidates)

    return _sort_and_cap(candidates, n, diversity=True)


def select_extended_leaders(df: pd.DataFrame, n: int = 8) -> pd.DataFrame:
    """Return extended-leader symbols for the monitoring section.

    Extended leaders are healthy names that are too extended for fresh entry.
    Shown for informational monitoring only — NOT in the primary action list.
    Ranked by composite_score desc (highest quality first).

    Parameters
    ----------
    df : snapshot DataFrame with bucket column.
    n  : max symbols to return.
    """
    if df.empty or "bucket" not in df.columns:
        return df.head(0)
    leaders = df[df["bucket"] == BUCKET_EXTENDED].copy()
    if leaders.empty:
        return leaders
    leaders["_score"] = leaders.apply(_safe_score, axis=1)
    leaders = leaders.sort_values("_score", ascending=False)
    return leaders.drop(columns=["_score"], errors="ignore").head(n)


def select_reversal_candidates(df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """Return speculative reversal candidates for the watch section.

    These are structurally weak names with some reversal characteristics.
    Explicitly NOT in the primary long list — kept in a separate section.
    Ranked by composite_score desc.

    Parameters
    ----------
    df : snapshot DataFrame with bucket column.
    n  : max symbols to return.
    """
    if df.empty or "bucket" not in df.columns:
        return df.head(0)
    candidates = df[df["bucket"] == BUCKET_REVERSAL].copy()
    if candidates.empty:
        return candidates
    candidates["_score"] = candidates.apply(_safe_score, axis=1)
    candidates = candidates.sort_values("_score", ascending=False)
    return candidates.drop(columns=["_score"], errors="ignore").head(n)


def select_portfolio_holdings(df: pd.DataFrame) -> pd.DataFrame:
    """Return all portfolio holdings for position-guidance review.

    Does NOT apply fresh/score filters — portfolio holdings are always shown
    regardless of state or score.  Sorted by state priority (active trades
    first) then composite_score.

    Parameters
    ----------
    df : snapshot DataFrame with bucket column.

    Returns
    -------
    DataFrame of portfolio holdings, sorted by (state_priority, score desc).
    """
    if df.empty:
        return df.head(0)

    # Try bucket column first; fall back to is_portfolio flag
    if "bucket" in df.columns:
        portfolio = df[df["bucket"] == BUCKET_PORTFOLIO].copy()
    elif "is_portfolio" in df.columns:
        portfolio = df[df["is_portfolio"].astype(bool)].copy()
    else:
        return df.head(0)

    if portfolio.empty:
        return portfolio

    # Sort: active trades first (TRIGGERED, ACCEPTED), then by score
    state_priority = {"TRIGGERED": 1, "ACCEPTED": 2, "CONFIRMED": 3, "ARMED": 4, "BASE": 5}
    portfolio["_state_pri"] = portfolio["state"].map(lambda s: state_priority.get(s, 9))
    portfolio["_score"] = portfolio.apply(_safe_score, axis=1)
    portfolio = portfolio.sort_values(["_state_pri", "_score"], ascending=[True, False])
    return portfolio.drop(columns=["_state_pri", "_score"], errors="ignore")


def select_top_setups(df: pd.DataFrame, n: int = TOP_N) -> pd.DataFrame:
    """Select the combined shortlist for the main dashboard cards section.

    Selects BREAKOUT_LONG first.  If fewer than MIN_FRESH_BREAKOUT breakout
    names are available, still shows what exists rather than filling with
    lower-quality names.  If budget remains after breakout names, fills with
    the best PULLBACK_LONG names (up to total n).

    This function is the primary interface for backward-compatible callers.
    New callers should use select_breakout_candidates() and
    select_pullback_candidates() directly.

    Parameters
    ----------
    df : snapshot DataFrame with eligibility, bucket, freshness, action columns.
    n  : maximum total setups to return.

    Returns
    -------
    DataFrame of up to n rows: breakout names first, then pullback fill.
    """
    if df.empty:
        return df.head(0)

    # If bucket column missing, fall back to legacy behaviour
    if "bucket" not in df.columns:
        return _legacy_select(df, n)

    breakout = select_breakout_candidates(df, n=n)
    n_remaining = n - len(breakout)

    if n_remaining > 0:
        # Avoid duplicating any symbol already in breakout list
        already_selected = set()
        sym_col = "user_symbol" if "user_symbol" in df.columns else "symbol"
        if sym_col in breakout.columns:
            already_selected = set(breakout[sym_col].tolist())

        pullback_candidates = df[df["bucket"] == BUCKET_PULLBACK].copy()
        if sym_col in pullback_candidates.columns and already_selected:
            pullback_candidates = pullback_candidates[
                ~pullback_candidates[sym_col].isin(already_selected)
            ]
        if "is_fresh" in pullback_candidates.columns:
            pullback_candidates = pullback_candidates[pullback_candidates["is_fresh"].astype(bool)]
        if "is_portfolio" in pullback_candidates.columns:
            pullback_candidates = pullback_candidates[~pullback_candidates["is_portfolio"].astype(bool)]
        pullback_candidates = _base_filter(pullback_candidates)
        pullback_fill = _sort_and_cap(pullback_candidates, n_remaining, diversity=True)

        if not pullback_fill.empty:
            result = pd.concat([breakout, pullback_fill], ignore_index=True)
            return result.head(n)

    return breakout.head(n)


# ── Packet-first selection API ───────────────────────────────────────────────


def _pkt_score(pkt: dict) -> float:
    """Extract composite_score from a packet as a sortable float."""
    v = pkt.get("composite_score", -1.0)
    try:
        f = float(v)
        return f if math.isfinite(f) else -1.0
    except (TypeError, ValueError):
        return -1.0


def _pkt_breakout_score(pkt: dict) -> float:
    """Score for ranking within the breakout bucket.

    BASE/ARMED: use setup_score (dedicated base-quality model output).
    TRIGGERED/ACCEPTED: use composite_score (trade-quality model output).
    Falls back to composite_score when setup_score is absent.
    """
    state = str(pkt.get("state", ""))
    if state in {"BASE", "ARMED"}:
        v = pkt.get("setup_score")
        try:
            f = float(v)
            if math.isfinite(f):
                return f
        except (TypeError, ValueError):
            pass
    return _pkt_score(pkt)


def _pkt_structural_tiebreaker(pkt: dict) -> tuple[int, float, float]:
    """Secondary sort key for breakout candidates with similar model scores.

    Returns a tuple used as a tiebreaker after (action_tier, -model_score).
    Lower values rank higher (used in ascending sort).

    Three components (all secondary — model score is primary):
      1. not_near_pivot   : 0 if abs(dist_to_pivot_atr) <= 0.5, else 1
                             Near-pivot names rank before distant ones.
      2. atr_compression  : lower = more compressed = better.
                             Normalised to [0, 1] from raw percentile.
                             Names with compressed ATR rank above noisy ones.
      3. rs_penalty       : 0 if rs63 > 0, else small positive penalty.
                             Outperformers rank ahead of flat/lagging names.

    Note: this is a purely ordinal tiebreaker for selector ranking.
    It is not a model score and is not stored in the packet.
    """
    # Component 1: proximity to pivot
    dist_v = pkt.get("dist_to_pivot_atr", math.nan)
    try:
        dist_f = float(dist_v)
        not_near_pivot = 0 if (math.isfinite(dist_f) and abs(dist_f) <= 0.5) else 1
    except (TypeError, ValueError):
        not_near_pivot = 1

    # Component 2: ATR compression (lower percentile = better)
    atr_v = pkt.get("atr_compression_pct", math.nan)
    try:
        atr_f = float(atr_v)
        atr_norm = atr_f / 100.0 if math.isfinite(atr_f) else 0.5
    except (TypeError, ValueError):
        atr_norm = 0.5

    # Component 3: RS penalty for lagging names
    rs_v = pkt.get("daily_rs_63", math.nan)
    try:
        rs_f = float(rs_v)
        rs_penalty = 0.0 if (math.isfinite(rs_f) and rs_f > 0) else 0.1
    except (TypeError, ValueError):
        rs_penalty = 0.0

    return (not_near_pivot, atr_norm, rs_penalty)


def _pkt_primary_group(pkt: dict) -> str:
    g = str(pkt.get("groups", ""))
    return g.split(",")[0].strip() if g else "other"


def _pkt_sort_and_cap(
    packets: list[dict],
    n: int,
    score_fn=None,
    diversity: bool = True,
    use_structural_tiebreaker: bool = False,
) -> list[dict]:
    """Sort packets by (action tier, score desc, structural tiebreaker), apply diversity cap, return top-n.

    Parameters
    ----------
    use_structural_tiebreaker : when True, adds a secondary structural sort key
        (pivot proximity, ATR compression, RS) as a tiebreaker after model score.
        Used for breakout bucket where near-pivot, compressed, outperforming names
        should be preferred among candidates with similar model scores.
        This is ordinal ranking only — NOT a model score.
    """
    if not packets:
        return []
    sf = score_fn if score_fn is not None else _pkt_score
    if use_structural_tiebreaker:
        keyed = sorted(
            packets,
            key=lambda p: (
                _BREAKOUT_TIER.get(str(p.get("action_label", "")), 9),
                -sf(p),
                _pkt_structural_tiebreaker(p),
            ),
        )
    else:
        keyed = sorted(
            packets,
            key=lambda p: (
                _BREAKOUT_TIER.get(str(p.get("action_label", "")), 9),
                -sf(p),
            ),
        )
    if not diversity:
        return keyed[:n]

    selected: list[dict] = []
    group_counts: dict[str, int] = {}
    for pkt in keyed:
        if len(selected) >= n:
            break
        g = _pkt_primary_group(pkt)
        if group_counts.get(g, 0) >= MAX_PER_GROUP:
            continue
        selected.append(pkt)
        group_counts[g] = group_counts.get(g, 0) + 1
    return selected[:n]


def select_packets(
    all_packets: list[dict],
    n_breakout: int = TOP_N_BREAKOUT,
    n_pullback: int = TOP_N_PULLBACK,
    n_extended: int = 8,
    n_reversal: int = 3,
) -> PacketSelections:
    """Select and rank packets from the full symbol list by bucket.

    This is the packet-first selector.  It receives a list of lightweight
    packets (produced by build_all_lightweight_packets) and groups them by
    their pre-computed ``bucket`` field.  No DataFrame access, no recomputation
    of eligibility or scores.

    Parameters
    ----------
    all_packets : list of lightweight packet dicts (from build_all_lightweight_packets).
    n_breakout  : max breakout-long candidates to return.
    n_pullback  : max pullback-long candidates to return.
    n_extended  : max extended-leader symbols to return.
    n_reversal  : max reversal-speculative symbols to return.

    Returns
    -------
    PacketSelections dict with keys:
      ``breakout``  — top breakout-long packets (fresh, eligible, non-portfolio)
      ``pullback``  — top pullback-long packets (fresh, eligible, non-portfolio)
      ``extended``  — top extended-leader packets (informational)
      ``reversal``  — top reversal-speculative packets (informational)
      ``portfolio`` — all portfolio-hold packets
      ``excluded``  — all excluded packets (for diagnostics / artifacts)
      ``top``       — combined breakout + pullback (for backward-compat callers)
    """
    # Partition by bucket
    by_bucket: dict[str, list[dict]] = {
        BUCKET_BREAKOUT:   [],
        BUCKET_PULLBACK:   [],
        BUCKET_EXTENDED:   [],
        BUCKET_REVERSAL:   [],
        BUCKET_PORTFOLIO:  [],
        BUCKET_EXCLUDED:   [],
        BUCKET_NON_EQUITY: [],
    }
    for pkt in all_packets:
        b = str(pkt.get("bucket", BUCKET_EXCLUDED))
        by_bucket.setdefault(b, []).append(pkt)

    # Breakout: fresh only, non-portfolio, ranked by setup_score/composite_score
    bo_candidates = [
        p for p in by_bucket[BUCKET_BREAKOUT]
        if bool(p.get("is_fresh", False)) and not bool(p.get("is_portfolio", False))
           and str(p.get("action_label", "")) != ACTION_AVOID
    ]
    breakout_selected = _pkt_sort_and_cap(
        bo_candidates, n_breakout,
        score_fn=_pkt_breakout_score,
        diversity=True,
        use_structural_tiebreaker=True,
    )

    # Pullback: fresh only, non-portfolio, ranked by composite_score
    pb_candidates = [
        p for p in by_bucket[BUCKET_PULLBACK]
        if bool(p.get("is_fresh", False)) and not bool(p.get("is_portfolio", False))
           and str(p.get("action_label", "")) != ACTION_AVOID
    ]
    # Exclude symbols already in breakout list
    bo_syms = {p.get("symbol") for p in breakout_selected}
    pb_candidates = [p for p in pb_candidates if p.get("symbol") not in bo_syms]
    pullback_selected = _pkt_sort_and_cap(pb_candidates, n_pullback, diversity=True)

    # Extended: informational, ranked by composite_score desc
    ext_sorted = sorted(by_bucket[BUCKET_EXTENDED], key=_pkt_score, reverse=True)
    extended_selected = ext_sorted[:n_extended]

    # Reversal: informational, ranked by composite_score desc
    rev_sorted = sorted(by_bucket[BUCKET_REVERSAL], key=_pkt_score, reverse=True)
    reversal_selected = rev_sorted[:n_reversal]

    # Portfolio: all holdings, sorted by active-trade state priority then score
    _state_pri = {"TRIGGERED": 1, "ACCEPTED": 2, "CONFIRMED": 3, "ARMED": 4, "BASE": 5}
    portfolio_sorted = sorted(
        by_bucket[BUCKET_PORTFOLIO],
        key=lambda p: (_state_pri.get(str(p.get("state", "")), 9), -_pkt_score(p)),
    )

    top = breakout_selected + pullback_selected

    return {
        "breakout":  breakout_selected,
        "pullback":  pullback_selected,
        "extended":  extended_selected,
        "reversal":  reversal_selected,
        "portfolio": portfolio_sorted,
        "excluded":  by_bucket[BUCKET_EXCLUDED],
        "top":       top,
    }


def _pkt_trade_plan(pkt: dict) -> dict:
    tp = pkt.get("trade_plan", {})
    return tp if isinstance(tp, dict) else {}


def _surface_blockers(pkt: dict, bucket: str) -> list[str]:
    blockers: list[str] = []
    if not bool(pkt.get("coherence_ok", False)):
        blockers.extend([str(issue) for issue in pkt.get("coherence_issues", []) if str(issue)])
    if not bool(pkt.get("packet_complete_for_surface", False)):
        blockers.extend([str(issue) for issue in pkt.get("packet_completeness_issues", []) if str(issue)])

    tp = _pkt_trade_plan(pkt)
    entry_style = str(tp.get("entry_style", ""))
    actionable_now = bool(tp.get("actionable_now", False))
    action_label = str(pkt.get("action_label", ""))
    setup_key = str(pkt.get("setup_key", ""))

    if bucket == BUCKET_BREAKOUT:
        if setup_key not in {"fresh_breakout", "breakout_watch"}:
            blockers.append("not_a_true_breakout_packet")
        if entry_style != "breakout":
            blockers.append("breakout_bucket_requires_breakout_entry_style")
        if action_label == ACTION_NOW and not actionable_now:
            blockers.append("actionable_now_without_live_entry")
        if bool(pkt.get("is_extended", False)):
            blockers.append("extended_name_cannot_surface_as_breakout")
    elif bucket == BUCKET_PULLBACK:
        if setup_key not in {"reclaim_pullback", "constructive_pullback", "aged_breakout_pullback"}:
            blockers.append("not_a_true_pullback_packet")
        if entry_style not in {"pullback", "reclaim"}:
            blockers.append("pullback_bucket_requires_pullback_or_reclaim_entry_style")
    return blockers


def _mark_surface_status(pkt: dict, *, section: str | None, surfaced: bool, reason: str = "", blockers: list[str] | None = None) -> None:
    pkt["surfaced_in_top"] = surfaced
    pkt["surface_section"] = section
    pkt["not_surfaced_reason"] = "" if surfaced else reason
    pkt["selector_blockers"] = blockers or []


def _select_packets_canonical(
    all_packets: list[dict],
    n_breakout: int = TOP_N_BREAKOUT,
    n_pullback: int = TOP_N_PULLBACK,
    n_extended: int = 8,
    n_reversal: int = 3,
) -> PacketSelections:
    by_bucket: dict[str, list[dict]] = {
        BUCKET_BREAKOUT: [],
        BUCKET_PULLBACK: [],
        BUCKET_EXTENDED: [],
        BUCKET_REVERSAL: [],
        BUCKET_PORTFOLIO: [],
        BUCKET_EXCLUDED: [],
        BUCKET_NON_EQUITY: [],
    }
    for pkt in all_packets:
        pkt["surfaced_in_top"] = False
        pkt["surface_section"] = None
        pkt["not_surfaced_reason"] = ""
        pkt["selector_blockers"] = []
        by_bucket.setdefault(str(pkt.get("bucket", BUCKET_EXCLUDED)), []).append(pkt)

    breakout_pool: list[dict] = []
    for pkt in by_bucket[BUCKET_BREAKOUT]:
        blockers = _surface_blockers(pkt, BUCKET_BREAKOUT)
        if blockers:
            _mark_surface_status(pkt, section=None, surfaced=False, reason="; ".join(blockers), blockers=blockers)
            continue
        breakout_pool.append(pkt)

    breakout_selected = _pkt_sort_and_cap(
        breakout_pool,
        n_breakout,
        score_fn=_pkt_breakout_score,
        diversity=True,
        use_structural_tiebreaker=True,
    )
    breakout_syms = {pkt.get("symbol") for pkt in breakout_selected}
    for pkt in breakout_selected:
        _mark_surface_status(pkt, section="breakout", surfaced=True)
    for pkt in breakout_pool:
        if pkt.get("symbol") not in breakout_syms:
            _mark_surface_status(
                pkt,
                section=None,
                surfaced=False,
                reason="ranked_below_breakout_cutoff",
                blockers=[],
            )

    pullback_pool: list[dict] = []
    for pkt in by_bucket[BUCKET_PULLBACK]:
        if pkt.get("symbol") in breakout_syms:
            _mark_surface_status(pkt, section=None, surfaced=False, reason="already_selected_in_breakout", blockers=[])
            continue
        blockers = _surface_blockers(pkt, BUCKET_PULLBACK)
        if blockers:
            _mark_surface_status(pkt, section=None, surfaced=False, reason="; ".join(blockers), blockers=blockers)
            continue
        pullback_pool.append(pkt)

    pullback_selected = _pkt_sort_and_cap(pullback_pool, n_pullback, diversity=True)
    pullback_syms = {pkt.get("symbol") for pkt in pullback_selected}
    for pkt in pullback_selected:
        _mark_surface_status(pkt, section="pullback", surfaced=True)
    for pkt in pullback_pool:
        if pkt.get("symbol") not in pullback_syms:
            _mark_surface_status(
                pkt,
                section=None,
                surfaced=False,
                reason="ranked_below_pullback_cutoff",
                blockers=[],
            )

    ext_sorted = sorted(by_bucket[BUCKET_EXTENDED], key=_pkt_score, reverse=True)
    extended_selected = ext_sorted[:n_extended]
    rev_sorted = sorted(by_bucket[BUCKET_REVERSAL], key=_pkt_score, reverse=True)
    reversal_selected = rev_sorted[:n_reversal]

    _state_pri = {"TRIGGERED": 1, "ACCEPTED": 2, "CONFIRMED": 3, "ARMED": 4, "BASE": 5}
    portfolio_sorted = sorted(
        by_bucket[BUCKET_PORTFOLIO],
        key=lambda p: (_state_pri.get(str(p.get("state", "")), 9), -_pkt_score(p)),
    )

    for pkt in by_bucket[BUCKET_EXCLUDED]:
        if not pkt.get("not_surfaced_reason"):
            pkt["not_surfaced_reason"] = str(pkt.get("rejection_reasons", "")) or "failed_eligibility"

    top = breakout_selected + pullback_selected
    unselected = [
        pkt for pkt in all_packets
        if not bool(pkt.get("surfaced_in_top", False))
        and str(pkt.get("bucket")) in {BUCKET_BREAKOUT, BUCKET_PULLBACK}
    ]

    return {
        "breakout": breakout_selected,
        "pullback": pullback_selected,
        "extended": extended_selected,
        "reversal": reversal_selected,
        "portfolio": portfolio_sorted,
        "excluded": by_bucket[BUCKET_EXCLUDED],
        "top": top,
        "all": list(all_packets),
        "unselected": unselected,
    }


select_packets = _select_packets_canonical


# ── Legacy fallback ───────────────────────────────────────────────────────────

def _legacy_select(df: pd.DataFrame, n: int = TOP_N) -> pd.DataFrame:
    """Legacy selection without bucket column (backward compatibility)."""
    if df.empty or "action_label" not in df.columns:
        return df.head(0)

    candidates = df[df["action_label"] != ACTION_AVOID].copy()
    if "is_non_equity" in candidates.columns:
        candidates = candidates[~candidates["is_non_equity"].astype(bool)]

    if candidates.empty:
        return candidates.head(0)

    _tier_map: dict[str, int] = {
        ACTION_NOW: 1,
        ACTION_BREAKOUT: 2,
        ACTION_PULLBACK: 3,
        ACTION_EXTENDED: 4,
        ACTION_AVOID: 5,
    }

    candidates["_tier"] = candidates["action_label"].map(lambda x: _tier_map.get(x, 9))
    candidates["_score"] = candidates.apply(_safe_score, axis=1)
    candidates["_group"] = candidates.apply(_primary_group, axis=1)
    candidates = candidates.sort_values(["_tier", "_score"], ascending=[True, False])

    selected: list[pd.Series] = []
    group_counts: dict[str, int] = {}

    for _, row in candidates.iterrows():
        if len(selected) >= n:
            break
        g = row["_group"]
        if group_counts.get(g, 0) >= MAX_PER_GROUP:
            continue
        selected.append(row)
        group_counts[g] = group_counts.get(g, 0) + 1

    if not selected:
        return candidates.head(0)

    result = pd.DataFrame(selected)
    return result.drop(columns=["_tier", "_score", "_group"], errors="ignore").head(n)
