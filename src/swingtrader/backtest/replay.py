"""Historical signal replay — converts state machine history into a P&L log.

This is a simulation, not a live trading system. It answers:
  "If I had bought every TRIGGERED bar and held to CONFIRMED or stop,
   what would the aggregate P&L look like?"

Assumptions
-----------
- Entry price = close on the TRIGGERED bar (market-on-close proxy)
- Target exit = first bar where state == CONFIRMED  (close used as exit price)
- Stop exit   = first bar where close ≤ trigger_pivot - atr_stop × atr_at_trigger
- Timeout exit = after max_bars_after_trigger bars with no resolution
- No slippage, commissions, or position sizing — this is a research tool, not a
  simulator for real capital.
- Survivorship bias: only symbols present in the current universe are evaluated.
  Point-in-time universe membership is NOT enforced.

All of these limitations are noted in the output summary dict.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from swingtrader.utils.config import REPO_ROOT
from swingtrader.utils.logging import get_logger

log = get_logger(__name__)

_STATES_DIR = REPO_ROOT / "data" / "states"
_FEATURES_DIR = REPO_ROOT / "data" / "features"

ENTRY_TRIGGERED = "TRIGGERED"
EXIT_CONFIRMED = "CONFIRMED"
EXIT_STOP = "STOP"
EXIT_TIMEOUT = "TIMEOUT"


@dataclass
class ReplayTrade:
    """One simulated trade."""

    symbol: str
    trigger_date: pd.Timestamp
    entry_price: float
    trigger_pivot: float
    atr_at_trigger: float
    target_price: float    # trigger_pivot + atr_gain_target × ATR
    stop_price: float      # trigger_pivot - atr_stop × ATR

    exit_date: pd.Timestamp | None = None
    exit_price: float | None = None
    exit_reason: str | None = None

    @property
    def pnl_pct(self) -> float | None:
        if self.exit_price is None or self.entry_price <= 0:
            return None
        return (self.exit_price - self.entry_price) / self.entry_price

    @property
    def pnl_atr(self) -> float | None:
        if self.exit_price is None or self.atr_at_trigger <= 0:
            return None
        return (self.exit_price - self.entry_price) / self.atr_at_trigger

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "trigger_date": self.trigger_date,
            "entry_price": self.entry_price,
            "trigger_pivot": self.trigger_pivot,
            "atr_at_trigger": self.atr_at_trigger,
            "target_price": self.target_price,
            "stop_price": self.stop_price,
            "exit_date": self.exit_date,
            "exit_price": self.exit_price,
            "exit_reason": self.exit_reason,
            "pnl_pct": self.pnl_pct,
            "pnl_atr": self.pnl_atr,
        }


def replay_symbol(
    symbol: str,
    states_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    *,
    atr_stop: float = 1.0,
    atr_target: float = 2.0,
    max_bars: int = 20,
) -> list[ReplayTrade]:
    """Replay all TRIGGERED signals for one symbol.

    Parameters
    ----------
    symbol : ticker string (for labelling)
    states_df : output of compute_states() — must have 'state', 'trigger_pivot',
                'trigger_atr' columns; DatetimeIndex
    prices_df : daily OHLCV DataFrame aligned to the same DatetimeIndex
    atr_stop : stop distance in ATR multiples (from trigger_pivot)
    atr_target : target distance in ATR multiples (used for annotation only;
                 CONFIRMED state is the actual exit signal)
    max_bars : timeout after this many bars if neither CONFIRMED nor stop hit

    Returns list of ReplayTrade (one per distinct trigger event).
    """
    if states_df.empty or prices_df.empty:
        return []

    if "state" not in states_df.columns:
        return []

    trades: list[ReplayTrade] = []
    in_trade: ReplayTrade | None = None
    bars_since_trigger: int = 0

    index = states_df.index.union(prices_df.index).sort_values().unique()

    for dt in index:
        if dt not in states_df.index or dt not in prices_df.index:
            continue

        state = str(states_df.loc[dt, "state"])
        close = float(prices_df.loc[dt, "close"]) if "close" in prices_df.columns else np.nan

        # Track open trade
        if in_trade is not None:
            bars_since_trigger += 1

            if not np.isfinite(close):
                continue

            # Stop hit?
            if close <= in_trade.stop_price:
                in_trade.exit_date = dt
                in_trade.exit_price = close
                in_trade.exit_reason = EXIT_STOP
                trades.append(in_trade)
                in_trade = None
                continue

            # Confirmed?
            if state == EXIT_CONFIRMED:
                in_trade.exit_date = dt
                in_trade.exit_price = close
                in_trade.exit_reason = EXIT_CONFIRMED
                trades.append(in_trade)
                in_trade = None
                continue

            # Timeout?
            if bars_since_trigger >= max_bars:
                in_trade.exit_date = dt
                in_trade.exit_price = close
                in_trade.exit_reason = EXIT_TIMEOUT
                trades.append(in_trade)
                in_trade = None
                continue

        # New trigger?
        if in_trade is None and state == ENTRY_TRIGGERED:
            # state_changed check: only enter on first bar of TRIGGERED
            if "state_changed" in states_df.columns and not bool(states_df.loc[dt, "state_changed"]):
                continue

            tp = states_df.loc[dt, "trigger_pivot"] if "trigger_pivot" in states_df.columns else np.nan
            atr = states_df.loc[dt, "trigger_atr"] if "trigger_atr" in states_df.columns else np.nan

            if not (np.isfinite(close) and np.isfinite(tp) and np.isfinite(atr) and atr > 0):
                continue

            in_trade = ReplayTrade(
                symbol=symbol,
                trigger_date=dt,
                entry_price=close,
                trigger_pivot=float(tp),
                atr_at_trigger=float(atr),
                target_price=float(tp) + atr_target * float(atr),
                stop_price=float(tp) - atr_stop * float(atr),
            )
            bars_since_trigger = 0

    # Close any still-open trade at end of history (treat as timeout)
    if in_trade is not None:
        in_trade.exit_reason = EXIT_TIMEOUT
        trades.append(in_trade)

    return trades


def replay_universe(
    states_dir: Path | None = None,
    features_dir: Path | None = None,
    *,
    atr_stop: float = 1.0,
    atr_target: float = 2.0,
    max_bars: int = 20,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Replay all symbols in the states directory. Returns trade log DataFrame."""
    states_dir = Path(states_dir or _STATES_DIR)
    features_dir = Path(features_dir or _FEATURES_DIR)

    all_trades: list[dict] = []
    n_symbols = 0

    for states_path in sorted(states_dir.glob("*.parquet")):
        sym = states_path.stem
        feat_path = features_dir / states_path.name

        try:
            states_df = pd.read_parquet(states_path)
            states_df.index = pd.to_datetime(states_df.index)
        except Exception as exc:
            log.warning("replay: skip %s (states read error: %s)", sym, exc)
            continue

        # Load price data from features (which has close) or a separate daily file
        daily_path = REPO_ROOT / "data" / "daily" / f"{sym}.parquet"
        try:
            if daily_path.exists():
                prices_df = pd.read_parquet(daily_path)
            else:
                prices_df = pd.read_parquet(feat_path)[["close"]] if feat_path.exists() else pd.DataFrame()
            prices_df.index = pd.to_datetime(prices_df.index)
        except Exception as exc:
            log.warning("replay: skip %s (price read error: %s)", sym, exc)
            continue

        # Optional date filter
        if start_date is not None:
            states_df = states_df[states_df.index >= start_date]
            prices_df = prices_df[prices_df.index >= start_date]
        if end_date is not None:
            states_df = states_df[states_df.index <= end_date]
            prices_df = prices_df[prices_df.index <= end_date]

        trades = replay_symbol(
            sym, states_df, prices_df,
            atr_stop=atr_stop,
            atr_target=atr_target,
            max_bars=max_bars,
        )
        all_trades.extend(t.to_dict() for t in trades)
        n_symbols += 1

    log.info("replay_universe: %d trades across %d symbols", len(all_trades), n_symbols)
    if not all_trades:
        return pd.DataFrame()
    return pd.DataFrame(all_trades).sort_values("trigger_date").reset_index(drop=True)


def replay_summary(trade_log: pd.DataFrame) -> dict:
    """Aggregate statistics from a replay trade log.

    Returns
    -------
    dict with:
        n_trades, win_rate, avg_pnl_pct, avg_pnl_atr,
        expectancy_atr, profit_factor,
        max_consecutive_losses, by_exit_reason,
        sharpe_approx (mean_pnl_atr / std_pnl_atr),
        survivorship_bias_caveat (always True)
    """
    if trade_log is None or trade_log.empty:
        return {"n_trades": 0, "survivorship_bias_caveat": True}

    closed = trade_log[trade_log["exit_reason"].notna()].copy()
    n = len(closed)
    if n == 0:
        return {"n_trades": 0, "survivorship_bias_caveat": True}

    pnl_atr = pd.to_numeric(closed["pnl_atr"], errors="coerce").dropna()
    pnl_pct = pd.to_numeric(closed["pnl_pct"], errors="coerce").dropna()
    winners = pnl_atr[pnl_atr > 0]
    losers = pnl_atr[pnl_atr <= 0]

    win_rate = len(winners) / n if n > 0 else float("nan")
    gross_profit = float(winners.sum()) if len(winners) > 0 else 0.0
    gross_loss = float(abs(losers.sum())) if len(losers) > 0 else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("nan")
    expectancy_atr = float(pnl_atr.mean()) if len(pnl_atr) > 0 else float("nan")
    sharpe_approx = (float(pnl_atr.mean()) / float(pnl_atr.std())
                     if len(pnl_atr) > 1 and pnl_atr.std() > 0 else float("nan"))

    # Max consecutive losses
    signs = (pnl_atr > 0).astype(int).tolist()
    max_loss_streak = 0
    streak = 0
    for s in signs:
        if s == 0:
            streak += 1
            max_loss_streak = max(max_loss_streak, streak)
        else:
            streak = 0

    return {
        "n_trades": n,
        "n_winners": len(winners),
        "n_losers": len(losers),
        "win_rate": round(win_rate, 4),
        "avg_pnl_pct": round(float(pnl_pct.mean()), 6) if len(pnl_pct) > 0 else float("nan"),
        "avg_pnl_atr": round(expectancy_atr, 4),
        "profit_factor": round(profit_factor, 4) if np.isfinite(profit_factor) else float("nan"),
        "expectancy_atr": round(expectancy_atr, 4),
        "sharpe_approx": round(sharpe_approx, 4) if np.isfinite(sharpe_approx) else float("nan"),
        "max_consecutive_losses": max_loss_streak,
        "by_exit_reason": closed["exit_reason"].value_counts().to_dict(),
        "survivorship_bias_caveat": True,
    }
