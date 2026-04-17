"""Trader-facing dashboard generation layer.

Modules:
  freshness  — classify setups as fresh / stale / extended
  levels     — compute entry, stop, target, and S/R levels
  action     — assign one of 5 rule-based action labels
  narrative  — generate short rule-based trade narrative
  charts     — produce PNG charts for top setups
  selector   — choose top 5-7 actionable setups
  packet     — assemble a structured per-symbol analysis packet
"""
