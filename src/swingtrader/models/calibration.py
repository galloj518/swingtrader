"""Probability calibration diagnostics.

All functions are pure; they only compute metrics from arrays.
They do not fit or modify models.

Calibration contract:
  A well-calibrated classifier predicts probability p for a set of events,
  and empirically ~p fraction of those events actually occur.

Metrics produced:
  brier_score     — mean squared error of probability estimates (lower = better)
  ece             — expected calibration error (weighted mean |predicted - actual|)
  reliability     — bin-by-bin fraction_positive vs mean_predicted_probability

Use these at the end of each walk-forward fold and after fitting the production
model to verify that isotonic calibration achieved its goal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Mean squared error between true labels and predicted probabilities.

    Range [0, 1]; 0 is perfect; a naive 50/50 classifier scores ~0.25.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_prob)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean((y_true[mask] - y_prob[mask]) ** 2))


def reliability_diagram(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bins: int = 10,
) -> dict:
    """Bin-by-bin calibration data for a reliability (calibration) diagram.

    Returns a dict with keys:
        bin_edges           — (n_bins+1,) array of bin boundaries
        bin_centers         — (n_bins,) midpoints
        mean_predicted_prob — mean predicted probability in each bin
        fraction_positive   — observed positive rate in each bin
        counts              — number of samples in each bin

    Bins with zero samples have NaN for mean_predicted_prob and fraction_positive.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_prob)
    y_true, y_prob = y_true[mask], y_prob[mask]

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    mean_pred = np.full(n_bins, np.nan)
    frac_pos = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        # Include right edge in last bin
        in_bin = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        cnt = int(in_bin.sum())
        counts[i] = cnt
        if cnt > 0:
            mean_pred[i] = float(y_prob[in_bin].mean())
            frac_pos[i] = float(y_true[in_bin].mean())

    return {
        "bin_edges": edges,
        "bin_centers": centers,
        "mean_predicted_prob": mean_pred,
        "fraction_positive": frac_pos,
        "counts": counts,
    }


def ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error — sample-weighted mean absolute calibration gap.

    ECE = Σ_b (|counts_b| / N) × |fraction_positive_b - mean_predicted_prob_b|

    Range [0, 1]; lower is better.  A perfectly calibrated model has ECE = 0.
    """
    rd = reliability_diagram(y_true, y_prob, n_bins=n_bins)
    counts = rd["counts"]
    n_total = counts.sum()
    if n_total == 0:
        return float("nan")

    fp = rd["fraction_positive"]
    mp = rd["mean_predicted_prob"]
    valid = np.isfinite(fp) & np.isfinite(mp)
    if not valid.any():
        return float("nan")

    gaps = np.abs(fp[valid] - mp[valid])
    weights = counts[valid] / n_total
    return float(np.dot(weights, gaps))


def calibration_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    label: str = "model",
    n_bins: int = 10,
) -> dict:
    """Convenience wrapper: returns a single dict with all calibration metrics."""
    bs = brier_score(y_true, y_prob)
    ece_val = ece(y_true, y_prob, n_bins=n_bins)
    rd = reliability_diagram(y_true, y_prob, n_bins=n_bins)

    y_arr = np.asarray(y_true, dtype=float)
    mask = np.isfinite(y_arr)
    prevalence = float(y_arr[mask].mean()) if mask.sum() > 0 else float("nan")
    # Brier score of a naive classifier always predicting prevalence
    brier_naive = float((prevalence ** 2) * (1 - prevalence) + ((1 - prevalence) ** 2) * prevalence) \
        if np.isfinite(prevalence) else float("nan")
    brier_skill = 1.0 - bs / brier_naive if (np.isfinite(brier_naive) and brier_naive > 0) else float("nan")

    return {
        "label": label,
        "n_samples": int(np.isfinite(np.asarray(y_true, dtype=float)).sum()),
        "prevalence": prevalence,
        "brier_score": bs,
        "brier_skill_score": brier_skill,
        "ece": ece_val,
        "reliability": rd,
    }


def aggregate_oos_reports(reports: list[dict]) -> dict:
    """Average numeric scalar metrics across a list of per-fold calibration_report dicts."""
    if not reports:
        return {}
    keys = ["brier_score", "brier_skill_score", "ece", "prevalence", "n_samples"]
    out: dict = {"n_folds": len(reports)}
    for k in keys:
        vals = [r[k] for r in reports if np.isfinite(r.get(k, float("nan")))]
        out[f"{k}_mean"] = float(np.mean(vals)) if vals else float("nan")
        out[f"{k}_std"] = float(np.std(vals)) if len(vals) > 1 else float("nan")
    return out


def reports_to_dataframe(reports: list[dict]) -> pd.DataFrame:
    """Convert list of calibration reports to a tidy DataFrame (one row per fold)."""
    rows = []
    for r in reports:
        row = {k: v for k, v in r.items() if not isinstance(v, dict)}
        rows.append(row)
    return pd.DataFrame(rows)
