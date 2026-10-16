"""Confidence intervals for per-group rates and between-group gaps.

- Group rates (selection rate, accuracy, TPR, FPR, ...) are proportions, so they get
  a Wilson score interval: accurate for small groups and rates near 0 or 1, where
  the textbook normal interval breaks down.
- Gaps (demographic parity / equalized odds difference) are a max-minus-min across
  groups, which has no closed form, so they get a stratified percentile bootstrap:
  resample rows within each group (keeping group sizes fixed) with a fixed seed.
"""

from __future__ import annotations

import math
import warnings
from statistics import NormalDist

import numpy as np
import pandas as pd


def wilson_interval(successes: int, trials: int, level: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a proportion. (nan, nan) when trials == 0."""
    if trials <= 0:
        return math.nan, math.nan
    z = NormalDist().inv_cdf(0.5 + level / 2)
    p = successes / trials
    denom = 1 + z**2 / trials
    centre = (p + z**2 / (2 * trials)) / denom
    half = z * math.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2)) / denom
    # Exact bounds at the edges (centre - half is ~1e-17, not 0, in floating point).
    lo = 0.0 if successes == 0 else max(0.0, centre - half)
    hi = 1.0 if successes == trials else min(1.0, centre + half)
    return lo, hi


def _group_rates(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Selection rate, TPR and FPR for each bootstrap replicate (rows of y and p)."""
    selection = p.mean(axis=1)
    positives = y.sum(axis=1)
    negatives = y.shape[1] - positives
    with np.errstate(invalid="ignore", divide="ignore"):
        tpr = (y & p).sum(axis=1) / positives
        fpr = (~y & p).sum(axis=1) / negatives
    return selection, np.where(positives > 0, tpr, np.nan), np.where(negatives > 0, fpr, np.nan)


def _gap(per_group: np.ndarray) -> np.ndarray:
    """Max minus min across groups (axis 0), ignoring groups undefined in a replicate."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN replicate -> NaN
        return np.nanmax(per_group, axis=0) - np.nanmin(per_group, axis=0)


def bootstrap_gaps(y_true, y_pred, groups, n_boot: int = 1000, level: float = 0.95,
                   seed: int = 42) -> dict[str, tuple[float, float]]:
    """Percentile bootstrap intervals for the demographic parity and equalized odds differences."""
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_pred).astype(bool)
    groups = pd.Series(np.asarray(groups)).astype(str).to_numpy()
    rng = np.random.default_rng(seed)

    selection, tpr, fpr = [], [], []
    for g in pd.unique(groups):
        idx = np.flatnonzero(groups == g)
        draws = idx[rng.integers(0, len(idx), size=(n_boot, len(idx)))]
        s, t, f = _group_rates(y_true[draws], y_pred[draws])
        selection.append(s), tpr.append(t), fpr.append(f)

    dpd = _gap(np.array(selection))
    eod = np.fmax(_gap(np.array(tpr)), _gap(np.array(fpr)))
    alpha = (1 - level) / 2
    return {
        "demographic_parity_difference": _percentiles(dpd, alpha),
        "equalized_odds_difference": _percentiles(eod, alpha),
    }


def _percentiles(values: np.ndarray, alpha: float) -> tuple[float, float]:
    values = values[~np.isnan(values)]
    if values.size == 0:
        return math.nan, math.nan
    lo, hi = np.quantile(values, [alpha, 1 - alpha])
    return float(lo), float(hi)
