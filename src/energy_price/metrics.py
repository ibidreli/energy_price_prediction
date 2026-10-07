"""Scores for the probabilistic forecast: P10, P50 and P90 of the price per quarter hour.

Decided on 2026-10-06, reasons and sources in docs/evaluation.md:
- main metric: mean pinball loss over P10, P50 and P90 (lower is better), used to choose between models;
- always reported with it: MAE of the median, coverage of the P10 to P90 interval (target 80 %) and its
  mean width.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

QUANTILES = (0.1, 0.5, 0.9)
TARGET_COVERAGE = QUANTILES[-1] - QUANTILES[0]

ArrayLike = npt.ArrayLike


def _arrays(*values: ArrayLike) -> list[np.ndarray]:
    arrays = [np.asarray(v, dtype=float) for v in values]
    if len({a.shape for a in arrays}) != 1:
        raise ValueError(f"inputs differ in shape: {[a.shape for a in arrays]}")
    if any(np.isnan(a).any() for a in arrays):
        raise ValueError("inputs contain NaN; drop or explain missing quarter hours before scoring")
    return arrays


def pinball_loss(actual: ArrayLike, predicted: ArrayLike, quantile: float) -> float:
    """Mean pinball loss of the forecast ``predicted`` for the ``quantile`` (between 0 and 1), in ct/kWh.

    Under-forecasts cost ``quantile`` per ct/kWh, over-forecasts ``1 - quantile``. For 0.5 it is half the MAE.
    """
    if not 0 < quantile < 1:
        raise ValueError(f"quantile must lie between 0 and 1, got {quantile}")
    y, q = _arrays(actual, predicted)
    diff = y - q
    return float(np.mean(np.maximum(quantile * diff, (quantile - 1) * diff)))


def mean_pinball_loss(actual: ArrayLike, forecasts: dict[float, ArrayLike]) -> float:
    """Pinball loss averaged over the quantiles in ``forecasts`` ({quantile: predicted values})."""
    return float(np.mean([pinball_loss(actual, predicted, q) for q, predicted in forecasts.items()]))


def coverage(actual: ArrayLike, lower: ArrayLike, upper: ArrayLike) -> float:
    """Share of actual values inside [lower, upper], both bounds included."""
    y, lo, hi = _arrays(actual, lower, upper)
    if (lo > hi).any():
        raise ValueError(f"lower bound above upper bound in {int((lo > hi).sum())} rows (crossing quantiles)")
    return float(np.mean((y >= lo) & (y <= hi)))


def mean_width(lower: ArrayLike, upper: ArrayLike) -> float:
    """Mean width of the interval [lower, upper] in ct/kWh."""
    lo, hi = _arrays(lower, upper)
    if (lo > hi).any():
        raise ValueError(f"lower bound above upper bound in {int((lo > hi).sum())} rows (crossing quantiles)")
    return float(np.mean(hi - lo))


def mae(actual: ArrayLike, predicted: ArrayLike) -> float:
    """Mean absolute error in ct/kWh."""
    y, p = _arrays(actual, predicted)
    return float(np.mean(np.abs(y - p)))


def score(actual: ArrayLike, p10: ArrayLike, p50: ArrayLike, p90: ArrayLike) -> dict[str, float]:
    """All agreed scores for one forecast: the main metric first, then the ones reported with it."""
    return {
        "mean_pinball": mean_pinball_loss(actual, {0.1: p10, 0.5: p50, 0.9: p90}),
        "mae_p50": mae(actual, p50),
        "coverage_p10_p90": coverage(actual, p10, p90),
        "width_p10_p90": mean_width(p10, p90),
    }
