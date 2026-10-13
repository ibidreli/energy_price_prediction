"""Scores for quantile forecasts P10, P50 and P90 of the balance energy price in ct/kWh.

The main metric is the mean pinball loss over the three quantiles: a proper scoring rule, so the
best score goes to the true quantiles, and it rates the median and the band at once. MAE of the
median, coverage of P10 to P90 (target 80 %) and the mean width are always reported with it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

QUANTILES = (0.1, 0.5, 0.9)
COLUMNS = {0.1: "q10", 0.5: "q50", 0.9: "q90"}
TARGET = "aep_ct_kwh"


def pinball(actual: np.ndarray | pd.Series, predicted: np.ndarray | pd.Series, tau: float) -> np.ndarray:
    """Pinball loss of every single forecast: tau * (y - q) if y >= q, else (1 - tau) * (q - y)."""
    diff = np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float)
    return np.maximum(tau * diff, (tau - 1) * diff)


def pinball_per_row(frame: pd.DataFrame, target: str = TARGET) -> pd.Series:
    """Pinball loss of each row, averaged over the three quantiles."""
    losses = [pinball(frame[target], frame[COLUMNS[tau]], tau) for tau in QUANTILES]
    return pd.Series(np.mean(losses, axis=0), index=frame.index, name="pinball")


def score(frame: pd.DataFrame, target: str = TARGET) -> dict[str, float]:
    """Mean pinball loss, its parts per quantile, MAE of P50, coverage and width of P10 to P90."""
    _check(frame, target)
    y = frame[target]
    out = {"pinball": float(pinball_per_row(frame, target).mean())}
    for tau in QUANTILES:
        out[f"pinball_{COLUMNS[tau]}"] = float(pinball(y, frame[COLUMNS[tau]], tau).mean())
    out["mae_q50"] = float((y - frame["q50"]).abs().mean())
    out["coverage_80"] = float(((y >= frame["q10"]) & (y <= frame["q90"])).mean())
    out["width_80"] = float((frame["q90"] - frame["q10"]).mean())
    out["n_quarter_hours"] = float(len(frame))
    return out


def skill_score(model_loss: float, reference_loss: float) -> float:
    """1 - model / reference. Above 0 the model is better than the reference, 0.10 means 10 % lower loss."""
    if reference_loss <= 0:
        raise ValueError("reference loss must be positive")
    return 1 - model_loss / reference_loss


def daily_pinball(frame: pd.DataFrame, day_column: str = "delivery_day", target: str = TARGET) -> pd.Series:
    """Mean pinball loss per delivery day. Days, not quarter hours, are the independent units."""
    _check(frame, target)
    return pinball_per_row(frame, target).groupby(frame[day_column]).mean()


def _check(frame: pd.DataFrame, target: str) -> None:
    columns = [target, *COLUMNS.values()]
    if frame[columns].isna().any().any():
        raise ValueError("forecasts or actual values contain missing values; select complete rows first")
    if (frame["q10"] > frame["q50"]).any() or (frame["q50"] > frame["q90"]).any():
        raise ValueError("quantiles cross: q10 <= q50 <= q90 is required")
