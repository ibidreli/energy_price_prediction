"""Bandit and online aggregation over existing models: explorative check, not a model candidate.

Issue #21, documented in docs/exploration/bandit/README.md. The strategies choose or weight the forecasts of existing models per delivery day. All of them
work on arrays with the axes day, quarter hour of the day, arm (model) and quantile:

- ``forecasts`` has the shape (days, slots, arms, 3) and holds P10, P50 and P90,
- ``losses`` has the same shape and holds the pinball loss of each of these forecasts,
- ``valid`` has the shape (days,) and is False for days without complete forecasts or prices.

Availability rule of the project: the decision for delivery day D uses only losses of days up to
D-2, because later prices are not published at the issue time (``energy_price.availability``).
Every strategy therefore reads the feedback through ``_known_day``.

These strategies get new feedback every day, while ``make backtest`` refits weekly. ``run.py`` in
this folder therefore evaluates them outside the standard workflow, on the same forecasts, days and
scores.
"""

from __future__ import annotations

from typing import Any

import numpy as np

ARMS = ("reference", "B1", "B2", "gbm_quantile")
SLOTS_PER_DAY = 96
SEED = 0
TUNE_SEEDS = 20
TEST_SEEDS = 200
FEEDBACK_LAG_DAYS = 2
EXPLORATION_FLOOR = 0.05  # minimum share per arm, as in the production bandit that inspired the test

HALF_LIVES = (7, 14, 28, 56, np.inf)
ETAS = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
_HEDGE_GRID = [{"eta": eta, "half_life": half_life} for eta in ETAS for half_life in HALF_LIVES]
_HALF_LIFE_GRID = [{"half_life": half_life} for half_life in HALF_LIVES]

# Settings tried on April and May (`make bandit-exploration STEP=tune`).
GRID: dict[str, list[dict[str, float]]] = {
    "hedge_global": _HEDGE_GRID,
    "hedge_hour": _HEDGE_GRID,
    "hedge_slot": _HEDGE_GRID,
    "follow_leader_slot": _HALF_LIFE_GRID,
    "thompson_bandit": _HALF_LIFE_GRID,
    "thompson_bandit_no_floor": _HALF_LIFE_GRID,
    "thompson_full_info": _HALF_LIFE_GRID,
}

# Lowest pinball loss on April and May (docs/exploration/bandit/results/tune_selection.csv).
CHOSEN: dict[str, dict[str, float]] = {
    "hedge_global": {"eta": 0.3, "half_life": 7},
    "hedge_hour": {"eta": 0.1, "half_life": 14},
    "hedge_slot": {"eta": 0.1, "half_life": 28},
    "follow_leader_slot": {"half_life": 28},
    "thompson_bandit": {"half_life": np.inf},
    "thompson_bandit_no_floor": {"half_life": np.inf},
    "thompson_full_info": {"half_life": np.inf},
}


def hour_groups(n_slots: int = SLOTS_PER_DAY) -> np.ndarray:
    """Group index of every quarter hour: its hour of the day."""
    return np.arange(n_slots) // 4


def decay(half_life: float) -> float:
    """Daily factor for old observations; 1 keeps them forever."""
    return 1.0 if np.isinf(half_life) else 0.5 ** (1 / half_life)


def _known_day(day: int, valid: np.ndarray) -> int | None:
    """The newest day whose prices became known for the decision on ``day``, if it has any."""
    known = day - FEEDBACK_LAG_DAYS
    return known if known >= 0 and valid[known] else None


def hedge(forecasts: np.ndarray, losses: np.ndarray, valid: np.ndarray, groups: np.ndarray, eta: float,
          half_life: float) -> np.ndarray:
    """Exponential weights per group and quantile; the forecast is the weighted mean of the arms."""
    n_days, n_slots, n_arms, n_quantiles = forecasts.shape
    n_groups = int(groups.max()) + 1
    cumulative = np.zeros((n_groups, n_arms, n_quantiles))
    out = np.full((n_days, n_slots, n_quantiles), np.nan)
    factor = decay(half_life)
    for day in range(n_days):
        cumulative *= factor
        known = _known_day(day, valid)
        if known is not None:
            for group in range(n_groups):
                cumulative[group] += losses[known, groups == group].mean(axis=0)
        weights = np.exp(-eta * (cumulative - cumulative.min(axis=1, keepdims=True)))
        weights /= weights.sum(axis=1, keepdims=True)
        out[day] = (weights[groups] * forecasts[day]).sum(axis=1)
    return np.sort(out, axis=-1)


def follow_leader(forecasts: np.ndarray, losses: np.ndarray, valid: np.ndarray, half_life: float) -> np.ndarray:
    """Per quarter hour the arm with the lowest decayed loss so far, without any random choice."""
    n_days, n_slots, n_arms, n_quantiles = forecasts.shape
    joint = losses.mean(axis=-1)
    cumulative = np.zeros((n_slots, n_arms))
    out = np.full((n_days, n_slots, n_quantiles), np.nan)
    factor = decay(half_life)
    rows = np.arange(n_slots)
    for day in range(n_days):
        cumulative *= factor
        known = _known_day(day, valid)
        if known is not None:
            cumulative += joint[known]
        out[day] = forecasts[day, rows, cumulative.argmin(axis=1)]
    return out


def thompson(forecasts: np.ndarray, losses: np.ndarray, valid: np.ndarray, half_life: float, full_info: bool,
             floor: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian Thompson sampling per quarter hour; returns the forecasts and the chosen arms.

    One draw per arm from the belief about its mean loss, the lowest draw wins. ``full_info=False``
    is the real bandit: it learns only from the arm it chose. ``floor`` is the minimum share of
    every arm. The belief starts at the mean loss over all arms of the slot, with a weight of one
    observation, and its spread is the pooled loss variance, at least 1.
    """
    n_days, n_slots, n_arms, n_quantiles = forecasts.shape
    rng = np.random.default_rng(seed)
    joint = losses.mean(axis=-1)
    weight, total, square = (np.zeros((n_slots, n_arms)) for _ in range(3))
    out = np.full((n_days, n_slots, n_quantiles), np.nan)
    choices = np.zeros((n_days, n_slots), dtype=int)
    factor = decay(half_life)
    rows = np.arange(n_slots)
    for day in range(n_days):
        weight, total, square = weight * factor, total * factor, square * factor
        known = _known_day(day, valid)
        if known is not None:
            seen = np.ones_like(weight, dtype=bool) if full_info else np.eye(n_arms, dtype=bool)[choices[known]]
            weight += seen
            total += np.where(seen, joint[known], 0.0)
            square += np.where(seen, joint[known] ** 2, 0.0)
        count = np.maximum(weight.sum(axis=1), 1e-12)
        pooled = total.sum(axis=1) / count
        variance = np.maximum(square.sum(axis=1) / count - pooled ** 2, 1.0)
        mean = (pooled[:, None] + total) / (1 + weight)
        draw = rng.normal(mean, np.sqrt(variance[:, None] / (1 + weight)))
        choice = draw.argmin(axis=1)
        explore = rng.random(n_slots) < floor * n_arms
        choice[explore] = rng.integers(0, n_arms, explore.sum())
        choices[day] = choice
        out[day] = forecasts[day, rows, choice]
    return out, choices


def run_strategy(name: str, forecasts: np.ndarray, losses: np.ndarray, valid: np.ndarray,
                 settings: dict[str, Any], seed: int = SEED) -> np.ndarray:
    """Forecasts of one strategy from ``GRID``. Thompson variants depend on ``seed``."""
    n_slots = forecasts.shape[1]
    groups = {"hedge_global": np.zeros(n_slots, dtype=int), "hedge_hour": hour_groups(n_slots),
              "hedge_slot": np.arange(n_slots)}
    if name in groups:
        return hedge(forecasts, losses, valid, groups[name], settings["eta"], settings["half_life"])
    if name == "follow_leader_slot":
        return follow_leader(forecasts, losses, valid, settings["half_life"])
    variants = {"thompson_bandit": (False, EXPLORATION_FLOOR), "thompson_bandit_no_floor": (False, 0.0),
                "thompson_full_info": (True, EXPLORATION_FLOOR)}
    if name in variants:
        full_info, floor = variants[name]
        return thompson(forecasts, losses, valid, settings["half_life"], full_info, floor, seed)[0]
    raise ValueError(f"unknown strategy {name!r}")
