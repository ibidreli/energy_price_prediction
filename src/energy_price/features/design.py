"""Feature sets for the models: from the shared table to the columns a model is trained on.

Every model picks a feature set by name, so all models use the same, tested definitions. A set is a
list of blocks; a new block is a function ``table -> dict of columns`` registered in ``BLOCKS``.

- ``B1`` calendar: hour dummies, a linear trend within each hour (position x hour), weekend, PV
  share of cantons with a holiday.
- ``B2`` B1 plus radiation (per 100 W/m²) and radiation change x position, because rising sun tips
  the system long towards :45 and falling sun short (EDA, H8).

Every column comes from ``features.dataset.build_table`` and is known at D-1, 11:00.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

HOURS = range(24)


def hour_dummies(table: pd.DataFrame) -> dict[str, pd.Series]:
    """One column per hour 01 to 23 (hour 00 is the reference level)."""
    return {f"hour_{h:02d}": (table["hour"] == h).astype(float) for h in HOURS if h != 0}


def trend_within_hour(table: pd.DataFrame) -> dict[str, pd.Series]:
    """Position of the quarter hour in the hour (-1.5 to 1.5), separately for every hour (EDA H8)."""
    return {f"trend_{h:02d}": table["position"] * (table["hour"] == h) for h in HOURS}


def day_type(table: pd.DataFrame) -> dict[str, pd.Series]:
    """Weekend and the share of the installed PV capacity in cantons with a holiday."""
    return {"weekend": table["weekend"], "holiday_pv_share": table["holiday_pv_share"]}


def radiation(table: pd.DataFrame) -> dict[str, pd.Series]:
    """PV-weighted radiation forecast per 100 W/m² and its change over the hour x position."""
    return {
        "radiation_100": table["radiation"] / 100,
        "radiation_change_100_x_position": table["radiation_change"] / 100 * table["position"],
    }


BLOCKS: dict[str, Callable[[pd.DataFrame], dict[str, pd.Series]]] = {
    "hour": hour_dummies,
    "trend": trend_within_hour,
    "day_type": day_type,
    "radiation": radiation,
}

FEATURE_SETS: dict[str, list[str]] = {
    "B1": ["hour", "trend", "day_type"],
    "B2": ["hour", "trend", "day_type", "radiation"],
}


def design_matrix(table: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    """Columns of ``feature_set`` for the rows of ``table`` (without intercept, missing values kept)."""
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"unknown feature set {feature_set!r}, expected one of {tuple(FEATURE_SETS)}")
    columns: dict[str, pd.Series] = {}
    for block in FEATURE_SETS[feature_set]:
        columns |= BLOCKS[block](table)
    return pd.DataFrame(columns, index=table.index)
