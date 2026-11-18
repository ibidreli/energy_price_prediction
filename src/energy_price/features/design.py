"""Feature sets for the models: from the shared table to the columns a model is trained on.

Every model picks a feature set by name, so all models use the same, tested definitions. A set is a
list of blocks; a new block is a function ``table -> dict of columns`` registered in ``BLOCKS``.

- ``B1`` calendar: hour dummies, a linear trend within each hour (position x hour), weekend, PV
  share of cantons with a holiday.
- ``B2`` B1 plus radiation (per 100 W/m²) and radiation change x position, because rising sun tips
  the system long towards :45 and falling sun short (EDA, H8).
- ``B3`` B2 plus the price level until D-2: mean of D-2, mean of the last 7 days and P10, P50, P90 of
  the same clock time over 56 days. Rows before 2026-01-30 have no value (minimum history, 28 days).
- ``B4`` B3 plus system balance until D-2 (mean TSI per 100 MW, share of short quarter hours,
  direction at the same clock time on D-2 and D-7) and cloud cover of the run D-2 18 UTC (per 100 %).
- ``B5`` B4 plus the ENTSO-E load (issue #37): day-ahead forecast for D in GW, its change over the
  hour x position (the consumption part of H8) and the mean actual load of D-2.
- ``B6`` B5 plus the day-ahead prices of D-1 per 100 EUR/MWh: CH at the same clock time, the CH daily
  mean and the spreads of the daily means to DE-LU, FR, IT-North and AT. Never the price of D.

B5 and B6 only have values where ``known_at_issue_entsoe`` allows them: in the history only once the
live check confirmed a series (``entsoe.HISTORY_RULES``, docs/data.md, section 7). Until then the
load columns are missing before the live period, and ``complete_days`` drops those days. The block
``residual_load`` is registered but in no set yet, because whether ENTSO-E has a solar and wind
forecast for Switzerland is only known from the live fetch.

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


def price_level(table: pd.DataFrame) -> dict[str, pd.Series]:
    """Price level until D-2 in ct/kWh: daily mean, 7-day mean and the quantiles of the reference."""
    columns = ["price_mean_d2", "price_mean_7d", "price_q10_56d", "price_q50_56d", "price_q90_56d"]
    return {column: table[column] for column in columns}


def system_balance(table: pd.DataFrame) -> dict[str, pd.Series]:
    """Mean TSI on D-2 per 100 MW, share short on D-2, direction at the same clock time on D-2 and D-7."""
    return {
        "tsi_mean_d2_100": table["tsi_mean_d2"] / 100,
        "short_share_d2": table["short_share_d2"],
        "direction_d2": table["direction_d2"],
        "direction_d7": table["direction_d7"],
    }


def cloud_cover(table: pd.DataFrame) -> dict[str, pd.Series]:
    """PV-weighted total cloud cover forecast per 100 %."""
    return {"cloud_cover_100": table["cloud_cover"] / 100}


def load_forecast(table: pd.DataFrame) -> dict[str, pd.Series]:
    """ENTSO-E load in GW: day-ahead forecast for D, its change over the hour x position, mean actual load of D-2."""
    return {
        "load_forecast_gw": table["load_forecast_mw"] / 1000,
        "load_forecast_change_gw_x_position": table["load_forecast_change_mw"] / 1000 * table["position"],
        "load_actual_mean_d2_gw": table["load_actual_mean_d2_mw"] / 1000,
    }


def residual_load(table: pd.DataFrame) -> dict[str, pd.Series]:
    """ENTSO-E residual load forecast (load - solar - wind) in GW and its change over the hour x position."""
    return {
        "residual_load_gw": table["residual_load_mw"] / 1000,
        "residual_load_change_gw_x_position": table["residual_load_change_mw"] / 1000 * table["position"],
    }


def day_ahead_prices(table: pd.DataFrame) -> dict[str, pd.Series]:
    """Day-ahead prices of D-1 per 100 EUR/MWh: CH at the same clock time, CH daily mean, spreads to neighbours."""
    columns = ["da_price_ch_d1", "da_price_ch_mean_d1", "da_spread_de_lu_d1", "da_spread_fr_d1",
               "da_spread_it_north_d1", "da_spread_at_d1"]
    return {f"{column}_100": table[column] / 100 for column in columns}


BLOCKS: dict[str, Callable[[pd.DataFrame], dict[str, pd.Series]]] = {
    "hour": hour_dummies,
    "trend": trend_within_hour,
    "day_type": day_type,
    "radiation": radiation,
    "price_level": price_level,
    "system_balance": system_balance,
    "cloud_cover": cloud_cover,
    "load_forecast": load_forecast,
    "residual_load": residual_load,
    "day_ahead_prices": day_ahead_prices,
}

FEATURE_SETS: dict[str, list[str]] = {
    "B1": ["hour", "trend", "day_type"],
    "B2": ["hour", "trend", "day_type", "radiation"],
    "B3": ["hour", "trend", "day_type", "radiation", "price_level"],
    "B4": ["hour", "trend", "day_type", "radiation", "price_level", "system_balance", "cloud_cover"],
    "B5": ["hour", "trend", "day_type", "radiation", "price_level", "system_balance", "cloud_cover",
           "load_forecast"],
    "B6": ["hour", "trend", "day_type", "radiation", "price_level", "system_balance", "cloud_cover",
           "load_forecast", "day_ahead_prices"],
}


def design_matrix(table: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    """Columns of ``feature_set`` for the rows of ``table`` (without intercept, missing values kept)."""
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"unknown feature set {feature_set!r}, expected one of {tuple(FEATURE_SETS)}")
    columns: dict[str, pd.Series] = {}
    for block in FEATURE_SETS[feature_set]:
        columns |= BLOCKS[block](table)
    return pd.DataFrame(columns, index=table.index)
