"""Modelling table: one row per quarter hour of a delivery day with target and features.

Every feature is known when the forecast for delivery day D is issued (D-1, 11:00 Swiss time):

- Calendar: local hour, position of the quarter hour in the hour, weekend, and the share of the
  installed PV capacity in cantons with a legal holiday. Fixed years in advance.
- Weather: PV-weighted global radiation of the six sites and its change from the previous to the
  next hour. Only from the run of D-2 18 UTC (``weather_forecasts.parquet`` holds no other run),
  available about nine hours before the issue time (docs/data.md, section 1).

The target ``aep_ct_kwh`` is the price of the quarter hour itself and never a feature. The same holds
for the direction label ``short`` (1 if the control area was short, TSI < 0): it is a second training
target for models that forecast the direction first, never an input. Past prices
enter the models only as training targets, selected with ``known_at_issue`` (data until D-2).
Rows follow Swiss local time: the switch to summer time has 92, the switch to winter time
100 quarter hours.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from energy_price.paths import ROOT

TZ = "Europe/Zurich"
# Owner decision of 2026-10-01: the single price system applies since 2026-01-01.
FIRST_DAY = pd.Timestamp("2026-01-01")
TARGET = "aep_ct_kwh"
DIRECTION = "short"
RADIATION = "shortwave_radiation"


@dataclass(frozen=True)
class Inputs:
    prices: pd.DataFrame
    weather: pd.DataFrame
    sites: pd.DataFrame
    holidays: pd.DataFrame
    pv_capacity: pd.DataFrame
    balance: pd.DataFrame | None = None


def load_inputs(root: Path = ROOT) -> Inputs:
    """Read the processed tables (``make data``) and the meta files of the repository."""
    processed, meta = root / "data" / "processed", root / "data" / "meta"
    return Inputs(
        prices=pd.read_parquet(processed / "balance_prices.parquet"),
        weather=pd.read_parquet(processed / "weather_forecasts.parquet"),
        sites=pd.read_csv(meta / "weather_sites.csv"),
        holidays=pd.read_csv(meta / "holidays_ch.csv"),
        pv_capacity=pd.read_csv(meta / "pv_capacity_by_canton.csv"),
        balance=pd.read_parquet(processed / "control_area_balance.parquet"),
    )


def single_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Quarter hours since 2026-01-01 with a single balance energy price."""
    local_day = prices["timestamp_local"].dt.tz_localize(None).dt.normalize()
    keep = (local_day >= FIRST_DAY) & prices[TARGET].notna()
    return prices.loc[keep, ["timestamp_utc", "timestamp_local", TARGET]].reset_index(drop=True)


def direction_label(balance: pd.DataFrame) -> pd.DataFrame:
    """Per quarter hour: 1.0 if the control area was short (TSI < 0), 0.0 if long, NaN without TSI.

    Same definition as the EDA (``eda.py``). In 2026 the price was never negative when short (price A)
    and never above 27.6 ct/kWh when long (price B), see docs/overview.md.
    """
    tsi = balance["tsi_mw"]
    short = (tsi < 0).astype(float).where(tsi.notna())
    return pd.DataFrame({"timestamp_utc": balance["timestamp_utc"], DIRECTION: short})


def quarter_hours(days: list[pd.Timestamp]) -> pd.DataFrame:
    """All quarter hours of the given local delivery days, 92 to 100 per day."""
    frames = []
    for day in days:
        start = pd.Timestamp(day).tz_localize(TZ)
        end = (pd.Timestamp(day) + pd.Timedelta(days=1)).tz_localize(TZ)
        stamps = pd.date_range(start, end, freq="15min", inclusive="left")
        frames.append(pd.DataFrame({"delivery_day": pd.Timestamp(day), "timestamp_local": stamps}))
    out = pd.concat(frames, ignore_index=True)
    out["timestamp_utc"] = out["timestamp_local"].dt.tz_convert("UTC")
    return out


def calendar_features(stamps_local: pd.Series) -> pd.DataFrame:
    """Hour, position in the hour (-1.5, -0.5, 0.5, 1.5 for :00 to :45) and weekend flag."""
    return pd.DataFrame({
        "hour": stamps_local.dt.hour,
        "position": stamps_local.dt.minute / 15 - 1.5,
        "weekend": (stamps_local.dt.dayofweek >= 5).astype(float),
    }, index=stamps_local.index)


def holiday_pv_share(holidays: pd.DataFrame, pv_capacity: pd.DataFrame) -> pd.Series:
    """Per date: share of the installed PV capacity in cantons with a legal holiday (0 to 1)."""
    if holidays.duplicated(["date", "canton"]).any():
        raise ValueError("holidays lists a canton twice on the same date")
    merged = holidays.merge(pv_capacity[["canton", "share"]], on="canton", how="left", validate="many_to_one")
    if merged["share"].isna().any():
        missing = sorted(merged.loc[merged["share"].isna(), "canton"].unique())
        raise ValueError(f"no PV share for cantons {missing}")
    return merged.groupby(pd.to_datetime(merged["date"]))["share"].sum().rename("holiday_pv_share")


def radiation_by_hour(weather: pd.DataFrame, sites: pd.DataFrame) -> pd.DataFrame:
    """PV-weighted radiation per delivery day and hour stamp, and its change over the hour.

    A stamp h:00 holds the mean of the preceding hour h-1:00 to h:00. The stamp D 00:00 is the last
    hour of D-1 and is dropped here: it belongs to the run for D, which appears after the issue time
    for D-1 (the same leak the EDA found in the first round). Sites with a missing value are left out
    and the weights of the others renormalised; if all sites miss a value it stays missing.

    ``radiation_change`` is (next hour - previous hour) / 2 in W/m² per hour, one-sided at the first
    and last hour of the day.
    """
    weights = sites.set_index("site")["pv_share"]
    unknown = set(weather["site"]) - set(weights.index)
    if unknown:
        raise ValueError(f"no PV share for sites {sorted(unknown)}")
    day_start = weather["delivery_day"].dt.tz_localize(TZ).dt.tz_convert("UTC")
    own = weather[weather["valid_time_utc"] > day_start]

    w = own["site"].map(weights)
    value = own[RADIATION]
    keys = [own["delivery_day"], own["valid_time_utc"]]
    numerator = (value * w).groupby(keys).sum(min_count=1)
    denominator = (w * value.notna()).groupby(keys).sum()
    hourly = (numerator / denominator.where(denominator > 0)).rename("radiation").reset_index()
    hourly = hourly.sort_values(["delivery_day", "valid_time_utc"], ignore_index=True)

    by_day = hourly.groupby("delivery_day")["radiation"]
    after, before = by_day.shift(-1), by_day.shift(1)
    change = (after - before) / 2
    change = change.fillna(after - hourly["radiation"]).fillna(hourly["radiation"] - before)
    hourly["radiation_change"] = change
    return hourly.rename(columns={"valid_time_utc": "hour_stamp_utc"})


def build_table(inputs: Inputs, days: list[pd.Timestamp] | None = None) -> pd.DataFrame:
    """Target and features for every quarter hour of ``days`` (default: all days with a price or weather).

    Columns: ``delivery_day``, ``timestamp_local``, ``timestamp_utc``, ``hour``, ``position``,
    ``weekend``, ``holiday_pv_share``, ``radiation``, ``radiation_change``, ``aep_ct_kwh``, ``short``.
    Missing radiation stays NaN; target and direction are NaN where no value exists yet.
    """
    prices = single_prices(inputs.prices)
    if days is None:
        price_days = prices["timestamp_local"].dt.tz_localize(None).dt.normalize()
        days = sorted(set(price_days) | set(inputs.weather["delivery_day"]))
        days = [d for d in days if d >= FIRST_DAY]
    table = quarter_hours(days)
    table = pd.concat([table, calendar_features(table["timestamp_local"])], axis=1)

    holidays = holiday_pv_share(inputs.holidays, inputs.pv_capacity)
    table["holiday_pv_share"] = table["delivery_day"].map(holidays).fillna(0.0)

    # Quarter hours h:00 to h:45 (UTC hour h) take the radiation stamped h+1:00, the mean over that hour.
    table["hour_stamp_utc"] = table["timestamp_utc"].dt.floor("h") + pd.Timedelta(hours=1)
    radiation = radiation_by_hour(inputs.weather, inputs.sites)
    radiation["delivery_day"] = radiation["delivery_day"].astype(table["delivery_day"].dtype)
    radiation["hour_stamp_utc"] = radiation["hour_stamp_utc"].astype(table["hour_stamp_utc"].dtype)
    table = table.merge(radiation, on=["delivery_day", "hour_stamp_utc"], how="left", validate="many_to_one")

    prices["timestamp_utc"] = prices["timestamp_utc"].astype(table["timestamp_utc"].dtype)
    table = table.merge(prices[["timestamp_utc", TARGET]], on="timestamp_utc", how="left", validate="one_to_one")

    if inputs.balance is None:
        table[DIRECTION] = np.nan
    else:
        label = direction_label(inputs.balance)
        label["timestamp_utc"] = label["timestamp_utc"].astype(table["timestamp_utc"].dtype)
        table = table.merge(label, on="timestamp_utc", how="left", validate="one_to_one")
    return table.drop(columns="hour_stamp_utc")


def complete_days(table: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """Delivery days on which every quarter hour has a value in ``columns``."""
    complete = table[columns].notna().all(axis=1).groupby(table["delivery_day"]).all()
    return complete.index[complete].to_numpy()
