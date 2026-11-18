"""Modelling table: one row per quarter hour of a delivery day with target and features.

Every feature is known when the forecast for delivery day D is issued (D-1, 11:00 Swiss time):

- Calendar: local hour, position of the quarter hour in the hour, weekend, and the share of the
  installed PV capacity in cantons with a legal holiday. Fixed years in advance.
- Weather: PV-weighted global radiation of the six sites and its change from the previous to the
  next hour, and PV-weighted total cloud cover. Only from the run of D-2 18 UTC
  (``weather_forecasts.parquet`` holds no other run), available about nine hours before the issue
  time (docs/data.md, section 1).
- Swissgrid history (``history_features``): price level, system balance and direction until the
  end of D-2. Every value is computed per delivery day through ``known_at_issue``, never with a
  fixed lag. Past prices are final here but were provisional at the issue time, so a backtest with
  them is slightly optimistic (docs/data.md, section 1).
- ENTSO-E (``entsoe_features``, issue #37): the day-ahead load forecast for D and its change over
  the hour, the week-ahead load forecast for D, the mean actual load of D-2, the solar and wind
  forecast for D and the residual load, the Swiss day-ahead price of D-1 at the same clock time
  and its daily mean, and the spreads of the daily means to DE-LU, FR, IT-North and AT. Every value
  comes from ``known_at_issue_entsoe``: in the live period only what our job had fetched by D-1
  11:00, before that only the series whose availability the live check confirmed. The price of D
  itself is never used (docs/data.md, section 7).

The target ``aep_ct_kwh`` is the price of the quarter hour itself and never a feature of that
quarter hour. The same holds for the direction label ``short`` (1 if the control area was short,
TSI < 0): it is a second training target for models that forecast the direction first, never an
input. Rows follow Swiss local time: the switch to summer time has 92, the switch to winter time
100 quarter hours. A feature "at the same clock time" meets both days, because the short day has no
02:xx and the long day has 02:xx twice (docs/data.md, section 1, "Zeitumstellung").
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from energy_price.availability import known_at_issue
from energy_price.data.entsoe import known_at_issue_entsoe, live_start
from energy_price.paths import ROOT

TZ = "Europe/Zurich"
# Owner decision of 2026-10-01: the single price system applies since 2026-01-01.
FIRST_DAY = pd.Timestamp("2026-01-01")
TARGET = "aep_ct_kwh"
DIRECTION = "short"
RADIATION = "shortwave_radiation"
CLOUD_COVER = "cloud_cover"
TSI = "tsi_mw"
QUARTER_HOUR = pd.Timedelta(minutes=15)

MEAN_WINDOW_DAYS = 7
QUANTILE_WINDOW_DAYS = 56  # the window of the historical reference (EDA H11)
# Fewest days behind the price quantiles. Four full weeks hold every weekday equally often and leave
# about 3 values beyond P10 and P90. Rows before 2026-01-30 therefore have no B3 or B4 features.
MIN_HISTORY_DAYS = 28
PRICE_QUANTILES = {0.1: "price_q10_56d", 0.5: "price_q50_56d", 0.9: "price_q90_56d"}
PRICE_COLUMNS = ["price_mean_d2", "price_mean_7d", *PRICE_QUANTILES.values()]
BALANCE_COLUMNS = ["tsi_mean_d2", "short_share_d2", "direction_d2", "direction_d7"]
HISTORY_COLUMNS = [*PRICE_COLUMNS, *BALANCE_COLUMNS]

SPREAD_ZONES = {"DE_LU": "da_spread_de_lu_d1", "FR": "da_spread_fr_d1", "IT_NORTH": "da_spread_it_north_d1",
                "AT": "da_spread_at_d1"}
SOLAR, WIND = {"B16"}, {"B18", "B19"}  # ENTSO-E production types: solar, wind offshore and onshore
ENTSOE_COLUMNS = [
    "load_forecast_mw", "load_forecast_change_mw", "load_week_ahead_mw", "load_actual_mean_d2_mw",
    "solar_forecast_mw", "wind_forecast_mw", "residual_load_mw", "residual_load_change_mw",
    "da_price_ch_d1", "da_price_ch_mean_d1", *SPREAD_ZONES.values(),
]


@dataclass(frozen=True)
class Inputs:
    prices: pd.DataFrame
    weather: pd.DataFrame
    sites: pd.DataFrame
    holidays: pd.DataFrame
    pv_capacity: pd.DataFrame
    balance: pd.DataFrame
    entsoe: pd.DataFrame | None = None


def load_inputs(root: Path = ROOT) -> Inputs:
    """Read the processed tables (``make data``) and the meta files of the repository."""
    processed, meta = root / "data" / "processed", root / "data" / "meta"
    entsoe = processed / "entsoe.parquet"
    return Inputs(
        prices=pd.read_parquet(processed / "balance_prices.parquet"),
        weather=pd.read_parquet(processed / "weather_forecasts.parquet"),
        sites=pd.read_csv(meta / "weather_sites.csv"),
        holidays=pd.read_csv(meta / "holidays_ch.csv"),
        pv_capacity=pd.read_csv(meta / "pv_capacity_by_canton.csv"),
        balance=pd.read_parquet(processed / "control_area_balance.parquet"),
        entsoe=pd.read_parquet(entsoe) if entsoe.exists() else None,
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


def slot(stamps_local: pd.Series) -> pd.Series:
    """Local clock time as minutes after midnight. Both passes of the repeated hour share a slot."""
    return stamps_local.dt.hour * 60 + stamps_local.dt.minute


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


def pv_weighted_by_hour(weather: pd.DataFrame, sites: pd.DataFrame, variable: str) -> pd.DataFrame:
    """PV-weighted ``variable`` per delivery day and hour stamp, from each delivery day's own run.

    The stamp D 00:00 is the last hour of D-1 and is dropped here: it belongs to the run for D, which
    appears after the issue time for D-1 (the same leak the EDA found in the first round). Sites with
    a missing value are left out and the weights of the others renormalised; if all sites miss a value
    it stays missing.
    """
    weights = sites.set_index("site")["pv_share"]
    unknown = set(weather["site"]) - set(weights.index)
    if unknown:
        raise ValueError(f"no PV share for sites {sorted(unknown)}")
    day_start = weather["delivery_day"].dt.tz_localize(TZ).dt.tz_convert("UTC")
    own = weather[weather["valid_time_utc"] > day_start]

    w = own["site"].map(weights)
    value = own[variable]
    keys = [own["delivery_day"], own["valid_time_utc"]]
    numerator = (value * w).groupby(keys).sum(min_count=1)
    denominator = (w * value.notna()).groupby(keys).sum()
    hourly = (numerator / denominator.where(denominator > 0)).rename(variable).reset_index()
    hourly = hourly.sort_values(["delivery_day", "valid_time_utc"], ignore_index=True)
    return hourly.rename(columns={"valid_time_utc": "hour_stamp_utc"})


def radiation_by_hour(weather: pd.DataFrame, sites: pd.DataFrame) -> pd.DataFrame:
    """PV-weighted radiation per delivery day and hour stamp, and its change over the hour.

    A stamp h:00 holds the mean of the preceding hour h-1:00 to h:00. ``radiation_change`` is
    (next hour - previous hour) / 2 in W/m² per hour, one-sided at the first and last hour of the day.
    """
    hourly = pv_weighted_by_hour(weather, sites, RADIATION).rename(columns={RADIATION: "radiation"})
    by_day = hourly.groupby("delivery_day")["radiation"]
    after, before = by_day.shift(-1), by_day.shift(1)
    change = (after - before) / 2
    change = change.fillna(after - hourly["radiation"]).fillna(hourly["radiation"] - before)
    hourly["radiation_change"] = change
    return hourly


def whole_days(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Rows since 2026-01-01 of the local days with a value in every quarter hour, with their ``day``.

    The expected number follows from the local day (92, 96 or 100), so a day with a gap is dropped
    entirely and never gives a mean over part of a day. Whether a day counts only depends on its own
    rows, and ``known_at_issue`` still decides per delivery day which of these days may be used.
    """
    day = frame["timestamp_local"].dt.tz_localize(None).dt.normalize()
    rows = frame.loc[frame[column].notna() & (day >= FIRST_DAY), ["timestamp_local", column]].assign(day=day)
    expected = {d: int(((d + pd.Timedelta(days=1)).tz_localize(TZ) - d.tz_localize(TZ)) / QUARTER_HOUR)
                for d in rows["day"].unique()}
    return rows[rows.groupby("day")[column].transform("size") == rows["day"].map(expected)]


def price_level(prices: pd.DataFrame, day: pd.Timestamp, stamps_local: pd.Series) -> pd.DataFrame:
    """Mean price on D-2 and over D-8 to D-2, and quantiles per slot over D-57 to D-2, for delivery day ``day``.

    ``prices`` comes from ``whole_days``. Without a whole D-2 all columns stay NaN, so old quantiles
    never continue after the data ends. If the window has no gap, the quantiles equal those of the
    historical reference (``models.shared.reference``), which allows a model to reproduce it. An
    incomplete day inside the window is left out here, while the reference uses its quarter hours.
    """
    out = pd.DataFrame(np.nan, index=stamps_local.index, columns=PRICE_COLUMNS)
    known = known_at_issue(prices, day.date())
    last = known[known["day"] == day - pd.Timedelta(days=2)]
    if last.empty:
        return out
    out["price_mean_d2"] = last[TARGET].mean()

    week = known[known["day"] >= day - pd.Timedelta(days=MEAN_WINDOW_DAYS + 1)]
    if week["day"].nunique() == MEAN_WINDOW_DAYS:
        out["price_mean_7d"] = week[TARGET].mean()

    window = known[known["day"] >= day - pd.Timedelta(days=QUANTILE_WINDOW_DAYS + 1)]
    if window["day"].nunique() >= MIN_HISTORY_DAYS:
        quantiles = window.groupby(slot(window["timestamp_local"]))[TARGET].quantile(list(PRICE_QUANTILES)).unstack()
        quantiles.columns = [PRICE_QUANTILES[tau] for tau in quantiles.columns]
        out[quantiles.columns] = quantiles.reindex(slot(stamps_local).to_numpy()).to_numpy()
    return out


def direction_at_clock(known: pd.DataFrame, day: pd.Timestamp, stamps_local: pd.Series) -> np.ndarray:
    """Direction on the local day ``day`` at the clock time of each stamp: +1 long, -1 short, as in the evaluation.

    Swissgrid's data has no gap and no duplicate in UTC. Local time does, because of the clock changes,
    which affects the lookup for a target two and seven days later (docs/data.md, "Zeitumstellung"):

    - Short day (e.g. 2026-03-29): clocks jump from 02:00 to 03:00, so 02:00 to 02:45 never happened
      and no direction exists. These targets get 0, "no information". A NaN would remove the whole
      delivery day from the comparison of all models.
    - Long day (e.g. 2026-10-25): clocks return from 03:00 to 02:00, so 02:00 to 02:45 happens twice,
      first in summer time and then in winter time, as two different quarter hours with their own TSI.
      They share a slot, so the value is the mean of both directions: -1 or +1 if they agree, else 0.

    A day without data stays NaN, in contrast to a clock time the day does not have.
    """
    rows = known[known["day"] == day]
    if rows.empty:
        return np.full(len(stamps_local), np.nan)
    direction = pd.Series(np.where(rows[TSI] < 0, -1.0, 1.0), index=slot(rows["timestamp_local"]).to_numpy())
    return direction.groupby(level=0).mean().reindex(slot(stamps_local).to_numpy()).fillna(0.0).to_numpy()


def system_balance(balance: pd.DataFrame, day: pd.Timestamp, stamps_local: pd.Series) -> pd.DataFrame:
    """Mean TSI and share of short quarter hours on D-2, direction at the same clock time on D-2 and D-7.

    ``balance`` comes from ``whole_days``. Without a whole D-2 all columns stay NaN.
    """
    out = pd.DataFrame(np.nan, index=stamps_local.index, columns=BALANCE_COLUMNS)
    known = known_at_issue(balance, day.date())
    d2 = day - pd.Timedelta(days=2)
    last = known[known["day"] == d2]
    if last.empty:
        return out
    out["tsi_mean_d2"] = last[TSI].mean()
    out["short_share_d2"] = (last[TSI] < 0).mean()
    out["direction_d2"] = direction_at_clock(known, d2, stamps_local)
    out["direction_d7"] = direction_at_clock(known, day - pd.Timedelta(days=7), stamps_local)
    return out


def history_features(table: pd.DataFrame, prices: pd.DataFrame, balance: pd.DataFrame) -> pd.DataFrame:
    """``HISTORY_COLUMNS`` for the rows of ``table``, each delivery day from what is known at its issue time."""
    prices, balance = whole_days(prices, TARGET), whole_days(balance, TSI)
    frames = []
    for day, rows in table.groupby("delivery_day"):
        stamps = rows["timestamp_local"]
        frames.append(pd.concat([price_level(prices, day, stamps), system_balance(balance, day, stamps)], axis=1))
    return pd.concat(frames).reindex(table.index)


def day_grid(day: pd.Timestamp) -> pd.DatetimeIndex:
    """The 92, 96 or 100 quarter hours of the local day ``day`` in UTC."""
    start = day.tz_localize(TZ).tz_convert("UTC")
    end = (day + pd.Timedelta(days=1)).tz_localize(TZ).tz_convert("UTC")
    return pd.date_range(start, end, freq="15min", inclusive="left")


def series_values(known: pd.DataFrame, series: str, zone: str = "CH", psr_types: set[str] | None = None) -> pd.Series:
    """Values of one ENTSO-E series per quarter hour (UTC).

    With ``psr_types`` (generation by production type) the matching groups are summed, otherwise the
    groups are averaged: one group for the load, minimum and maximum for the week-ahead forecast.
    """
    rows = known[(known["series"] == series) & (known["zone"] == zone)]
    if psr_types is not None:
        rows = rows[rows["group"].map(lambda group: group.partition("/")[2]).isin(psr_types)]
        return rows.groupby("timestamp_utc")["value"].sum(min_count=1)
    return rows.groupby("timestamp_utc")["value"].mean()


def whole_day_values(values: pd.Series, day: pd.Timestamp) -> pd.Series | None:
    """The values of the local day ``day`` if every quarter hour has one, else None (no partial means)."""
    out = values.reindex(day_grid(day))
    return None if out.isna().any() else out


def change_over_hour(values: pd.Series) -> pd.Series:
    """Per quarter hour: (mean of the next hour - mean of the previous hour) / 2 in MW per hour.

    One-sided at the first and last hour of the day, as ``radiation_change``. ``values`` holds the
    quarter hours of one day; hours are taken in UTC, so the switch days need no special case.
    """
    hours = values.groupby(values.index.floor("h")).mean()
    after, before = hours.shift(-1), hours.shift(1)
    change = ((after - before) / 2).fillna(after - hours).fillna(hours - before)
    return pd.Series(change.reindex(values.index.floor("h")).to_numpy(), index=values.index)


def entsoe_day(known: pd.DataFrame, day: pd.Timestamp, rows: pd.DataFrame) -> pd.DataFrame:
    """``ENTSOE_COLUMNS`` for the quarter hours ``rows`` of delivery day ``day`` from the selected rows ``known``."""
    out = pd.DataFrame(np.nan, index=rows.index, columns=ENTSOE_COLUMNS)
    grid, stamps = day_grid(day), rows["timestamp_utc"].to_numpy()

    def on_rows(values: pd.Series) -> np.ndarray:
        return values.reindex(stamps).to_numpy()

    load = series_values(known, "load_day_ahead").reindex(grid)
    solar = series_values(known, "wind_solar_day_ahead", psr_types=SOLAR).reindex(grid)
    wind = series_values(known, "wind_solar_day_ahead", psr_types=WIND).reindex(grid)
    residual = load - solar - wind
    out["load_forecast_mw"] = on_rows(load)
    out["load_forecast_change_mw"] = on_rows(change_over_hour(load))
    out["load_week_ahead_mw"] = on_rows(series_values(known, "load_week_ahead").reindex(grid))
    out["solar_forecast_mw"], out["wind_forecast_mw"] = on_rows(solar), on_rows(wind)
    out["residual_load_mw"] = on_rows(residual)
    out["residual_load_change_mw"] = on_rows(change_over_hour(residual))

    actual = whole_day_values(series_values(known, "load_actual"), day - pd.Timedelta(days=2))
    if actual is not None:
        out["load_actual_mean_d2_mw"] = actual.mean()

    d1 = day - pd.Timedelta(days=1)
    ch = whole_day_values(series_values(known, "price_day_ahead", "CH"), d1)
    if ch is not None:
        mean = ch.mean()
        out["da_price_ch_mean_d1"] = mean
        # Same clock time on D-1: both passes of a repeated hour are averaged. A clock time D-1 does not
        # have (02:xx after the short day) takes the daily mean of D-1.
        by_slot = ch.groupby(slot(pd.Series(ch.index.tz_convert(TZ))).to_numpy()).mean()
        out["da_price_ch_d1"] = by_slot.reindex(slot(rows["timestamp_local"]).to_numpy()).fillna(mean).to_numpy()
        for zone, column in SPREAD_ZONES.items():
            other = whole_day_values(series_values(known, "price_day_ahead", zone), d1)
            if other is not None:
                out[column] = mean - other.mean()
    return out


def entsoe_features(table: pd.DataFrame, entsoe: pd.DataFrame | None) -> pd.DataFrame:
    """``ENTSOE_COLUMNS`` for the rows of ``table``, each delivery day from ``known_at_issue_entsoe``.

    The features need only the local days D-2 to D, so each day gets only that window of the table,
    with the start of the live period taken from the whole table.
    """
    if entsoe is None or entsoe.empty:
        return pd.DataFrame(np.nan, index=table.index, columns=ENTSOE_COLUMNS)
    start = live_start(entsoe)
    by_day = dict(tuple(entsoe.groupby("local_day")))
    frames = []
    for day, rows in table.groupby("delivery_day"):
        window = [by_day[d] for d in (day - pd.Timedelta(days=2), day - pd.Timedelta(days=1), day) if d in by_day]
        if not window:
            frames.append(pd.DataFrame(np.nan, index=rows.index, columns=ENTSOE_COLUMNS))
            continue
        known = known_at_issue_entsoe(pd.concat(window), day.date(), live_from=start)
        frames.append(entsoe_day(known, day, rows))
    return pd.concat(frames).reindex(table.index)


def build_table(inputs: Inputs, days: list[pd.Timestamp] | None = None) -> pd.DataFrame:
    """Target and features for every quarter hour of ``days`` (default: all days with a price or weather).

    Columns: ``delivery_day``, ``timestamp_local``, ``timestamp_utc``, ``hour``, ``position``,
    ``weekend``, ``holiday_pv_share``, ``radiation``, ``radiation_change``, ``cloud_cover``,
    ``aep_ct_kwh``, ``short``, ``HISTORY_COLUMNS`` and ``ENTSOE_COLUMNS``. Missing weather stays NaN;
    so does the history as long as its days do not exist, and every ENTSO-E value that was not known
    at the issue time; target and direction are NaN where no value exists yet.
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
    # Cloud cover is a momentary value, so it lies at most 60 minutes next to its stamp (docs/eda.md).
    table["hour_stamp_utc"] = table["timestamp_utc"].dt.floor("h") + pd.Timedelta(hours=1)
    hour_keys = ["delivery_day", "hour_stamp_utc"]
    weather = radiation_by_hour(inputs.weather, inputs.sites).merge(
        pv_weighted_by_hour(inputs.weather, inputs.sites, CLOUD_COVER), on=hour_keys, validate="one_to_one")
    weather["delivery_day"] = weather["delivery_day"].astype(table["delivery_day"].dtype)
    weather["hour_stamp_utc"] = weather["hour_stamp_utc"].astype(table["hour_stamp_utc"].dtype)
    table = table.merge(weather, on=hour_keys, how="left", validate="many_to_one")

    prices["timestamp_utc"] = prices["timestamp_utc"].astype(table["timestamp_utc"].dtype)
    table = table.merge(prices[["timestamp_utc", TARGET]], on="timestamp_utc", how="left", validate="one_to_one")

    label = direction_label(inputs.balance)
    label["timestamp_utc"] = label["timestamp_utc"].astype(table["timestamp_utc"].dtype)
    table = table.merge(label, on="timestamp_utc", how="left", validate="one_to_one")
    table = table.drop(columns="hour_stamp_utc")
    table = pd.concat([table, history_features(table, prices, inputs.balance)], axis=1)
    return pd.concat([table, entsoe_features(table, inputs.entsoe)], axis=1)


def complete_days(table: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """Delivery days on which every quarter hour has a value in ``columns``."""
    complete = table[columns].notna().all(axis=1).groupby(table["delivery_day"]).all()
    return complete.index[complete].to_numpy()
