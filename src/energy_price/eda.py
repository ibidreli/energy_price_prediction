"""Reusable calculations of the exploratory analysis (notebooks/01_eda.ipynb).

Everything here describes the data. Nothing builds a model feature: where a calculation could become
one, the docstring says what is known at the issue time (see availability.py and docs/data.md).
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from energy_price.availability import known_at_issue

WORKDAY = "Werktag"
SATURDAY = "Samstag"
SUNDAY = "Sonntag"
HOLIDAY = "Feiertag"
BRIDGE_DAY = "Brückentag"
DAY_TYPES = [WORKDAY, BRIDGE_DAY, SATURDAY, SUNDAY, HOLIDAY]


def _require_tz(stamps: pd.Series, name: str) -> None:
    if not isinstance(stamps.dtype, pd.DatetimeTZDtype):
        raise TypeError(f"{name} must be timezone-aware")


def clock_slot(stamps: pd.Series) -> pd.Series:
    """Quarter hour of the local clock, 0 for 00:00 to 95 for 23:45.

    Uses the clock time, not the position in the day: on 29.03.2026 the slots 8 to 11 (02:00 to 02:45)
    do not exist, on 25.10.2026 they occur twice.
    """
    _require_tz(stamps, "stamps")
    return (stamps.dt.hour * 4 + stamps.dt.minute // 15).astype("int64")


def holiday_pv_share(holidays: pd.DataFrame, pv_capacity: pd.DataFrame) -> pd.Series:
    """Per date: share of the installed PV capacity that lies in cantons with a legal holiday.

    ``holidays`` has the columns ``date`` and ``canton`` (data/meta/holidays_ch.csv),
    ``pv_capacity`` the columns ``canton`` and ``share`` (data/meta/pv_capacity_by_canton.csv).
    Known years in advance, so usable as a feature without restriction.
    """
    if holidays.duplicated(["date", "canton"]).any():
        raise ValueError("holidays lists a canton twice on the same date")
    share = holidays.merge(pv_capacity[["canton", "share"]], on="canton", how="left", validate="many_to_one")
    if share["share"].isna().any():
        missing = sorted(share.loc[share["share"].isna(), "canton"].unique())
        raise ValueError(f"no PV share for cantons {missing}")
    dates = pd.to_datetime(share["date"]).dt.date
    return share.groupby(dates)["share"].sum().rename("holiday_pv_share")


def day_types(days: pd.Series, pv_share: pd.Series, threshold: float = 0.5) -> pd.Series:
    """Classify calendar days as workday, bridge day, Saturday, Sunday or holiday.

    A Monday to Friday counts as holiday when at least ``threshold`` of the PV capacity has a legal
    holiday (``holiday_pv_share``). A holiday on a weekend stays Saturday or Sunday. A bridge day is a
    remaining workday whose previous and next day are both free (weekend or holiday), e.g. the Friday
    after Ascension.
    """
    # A Timestamp index would never match the date lookups below and turn every holiday into a workday.
    if not all(isinstance(d, dt.date) and not isinstance(d, dt.datetime) for d in pv_share.index):
        raise TypeError("pv_share must be indexed by datetime.date, as returned by holiday_pv_share")
    dates = pd.to_datetime(pd.Series(days)).dt.date
    one = pd.Timedelta(days=1)

    def is_holiday(day) -> bool:
        return float(pv_share.get(day, 0.0)) >= threshold

    def is_free(stamp: pd.Timestamp) -> bool:
        return stamp.weekday() >= 5 or is_holiday(stamp.date())

    def classify(day) -> str:
        stamp = pd.Timestamp(day)
        if stamp.weekday() == 5:
            return SATURDAY
        if stamp.weekday() == 6:
            return SUNDAY
        if is_holiday(day):
            return HOLIDAY
        if is_free(stamp - one) and is_free(stamp + one):
            return BRIDGE_DAY
        return WORKDAY

    return pd.Series([classify(d) for d in dates], index=dates.index, name="day_type")


def direction_matches(frame: pd.DataFrame, lags_days: list[int], value: str = "tsi_mw",
                      time_column: str = "timestamp_local") -> pd.DataFrame:
    """Compare the system direction (short if ``value`` < 0) with the same clock time ``lag`` days earlier.

    One row per (date, slot, lag) where both values exist, with the boolean column ``match``.
    A lag of at least 2 days is known at the issue time for every target quarter hour; a lag of 1 day
    is not (D-1 is not yet published at 11:00).
    """
    _require_tz(frame[time_column], time_column)
    data = pd.DataFrame({
        "date": pd.to_datetime(frame[time_column].dt.date),
        "slot": clock_slot(frame[time_column]),
        "short": (frame[value] < 0).astype(float),
    })[frame[value].notna()]
    # The repeated hour on the switch to winter time gives two values per slot. If they disagree there is
    # no majority, and the slot is left out like a missing value.
    share = data.groupby(["date", "slot"])["short"].mean().unstack()
    wide = (share > 0.5).where(share.notna() & (share != 0.5))
    wide = wide.asfreq("D")

    rows = []
    for lag in lags_days:
        earlier = wide.shift(lag)
        both = wide.notna() & earlier.notna()
        match = (wide == earlier).where(both)
        long = match.stack().dropna().rename("match").reset_index()  # pandas 3 keeps NaN in stack()
        long["lag_days"] = lag
        rows.append(long)
    out = pd.concat(rows, ignore_index=True)
    out["date"] = out["date"].dt.date
    out["match"] = out["match"].astype(bool)
    return out[["date", "slot", "lag_days", "match"]]


def extreme_events(frame: pd.DataFrame, low: float, high: float, value: str = "aep_ct_kwh",
                   time_column: str = "timestamp_utc") -> pd.DataFrame:
    """Group consecutive quarter hours below ``low`` or above ``high`` into events.

    Returns one row per event: ``kind`` ("low" or "high"), ``start``, ``end`` (last quarter hour),
    ``quarter_hours`` and ``extreme`` (minimum for low, maximum for high).
    """
    columns = ["kind", "start", "end", "quarter_hours", "extreme"]
    data = frame[[time_column, value]].sort_values(time_column)
    kind = pd.Series(None, index=data.index, dtype="object")
    kind[data[value] < low] = "low"
    kind[data[value] > high] = "high"
    flagged = data.assign(kind=kind)[kind.notna()]
    if flagged.empty:
        return pd.DataFrame(columns=columns)

    new_event = (flagged[time_column].diff() != pd.Timedelta(minutes=15)) | (flagged["kind"] != flagged["kind"].shift())
    events = flagged.groupby(new_event.cumsum()).agg(
        kind=("kind", "first"), start=(time_column, "min"), end=(time_column, "max"),
        quarter_hours=(value, "size"), minimum=(value, "min"), maximum=(value, "max"),
    )
    events["extreme"] = events["minimum"].where(events["kind"] == "low", events["maximum"])
    return events[columns].reset_index(drop=True)


def within_hour_trend(frame: pd.DataFrame, value: str = "tsi_mw", time_column: str = "timestamp_local") -> pd.DataFrame:
    """Values of the four quarter hours of every local clock hour and their change from :00 to :45.

    One row per date and hour with the columns ``date``, ``hour``, ``q00``, ``q15``, ``q30``, ``q45`` and
    ``trend`` (``q45 - q00``). Hours with a missing quarter hour are left out, and so is the repeated hour
    on the switch to winter time: its two passes would otherwise be mixed into one row.
    """
    _require_tz(frame[time_column], time_column)
    stamps = frame[time_column]
    data = pd.DataFrame({"date": stamps.dt.date, "hour": stamps.dt.hour, "minute": stamps.dt.minute, "value": frame[value]})
    grouped = data.groupby(["date", "hour", "minute"])["value"]
    counts = grouped.size().unstack()
    wide = grouped.first().unstack().reindex(columns=[0, 15, 30, 45])
    complete = (counts == 1).all(axis=1) & wide.notna().all(axis=1)
    wide = wide[complete]
    wide.columns = ["q00", "q15", "q30", "q45"]
    wide["trend"] = wide["q45"] - wide["q00"]
    return wide.reset_index()


def weather_to_quarter_hours(weather: pd.DataFrame, weights: pd.Series, columns: list[str]) -> pd.DataFrame:
    """PV-weighted mean over the sites, spread from hours to quarter hours (index ``timestamp_utc``).

    ``weights`` maps site to PV share. A site with a missing value is left out and the weights of the
    others are renormalised; if all sites miss a value, it stays missing.

    Mapping: the quarter hours from h-1:00 to h-1:45 take the value stamped h:00. This matches radiation,
    sunshine duration and precipitation, which describe the preceding hour. Momentary values (cloud cover,
    temperature, wind) are then at most 45 minutes away from their stamp.

    The value stamped D 00:00 exists twice, as the last hour of delivery day D-1 and as the first of D.
    Only the row of the delivery day that contains the quarter hours is kept: the run for D is published
    after the issue time for D-1 and must not reach D-1 23:00 to 23:45.

    The weather table only holds the run of D-2 18 UTC, so every value is known at the issue time.
    """
    unknown = set(weather["site"]) - set(weights.index)
    if unknown:
        raise ValueError(f"no weight for sites {sorted(unknown)}")
    weather = _rows_of_own_delivery_day(weather)
    w = weather["site"].map(weights)
    values = weather[columns]
    numerator = values.mul(w, axis=0).groupby(weather["valid_time_utc"]).sum(min_count=1)
    denominator = values.notna().mul(w, axis=0).groupby(weather["valid_time_utc"]).sum()
    return _hours_to_quarter_hours(numerator / denominator.where(denominator > 0))


def _rows_of_own_delivery_day(weather: pd.DataFrame) -> pd.DataFrame:
    """Keep the stamp D 00:00 only in the run of the delivery day whose last hour it describes.

    Reason in the docstring of ``weather_to_quarter_hours``.
    """
    covered_day = (weather["valid_time_local"] - pd.Timedelta(hours=1)).dt.tz_localize(None).dt.normalize()
    weather = weather[covered_day == pd.to_datetime(weather["delivery_day"])]
    if weather.duplicated(["site", "valid_time_utc"]).any():
        raise ValueError("more than one value per site and hour after selecting the delivery day")
    return weather


def _hours_to_quarter_hours(hourly: pd.DataFrame) -> pd.DataFrame:
    """The quarter hours h-1:00 to h-1:45 take the value stamped h:00 (index ``timestamp_utc``)."""
    start = hourly.index.min() - pd.Timedelta(hours=1)
    stamps = pd.date_range(start, hourly.index.max(), freq="15min", inclusive="left", name="timestamp_utc")
    out = hourly.reindex(stamps.floor("h") + pd.Timedelta(hours=1))
    out.index = stamps
    return out


def site_spread(weather: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Standard deviation over the sites per hour (unweighted, ddof=0), spread to quarter hours.

    A measure of how differently the forecast sees the regions. Missing if fewer than two sites have a
    value. Same delivery-day selection and mapping as ``weather_to_quarter_hours``, so it is known at the
    issue time as well.
    """
    weather = _rows_of_own_delivery_day(weather)
    grouped = weather.groupby("valid_time_utc")[columns]
    return _hours_to_quarter_hours(grouped.std(ddof=0).where(grouped.count() >= 2))


def daily_mean_abs_change(values: pd.Series, tz: str = "Europe/Zurich") -> pd.Series:
    """Mean absolute change between consecutive hours, per local calendar day.

    ``values`` is quarter-hourly with a timezone-aware index, as from ``weather_to_quarter_hours``; the
    four quarter hours of an hour share one value, so the first one stands for the hour. Changes across
    midnight are left out, so each day only uses its own hours.
    """
    if values.index.tz is None:
        raise TypeError("values must have a timezone-aware index")
    # Floor in UTC: flooring a local index fails on the repeated hour of the switch to winter time.
    utc = values.index.tz_convert("UTC")
    hourly = values.groupby(utc.floor("h")).first()
    local_day = pd.Series(hourly.index.tz_convert(tz).date, index=hourly.index)
    consecutive = hourly.index.to_series().diff() == pd.Timedelta(hours=1)  # no bridging over missing hours
    change = hourly.diff().abs()[local_day.eq(local_day.shift()) & consecutive]
    return change.groupby(local_day[change.index]).mean().rename("mean_abs_change")


def same_slot_pairs(frame: pd.DataFrame, lags_days: list[int], value: str = "aep_ct_kwh",
                    time_column: str = "timestamp_local") -> pd.DataFrame:
    """Each quarter hour next to the value at the same local clock time ``lag`` days earlier.

    Columns ``date``, ``slot``, ``lag_days``, ``value`` and ``earlier``; only pairs where both exist. The two
    passes of the repeated hour on the switch to winter time are averaged. A lag of at least 2 days is known
    at the issue time for every target quarter hour (end of D-2); a lag of 1 day is not.
    """
    _require_tz(frame[time_column], time_column)
    if not lags_days or any(lag < 1 for lag in lags_days):
        raise ValueError(f"lags_days must be a non-empty list of lags of at least 1 day, got {lags_days}")
    data = pd.DataFrame({
        "date": pd.to_datetime(frame[time_column].dt.date),
        "slot": clock_slot(frame[time_column]),
        "value": frame[value],
    })
    wide = data.groupby(["date", "slot"])["value"].mean().unstack().asfreq("D")
    rows = []
    for lag in lags_days:
        pairs = pd.DataFrame({"value": wide.stack(), "earlier": wide.shift(lag).stack()}).dropna()
        rows.append(pairs.assign(lag_days=lag).reset_index())
    out = pd.concat(rows, ignore_index=True)
    out["date"] = out["date"].dt.date
    return out[["date", "slot", "lag_days", "value", "earlier"]]


def past_slot_quantiles(frame: pd.DataFrame, day: dt.date, window_days: int,
                        quantiles: tuple[float, ...] = (0.1, 0.5, 0.9), value: str = "aep_ct_kwh",
                        time_column: str = "timestamp_local") -> pd.DataFrame:
    """Quantiles of ``value`` per local clock slot over the last ``window_days`` days known for ``day``.

    Uses ``known_at_issue``: the window ends with D-2 and covers the calendar days D-1-window_days to D-2.
    Index: slot (0 to 95), columns: the quantiles. Every quarter hour counts once, so on the switch to
    winter time both passes of the repeated hour enter (unlike ``same_slot_pairs``, which averages them).
    """
    if window_days < 1:
        raise ValueError(f"window_days must be at least 1, got {window_days}")
    known = known_at_issue(frame, day, time_column)
    first = day - dt.timedelta(days=1 + window_days)
    known = known[known[time_column].dt.date >= first]
    if known.empty:
        raise ValueError(f"no values known for {day} in the {window_days} days up to D-2")
    return known.groupby(clock_slot(known[time_column]))[value].quantile(list(quantiles)).unstack()


def ks_distance(a: pd.Series, b: pd.Series) -> float:
    """Kolmogorov-Smirnov distance: largest gap between the two empirical distribution functions (0 to 1).

    No p-value on purpose: neighbouring quarter hours are strongly dependent, a standard test would be far
    too confident. Missing values are dropped; an empty sample is an error.
    """
    x, y = (np.sort(v[~np.isnan(v)]) for v in (np.asarray(a, dtype=float), np.asarray(b, dtype=float)))
    if len(x) == 0 or len(y) == 0:
        raise ValueError("both samples need at least one value")
    grid = np.concatenate([x, y])
    cdf_x = np.searchsorted(x, grid, side="right") / len(x)
    cdf_y = np.searchsorted(y, grid, side="right") / len(y)
    return float(np.max(np.abs(cdf_x - cdf_y)))
