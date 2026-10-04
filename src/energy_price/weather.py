"""Turn the stored Open-Meteo run files into one table of weather forecasts per delivery day.

Each run file holds the forecast of one model run for all sites. For delivery day D we keep the
hourly values from D 00:00 to D+1 00:00 Swiss time (inclusive), because radiation, sunshine and
precipitation are averages or sums over the preceding hour: the value at D+1 00:00 describes the
last hour of D. Temperature, cloud cover, humidity and wind are instantaneous values.

``snow_depth`` is downloaded but not used: the archive lacks it for most 18 UTC runs.
Single missing values in the archive stay NaN and are reported by ``missing_values``;
structural problems (wrong run, wrong number of hours) raise ``WeatherFormatError``.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd

from energy_price.fetch_weather import RUN_DAYS_BEFORE_DELIVERY, VARIABLES, run_for_delivery_day

TZ = "Europe/Zurich"
UNRELIABLE_VARIABLES = ["snow_depth"]
WEATHER_VARIABLES = [v for v in VARIABLES if v not in UNRELIABLE_VARIABLES]
OUTPUT_COLUMNS = ["delivery_day", "run_utc", "site", "valid_time_utc", "valid_time_local", *WEATHER_VARIABLES]


class WeatherFormatError(ValueError):
    """A stored run file does not match the expected structure or misses values for the delivery day."""


def read_run_file(path: str | Path) -> pd.DataFrame:
    """Hourly forecasts of one run for its delivery day, all sites, in long format."""
    path = Path(path)
    record = json.loads(path.read_text())
    run = pd.Timestamp(record["run_utc"])
    day = (run + pd.Timedelta(days=RUN_DAYS_BEFORE_DELIVERY)).date()
    if run_for_delivery_day(day) != run:
        raise WeatherFormatError(f"{path.name}: run {run} is not the run used for any delivery day")

    start = pd.Timestamp(day).tz_localize(TZ)
    end = pd.Timestamp(day + dt.timedelta(days=1)).tz_localize(TZ)

    frames = []
    for site, loc in zip(record["sites"], record["locations"], strict=True):
        hourly = pd.DataFrame(loc["hourly"])
        hourly["valid_time_utc"] = pd.to_datetime(hourly.pop("time"), format="%Y-%m-%dT%H:%M").dt.tz_localize("UTC")
        hourly = hourly[(hourly["valid_time_utc"] >= start) & (hourly["valid_time_utc"] <= end)]
        expected = int((end - start) / pd.Timedelta(hours=1)) + 1
        if len(hourly) != expected:
            raise WeatherFormatError(f"{path.name}, {site}: {len(hourly)} hours for {day}, expected {expected}")
        hourly["site"] = site
        frames.append(hourly)

    out = pd.concat(frames, ignore_index=True)
    out["delivery_day"] = pd.Timestamp(day)
    out["run_utc"] = run
    out["valid_time_local"] = out["valid_time_utc"].dt.tz_convert(TZ)
    return out[OUTPUT_COLUMNS]


def load_weather(run_dir: str | Path) -> pd.DataFrame:
    """All run files below ``run_dir`` in one table, sorted by delivery day, site and time."""
    files = sorted(Path(run_dir).glob("run_*.json"))
    if not files:
        raise FileNotFoundError(f"no run files found in {run_dir}")
    weather = pd.concat([read_run_file(f) for f in files], ignore_index=True)
    duplicated = weather.duplicated(["delivery_day", "site", "valid_time_utc"])
    if duplicated.any():
        raise WeatherFormatError(f"{duplicated.sum()} duplicate rows across run files")
    return weather.sort_values(["delivery_day", "site", "valid_time_utc"], ignore_index=True)


def missing_values(weather: pd.DataFrame) -> pd.DataFrame:
    """Missing hourly values per delivery day and variable, summed over all sites."""
    counts = weather.groupby("delivery_day")[WEATHER_VARIABLES].agg(lambda s: s.isna().sum())
    long = counts.stack().rename("missing").reset_index().rename(columns={"level_1": "variable"})
    return long[long["missing"] > 0].reset_index(drop=True)
