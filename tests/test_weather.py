import datetime as dt
import json
from pathlib import Path

import pandas as pd
import pytest

from energy_price.fetch_weather import VARIABLES, run_for_delivery_day
from energy_price.weather import WeatherFormatError, load_weather, missing_values, read_run_file


def write_run(directory: Path, run: pd.Timestamp, sites: list[str], hours: int = 96, gap: str | None = None) -> Path:
    """A run file as written by fetch_weather: hourly UTC values from the run time on."""
    times = pd.date_range(run, periods=hours, freq="h")
    locations = []
    for i, _ in enumerate(sites):
        hourly = {"time": [t.strftime("%Y-%m-%dT%H:%M") for t in times]}
        hourly |= {v: [float(i)] * hours for v in VARIABLES}
        if gap:
            hourly[gap] = [None] * hours
        locations.append({"latitude": 47.0, "longitude": 8.0, "hourly": hourly})
    path = directory / f"run_{run.strftime('%Y-%m-%dT%H')}.json"
    path.write_text(json.dumps({"run_utc": run.isoformat(), "sites": sites, "locations": locations}))
    return path


def test_read_run_file_keeps_the_delivery_day_including_the_hour_ending_at_midnight(tmp_path):
    f = write_run(tmp_path, run_for_delivery_day(dt.date(2026, 1, 16)), ["A", "B"])

    df = read_run_file(f)
    a = df[df["site"] == "A"]

    assert len(a) == 25
    assert a["valid_time_local"].iloc[0] == pd.Timestamp("2026-01-16 00:00", tz="Europe/Zurich")
    assert a["valid_time_local"].iloc[-1] == pd.Timestamp("2026-01-17 00:00", tz="Europe/Zurich")
    assert (df["delivery_day"] == pd.Timestamp("2026-01-16")).all()


def test_read_run_file_handles_the_short_day_in_spring(tmp_path):
    f = write_run(tmp_path, run_for_delivery_day(dt.date(2026, 3, 29)), ["A"])

    assert len(read_run_file(f)) == 24


def test_read_run_file_handles_the_long_day_in_autumn(tmp_path):
    f = write_run(tmp_path, run_for_delivery_day(dt.date(2026, 10, 25)), ["A"])

    df = read_run_file(f)

    assert len(df) == 26
    assert df["valid_time_utc"].is_unique
    assert df["valid_time_local"].dt.strftime("%H:%M").tolist().count("02:00") == 2


def test_read_run_file_rejects_a_run_that_is_not_used_for_any_delivery_day(tmp_path):
    f = write_run(tmp_path, pd.Timestamp("2026-01-14 12:00", tz="UTC"), ["A"])

    with pytest.raises(WeatherFormatError, match="not the run used"):
        read_run_file(f)


def test_read_run_file_rejects_a_run_that_does_not_reach_the_end_of_the_day(tmp_path):
    f = write_run(tmp_path, run_for_delivery_day(dt.date(2026, 1, 16)), ["A"], hours=40)

    with pytest.raises(WeatherFormatError, match="hours for 2026-01-16"):
        read_run_file(f)


def test_missing_values_are_kept_and_reported(tmp_path):
    write_run(tmp_path, run_for_delivery_day(dt.date(2026, 6, 24)), ["A", "B"], gap="shortwave_radiation")

    weather = load_weather(tmp_path)
    report = missing_values(weather)

    assert weather["shortwave_radiation"].isna().all()
    assert report.to_dict("records") == [{"delivery_day": pd.Timestamp("2026-06-24"), "variable": "shortwave_radiation", "missing": 50}]


def test_snow_depth_is_not_in_the_processed_table(tmp_path):
    write_run(tmp_path, run_for_delivery_day(dt.date(2026, 1, 16)), ["A"])

    assert "snow_depth" not in load_weather(tmp_path).columns
