import csv
import json
import os
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone

import pytest

from energy_price.data.energy_charts_json import parse
from energy_price.data.entsoe_xml import EntsoeResponseError, data_checksum, is_complete, local_day_bounds
from energy_price.data.fetch_energy_charts import (
    PAUSE,
    SERIES,
    fetch,
    history_requests,
    live_requests,
    run,
    url,
)
from energy_price.paths import ROOT
from entsoe_fixtures import ec_document

HOUR, QUARTER = timedelta(hours=1), timedelta(minutes=15)
DAY = date(2026, 11, 25)
LOAD = SERIES[0]


def day_doc(day: date = DAY, step: timedelta = HOUR, **kwargs) -> bytes:
    return ec_document(*local_day_bounds(day), step, **kwargs)


def days_doc(first: date, last: date, step: timedelta = HOUR, **kwargs) -> bytes:
    return ec_document(local_day_bounds(first)[0], local_day_bounds(last)[1], step, **kwargs)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class Clock:
    def __init__(self, start: datetime):
        self.now = start

    def __call__(self) -> datetime:
        self.now += timedelta(seconds=1)
        return self.now


def run_once(tmp_path, responses, requests=None, start=datetime(2026, 11, 24, 9, 37, tzinfo=timezone.utc)):
    calls, sleeps = [], []

    def get(address):
        calls.append(address)
        response = responses[len(calls) - 1]
        if isinstance(response, Exception):
            raise response
        return response

    requests = requests or live_requests(date(2026, 11, 24), [LOAD])
    rows = run(requests, tmp_path, "live", get=get, now=Clock(start), sleep=sleeps.append)
    return rows, calls, sleeps


@pytest.mark.parametrize(("step", "count"), [(HOUR, 24), (QUARTER, 96)])
def test_a_whole_day_gives_one_point_per_row(step, count):
    response = parse(day_doc(step=step, value=float), "load")
    assert len(response.points) == count
    assert response.points[0].start_utc == local_day_bounds(DAY)[0]
    assert response.points[-1].end_utc == local_day_bounds(DAY)[1]
    assert response.groups == ["load"] and is_complete(response.points, DAY)


@pytest.mark.parametrize(("day", "rows"), [(date(2026, 3, 29), 23), (date(2026, 10, 25), 25)])
def test_hourly_rows_cover_the_switch_days(day, rows):
    response = parse(day_doc(day), "load")
    assert len(response.points) == rows and is_complete(response.points, day)


def test_each_row_lasts_until_its_neighbour_across_the_quarter_hour_switch():
    # The header says PT1H, as Energy-Charts does for a Swiss price response across 2026-11-04.
    hourly = ec_document(*local_day_bounds(date(2026, 11, 3)), HOUR, series_id="day_ahead_price")
    quarter = ec_document(*local_day_bounds(date(2026, 11, 4)), QUARTER, series_id="day_ahead_price")
    first, second = json.loads(hourly), json.loads(quarter)
    first["data"] += second["data"]
    response = parse(json.dumps(first).encode(), "day_ahead_price")
    by_time = sorted(response.points, key=lambda p: p.start_utc)
    assert [p.resolution for p in by_time] == ["PT60M"] * 24 + ["PT15M"] * 96
    assert is_complete(response.points, date(2026, 11, 3)) and is_complete(response.points, date(2026, 11, 4))


def test_null_values_and_missing_rows_stay_gaps():
    with_null = parse(day_doc(step=QUARTER, value=lambda i: None if i == 40 else 1.0), "load")
    assert len(with_null.points) == 95 and not is_complete(with_null.points, DAY)

    doc = json.loads(day_doc(step=QUARTER))
    del doc["data"][40]
    missing = parse(json.dumps(doc).encode(), "load")
    assert len(missing.points) == 95 and not is_complete(missing.points, DAY)
    assert all(p.end_utc - p.start_utc == QUARTER for p in missing.points)


def test_a_missing_series_gives_no_points():
    assert parse(day_doc(series_id="solar"), "load").points == ()


@pytest.mark.parametrize("content", [b"no content available", b'{"unix_seconds": [1], "price": [1]}', b"[]"])
def test_responses_that_are_not_v2_json_are_rejected(content):
    with pytest.raises(EntsoeResponseError):
        parse(content, "load")


def test_checksum_ignores_the_generation_time():
    first = parse(day_doc(generated="2026-11-24T08:00:00+01:00"), "load")
    second = parse(day_doc(generated="2026-11-24T10:37:00+01:00"), "load")
    assert data_checksum(first.points) == data_checksum(second.points)


def test_fetch_module_needs_only_the_standard_library():
    code = ("import sys, energy_price.data.fetch_energy_charts; "
            "bad = [m for m in ('pandas', 'numpy', 'pyarrow') if m in sys.modules]; sys.exit(', '.join(bad) or None)")
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_series_cover_the_issue_except_the_week_ahead_forecast():
    assert {(s.name, s.zone) for s in SERIES} == {
        ("load_day_ahead", "CH"), ("solar_day_ahead", "CH"), ("wind_day_ahead", "CH"), ("load_actual", "CH"),
        *(("price_day_ahead", zone) for zone in ["CH", "DE_LU", "FR", "IT_NORTH", "AT"]),
    }


def test_a_live_run_asks_once_per_series_for_d_minus_2_to_d():
    requests = live_requests(date(2026, 11, 24))
    assert len(requests) == len(SERIES)
    assert all(r.days == (date(2026, 11, 23), date(2026, 11, 24), date(2026, 11, 25)) for r in requests)
    assert url(requests[0]) == ("https://api.energy-charts.info/v2/public_power_forecast?country=ch&"
                                "forecast_type=day-ahead&production_type=load&start=2026-11-23&end=2026-11-25")


def test_price_urls_use_the_bidding_zone_names_of_energy_charts():
    request = next(r for r in live_requests(date(2026, 11, 24)) if r.series.zone == "IT_NORTH")
    assert url(request) == "https://api.energy-charts.info/v2/price?bzn=IT-North&start=2026-11-23&end=2026-11-25"


def test_history_comes_in_local_calendar_months_with_inclusive_ends():
    requests = [r for r in history_requests(date(2026, 1, 1), date(2026, 3, 15)) if r.series == LOAD]
    assert [url(r).split("&start=")[1] for r in requests] == [
        "2026-01-01&end=2026-01-31", "2026-02-01&end=2026-02-28", "2026-03-01&end=2026-03-15"]


def test_one_snapshot_per_response_and_one_log_row_per_day(tmp_path):
    rows, calls, _ = run_once(tmp_path, [(200, days_doc(date(2026, 11, 23), DAY))])
    manifest = read_csv(tmp_path / "manifest.csv")
    assert len(calls) == 1 and len(manifest) == 1
    assert manifest[0]["file"] == "load_day_ahead_CH_2026-11-23_2026-11-25_2026-11-24T09-37-01Z.json"
    assert manifest[0]["provider"] == "energy_charts" and manifest[0]["delivery_days"].split() == [
        "2026-11-23", "2026-11-24", "2026-11-25"]
    assert [(r["delivery_day"], r["complete"], r["points"]) for r in rows] == [
        ("2026-11-23", True, 24), ("2026-11-24", True, 24), ("2026-11-25", True, 24)]


def test_a_forecast_not_yet_published_shows_as_incomplete_day(tmp_path):
    rows, _, _ = run_once(tmp_path, [(200, days_doc(date(2026, 11, 23), date(2026, 11, 24)))])
    assert [r["complete"] for r in rows] == [True, True, False]


def test_an_unchanged_response_is_logged_but_not_stored_again(tmp_path):
    run_once(tmp_path, [(200, days_doc(date(2026, 11, 23), DAY, generated="2026-11-24T08:17:00+01:00"))])
    rows, _, _ = run_once(tmp_path, [(200, days_doc(date(2026, 11, 23), DAY, generated="2026-11-24T10:37:00+01:00"))])
    assert not any(r["changed"] for r in rows)
    assert len(read_csv(tmp_path / "manifest.csv")) == 1 and len(read_csv(tmp_path / "fetch_log.csv")) == 6


def test_no_content_is_no_data_without_snapshot(tmp_path):
    rows, _, _ = run_once(tmp_path, [(404, b"no content available")])
    assert {r["status"] for r in rows} == {"no_data"} and not (tmp_path / "manifest.csv").exists()


def test_rate_limits_are_retried_after_one_and_two_minutes(tmp_path):
    rows, calls, sleeps = run_once(tmp_path, [(429, b"Too Many Requests"), (200, days_doc(date(2026, 11, 23), DAY))])
    assert rows[0]["status"] == "ok" and len(calls) == 2 and sleeps == [60.0]

    rows, calls, sleeps = run_once(tmp_path, [(429, b"Too Many Requests")] * 3)
    assert rows[0]["status"] == "http_error" and sleeps == [60.0, 120.0]
    assert "429" in rows[0]["message"]


def test_requests_are_spaced(tmp_path):
    requests = live_requests(date(2026, 11, 24), SERIES[:3])
    doc = days_doc(date(2026, 11, 23), DAY)
    _, calls, sleeps = run_once(tmp_path, [(200, doc)] * 3, requests=requests)
    assert len(calls) == 3 and sleeps == [PAUSE, PAUSE]


def test_retry_returns_ok_after_a_network_error():
    responses = iter([OSError("reset"), (200, day_doc())])

    def get(address):
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        return response

    request = live_requests(date(2026, 11, 24), [LOAD])[0]
    assert fetch(request, get=get, sleep=lambda s: None).status == "ok"
