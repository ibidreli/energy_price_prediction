import datetime as dt
import io
import json

import pandas as pd
import pytest

from energy_price import fetch_weather as fw

SITES = pd.DataFrame({"site": ["A", "B"], "lat": [47.0, 46.5], "lon": [8.0, 7.0]})


def test_run_for_delivery_day_is_d_minus_2_at_18_utc():
    assert fw.run_for_delivery_day(dt.date(2026, 1, 16)) == pd.Timestamp("2026-01-14 18:00", tz="UTC")


@pytest.mark.parametrize(
    ("day", "issue"),
    [
        (dt.date(2026, 3, 29), "2026-03-28 11:00+01:00"),  # issued before the switch to summer time
        (dt.date(2026, 3, 30), "2026-03-29 11:00+02:00"),
        (dt.date(2026, 10, 26), "2026-10-25 11:00+01:00"),
    ],
)
def test_issue_time_is_d_minus_1_at_11_local(day, issue):
    assert fw.issue_time(day) == pd.Timestamp(issue)


def test_every_run_of_2026_is_available_at_least_9_hours_before_issue():
    days = fw.delivery_days(dt.date(2026, 1, 1), dt.date(2026, 12, 31))
    margins = [fw.issue_time(d) - (fw.run_for_delivery_day(d) + fw.AVAILABLE_AFTER) for d in days]

    assert all(fw.run_is_available_at_issue(d) for d in days)
    assert min(margins) >= pd.Timedelta(hours=9)


def test_request_url_asks_for_the_run_all_sites_and_variables():
    url = fw.request_url(pd.Timestamp("2026-01-14 18:00", tz="UTC"), SITES)

    assert "run=2026-01-14T18:00" in url
    assert "latitude=47.0000,46.5000" in url and "longitude=8.0000,7.0000" in url
    assert f"hourly={','.join(fw.VARIABLES)}" in url
    assert "models=ecmwf_ifs" in url and "timezone=UTC" in url


def _location(lat: float) -> dict:
    return {"latitude": lat, "longitude": 8.0, "hourly": {"time": ["2026-01-14T18:00"], **{v: [1.0] for v in fw.VARIABLES}}}


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_fetch_run_retries_after_a_truncated_response(monkeypatch):
    good = json.dumps([_location(47.0), _location(46.5)]).encode()
    responses = [FakeResponse(good[:50]), FakeResponse(good)]
    monkeypatch.setattr(fw.urllib.request, "urlopen", lambda url, timeout: responses.pop(0))
    monkeypatch.setattr(fw.time, "sleep", lambda s: None)

    record = fw.fetch_run(pd.Timestamp("2026-01-14 18:00", tz="UTC"), SITES)

    assert record["sites"] == ["A", "B"]
    assert record["run_utc"] == "2026-01-14T18:00:00+00:00"
    assert not responses


def test_fetch_run_rejects_a_response_for_fewer_sites(monkeypatch):
    body = json.dumps([_location(47.0)]).encode()
    monkeypatch.setattr(fw.urllib.request, "urlopen", lambda url, timeout: FakeResponse(body))

    with pytest.raises(fw.WeatherFetchError, match="1 locations returned, 2 requested"):
        fw.fetch_run(pd.Timestamp("2026-01-14 18:00", tz="UTC"), SITES)


def test_fetch_run_rejects_an_api_error(monkeypatch):
    body = json.dumps({"error": True, "reason": "run not available"}).encode()
    monkeypatch.setattr(fw.urllib.request, "urlopen", lambda url, timeout: FakeResponse(body))

    with pytest.raises(fw.WeatherFetchError, match="run not available"):
        fw.fetch_run(pd.Timestamp("2026-01-14 18:00", tz="UTC"), SITES)
