import csv
import os
import subprocess
import sys
import urllib.error
from datetime import date, datetime, timedelta, timezone

import pytest

from energy_price.data import fetch_entsoe
from energy_price.data.entsoe_xml import local_day_bounds
from energy_price.data.fetch_entsoe import (
    SERIES,
    Request,
    ceil_second,
    fetch,
    history_requests,
    live_requests,
    main,
    public_url,
    read_token,
    request_params,
    run,
)
from energy_price.paths import ROOT
from entsoe_fixtures import acknowledgement, gl_document, period, time_series

DAY = date(2026, 11, 19)
LOAD = SERIES[0]
TOKEN = "SECRET-TOKEN-123"


def load_response(day: date = DAY, value: float = 7000.0, created: str = "2026-11-18T09:00:00Z") -> bytes:
    start, end = local_day_bounds(day)
    hours = int((end - start) / timedelta(hours=1))
    return gl_document([time_series([period(start, end, "PT60M", [value] * hours)])], created=created)


class Clock:
    def __init__(self, start: datetime):
        self.now = start

    def __call__(self) -> datetime:
        self.now += timedelta(seconds=1)
        return self.now


def run_once(tmp_path, responses, requests=None, token=TOKEN, start=datetime(2026, 11, 18, 9, 37, tzinfo=timezone.utc)):
    """Run with a fake download that returns ``responses`` in order (an exception is raised instead)."""
    calls = []

    def get(params, token):
        calls.append(params)
        response = responses[len(calls) - 1]
        if isinstance(response, Exception):
            raise response
        return response

    requests = requests or [Request(LOAD, *local_day_bounds(DAY), delivery_day=DAY)]
    rows = run(requests, tmp_path, "live", token, get=get, now=Clock(start), sleep=lambda s: None)
    return rows, calls


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_fetch_module_needs_only_the_standard_library():
    code = ("import sys, energy_price.data.fetch_entsoe; "
            "bad = [m for m in ('pandas', 'numpy', 'pyarrow') if m in sys.modules]; sys.exit(', '.join(bad) or None)")
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_a_live_run_asks_for_d_minus_2_to_d_of_the_next_forecast():
    requests = live_requests(date(2026, 11, 18))
    assert len(requests) == len(SERIES) * 3
    days = {r.delivery_day for r in requests if r.series == LOAD}
    assert days == {date(2026, 11, 17), date(2026, 11, 18), date(2026, 11, 19)}


def test_request_periods_are_local_days_in_utc():
    request = next(r for r in live_requests(date(2026, 3, 28)) if r.delivery_day == date(2026, 3, 29))
    params = request_params(request)
    assert (params["periodStart"], params["periodEnd"]) == ("202603282300", "202603292200")


def test_series_cover_load_generation_and_the_prices_of_five_zones():
    assert {(s.name, s.zone) for s in SERIES if s.name != "price_day_ahead"} == {
        ("load_day_ahead", "CH"), ("load_week_ahead", "CH"), ("load_actual", "CH"), ("wind_solar_day_ahead", "CH")}
    assert {s.zone for s in SERIES if s.name == "price_day_ahead"} == {"CH", "DE_LU", "FR", "IT_NORTH", "AT"}
    prices = [dict(s.params) for s in SERIES if s.name == "price_day_ahead"]
    assert all(p["documentType"] == "A44" and p["in_Domain"] == p["out_Domain"] for p in prices)


def test_history_comes_in_local_calendar_months():
    requests = [r for r in history_requests(date(2026, 1, 1), date(2026, 3, 15)) if r.series == LOAD]
    assert [(r.start.isoformat(), r.end.isoformat()) for r in requests] == [
        ("2025-12-31T23:00:00+00:00", "2026-01-31T23:00:00+00:00"),
        ("2026-01-31T23:00:00+00:00", "2026-02-28T23:00:00+00:00"),
        ("2026-02-28T23:00:00+00:00", "2026-03-15T23:00:00+00:00"),
    ]
    assert all(r.delivery_day is None for r in requests)


def test_a_new_response_is_stored_with_manifest_and_log(tmp_path):
    rows, _ = run_once(tmp_path, [(200, load_response())])
    manifest = read_csv(tmp_path / "manifest.csv")
    log = read_csv(tmp_path / "fetch_log.csv")
    assert len(manifest) == 1 and len(log) == 1
    assert (tmp_path / manifest[0]["file"]).read_bytes() == load_response()
    assert manifest[0]["file"] == "load_day_ahead_CH_2026-11-19_2026-11-18T09-37-01Z.xml"
    assert manifest[0]["delivery_day"] == "2026-11-19" and manifest[0]["points"] == "24"
    assert log[0]["status"] == "ok" and log[0]["complete"] == "True" and log[0]["changed"] == "True"
    assert log[0]["fetched_at_local"] == "2026-11-18T10:37:01+01:00"


def test_an_unchanged_response_is_logged_but_not_stored_again(tmp_path):
    run_once(tmp_path, [(200, load_response())])
    rows, _ = run_once(tmp_path, [(200, load_response(created="2026-11-18T10:00:00Z"))])
    assert not rows[0]["changed"]
    assert len(read_csv(tmp_path / "manifest.csv")) == 1
    assert len(read_csv(tmp_path / "fetch_log.csv")) == 2
    rows, _ = run_once(tmp_path, [(200, load_response(value=7100.0))])
    assert rows[0]["changed"]
    assert len(read_csv(tmp_path / "manifest.csv")) == 2


def test_no_data_is_logged_without_snapshot(tmp_path):
    rows, _ = run_once(tmp_path, [(200, acknowledgement())])
    assert rows[0]["status"] == "no_data" and not rows[0]["complete"]
    assert not (tmp_path / "manifest.csv").exists()
    assert list(tmp_path.glob("*.xml")) == []


def test_an_incomplete_day_is_logged_as_incomplete(tmp_path):
    start, end = local_day_bounds(DAY)
    half = gl_document([time_series([period(start, start + timedelta(hours=12), "PT60M", [1.0] * 12)])])
    rows, _ = run_once(tmp_path, [(200, half)])
    assert rows[0]["status"] == "ok" and not rows[0]["complete"]


@pytest.mark.parametrize(("response", "status"), [
    ((401, b"Unauthorized"), "http_error"),
    ((400, acknowledgement(code="999", message="no data")), "no_data"),
    ((400, acknowledgement(code="400", message="bad request")), "http_error"),
    ((200, b"<html>maintenance</html>"), "parse_error"),
    (urllib.error.URLError("timed out"), "network_error"),
])
def test_failures_are_logged_with_their_status(tmp_path, response, status):
    rows, _ = run_once(tmp_path, [response, response])
    assert rows[0]["status"] == status
    assert len(read_csv(tmp_path / "fetch_log.csv")) == 1


def test_server_errors_are_retried_once():
    responses = iter([(503, b"busy"), (200, load_response())])
    result = fetch({}, TOKEN, get=lambda params, token: next(responses), sleep=lambda s: None)
    assert result.status == "ok"


def test_the_token_is_never_written(tmp_path):
    leaky = (401, f"invalid token {TOKEN}".encode())
    run_once(tmp_path, [(200, load_response())])
    run_once(tmp_path, [leaky, leaky])
    run_once(tmp_path, [urllib.error.URLError(f"failed for {TOKEN}")] * 2)
    written = b"".join(path.read_bytes() for path in tmp_path.iterdir())
    assert TOKEN.encode() not in written and b"securityToken" not in written
    assert "***" in read_csv(tmp_path / "fetch_log.csv")[1]["message"]


def test_the_public_url_has_no_token():
    url = public_url(request_params(Request(LOAD, *local_day_bounds(DAY), delivery_day=DAY)))
    assert "securityToken" not in url and "documentType=A65" in url


def test_without_token_every_request_is_logged_and_nothing_is_downloaded(tmp_path):
    rows, calls = run_once(tmp_path, [], token=None)
    assert calls == [] and rows[0]["status"] == "no_token"


def test_main_fails_after_writing_the_log(tmp_path, monkeypatch):
    monkeypatch.delenv(fetch_entsoe.TOKEN_VARIABLE, raising=False)
    monkeypatch.chdir(tmp_path)
    assert main(["--out", str(tmp_path / "snapshots")]) == 1
    log = read_csv(tmp_path / "snapshots" / "fetch_log.csv")
    assert len(log) == len(SERIES) * 3 and {row["status"] for row in log} == {"no_token"}


def test_the_token_is_read_from_the_environment_before_dotenv(tmp_path):
    dotenv = tmp_path / ".env"
    dotenv.write_text("WANDB_ENTITY=x\nENTSOE_API_TOKEN = 'from-file'\n", encoding="utf-8")
    assert read_token({}, dotenv) == "from-file"
    assert read_token({"ENTSOE_API_TOKEN": "from-env"}, dotenv) == "from-env"
    assert read_token({}, tmp_path / "missing") is None


def test_fetch_times_are_rounded_up_to_the_second():
    assert ceil_second(datetime(2026, 11, 18, 9, 59, 59, 1, tzinfo=timezone.utc)) == datetime(
        2026, 11, 18, 10, tzinfo=timezone.utc)
    exact = datetime(2026, 11, 18, 10, tzinfo=timezone.utc)
    assert ceil_second(exact) == exact
