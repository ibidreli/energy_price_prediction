"""Download ENTSO-E load, generation forecasts and day-ahead prices and keep every distinct version.

Issue #37. Source: ENTSO-E Transparency Platform, REST API, bidding zone Switzerland and the day-ahead
prices of DE-LU, FR, IT-North and AT. The token comes from the environment variable
``ENTSOE_API_TOKEN`` (locally also from ``.env``, on GitHub Actions from the repository secret). It is
never written anywhere: logged URLs leave it out and error messages have it masked.

ENTSO-E does not say when a value was published, so the time of our own download is the evidence
(docs/data.md, section 7). A live run asks, for every series, for the local day before, of and after
the fetch day. For a forecast issued at D-1 these are D-2 (actual load), D-1 (day-ahead prices) and
D (forecasts). A response is stored as a snapshot when its values differ from the last snapshot of
the same series and day. ``createdDateTime`` and the document id change with every request, so the
comparison uses a checksum of the parsed values (``data_sha256``), not of the bytes.

``fetch_log.csv`` records every request, also failed and unchanged ones, with ``status`` (ok, no_data,
http_error, network_error, parse_error, no_token) and ``complete``: whether every quarter hour of the
requested period is present. ``make entsoe-availability`` evaluates it. The run exits with code 1 if
any request failed, so the scheduled job turns red, but only after the log was written.

Only the standard library is used, so the GitHub Actions job runs without installing the project.

Usage:
``python -m energy_price.data.fetch_entsoe --out data/entsoe/snapshots``  (live run, scheduled)
``python -m energy_price.data.fetch_entsoe --out data/entsoe/backfill --history-from 2026-01-01``
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from energy_price.data import entsoe_xml
from energy_price.data.entsoe_xml import TZ, EntsoeResponseError, Response, local_day_bounds, local_midnight

API_URL = "https://web-api.tp.entsoe.eu/api"
TOKEN_VARIABLE = "ENTSOE_API_TOKEN"
USER_AGENT = "energy-price-research (python urllib)"
ZONES = {
    "CH": "10YCH-SWISSGRIDZ",
    "DE_LU": "10Y1001A1001A82H",
    "FR": "10YFR-RTE------C",
    "IT_NORTH": "10Y1001A1001A73I",
    "AT": "10YAT-APG------L",
}
# Local days relative to the fetch day T. For the forecast of D = T+1 they hold D-2, D-1 and D.
LIVE_DAY_OFFSETS = (-1, 0, 1)
ERRORS = {"http_error", "network_error", "parse_error", "no_token"}
RETRY_STATUS = {429, 500, 502, 503, 504}
MANIFEST_COLUMNS = [
    "file", "source", "series", "zone", "delivery_day", "period_start_utc", "period_end_utc", "fetched_at_utc",
    "url", "sha256", "data_sha256", "points", "groups", "first_timestamp_utc", "last_timestamp_utc",
]
FETCH_LOG_COLUMNS = [
    "fetched_at_utc", "fetched_at_local", "source", "series", "zone", "delivery_day", "period_start_utc",
    "period_end_utc", "status", "http_status", "points", "groups", "complete", "data_sha256", "changed",
    "snapshot", "message",
]


@dataclass(frozen=True)
class Series:
    name: str
    zone: str
    params: tuple[tuple[str, str], ...]


SERIES = (
    Series("load_day_ahead", "CH", (("documentType", "A65"), ("processType", "A01"),
                                    ("outBiddingZone_Domain", ZONES["CH"]))),
    Series("load_week_ahead", "CH", (("documentType", "A65"), ("processType", "A31"),
                                     ("outBiddingZone_Domain", ZONES["CH"]))),
    Series("load_actual", "CH", (("documentType", "A65"), ("processType", "A16"),
                                 ("outBiddingZone_Domain", ZONES["CH"]))),
    Series("wind_solar_day_ahead", "CH", (("documentType", "A69"), ("processType", "A01"),
                                          ("in_Domain", ZONES["CH"]))),
    *(Series("price_day_ahead", zone, (("documentType", "A44"), ("in_Domain", eic), ("out_Domain", eic)))
      for zone, eic in ZONES.items()),
)


@dataclass(frozen=True)
class Request:
    series: Series
    start: datetime
    end: datetime
    delivery_day: date | None = None  # the local day of a live request, None for a history block


@dataclass(frozen=True)
class Result:
    status: str
    http_status: int | None = None
    content: bytes = b""
    response: Response | None = None
    message: str = ""


def live_requests(fetch_day: date, series: Iterable[Series] = SERIES) -> list[Request]:
    """One request per series and local day ``fetch_day`` + ``LIVE_DAY_OFFSETS``."""
    out = []
    for item in series:
        for offset in LIVE_DAY_OFFSETS:
            day = fetch_day + timedelta(days=offset)
            out.append(Request(item, *local_day_bounds(day), delivery_day=day))
    return out


def history_requests(first: date, last: date, series: Iterable[Series] = SERIES) -> list[Request]:
    """One request per series and local calendar month, from ``first`` to ``last`` inclusive."""
    blocks, start = [], first
    while start <= last:
        month_end = date(start.year + start.month // 12, start.month % 12 + 1, 1)
        end = min(month_end, last + timedelta(days=1))
        blocks.append((local_midnight(start), local_midnight(end)))
        start = end
    return [Request(item, start, end) for item in series for start, end in blocks]


def period_params(start: datetime, end: datetime) -> dict[str, str]:
    return {"periodStart": start.strftime("%Y%m%d%H%M"), "periodEnd": end.strftime("%Y%m%d%H%M")}


def request_params(request: Request) -> dict[str, str]:
    return {**dict(request.series.params), **period_params(request.start, request.end)}


def public_url(params: Mapping[str, str]) -> str:
    """The request URL without the token, as it is written to the manifest."""
    return f"{API_URL}?{urllib.parse.urlencode(params)}"


def mask(message: str, token: str | None) -> str:
    """One line of at most 300 characters, with the token replaced."""
    if token:
        message = message.replace(token, "***")
    return " ".join(message.split())[:300]


def read_token(environ: Mapping[str, str] = os.environ, dotenv: Path = Path(".env")) -> str | None:
    """``ENTSOE_API_TOKEN`` from the environment, otherwise from a local ``.env`` (ignored by git)."""
    token = environ.get(TOKEN_VARIABLE, "").strip()
    if token:
        return token
    if dotenv.is_file():
        for line in dotenv.read_text(encoding="utf-8").splitlines():
            key, sep, value = line.partition("=")
            if sep and key.strip() == TOKEN_VARIABLE:
                return value.strip().strip("\"'") or None
    return None


def download(params: Mapping[str, str], token: str, timeout: float = 60) -> tuple[int, bytes]:
    """HTTP status and body. An HTTP error is returned, not raised: its body says what went wrong."""
    query = urllib.parse.urlencode({"securityToken": token, **params})
    request = urllib.request.Request(f"{API_URL}?{query}", headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def classify(http_status: int, content: bytes, token: str | None) -> Result:
    try:
        parsed = entsoe_xml.parse(content)
    except EntsoeResponseError as exc:
        if http_status == 200:
            return Result("parse_error", http_status, content, message=mask(str(exc), token))
        body = content[:200].decode("utf-8", "replace")
        return Result("http_error", http_status, content, message=mask(f"HTTP {http_status}: {exc}; {body}", token))
    if parsed.no_data:
        return Result("no_data", http_status, content, parsed, mask(parsed.reason, token))
    if http_status != 200:
        return Result("http_error", http_status, content, message=f"HTTP {http_status} with data")
    return Result("ok", http_status, content, parsed)


def fetch(params: Mapping[str, str], token: str, get: Callable = download, retries: int = 1,
          wait: float = 10.0, sleep: Callable[[float], None] = time.sleep) -> Result:
    """Download and classify; retry once after network errors, rate limits and server errors."""
    result = Result("network_error")
    for attempt in range(retries + 1):
        try:
            http_status, content = get(params, token)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            result = Result("network_error", message=mask(f"{type(exc).__name__}: {exc}", token))
        else:
            result = classify(http_status, content, token)
            if result.status != "http_error" or http_status not in RETRY_STATUS:
                return result
        if attempt < retries:
            sleep(wait)
    return result


def ceil_second(stamp: datetime) -> datetime:
    """Round up: the stored fetch time must never lie before the moment the data was actually there."""
    whole = stamp.replace(microsecond=0)
    return whole + timedelta(seconds=1) if stamp.microsecond else whole


def latest_checksums(manifest: Path) -> dict[tuple[str, str, str, str], str]:
    """``data_sha256`` of the newest snapshot per series, zone and requested period."""
    if not manifest.exists():
        return {}
    with manifest.open(newline="", encoding="utf-8") as f:
        return {(r["series"], r["zone"], r["period_start_utc"], r["period_end_utc"]): r["data_sha256"]
                for r in csv.DictReader(f)}


def append_row(path: Path, columns: list[str], row: dict) -> None:
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def snapshot_name(request: Request, fetched: datetime) -> str:
    stamp = fetched.strftime("%Y-%m-%dT%H-%M-%SZ")
    if request.delivery_day is not None:
        period = request.delivery_day.isoformat()
    else:
        period = f"{request.start.astimezone(TZ).date()}_{request.end.astimezone(TZ).date()}"
    return f"{request.series.name}_{request.series.zone}_{period}_{stamp}.xml"


def run(requests: list[Request], out: Path, source: str, token: str | None,
        get: Callable = download, now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        pause: float = 0.2, sleep: Callable[[float], None] = time.sleep) -> list[dict]:
    """Fetch every request, store changed responses and log every request. Returns the log rows."""
    out.mkdir(parents=True, exist_ok=True)
    manifest, log = out / "manifest.csv", out / "fetch_log.csv"
    latest = latest_checksums(manifest)
    rows = []
    for i, request in enumerate(requests):
        if i and pause and token:
            sleep(pause)  # well below the limit of 400 requests per minute
        params = request_params(request)
        result = fetch(params, token, get=get, sleep=sleep) if token else Result(
            "no_token", message=f"{TOKEN_VARIABLE} is not set")
        fetched = ceil_second(now())
        key = (request.series.name, request.series.zone, request.start.isoformat(), request.end.isoformat())
        points = result.response.points if result.response is not None else ()
        data_sha = entsoe_xml.data_checksum(points) if result.status == "ok" else ""
        changed = result.status == "ok" and latest.get(key) != data_sha
        name = ""
        if changed:
            name = snapshot_name(request, fetched)
            (out / name).write_bytes(result.content)
            append_row(manifest, MANIFEST_COLUMNS, {
                "file": name, "source": source, "series": request.series.name, "zone": request.series.zone,
                "delivery_day": request.delivery_day.isoformat() if request.delivery_day else "",
                "period_start_utc": key[2], "period_end_utc": key[3], "fetched_at_utc": fetched.isoformat(),
                "url": public_url(params), "sha256": hashlib.sha256(result.content).hexdigest(),
                "data_sha256": data_sha, "points": len(points), "groups": " ".join(result.response.groups),
                "first_timestamp_utc": min(p.start_utc for p in points).isoformat() if points else "",
                "last_timestamp_utc": max(p.start_utc for p in points).isoformat() if points else "",
            })
            latest[key] = data_sha
        complete = entsoe_xml.covers(points, request.start, request.end) if result.status == "ok" else False
        row = {
            "fetched_at_utc": fetched.isoformat(), "fetched_at_local": fetched.astimezone(TZ).isoformat(),
            "source": source, "series": request.series.name, "zone": request.series.zone,
            "delivery_day": request.delivery_day.isoformat() if request.delivery_day else "",
            "period_start_utc": key[2], "period_end_utc": key[3], "status": result.status,
            "http_status": result.http_status or "", "points": len(points),
            "groups": " ".join(result.response.groups) if result.response else "", "complete": complete,
            "data_sha256": data_sha, "changed": changed, "snapshot": name, "message": result.message,
        }
        append_row(log, FETCH_LOG_COLUMNS, row)
        rows.append(row)
        period = request.delivery_day or f"{request.start.astimezone(TZ).date()}..{request.end.astimezone(TZ).date()}"
        print(f"{request.series.name} {request.series.zone} {period}: {result.status}, {len(points)} points"
              f"{', complete' if complete else ''}{', new snapshot' if changed else ''}"
              f"{f' ({result.message})' if result.status in ERRORS else ''}")
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="snapshot directory")
    parser.add_argument("--history-from", type=date.fromisoformat,
                        help="download the history from this local day on, in monthly blocks (backfill)")
    parser.add_argument("--history-to", type=date.fromisoformat,
                        help="last local day of the history (default: tomorrow)")
    args = parser.parse_args(argv)

    today = datetime.now(TZ).date()
    if args.history_from:
        requests = history_requests(args.history_from, args.history_to or today + timedelta(days=1))
        source = "backfill"
    else:
        requests, source = live_requests(today), "live"
    rows = run(requests, args.out, source, read_token())
    failed = [row for row in rows if row["status"] in ERRORS]
    print(f"{len(rows)} requests, {len(failed)} failed, "
          f"{sum(row['changed'] for row in rows)} new snapshots in {args.out}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
