"""Store every changed API response as a snapshot and log every request (issue #37).

Shared by the two sources of the issue #37 data: ``fetch_energy_charts`` (public, no key) and
``fetch_entsoe`` (ENTSO-E REST API, token). Each provider has its own folder below ``data/entsoe``
with ``snapshots/`` (scheduled live runs) and ``backfill/`` (one-time history).

Neither source says when a value was published, so the time of our own download is the evidence for
the availability at D-1 11:00 (docs/data.md, section 7). A response is stored when its values differ
from the last snapshot of the same series and period. The comparison uses a checksum of the parsed
values (``data_sha256``): responses carry their creation time, so the bytes change with every request.

``fetch_log.csv`` gets one row per request and local day of a live request, also for failed and
unchanged requests, with ``status`` (ok, no_data, http_error, network_error, parse_error, no_token)
and ``complete``: whether every quarter hour of the day is present.

Only the standard library is used, so the GitHub Actions job runs without installing the project.
"""

from __future__ import annotations

import csv
import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from energy_price.data.entsoe_xml import TZ, Response, covers, data_checksum, local_day_bounds

ERRORS = {"http_error", "network_error", "parse_error", "no_token"}
MANIFEST_COLUMNS = [
    "file", "provider", "source", "series", "zone", "delivery_days", "period_start_utc", "period_end_utc",
    "fetched_at_utc", "url", "sha256", "data_sha256", "points", "groups", "first_timestamp_utc", "last_timestamp_utc",
]
FETCH_LOG_COLUMNS = [
    "fetched_at_utc", "fetched_at_local", "provider", "source", "series", "zone", "delivery_day", "period_start_utc",
    "period_end_utc", "status", "http_status", "points", "groups", "complete", "data_sha256", "changed",
    "snapshot", "message",
]


@dataclass(frozen=True)
class Series:
    name: str
    zone: str
    params: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class Request:
    series: Series
    start: datetime
    end: datetime
    days: tuple[date, ...] = ()  # local days logged one by one (live run); empty: one row for the period


@dataclass(frozen=True)
class Result:
    status: str
    http_status: int | None = None
    content: bytes = b""
    response: Response | None = None
    message: str = ""


def window(series: Series, first: date, last: date) -> Request:
    """A live request for the local days ``first`` to ``last``, each logged on its own."""
    days = tuple(first + timedelta(days=i) for i in range((last - first).days + 1))
    return Request(series, local_day_bounds(first)[0], local_day_bounds(last)[1], days)


def months(first: date, last: date) -> list[tuple[datetime, datetime]]:
    """Local calendar months from ``first`` to ``last`` inclusive, as UTC bounds."""
    blocks, start = [], first
    while start <= last:
        month_end = date(start.year + start.month // 12, start.month % 12 + 1, 1)
        end = min(month_end, last + timedelta(days=1))
        blocks.append((local_day_bounds(start)[0], local_day_bounds(end)[0]))
        start = end
    return blocks


def mask(message: str, secret: str | None = None) -> str:
    """One line of at most 300 characters, with ``secret`` replaced."""
    if secret:
        message = message.replace(secret, "***")
    return " ".join(message.split())[:300]


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


def period_label(request: Request) -> str:
    first, last = request.start.astimezone(TZ).date(), (request.end - timedelta(seconds=1)).astimezone(TZ).date()
    return first.isoformat() if first == last else f"{first}_{last}"


def snapshot_name(request: Request, fetched: datetime, suffix: str) -> str:
    stamp = fetched.strftime("%Y-%m-%dT%H-%M-%SZ")
    return f"{request.series.name}_{request.series.zone}_{period_label(request)}_{stamp}{suffix}"


def run(requests: list[Request], out: Path, provider: str, source: str, fetch_one: Callable[[Request], Result],
        url_of: Callable[[Request], str], suffix: str,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        pause: float = 0.0, sleep: Callable[[float], None] = time.sleep) -> list[dict]:
    """Fetch every request, store changed responses and log every request and day. Returns the log rows."""
    out.mkdir(parents=True, exist_ok=True)
    manifest, log = out / "manifest.csv", out / "fetch_log.csv"
    latest = latest_checksums(manifest)
    rows = []
    for i, request in enumerate(requests):
        if i and pause:
            sleep(pause)
        result = fetch_one(request)
        fetched = ceil_second(now())
        key = (request.series.name, request.series.zone, request.start.isoformat(), request.end.isoformat())
        points = result.response.points if result.response is not None else ()
        groups = " ".join(result.response.groups) if result.response is not None else ""
        data_sha = data_checksum(points) if result.status == "ok" else ""
        changed = result.status == "ok" and latest.get(key) != data_sha
        name = ""
        if changed:
            name = snapshot_name(request, fetched, suffix)
            (out / name).write_bytes(result.content)
            append_row(manifest, MANIFEST_COLUMNS, {
                "file": name, "provider": provider, "source": source, "series": request.series.name,
                "zone": request.series.zone, "delivery_days": " ".join(d.isoformat() for d in request.days),
                "period_start_utc": key[2], "period_end_utc": key[3], "fetched_at_utc": fetched.isoformat(),
                "url": url_of(request), "sha256": hashlib.sha256(result.content).hexdigest(),
                "data_sha256": data_sha, "points": len(points), "groups": groups,
                "first_timestamp_utc": min(p.start_utc for p in points).isoformat() if points else "",
                "last_timestamp_utc": max(p.start_utc for p in points).isoformat() if points else "",
            })
            latest[key] = data_sha

        periods = [(day, *local_day_bounds(day)) for day in request.days] or [(None, request.start, request.end)]
        for day, start, end in periods:
            inside = [p for p in points if start <= p.start_utc < end]
            row = {
                "fetched_at_utc": fetched.isoformat(), "fetched_at_local": fetched.astimezone(TZ).isoformat(),
                "provider": provider, "source": source, "series": request.series.name, "zone": request.series.zone,
                "delivery_day": day.isoformat() if day else "", "period_start_utc": start.isoformat(),
                "period_end_utc": end.isoformat(), "status": result.status, "http_status": result.http_status or "",
                "points": len(inside), "groups": groups,
                "complete": result.status == "ok" and covers(inside, start, end),
                "data_sha256": data_sha, "changed": changed, "snapshot": name, "message": result.message,
            }
            append_row(log, FETCH_LOG_COLUMNS, row)
            rows.append(row)
        complete_days = sum(r["complete"] for r in rows[-len(periods):])
        print(f"{provider} {request.series.name} {request.series.zone} {period_label(request)}: {result.status}, "
              f"{len(points)} points, {complete_days}/{len(periods)} complete{', new snapshot' if changed else ''}"
              f"{f' ({result.message})' if result.status in ERRORS else ''}")
    return rows
