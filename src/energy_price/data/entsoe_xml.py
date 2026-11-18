"""Read ENTSO-E Transparency Platform responses with the standard library only.

The scheduled fetch (``fetch_entsoe``) runs on GitHub Actions without installing the project, so this
module and ``fetch_entsoe`` import nothing outside the standard library (``tests/test_fetch_entsoe.py``
checks it). ``energy_price.data.entsoe`` builds the pandas tables on top of it.

A response is one of two documents:

- a market document (``GL_MarketDocument`` for load and generation, ``Publication_MarketDocument``
  for prices) with one or more ``TimeSeries``. Each holds ``Period`` blocks with a time interval in
  UTC, a resolution (``PT15M``, ``PT60M``, ``P1D``, ...) and numbered ``Point`` elements;
- an ``Acknowledgement_MarketDocument`` with a reason instead of data. Code 999 means that no data
  matches the request, the normal answer before a value is published.

Curve type A03 ("variable sized block") leaves out a point whose value equals the previous one; the
value holds until the next listed position or the end of the period. A01 lists every position, so a
missing position stays a gap. ``parse`` expands both to one ``Point`` per resolution step. Any other
curve type is rejected rather than guessed.
"""

from __future__ import annotations

import hashlib
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

TZ = ZoneInfo("Europe/Zurich")
QUARTER_HOUR = timedelta(minutes=15)
NO_DATA = "999"
FIXED_RESOLUTION = re.compile(r"^PT(\d+)([MH])$")
DAY_RESOLUTION = re.compile(r"^P(\d+)D$")


class EntsoeResponseError(ValueError):
    """The response is neither readable data nor a 'no matching data' acknowledgement."""


@dataclass(frozen=True)
class Point:
    """One value over ``start_utc`` to ``end_utc`` (MW for load and generation, EUR/MWh for prices)."""

    business_type: str
    psr_type: str
    resolution: str
    start_utc: datetime
    end_utc: datetime
    value: float

    @property
    def group(self) -> str:
        """Business type, with the production type if there is one (e.g. ``A94/B16`` for solar)."""
        return f"{self.business_type}/{self.psr_type}" if self.psr_type else self.business_type


@dataclass(frozen=True)
class Response:
    points: tuple[Point, ...]
    no_data: bool = False
    reason: str = ""

    @property
    def groups(self) -> list[str]:
        return sorted({point.group for point in self.points})


def local_name(tag: str) -> str:
    """Tag without its namespace: ENTSO-E changes the namespace version between document releases."""
    return tag.rsplit("}", 1)[-1]


def child(element: ET.Element, name: str) -> ET.Element | None:
    return next((c for c in element if local_name(c.tag) == name), None)


def text(element: ET.Element | None, name: str, default: str = "") -> str:
    found = None if element is None else child(element, name)
    return found.text.strip() if found is not None and found.text else default


def parse_time(value: str) -> datetime:
    """``2026-11-17T23:00Z`` (with or without seconds) as an aware UTC datetime."""
    try:
        stamp = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise EntsoeResponseError(f"unreadable time {value!r}") from exc
    if stamp.tzinfo is None:
        raise EntsoeResponseError(f"time without offset {value!r}")
    return stamp.astimezone(timezone.utc)


def local_midnight(day: date) -> datetime:
    """Start of the local day ``day`` in UTC."""
    return datetime(day.year, day.month, day.day, tzinfo=TZ).astimezone(timezone.utc)


def local_day_bounds(day: date) -> tuple[datetime, datetime]:
    """Start and end of the local day ``day`` in UTC: 23, 24 or 25 hours apart."""
    return local_midnight(day), local_midnight(day + timedelta(days=1))


def steps(start: datetime, end: datetime, resolution: str) -> list[tuple[datetime, datetime]]:
    """The resolution steps of a period. Days follow the local calendar (23 or 25 hours on switch days)."""
    fixed = FIXED_RESOLUTION.match(resolution)
    if fixed:
        size = int(fixed[1])
        step = timedelta(minutes=size) if fixed[2] == "M" else timedelta(hours=size)
        count, rest = divmod(end - start, step)
        if rest or count <= 0:
            raise EntsoeResponseError(f"period {start} to {end} is no multiple of {resolution}")
        return [(start + i * step, start + (i + 1) * step) for i in range(count)]
    days = DAY_RESOLUTION.match(resolution)
    if days:
        out, current = [], start
        while current < end:
            local = current.astimezone(TZ)
            following = local.date() + timedelta(days=int(days[1]))
            nxt = datetime(following.year, following.month, following.day, local.hour, local.minute, tzinfo=TZ)
            out.append((current, min(nxt.astimezone(timezone.utc), end)))
            current = out[-1][1]
        return out
    raise EntsoeResponseError(f"unsupported resolution {resolution!r}")


def period_points(period: ET.Element, business_type: str, psr_type: str, curve_type: str) -> list[Point]:
    interval = child(period, "timeInterval")
    start, end = parse_time(text(interval, "start")), parse_time(text(interval, "end"))
    resolution = text(period, "resolution")
    grid = steps(start, end, resolution)
    listed: dict[int, float] = {}
    for point in period:
        if local_name(point.tag) != "Point":
            continue
        raw = text(point, "quantity") or text(point, "price.amount")
        try:
            position, value = int(text(point, "position")), float(raw)
        except ValueError as exc:
            raise EntsoeResponseError(f"unreadable point in period starting {start}") from exc
        if not 1 <= position <= len(grid):
            raise EntsoeResponseError(f"position {position} outside the period of {len(grid)} steps")
        if position in listed:
            raise EntsoeResponseError(f"position {position} listed twice")
        listed[position] = value

    positions = sorted(listed)
    out = []
    for i, position in enumerate(positions):
        if curve_type == "A03":
            last = positions[i + 1] - 1 if i + 1 < len(positions) else len(grid)
        else:
            last = position
        for p in range(position, last + 1):
            step_start, step_end = grid[p - 1]
            out.append(Point(business_type, psr_type, resolution, step_start, step_end, listed[position]))
    return out


def unique(points: list[Point]) -> tuple[Point, ...]:
    """Drop exact repeats; two different values for the same step are an error, never a silent choice."""
    seen: dict[tuple, Point] = {}
    for point in points:
        key = (point.group, point.resolution, point.start_utc)
        if key in seen and seen[key].value != point.value:
            raise EntsoeResponseError(f"two values for {point.group} at {point.start_utc}")
        seen.setdefault(key, point)
    return tuple(sorted(seen.values(), key=lambda p: (p.group, p.resolution, p.start_utc)))


def parse(content: bytes) -> Response:
    """Points of a market document, or ``no_data`` for an acknowledgement with reason code 999."""
    try:
        root = ET.fromstring(content)
    except ET.ParseError as exc:
        raise EntsoeResponseError(f"not XML: {exc}") from exc
    kind = local_name(root.tag)
    if kind == "Acknowledgement_MarketDocument":
        reasons = [(text(r, "code"), text(r, "text")) for r in root if local_name(r.tag) == "Reason"]
        reason = "; ".join(f"{code}: {message}" for code, message in reasons)
        if any(code == NO_DATA for code, _ in reasons):
            return Response(points=(), no_data=True, reason=reason)
        raise EntsoeResponseError(f"acknowledgement without data: {reason}")
    if not kind.endswith("MarketDocument"):
        raise EntsoeResponseError(f"unexpected document {kind}")

    points: list[Point] = []
    for series in root.iter():
        if local_name(series.tag) != "TimeSeries":
            continue
        curve_type = text(series, "curveType", "A01")
        if curve_type not in {"A01", "A03"}:
            raise EntsoeResponseError(f"unsupported curve type {curve_type}")
        business_type = text(series, "businessType")
        psr_type = text(child(series, "MktPSRType"), "psrType")
        for period in series:
            if local_name(period.tag) == "Period":
                points += period_points(period, business_type, psr_type, curve_type)
    return Response(points=unique(points))


def covers(points: tuple[Point, ...] | list[Point], start: datetime, end: datetime) -> bool:
    """True if there is at least one group and every group has a value in every quarter hour from start to end."""
    covered: dict[str, set[datetime]] = {}
    for point in points:
        stamps = covered.setdefault(point.group, set())
        stamp = point.start_utc
        while stamp < point.end_utc:
            stamps.add(stamp)
            stamp += QUARTER_HOUR
    needed, stamp = [], start
    while stamp < end:
        needed.append(stamp)
        stamp += QUARTER_HOUR
    return bool(covered) and all(stamps.issuperset(needed) for stamps in covered.values())


def is_complete(points: tuple[Point, ...] | list[Point], day: date) -> bool:
    """True if every group has all 92, 96 or 100 quarter hours of the local day ``day``."""
    return covers(points, *local_day_bounds(day))


def data_checksum(points: tuple[Point, ...] | list[Point]) -> str:
    """SHA-256 of the values only. ``createdDateTime`` and the document id change with every request,
    so a checksum of the raw bytes would store a new snapshot each time although nothing changed."""
    lines = sorted(
        f"{p.group}|{p.resolution}|{p.start_utc.isoformat()}|{p.end_utc.isoformat()}|{p.value!r}" for p in points
    )
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
