"""Read Energy-Charts API v2 responses with the standard library only.

Energy-Charts (Fraunhofer ISE, https://api.energy-charts.info) publishes the issue #37 data without a
key. A v2 response is JSON with ``data``: a list of rows ``{"timestamp": ISO 8601 with offset,
"values": {series id: value}}``. ``parse`` turns one series id into the same ``Point`` objects as
the ENTSO-E parser, so both sources share snapshots, checksums and the completeness check.

The ``resolution`` in the header is not reliable: a Swiss price response across the switch to
quarter hours on 2026-11-04 says ``PT1H`` but holds 15-minute rows from that day on. Each row therefore
lasts until the nearer of its neighbours, at most one hour. A missing row stays a gap, and so does a
``null`` value.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from energy_price.data.entsoe_xml import EntsoeResponseError, Point, Response, unique

MAX_STEP = timedelta(hours=1)
STEPS = {timedelta(minutes=15), timedelta(minutes=30), timedelta(hours=1)}


def parse_timestamp(value: str) -> datetime:
    try:
        stamp = datetime.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise EntsoeResponseError(f"unreadable timestamp {value!r}") from exc
    if stamp.tzinfo is None:
        raise EntsoeResponseError(f"timestamp without offset {value!r}")
    return stamp.astimezone(timezone.utc)


def row_steps(stamps: list[datetime], header: timedelta) -> list[timedelta]:
    """Length of each row from the distance to the nearer neighbour.

    The longest of 1 h, 30 and 15 minutes that divides that distance, so a row never covers a
    quarter hour that is missing: 45 minutes to both neighbours gives 15 minutes, not 30.
    """
    if len(stamps) == 1:
        return [min(header, MAX_STEP)]
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    out = []
    for i, stamp in enumerate(stamps):
        nearest = min(gaps[max(i - 1, 0):i + 1])
        step = next((s for s in sorted(STEPS, reverse=True) if nearest % s == timedelta(0)), None)
        if step is None:
            raise EntsoeResponseError(f"rows {nearest} apart at {stamp}")
        out.append(step)
    return out


def parse(content: bytes, series_id: str) -> Response:
    """Points of ``series_id``; a response without that series or without rows has no points."""
    try:
        data = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EntsoeResponseError(f"not JSON: {exc}") from exc
    if not isinstance(data, dict) or not str(data.get("schema_version", "")).startswith("2."):
        raise EntsoeResponseError("not an Energy-Charts v2 response")
    rows = [row for row in data.get("data") or [] if (row.get("values") or {}).get(series_id) is not None]
    if not rows:
        return Response(points=())
    stamps = [parse_timestamp(row["timestamp"]) for row in data["data"]]
    if any(b <= a for a, b in zip(stamps, stamps[1:])):
        raise EntsoeResponseError("timestamps are not increasing")
    header = timedelta(minutes=int(data.get("interval_minutes") or 60))
    steps = dict(zip(stamps, row_steps(stamps, header)))
    points = []
    for row in rows:
        start = parse_timestamp(row["timestamp"])
        step = steps[start]
        try:
            value = float(row["values"][series_id])
        except (TypeError, ValueError) as exc:
            raise EntsoeResponseError(f"unreadable value at {start}") from exc
        points.append(Point(series_id, "", f"PT{int(step.total_seconds() // 60)}M", start, start + step, value))
    return Response(points=unique(points))
