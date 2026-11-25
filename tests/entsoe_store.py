"""Store synthetic responses the way the scheduled job does, for either source of the issue #37 data."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime, timedelta
from pathlib import Path

from energy_price.data import fetch_energy_charts, fetch_entsoe
from energy_price.data.entsoe_xml import TZ, local_day_bounds
from energy_price.data.snapshots import Request, window
from entsoe_fixtures import ec_document, gl_document, period, price_document, price_series, time_series

HOUR = timedelta(hours=1)


def local(day: date, hour: int, minute: int = 0, second: int = 0) -> datetime:
    return datetime(day.year, day.month, day.day, hour, minute, second, tzinfo=TZ)


def response(provider: str, name: str, zone: str, first: date, last: date,
             value: Callable[[int], float | None]) -> bytes:
    """Hourly values over the local days ``first`` to ``last``; ``value`` gets the hour index."""
    start, end = local_day_bounds(first)[0], local_day_bounds(last)[1]
    if provider == "energy_charts":
        wanted = fetch_energy_charts.series_id(fetch_energy_charts.series_by_key()[(name, zone)])
        return ec_document(start, end, HOUR, value=value, series_id=wanted)
    values = [value(i) for i in range(int((end - start) / HOUR))]
    if name == "price_day_ahead":
        return price_document([price_series([period(start, end, "PT60M", values, "price.amount")], "A01")])
    return gl_document([time_series([period(start, end, "PT60M", values)])])


def store(root: Path, name: str, zone: str, fetched: datetime, first: date, last: date | None = None,
          value: Callable[[int], float | None] = lambda hour: 1.0, source: str = "live",
          provider: str = "energy_charts", content: bytes | None = None) -> None:
    """One response for the local days ``first`` to ``last``, stored and logged at ``fetched``."""
    last = last or first
    content = content if content is not None else response(provider, name, zone, first, last, value)
    module = fetch_energy_charts if provider == "energy_charts" else fetch_entsoe
    series = next(s for s in module.SERIES if s.name == name and s.zone == zone)
    if source == "live":
        request = window(series, first, last)
    else:
        request = Request(series, local_day_bounds(first)[0], local_day_bounds(last)[1])
    folder = root / provider / ("snapshots" if source == "live" else "backfill")
    if provider == "energy_charts":
        fetch_energy_charts.run([request], folder, source, get=lambda url: (200, content), now=lambda: fetched,
                                pause=0, sleep=lambda s: None)
    else:
        fetch_entsoe.run([request], folder, source, "token", get=lambda params, token: (200, content),
                         now=lambda: fetched, sleep=lambda s: None)
