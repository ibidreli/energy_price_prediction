"""Download the issue #37 data from Energy-Charts (public API, no key) and keep every distinct version.

Energy-Charts (Fraunhofer ISE, https://api.energy-charts.info) publishes for Switzerland the day-ahead
forecasts of load, solar and wind, the actual load, and the day-ahead prices of CH, DE-LU, FR,
IT-North and AT, without an account or key. The load and generation values are presumably taken over
from ENTSO-E; the prices carry the licence "CC BY 4.0 from Bundesnetzagentur | SMARD.de". The
week-ahead load forecast is not available there; ``fetch_entsoe`` has it once a token exists.

A live run asks once per series for the local days before, of and after the fetch day and logs each
day on its own. For the forecast issued at D-1 these are D-2 (actual load), D-1 (prices) and D
(forecasts). One request covers the three days, because Energy-Charts limits the request rate: it
answered ``429 Too Many Requests`` after about a dozen quick requests on 2026-11-24. The run therefore
pauses between requests and retries a limited request after 1 and 2 minutes.

Snapshots, log and change detection: ``energy_price.data.snapshots``. Only the standard library is
used, so the GitHub Actions job runs without installing the project.

Usage:
``python -m energy_price.data.fetch_energy_charts --out data/entsoe/energy_charts/snapshots``
``python -m energy_price.data.fetch_energy_charts --out data/entsoe/energy_charts/backfill --history-from 2026-01-01``
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from energy_price.data import energy_charts_json, snapshots
from energy_price.data.entsoe_xml import TZ, EntsoeResponseError
from energy_price.data.snapshots import ERRORS, Request, Result, Series, mask, months, window

PROVIDER = "energy_charts"
API_URL = "https://api.energy-charts.info/v2"
USER_AGENT = "energy-price-research (python urllib)"
PRICE_ZONES = {"CH": "CH", "DE_LU": "DE-LU", "FR": "FR", "IT_NORTH": "IT-North", "AT": "AT"}
FORECAST = (("endpoint", "public_power_forecast"), ("country", "ch"), ("forecast_type", "day-ahead"))
SERIES = (
    Series("load_day_ahead", "CH", (*FORECAST, ("production_type", "load"), ("id", "load"))),
    Series("solar_day_ahead", "CH", (*FORECAST, ("production_type", "solar"), ("id", "solar"))),
    Series("wind_day_ahead", "CH", (*FORECAST, ("production_type", "wind_onshore"), ("id", "wind_onshore"))),
    Series("load_actual", "CH", (("endpoint", "public_power"), ("country", "ch"), ("id", "load"))),
    *(Series("price_day_ahead", zone, (("endpoint", "price"), ("bzn", bzn), ("id", "day_ahead_price")))
      for zone, bzn in PRICE_ZONES.items()),
)
PAUSE = 8.0  # seconds between requests
RETRY_WAITS = (60.0, 120.0)  # after 429, server errors and network errors
RETRY_STATUS = {429, 500, 502, 503, 504}


def series_id(series: Series) -> str:
    """The key of the wanted values in the response, e.g. ``load`` or ``day_ahead_price``."""
    return dict(series.params)["id"]


def series_by_key() -> dict[tuple[str, str], Series]:
    return {(s.name, s.zone): s for s in SERIES}


def live_requests(fetch_day: date, series: Iterable[Series] = SERIES) -> list[Request]:
    """One request per series for the local days before, of and after ``fetch_day``."""
    return [window(item, fetch_day - timedelta(days=1), fetch_day + timedelta(days=1)) for item in series]


def history_requests(first: date, last: date, series: Iterable[Series] = SERIES) -> list[Request]:
    """One request per series and local calendar month, from ``first`` to ``last`` inclusive."""
    return [Request(item, start, end) for item in series for start, end in months(first, last)]


def url(request: Request) -> str:
    """The request URL. Days are local and both ends inclusive, as the API reads them."""
    params = dict(request.series.params)
    endpoint = params.pop("endpoint")
    params.pop("id")
    params["start"] = request.start.astimezone(TZ).date().isoformat()
    params["end"] = (request.end - timedelta(seconds=1)).astimezone(TZ).date().isoformat()
    return f"{API_URL}/{endpoint}?{urllib.parse.urlencode(params)}"


def download(address: str, timeout: float = 60) -> tuple[int, bytes]:
    """HTTP status and body. An HTTP error is returned, not raised."""
    request = urllib.request.Request(address, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def classify(http_status: int, content: bytes, wanted: str) -> Result:
    if http_status == 404 and b"no content available" in content:
        return Result("no_data", http_status, content, message="no content available")
    if http_status != 200:
        body = content[:200].decode("utf-8", "replace")
        return Result("http_error", http_status, content, message=mask(f"HTTP {http_status}: {body}"))
    try:
        return Result("ok", http_status, content, energy_charts_json.parse(content, wanted))
    except EntsoeResponseError as exc:
        return Result("parse_error", http_status, content, message=mask(str(exc)))


def fetch(request: Request, get: Callable[[str], tuple[int, bytes]] = download,
          waits: tuple[float, ...] = RETRY_WAITS, sleep: Callable[[float], None] = time.sleep) -> Result:
    """Download and classify; retry after rate limits, server errors and network errors."""
    address, wanted = url(request), series_id(request.series)
    result = Result("network_error")
    for attempt in range(len(waits) + 1):
        try:
            http_status, content = get(address)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            result = Result("network_error", message=mask(f"{type(exc).__name__}: {exc}"))
        else:
            result = classify(http_status, content, wanted)
            if result.status != "http_error" or http_status not in RETRY_STATUS:
                return result
        if attempt < len(waits):
            sleep(waits[attempt])
    return result


def run(requests: list[Request], out: Path, source: str, get: Callable[[str], tuple[int, bytes]] = download,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc), pause: float = PAUSE,
        sleep: Callable[[float], None] = time.sleep) -> list[dict]:
    """Fetch every request, store changed responses and log every request and day."""
    return snapshots.run(requests, out, PROVIDER, source, lambda r: fetch(r, get=get, sleep=sleep), url, ".json",
                         now=now, pause=pause, sleep=sleep)


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
    rows = run(requests, args.out, source)
    failed = [row for row in rows if row["status"] in ERRORS]
    print(f"{len(rows)} log rows, {len(failed)} failed, "
          f"{len({row['snapshot'] for row in rows if row['snapshot']})} new snapshots in {args.out}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
