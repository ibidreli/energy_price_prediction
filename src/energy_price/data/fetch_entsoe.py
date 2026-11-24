"""Download the issue #37 data directly from the ENTSO-E REST API (needs a token).

The second source next to ``fetch_energy_charts``, which needs no key. It also covers the
week-ahead load forecast, which Energy-Charts does not have, and shows whether ENTSO-E publishes
earlier than Energy-Charts. The scheduled job runs it only when the repository secret
``ENTSOE_API_TOKEN`` exists; locally the token comes from the environment or ``.env``. It is never
written anywhere: logged URLs leave it out and error messages have it masked.

A live run asks, for every series, for the local day before, of and after the fetch day, one request
per day. Snapshots, log and change detection: ``energy_price.data.snapshots``.

Only the standard library is used, so the GitHub Actions job runs without installing the project.

Usage:
``python -m energy_price.data.fetch_entsoe --out data/entsoe/entsoe_api/snapshots``
``python -m energy_price.data.fetch_entsoe --out data/entsoe/entsoe_api/backfill --history-from 2026-01-01``
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Mapping
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from energy_price.data import entsoe_xml, snapshots
from energy_price.data.entsoe_xml import TZ, EntsoeResponseError, local_day_bounds
from energy_price.data.snapshots import (  # noqa: F401  (ceil_second is re-exported for the tests)
    ERRORS,
    Request,
    Result,
    Series,
    ceil_second,
    mask,
    months,
)

PROVIDER = "entsoe_api"
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
RETRY_STATUS = {429, 500, 502, 503, 504}

SERIES = (
    Series("load_day_ahead", "CH", (("documentType", "A65"), ("processType", "A01"),
                                    ("outBiddingZone_Domain", ZONES["CH"]))),
    Series("load_week_ahead", "CH", (("documentType", "A65"), ("processType", "A31"),
                                     ("outBiddingZone_Domain", ZONES["CH"]))),
    Series("load_actual", "CH", (("documentType", "A65"), ("processType", "A16"),
                                 ("outBiddingZone_Domain", ZONES["CH"]))),
    # Solar (B16) and wind (B18, B19) come in one document; the loader splits them into two series.
    Series("wind_solar_day_ahead", "CH", (("documentType", "A69"), ("processType", "A01"),
                                          ("in_Domain", ZONES["CH"]))),
    *(Series("price_day_ahead", zone, (("documentType", "A44"), ("in_Domain", eic), ("out_Domain", eic)))
      for zone, eic in ZONES.items()),
)


def live_requests(fetch_day: date, series: Iterable[Series] = SERIES) -> list[Request]:
    """One request per series and local day ``fetch_day`` + ``LIVE_DAY_OFFSETS``."""
    out = []
    for item in series:
        for offset in LIVE_DAY_OFFSETS:
            day = fetch_day + timedelta(days=offset)
            out.append(Request(item, *local_day_bounds(day), days=(day,)))
    return out


def history_requests(first: date, last: date, series: Iterable[Series] = SERIES) -> list[Request]:
    """One request per series and local calendar month, from ``first`` to ``last`` inclusive."""
    return [Request(item, start, end) for item in series for start, end in months(first, last)]


def period_params(start: datetime, end: datetime) -> dict[str, str]:
    return {"periodStart": start.strftime("%Y%m%d%H%M"), "periodEnd": end.strftime("%Y%m%d%H%M")}


def request_params(request: Request) -> dict[str, str]:
    return {**dict(request.series.params), **period_params(request.start, request.end)}


def public_url(params: Mapping[str, str]) -> str:
    """The request URL without the token, as it is written to the manifest."""
    return f"{API_URL}?{urllib.parse.urlencode(params)}"


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


def run(requests: list[Request], out: Path, source: str, token: str | None,
        get: Callable = download, now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        pause: float = 0.2, sleep: Callable[[float], None] = time.sleep) -> list[dict]:
    """Fetch every request with the token, store changed responses and log every request."""
    def fetch_one(request: Request) -> Result:
        if not token:
            return Result("no_token", message=f"{TOKEN_VARIABLE} is not set")
        return fetch(request_params(request), token, get=get, sleep=sleep)

    # well below the limit of 400 requests per minute
    return snapshots.run(requests, out, PROVIDER, source, fetch_one, lambda r: public_url(request_params(r)),
                         ".xml", now=now, pause=pause if token else 0.0, sleep=sleep)


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
    print(f"{len(rows)} log rows, {len(failed)} failed, "
          f"{len({row['snapshot'] for row in rows if row['snapshot']})} new snapshots in {args.out}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
