"""Download archived ECMWF IFS weather forecasts from the Open-Meteo Single Runs API.

For delivery day D (Swiss local date) we use the model run initialised on D-2 at 18:00 UTC.
Open-Meteo states that global model runs are available about 4 to 6 hours after initialisation,
so this run exists by 00:00 UTC on D-1, at least 9 hours before the forecast is issued at
D-1 11:00 Swiss time. Later runs are never used, which rules out look-ahead.

One JSON file per run is stored unchanged, together with the request and the download time.

Usage: ``python -m energy_price.fetch_weather --sites data/meta/weather_sites.csv --out data/weather/ecmwf_ifs --start 2026-01-01``
"""

from __future__ import annotations

import argparse
import datetime as dt
import http.client
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

from energy_price.availability import issue_time

API_URL ="https://single-runs-api.open-meteo.com/v1/forecast"
MODEL = "ecmwf_ifs"
VARIABLES = [
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "direct_normal_irradiance",
    "sunshine_duration",
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "snowfall",
    "snow_depth",
    "wind_speed_10m",
    "wind_speed_100m",
]
TZ = "Europe/Zurich"
RUN_HOUR_UTC = 18
RUN_DAYS_BEFORE_DELIVERY = 2
FORECAST_DAYS = 4  # 96 hours from D-2 18:00 UTC, covers the whole delivery day D
# Conservative availability per Open-Meteo documentation (global models: 4 to 6 hours).
AVAILABLE_AFTER = pd.Timedelta(hours=6)


class WeatherFetchError(RuntimeError):
    """The API returned an error or a response that does not match the request."""


def run_for_delivery_day(day: dt.date) -> pd.Timestamp:
    """Initialisation time (UTC) of the model run used for delivery day ``day``."""
    run_date = day - dt.timedelta(days=RUN_DAYS_BEFORE_DELIVERY)
    return pd.Timestamp(run_date.isoformat(), tz="UTC") + pd.Timedelta(hours=RUN_HOUR_UTC)


def run_is_available_at_issue(day: dt.date) -> bool:
    return run_for_delivery_day(day) + AVAILABLE_AFTER <= issue_time(day)


def request_url(run: pd.Timestamp, sites: pd.DataFrame) -> str:
    params = {
        "latitude": ",".join(f"{v:.4f}" for v in sites["lat"]),
        "longitude": ",".join(f"{v:.4f}" for v in sites["lon"]),
        "run": run.strftime("%Y-%m-%dT%H:%M"),
        "hourly": ",".join(VARIABLES),
        "models": MODEL,
        "timezone": "UTC",
        "forecast_days": FORECAST_DAYS,
    }
    return f"{API_URL}?{urllib.parse.urlencode(params, safe=',:')}"


def fetch_run(run: pd.Timestamp, sites: pd.DataFrame, retries: int = 4) -> dict:
    """Download one run for all sites. Retries on rate limits and server errors."""
    url = request_url(run, sites)
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                payload = json.load(response)
            break
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")
            if exc.code in (429, 500, 502, 503, 504) and attempt < retries:
                time.sleep(2**attempt * 5)
                continue
            raise WeatherFetchError(f"run {run}: HTTP {exc.code}: {body[:300]}") from exc
        except (urllib.error.URLError, http.client.HTTPException, TimeoutError, json.JSONDecodeError) as exc:
            # Dropped connections and truncated bodies are transient: retry, then give up loudly.
            if attempt < retries:
                time.sleep(2**attempt * 5)
                continue
            raise WeatherFetchError(f"run {run}: {type(exc).__name__}: {exc}") from exc

    if isinstance(payload, dict) and payload.get("error"):
        raise WeatherFetchError(f"run {run}: {payload.get('reason')}")
    locations = payload if isinstance(payload, list) else [payload]
    if len(locations) != len(sites):
        raise WeatherFetchError(f"run {run}: {len(locations)} locations returned, {len(sites)} requested")
    for site, loc in zip(sites["site"], locations):
        missing = [v for v in VARIABLES if v not in loc.get("hourly", {})]
        if missing:
            raise WeatherFetchError(f"run {run}, site {site}: variables missing {missing}")

    return {
        "run_utc": run.isoformat(),
        "model": MODEL,
        "fetched_at_utc": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
        "request_url": url,
        "sites": sites["site"].tolist(),
        "locations": locations,
    }


def run_path(out_dir: Path, run: pd.Timestamp) -> Path:
    return out_dir / f"run_{run.strftime('%Y-%m-%dT%H')}.json"


def delivery_days(start: dt.date, end: dt.date) -> list[dt.date]:
    return [start + dt.timedelta(days=i) for i in range((end - start).days + 1)]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sites", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--start", type=dt.date.fromisoformat, required=True, help="first delivery day")
    parser.add_argument("--end", type=dt.date.fromisoformat, help="last delivery day, default: tomorrow")
    args = parser.parse_args(argv)

    sites = pd.read_csv(args.sites)
    end = args.end or (pd.Timestamp.now(tz=TZ).date() + dt.timedelta(days=1))
    now = pd.Timestamp.now(tz="UTC")
    args.out.mkdir(parents=True, exist_ok=True)

    fetched = skipped = 0
    for day in delivery_days(args.start, end):
        if not run_is_available_at_issue(day):
            raise WeatherFetchError(f"run for {day} would not be available at the issue time")
        run = run_for_delivery_day(day)
        target = run_path(args.out, run)
        if target.exists() or run + AVAILABLE_AFTER > now:
            skipped += 1
            continue
        record = fetch_run(run, sites)
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(record))
        tmp.replace(target)
        fetched += 1
        time.sleep(0.3)
    print(f"weather runs: {fetched} downloaded, {skipped} skipped (already present or not yet available)")


if __name__ == "__main__":
    main()
