"""Download Swiss Energy-Charts reservoir JSON and archive each distinct version.

Usage: python -m energy_price.fetch_reservoirs --out data/reservoirs/snapshots --years 2026
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

import pandas as pd

from energy_price.fetch_swissgrid import append_row
from energy_price.reservoir import ReservoirFormatError, parse_payload

URL_TEMPLATE = "https://www.energy-charts.info/charts/filling_level/data/ch/year_storage_{year}.json"
MANIFEST_COLUMNS = [
    "file", "year", "url", "fetched_at_utc", "last_modified", "sha256",
    "weeks", "first_timestamp_utc", "last_timestamp_utc",
]
FETCH_LOG_COLUMNS = ["year", "url", "fetched_at_utc", "last_modified", "sha256", "changed", "snapshot"]


def download(year: int) -> tuple[bytes, str]:
    request = urllib.request.Request(
        URL_TEMPLATE.format(year=year),
        headers={"User-Agent": "Mozilla/5.0 (energy-price research project)"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read(), response.headers.get("Last-Modified", "")


def fetch_year(year: int, out: Path) -> Path:
    content, last_modified = download(year)
    fetched = pd.Timestamp.now(tz="UTC")
    payload = json.loads(content)
    frame = parse_payload(payload)
    if not (frame["timestamp_utc"].dt.year == year).all():
        raise ReservoirFormatError(f"response does not contain the requested year {year}")
    sha = hashlib.sha256(content).hexdigest()
    files = sorted(out.glob(f"filling-level-{year}_*.json"))
    previous = json.loads(files[-1].read_text(encoding="utf-8")) if files else None
    changed = previous is None or previous["sha256"] != sha
    out.mkdir(parents=True, exist_ok=True)
    if changed:
        path = out / f"filling-level-{year}_{fetched.strftime('%Y-%m-%dT%H-%M-%S-%fZ')}.json"
        record = dict(year=year, url=URL_TEMPLATE.format(year=year), fetched_at_utc=fetched.isoformat(),
                      last_modified=last_modified, sha256=sha, data=payload)
        # Keep the response verbatim inside the envelope, including whitespace and numeric precision.
        header = json.dumps({k: v for k, v in record.items() if k != "data"}, ensure_ascii=False)
        tmp = path.with_suffix(".json.tmp")
        try:
            tmp.write_text(header[:-1] + ', "data": ' + content.decode("utf-8") + '}\n', encoding="utf-8")
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)
        append_row(out / "manifest.csv", MANIFEST_COLUMNS, {
            **{key: record[key] for key in MANIFEST_COLUMNS if key in record},
            "file": path.name, "weeks": frame["timestamp_utc"].nunique(),
            "first_timestamp_utc": frame["timestamp_utc"].min().isoformat(),
            "last_timestamp_utc": frame["timestamp_utc"].max().isoformat(),
        })
    else:
        path = files[-1]
    append_row(out / "fetch_log.csv", FETCH_LOG_COLUMNS, {
        "year": year, "url": URL_TEMPLATE.format(year=year), "fetched_at_utc": fetched.isoformat(),
        "last_modified": last_modified, "sha256": sha, "changed": changed, "snapshot": path.name,
    })
    print(f"reservoirs {year}: {'new snapshot' if changed else 'unchanged'}, "
          f"{frame['timestamp_utc'].nunique()} weeks, last reference Sunday "
          f"{frame['reference_date'].max().date()}")
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--years", type=int, nargs="+", default=[2026])
    args = parser.parse_args(argv)
    for year in sorted(set(args.years)):
        fetch_year(year, args.out)


if __name__ == "__main__":
    main()
