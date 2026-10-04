"""Download the Swissgrid "control area balance" file and keep every distinct version as a snapshot.

Swissgrid overwrites provisional values with final ones later, so the file is not stable over
time. Each download is stored under its download time unless its content equals the latest
snapshot. ``manifest.csv`` records URL, download time, ``Last-Modified``, checksum and time range.
``fetch_log.csv`` records every download, changed or not, with ``days_behind``: how many days the
last complete local day lags behind the download date. It is the evidence for the D-2 cutoff.

Usage: ``python -m energy_price.fetch_swissgrid --out data/control_area_balance/snapshots``
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import urllib.request
from pathlib import Path

import pandas as pd

URL = "https://www.swissgrid.ch/dam/jcr:aeff36d3-9538-4d0c-b369-fc22172b8374/control-area-balance-2026.csv"
MANIFEST_COLUMNS = ["file", "url", "fetched_at_utc", "last_modified", "sha256", "rows", "first_timestamp_utc", "last_timestamp_utc"]
FETCH_LOG_COLUMNS = ["fetched_at_utc", "fetched_at_local", "last_modified", "sha256", "last_timestamp_local", "days_behind", "changed", "snapshot"]
TZ = "Europe/Zurich"


def download(url: str = URL) -> tuple[bytes, str]:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (energy-price research project)"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read(), response.headers.get("Last-Modified", "")


def time_range(content: bytes) -> tuple[int, str, str]:
    """Rows and first and last timestamp of a control area balance CSV (timestamps in UTC)."""
    frame = pd.read_csv(io.BytesIO(content), sep=";", usecols=["Date Time [UTC]"])
    stamps = pd.to_datetime(frame["Date Time [UTC]"], format="%d.%m.%Y %H:%M", utc=True)
    return len(frame), stamps.min().isoformat(), stamps.max().isoformat()


def latest_checksum(manifest: Path) -> str | None:
    if not manifest.exists():
        return None
    rows = list(csv.DictReader(manifest.open()))
    return rows[-1]["sha256"] if rows else None


def days_behind(fetched_at: pd.Timestamp, last_timestamp_utc: str) -> int:
    """Days between the local fetch date and the last complete local day in the file.

    1 means yesterday is complete. At the issue time D-1 11:00 the forecast relies on D-2 being
    complete, so a value above 1 at that time of day would break the cutoff in availability.py.
    """
    end = pd.Timestamp(last_timestamp_utc).tz_convert(TZ) + pd.Timedelta(minutes=15)
    last_complete_day = (end.normalize() - pd.Timedelta(days=1)).date()
    return (fetched_at.tz_convert(TZ).date() - last_complete_day).days


def append_row(path: Path, columns: list[str], row: dict) -> None:
    new_file = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="snapshot directory")
    args = parser.parse_args(argv)

    content, last_modified = download()
    fetched_at = pd.Timestamp.now(tz="UTC").floor("s")
    sha = hashlib.sha256(content).hexdigest()
    manifest = args.out / "manifest.csv"
    rows, first, last = time_range(content)
    changed = sha != latest_checksum(manifest)
    args.out.mkdir(parents=True, exist_ok=True)

    name = ""
    if changed:
        name = f"control-area-balance-2026_{fetched_at.strftime('%Y-%m-%dT%H-%M-%SZ')}.csv"
        (args.out / name).write_bytes(content)
        append_row(
            manifest,
            MANIFEST_COLUMNS,
            {
                "file": name,
                "url": URL,
                "fetched_at_utc": fetched_at.isoformat(),
                "last_modified": last_modified,
                "sha256": sha,
                "rows": rows,
                "first_timestamp_utc": first,
                "last_timestamp_utc": last,
            },
        )

    # Every download is logged, also unchanged ones: they show how far behind the file is at a given time.
    behind = days_behind(fetched_at, last)
    append_row(
        args.out / "fetch_log.csv",
        FETCH_LOG_COLUMNS,
        {
            "fetched_at_utc": fetched_at.isoformat(),
            "fetched_at_local": fetched_at.tz_convert(TZ).isoformat(),
            "last_modified": last_modified,
            "sha256": sha,
            "last_timestamp_local": pd.Timestamp(last).tz_convert(TZ).isoformat(),
            "days_behind": behind,
            "changed": changed,
            "snapshot": name,
        },
    )
    status = f"new snapshot {name}" if changed else "unchanged since the last snapshot"
    print(f"control area balance {status}: data until {pd.Timestamp(last).tz_convert(TZ)}, "
          f"{behind} day(s) behind (Last-Modified: {last_modified})")


if __name__ == "__main__":
    main()
