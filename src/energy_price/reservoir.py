"""Weekly Swiss reservoir energy from the official BFE CSV.

Keep every downloaded version: BFE can revise historical values. Weekly reports
use Monday dates; reference_date denotes the preceding Sunday reported as 'Sunday 24h'.
Historical Energy-Charts JSON snapshots remain readable.
Availability is established by the snapshot's fetch time, never the chart date.

Download: python -m energy_price.reservoir
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from energy_price.availability import issue_time
from energy_price.fetch_swissgrid import append_row

TZ = "Europe/Zurich"
BFE_URL = "https://www.bfe-ogd.ch/ogd17/ogd17_fuellungsgrad_speicherseen.csv"
BFE_REGIONS = {"valais": "Wallis", "grisons": "Graubuenden", "ticino": "Tessin",
               "other_switzerland": "UebrigCH", "switzerland": "TotalCH"}
MANIFEST_COLUMNS = [
    "file", "year", "url", "fetched_at_utc", "last_modified", "sha256",
    "weeks", "first_timestamp_utc", "last_timestamp_utc",
]
FETCH_LOG_COLUMNS = ["year", "url", "fetched_at_utc", "last_modified", "sha256", "changed", "snapshot"]
REGIONS = {
    "Wallis": "valais",
    "Grisons": "grisons",
    "Ticino": "ticino",
    "Other Switzerland": "other_switzerland",
}
SERIES = {
    **{name: (region, "stored_twh") for name, region in REGIONS.items()},
    **{f"{name} max. storage": (region, "capacity_twh") for name, region in REGIONS.items()
       if name != "Ticino"},
    "Tessin max. storage": ("ticino", "capacity_twh"),
    "Total Switzerland max. storage": ("switzerland", "capacity_twh"),
}
OUTPUT_COLUMNS = [
    "timestamp_utc", "timestamp_local", "reference_date", "region",
    "stored_twh", "capacity_twh", "filling_pct",
]


class ReservoirFormatError(ValueError):
    """The response does not match the expected Swiss weekly reservoir data."""


def download() -> tuple[bytes, str]:
    """Download BFE's all-years CSV; parse_bfe_csv selects the requested year."""
    request = urllib.request.Request(
        BFE_URL,
        headers={"User-Agent": "Mozilla/5.0 (energy-price research project)"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read(), response.headers.get("Last-Modified", "")


def fetch_year(year: int, out: Path) -> Path:
    """Validate and archive changed data; log every successful download."""
    content, last_modified = download()
    fetched = pd.Timestamp.now(tz="UTC")
    frame = parse_bfe_csv(content, year)
    sha = hashlib.sha256(content).hexdigest()
    files = sorted(out.glob(f"reservoirs-{year}_*.json"))
    previous = json.loads(files[-1].read_text(encoding="utf-8")) if files else None
    changed = previous is None or previous["sha256"] != sha
    status = "first snapshot"
    if previous is not None:
        old = read_snapshot(files[-1])[OUTPUT_COLUMNS].reset_index(drop=True)
        current = frame.reset_index(drop=True)
        added = len(set(current["reference_date"]) - set(old["reference_date"]))
        if added:
            status = f"{added} new weekly report(s)"
        else:
            try:
                pd.testing.assert_frame_equal(current, old, check_dtype=False, rtol=1e-12, atol=1e-12)
                status = "weekly values unchanged"
            except AssertionError:
                status = "weekly values revised"
    out.mkdir(parents=True, exist_ok=True)
    if changed:
        path = out / f"reservoirs-{year}_{fetched.strftime('%Y-%m-%dT%H-%M-%S-%fZ')}.json"
        metadata = dict(year=year, url=BFE_URL, fetched_at_utc=fetched.isoformat(),
                        last_modified=last_modified, sha256=sha, source_format="bfe_csv")
        tmp = path.with_suffix(".json.tmp")
        try:
            # Preserve the original CSV as text, including its original line endings.
            tmp.write_text(json.dumps({**metadata, "data": content.decode("utf-8")}, ensure_ascii=False) + "\n",
                           encoding="utf-8")
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)
        append_row(out / "manifest.csv", MANIFEST_COLUMNS, {
            **{key: metadata[key] for key in MANIFEST_COLUMNS if key in metadata},
            "file": path.name, "weeks": frame["timestamp_utc"].nunique(),
            "first_timestamp_utc": frame["timestamp_utc"].min().isoformat(),
            "last_timestamp_utc": frame["timestamp_utc"].max().isoformat(),
        })
    else:
        path = files[-1]
    append_row(out / "fetch_log.csv", FETCH_LOG_COLUMNS, {
        "year": year, "url": BFE_URL, "fetched_at_utc": fetched.isoformat(),
        "last_modified": last_modified, "sha256": sha, "changed": changed, "snapshot": path.name,
    })
    last_reference = frame["reference_date"].max().date()
    age = (fetched.tz_convert(TZ).date() - last_reference).days
    print(f"reservoirs {year}: {'new snapshot' if changed else 'unchanged'} (BFE; {status}), "
          f"{frame['timestamp_utc'].nunique()} weeks, last reference Sunday {last_reference}, "
          f"reference age {age} calendar days")
    return path


def parse_payload(payload: list[dict]) -> pd.DataFrame:
    """Validate chart series and derive Swiss totals without summing capacity lines."""
    try:
        if not isinstance(payload, list) or not payload:
            raise ValueError("expected a non-empty list of chart series")
        meta = payload[0]
        if meta["format"] != "Highcharts" or meta["timeZone"] != "UTC":
            raise ValueError("expected Highcharts format with UTC timestamps")
        if meta["y0AxisLabel"][0]["en"] != "Energy (TWh)":
            raise ValueError("expected Energy (TWh), refusing an unknown unit")
        stamps = pd.to_datetime(meta["xAxisValues"], unit="ms", utc=True)
        values = {}
        for series in payload:
            name = series["name"][0]["en"]
            if name not in SERIES or name in values:
                raise ValueError(f"unknown or duplicate series {name!r}")
            if len(series["data"]) != len(stamps):
                raise ValueError(f"{name}: data and timestamps have different lengths")
            if "xAxisValues" in series and series["xAxisValues"] != meta["xAxisValues"]:
                raise ValueError(f"{name}: timestamps do not match the shared axis")
            numbers = pd.to_numeric(pd.Series(series["data"], dtype="object"), errors="raise").astype(float)
            values[name] = numbers
        if set(values) != set(SERIES):
            raise ValueError(f"missing series {sorted(set(SERIES) - set(values))}")
        return _build_frame(stamps, values)
    except (KeyError, IndexError, TypeError, ValueError, OverflowError) as exc:
        raise ReservoirFormatError(f"invalid reservoir data: {exc}") from exc


def _build_frame(stamps: pd.DatetimeIndex, values: dict[str, pd.Series]) -> pd.DataFrame:
    """Common regional calculations and plausibility checks for both source formats."""
    if not len(stamps) or stamps.isna().any() or stamps.duplicated().any():
        raise ValueError("empty, invalid or duplicate timestamps")
    if not stamps.is_monotonic_increasing or not (stamps.dayofweek == 0).all():
        raise ValueError("expected increasing Monday timestamps for Sunday 24h")
    for numbers in values.values():
        if np.isinf(numbers).any() or (numbers < 0).any():
            raise ValueError("negative or infinite energy")
    frames = []
    for region in [*REGIONS.values(), "switzerland"]:
        columns = {column: values[name] for name, (r, column) in SERIES.items() if r == region}
        if region == "switzerland":
            columns["stored_twh"] = pd.concat([values[name] for name in REGIONS], axis=1).sum(
                axis=1, min_count=4
            )
            regional_capacity = pd.concat([
                values[name] for name, (r, col) in SERIES.items()
                if r != "switzerland" and col == "capacity_twh"
            ], axis=1).sum(axis=1, min_count=4)
            if ((regional_capacity - columns["capacity_twh"]).abs() > 0.002).any():
                raise ValueError("total capacity differs from regional capacities")
        frame = pd.DataFrame(columns)
        if (frame["capacity_twh"] <= 0).any():
            raise ValueError(f"{region}: capacity must be positive")
        if (frame["stored_twh"] > frame["capacity_twh"] + 0.002).any():
            raise ValueError(f"{region}: stored energy exceeds capacity")
        frame["filling_pct"] = 100 * frame["stored_twh"] / frame["capacity_twh"]
        frame["region"] = region
        frame["timestamp_utc"] = stamps
        frame["timestamp_local"] = stamps.tz_convert(TZ)
        frame["reference_date"] = (stamps.tz_localize(None).normalize() - pd.Timedelta(days=1))
        frames.append(frame[OUTPUT_COLUMNS])
    return pd.concat(frames, ignore_index=True).sort_values(["timestamp_utc", "region"], ignore_index=True)


def parse_bfe_csv(content: bytes, year: int) -> pd.DataFrame:
    """Read real weekly BFE reports, convert GWh to TWh and select one year."""
    try:
        raw = pd.read_csv(io.BytesIO(content), encoding="utf-8-sig")
        columns = {name: f"{BFE_REGIONS[region]}_{'max_' if column == 'capacity_twh' else ''}speicherinhalt_gwh"
                   for name, (region, column) in SERIES.items()}
        required = {"Datum", "TotalCH_speicherinhalt_gwh", *columns.values()}
        missing = required - set(raw.columns)
        if missing:
            raise ValueError(f"missing BFE columns: {sorted(missing)}")
        dates = pd.to_datetime(raw["Datum"], format="%Y-%m-%d", utc=True)
        energy_columns = [f"{prefix}_speicherinhalt_gwh" for prefix in BFE_REGIONS.values()]
        # Ignore prefilled calendar rows without any reported energy values.
        selected = (dates.dt.year == year) & raw[energy_columns].notna().any(axis=1)
        raw = raw.loc[selected].reset_index(drop=True)
        stamps = pd.DatetimeIndex(dates.loc[selected])
        if raw.empty:
            raise ValueError(f"no BFE reports for requested year {year}")
        values = {name: pd.to_numeric(raw[column], errors="raise").astype(float) / 1000
                  for name, column in columns.items()}
        frame = _build_frame(stamps, values)
        total = pd.to_numeric(raw["TotalCH_speicherinhalt_gwh"], errors="raise").astype(float) / 1000
        if np.isinf(total).any() or (total < 0).any():
            raise ValueError("negative or infinite Swiss total")
        computed = frame.loc[frame["region"] == "switzerland", "stored_twh"].reset_index(drop=True)
        if ((total - computed).abs() > 0.002).any():
            raise ValueError("Swiss total energy differs from regional sum")
        return frame
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ReservoirFormatError(f"invalid BFE reservoir CSV: {exc}") from exc


def read_snapshot(path: str | Path) -> pd.DataFrame:
    """Read a raw snapshot envelope, including the time this version was observed."""
    path = Path(path)
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        fetched = pd.Timestamp(record["fetched_at_utc"])
        if fetched.tzinfo is None or pd.isna(fetched):
            raise ValueError("fetched_at_utc must be timezone-aware")
        year = int(record["year"])
        source_format = record.get("source_format", "energy_charts_json")
        if source_format == "bfe_csv":
            if not isinstance(record["data"], str):
                raise ValueError("BFE snapshot data must contain CSV text")
            frame = parse_bfe_csv(record["data"].encode("utf-8"), year)
        elif source_format == "energy_charts_json":
            frame = parse_payload(record["data"])
        else:
            raise ValueError(f"unknown source format {source_format!r}")
        if not (frame["timestamp_utc"].dt.year == year).all():
            raise ValueError("timestamps do not match the requested year")
        return frame.assign(fetched_at_utc=fetched.tz_convert("UTC"), source_year=year, snapshot=path.name)
    except (KeyError, TypeError, ValueError) as exc:
        raise ReservoirFormatError(f"{path.name}: {exc}") from exc


def load_reservoirs(snapshot_dir: str | Path) -> pd.DataFrame:
    """All versions, so a backtest can select the version available at its issue time."""
    files = sorted(Path(snapshot_dir).glob("reservoirs-*.json"))
    if not files:
        raise FileNotFoundError(f"no reservoir snapshots in {snapshot_dir}; run make fetch-reservoirs")
    frame = pd.concat([read_snapshot(path) for path in files], ignore_index=True)
    if frame.duplicated(["source_year", "fetched_at_utc", "timestamp_utc", "region"]).any():
        raise ReservoirFormatError("duplicate reservoir observations across snapshots")
    return frame.sort_values(["fetched_at_utc", "timestamp_utc", "region"], ignore_index=True)


def latest_reservoir_report(frame: pd.DataFrame, delivery_day: dt.date) -> pd.DataFrame:
    """Latest weekly report observed by D-1 11:00, or no rows if no snapshot existed.

    This is deliberately conservative: a fetch today cannot prove availability in
    January. No interpolation or assumed publication delay is used. The returned
    five rows can be pivoted by region and reused for all quarter hours of day D.
    """
    cutoff = issue_time(delivery_day)
    for column in ["fetched_at_utc", "timestamp_utc"]:
        if not isinstance(frame[column].dtype, pd.DatetimeTZDtype):
            raise TypeError(f"{column} must be timezone-aware")
    known = frame[(frame["fetched_at_utc"] <= cutoff) & (frame["timestamp_utc"] <= cutoff)]
    if known.empty:
        return known.copy()
    # Respect each year's newest known snapshot, including deleted/corrected observations.
    latest = known.groupby("source_year")["fetched_at_utc"].transform("max")
    known = known[known["fetched_at_utc"] == latest]
    return known[known["timestamp_utc"] == known["timestamp_utc"].max()].copy()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download and archive Swiss reservoir data.")
    parser.add_argument("--out", type=Path, default=Path("data/reservoirs/snapshots"),
                        help="snapshot directory, relative to the working directory (default: %(default)s)")
    parser.add_argument("--years", type=int, nargs="+",
                        help="years to download (otherwise: current year in Europe/Zurich)")
    args = parser.parse_args(argv)
    years = args.years or [pd.Timestamp.now(tz=TZ).year]
    for year in sorted(set(years)):
        fetch_year(year, args.out)


if __name__ == "__main__":
    main()
