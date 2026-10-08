"""Weekly Swiss reservoir energy from Energy-Charts (upstream source: BFE).

Keep every downloaded version: BFE can revise historical values. Chart timestamps
are retained verbatim; reference_date denotes the Sunday reported as 'Sunday 24h'.
Availability is established by the snapshot's fetch time, never the chart date.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

from energy_price.availability import issue_time

TZ = "Europe/Zurich"
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
    """The JSON does not match the expected Swiss weekly reservoir data."""


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
        if not len(stamps) or stamps.isna().any() or stamps.duplicated().any():
            raise ValueError("empty, invalid or duplicate timestamps")
        if not stamps.is_monotonic_increasing or not (stamps.dayofweek == 0).all():
            raise ValueError("expected increasing Monday timestamps for Sunday 24h")
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
            if np.isinf(numbers).any() or (numbers < 0).any():
                raise ValueError(f"{name}: negative or infinite energy")
            values[name] = numbers
        if set(values) != set(SERIES):
            raise ValueError(f"missing series {sorted(set(SERIES) - set(values))}")
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
    except (KeyError, IndexError, TypeError, ValueError, OverflowError) as exc:
        raise ReservoirFormatError(f"invalid reservoir data: {exc}") from exc


def read_snapshot(path: str | Path) -> pd.DataFrame:
    """Read a raw snapshot envelope, including the time this version was observed."""
    path = Path(path)
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        fetched = pd.Timestamp(record["fetched_at_utc"])
        if fetched.tzinfo is None or pd.isna(fetched):
            raise ValueError("fetched_at_utc must be timezone-aware")
        frame = parse_payload(record["data"])
        year = int(record["year"])
        if not (frame["timestamp_utc"].dt.year == year).all():
            raise ValueError("timestamps do not match the requested year")
        return frame.assign(fetched_at_utc=fetched.tz_convert("UTC"), source_year=year, snapshot=path.name)
    except (KeyError, TypeError, ValueError) as exc:
        raise ReservoirFormatError(f"{path.name}: {exc}") from exc


def load_reservoirs(snapshot_dir: str | Path) -> pd.DataFrame:
    """All versions, so a backtest can select the version available at its issue time."""
    files = sorted(Path(snapshot_dir).glob("filling-level-*.json"))
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
