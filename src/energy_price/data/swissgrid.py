"""Read Swissgrid balance energy price files (monthly XML) into one quarter-hourly table.

Until 2025-12-31 Swissgrid published a two-price system (``BG-long``, ``BG-short``).
Since 2026-01-01 a single price applies (``BG-AEP``, GBGR v3.2 section 7.1).
``BG-AEP`` already appears from 2025-07 onwards, published alongside the two prices.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

TZ = "Europe/Zurich"
UNIT = "ct/kWh"
QUARTER_HOUR = pd.Timedelta(minutes=15)
SINGLE_PRICE_START = pd.Timestamp("2026-01-01", tz=TZ)

SERIES_COLUMNS = {
    "BG-AEP": "aep_ct_kwh",
    "BG-long": "long_ct_kwh",
    "BG-short": "short_ct_kwh",
}
REQUIRED_BY_REGIME = {
    "two_price": ["long_ct_kwh", "short_ct_kwh"],
    "single_price": ["aep_ct_kwh"],
}
OUTPUT_COLUMNS = ["timestamp_utc", "timestamp_local", *SERIES_COLUMNS.values(), "regime"]

# Without an offset a timestamp would silently be read as UTC and shift the data by one or two hours.
_HAS_OFFSET = re.compile(r"(Z|[+-]\d{2}:?\d{2})$")


class SwissgridFormatError(ValueError):
    """A raw file or the combined data does not match the expected Swissgrid format."""


def read_price_file(path: str | Path) -> pd.DataFrame:
    """Read one Swissgrid XML file into long format.

    Returns columns ``timestamp_utc`` (start of the quarter hour, UTC),
    ``series`` (``BG-AEP``, ``BG-long`` or ``BG-short``), ``value_ct_kwh`` and ``source_file``.
    """
    path = Path(path)
    root = ET.parse(path).getroot()

    frames = []
    for ts in root.iter("timeSeries"):
        name = ts.findtext("Report_TS_Titel_de")
        if name not in SERIES_COLUMNS:
            raise SwissgridFormatError(f"{path.name}: unknown series {name!r}")
        unit = ts.findtext("unit")
        if unit != UNIT:
            raise SwissgridFormatError(f"{path.name}: series {name} has unit {unit!r}, expected {UNIT!r}")
        data = ts.find("timeSeriesData")
        if data is None or len(data) == 0:
            raise SwissgridFormatError(f"{path.name}: series {name} has no data")

        times = [(dv.findtext("TIME") or "").strip() for dv in data]
        values = [(dv.findtext("VALUE") or "").strip() for dv in data]
        if "" in times or "" in values:
            raise SwissgridFormatError(f"{path.name}: series {name} has empty TIME or VALUE fields")
        naive = next((t for t in times if not _HAS_OFFSET.search(t)), None)
        if naive is not None:
            raise SwissgridFormatError(f"{path.name}: series {name} has a TIME without UTC offset: {naive!r}")
        try:
            timestamps = pd.to_datetime(times, format="ISO8601", utc=True)
            numbers = pd.to_numeric(pd.Series(values), errors="raise").astype("float64")
        except (ValueError, TypeError) as exc:
            raise SwissgridFormatError(f"{path.name}: series {name} has an unreadable row: {exc}") from exc
        if not np.isfinite(numbers).all():
            raise SwissgridFormatError(f"{path.name}: series {name} has non-finite values")

        frames.append(
            pd.DataFrame(
                {
                    "timestamp_utc": timestamps,
                    "series": name,
                    "value_ct_kwh": numbers.to_numpy(),
                    "source_file": path.name,
                }
            )
        )

    if not frames:
        raise SwissgridFormatError(f"{path.name}: no timeSeries found")

    result = pd.concat(frames, ignore_index=True)
    off_grid = result["timestamp_utc"] != result["timestamp_utc"].dt.floor("15min")
    if off_grid.any():
        first = result.loc[off_grid, "timestamp_utc"].iloc[0]
        raise SwissgridFormatError(f"{path.name}: {off_grid.sum()} timestamps not on the 15-minute grid, e.g. {first}")
    return result


def load_prices(raw_dir: str | Path) -> pd.DataFrame:
    """Read all XML files below ``raw_dir`` into one row per quarter hour.

    Raises ``SwissgridFormatError`` unless the data covers complete calendar months without
    duplicates or gaps, every quarter hour has the series its regime requires, and ``BG-AEP``
    has no holes once it has started.
    """
    files = sorted(Path(raw_dir).rglob("*.xml"))
    if not files:
        raise FileNotFoundError(f"no XML files found below {raw_dir}")

    long = pd.concat([read_price_file(f) for f in files], ignore_index=True)

    duplicated = long.duplicated(["timestamp_utc", "series"], keep=False)
    if duplicated.any():
        sources = sorted(long.loc[duplicated, "source_file"].unique())
        raise SwissgridFormatError(f"{duplicated.sum()} duplicate quarter hours in: {', '.join(sources)}")

    wide = (
        long.pivot(index="timestamp_utc", columns="series", values="value_ct_kwh")
        .reindex(columns=list(SERIES_COLUMNS))
        .rename(columns=SERIES_COLUMNS)
        .sort_index()
    )
    wide.columns.name = None

    _check_complete_months(wide.index)

    wide = wide.reset_index()
    wide["timestamp_local"] = wide["timestamp_utc"].dt.tz_convert(TZ)
    wide["regime"] = pd.Categorical(
        np.where(wide["timestamp_utc"] >= SINGLE_PRICE_START, "single_price", "two_price"),
        categories=list(REQUIRED_BY_REGIME),
    )

    for regime, columns in REQUIRED_BY_REGIME.items():
        rows = wide["regime"] == regime
        for column in columns:
            gaps = rows & wide[column].isna()
            if gaps.any():
                first = wide.loc[gaps, "timestamp_local"].iloc[0]
                raise SwissgridFormatError(f"{gaps.sum()} {regime} quarter hours without {column}, first {first}")

    aep_started = wide["aep_ct_kwh"].notna().cummax()
    aep_holes = aep_started & wide["aep_ct_kwh"].isna()
    if aep_holes.any():
        first = wide.loc[aep_holes, "timestamp_local"].iloc[0]
        raise SwissgridFormatError(f"{aep_holes.sum()} quarter hours without aep_ct_kwh after it started, first {first}")

    return wide[OUTPUT_COLUMNS]


def _check_complete_months(index: pd.DatetimeIndex) -> None:
    """Require a gap-free grid from local 00:00 on the 1st of a month to the end of a month."""
    start = index[0].tz_convert(TZ)
    end = (index[-1] + QUARTER_HOUR).tz_convert(TZ)
    for label, stamp in (("starts", start), ("ends", end)):
        if stamp != stamp.normalize() or stamp.day != 1:
            raise SwissgridFormatError(f"data {label} at {stamp}, expected local 00:00 on the 1st of a month")

    expected = pd.date_range(index[0], index[-1], freq=QUARTER_HOUR)
    missing = expected.difference(index)
    if len(missing) > 0:
        raise SwissgridFormatError(f"{len(missing)} quarter hours missing, first {missing[0]}, last {missing[-1]}")
