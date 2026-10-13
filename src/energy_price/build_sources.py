"""Build the processed tables for the additional data sources.

Usage:
``python -m energy_price.build_sources cab --src data/control_area_balance/snapshots --out data/processed/control_area_balance.parquet``
``python -m energy_price.build_sources weather --src data/weather/ecmwf_ifs --out data/processed/weather_forecasts.parquet``
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from energy_price.control_area_balance import latest_snapshot, read_snapshot
from energy_price.reservoir import load_reservoirs
from energy_price.weather import load_weather, missing_values


def write_parquet(frame: pd.DataFrame, out: Path) -> None:
    """Write via a temporary file so an aborted run never leaves a half-written table."""
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".parquet.tmp")
    try:
        frame.to_parquet(tmp, index=False)
        tmp.replace(out)
    finally:
        tmp.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("kind", choices=["cab", "weather", "reservoirs"])
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.kind == "cab":
        snapshot = latest_snapshot(args.src)
        table = read_snapshot(snapshot).assign(snapshot=snapshot.name)
        write_parquet(table, args.out)
        print(f"wrote {args.out}: {len(table)} quarter hours from {snapshot.name}")
    elif args.kind == "reservoirs":
        table = load_reservoirs(args.src)
        write_parquet(table, args.out)
        print(f"wrote {args.out}: {table['timestamp_utc'].nunique()} weeks, "
              f"{table['snapshot'].nunique()} snapshots, {len(table)} rows")
    else:
        table = load_weather(args.src)
        write_parquet(table, args.out)
        gaps = missing_values(table)
        gaps.to_csv(args.out.with_name("weather_missing.csv"), index=False)
        print(f"wrote {args.out}: {table['delivery_day'].nunique()} delivery days, {table['site'].nunique()} sites, "
              f"{len(gaps)} day/variable combinations with missing values")


if __name__ == "__main__":
    main()
