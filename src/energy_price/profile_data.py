"""Automated first-pass profile (fg-data-profiling) of every processed table, one HTML report each.

Usage: ``make profile`` (runs in the separate ``.venv-profile``, see Makefile).

fg-data-profiling 4.20 requires pandas < 3, the project uses pandas 3. This module therefore runs in its
own environment and imports nothing from the rest of the package. Checked on 2026-10-05: pandas 2.3.3
reads the parquet files written by pandas 3.0.6 with identical shape, values, missing counts, time zones
and categories; only the ``str`` columns come back as ``object``.

The report shows where to look, not what follows from it. It does not replace notebooks/01_eda.ipynb.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from data_profiling import ProfileReport

PROCESSED = Path("data/processed")
META = Path("data/meta")
SINGLE_PRICE_FROM = pd.Timestamp("2026-01-01", tz="Europe/Zurich")


def tables() -> dict[str, tuple[str, pd.DataFrame]]:
    """(title, frame) per report name. Prices are limited to 2026, as agreed with the owner."""
    prices = pd.read_parquet(PROCESSED / "balance_prices.parquet")
    prices = prices[prices["timestamp_local"] >= SINGLE_PRICE_FROM].drop(columns=["long_ct_kwh", "short_ct_kwh", "regime"])
    # The snapshot name is the same in every row and says nothing about the data.
    cab = pd.read_parquet(PROCESSED / "control_area_balance.parquet").drop(columns=["snapshot"])
    weather = pd.read_parquet(PROCESSED / "weather_forecasts.parquet")
    holidays = pd.read_csv(META / "holidays_ch.csv", parse_dates=["date"])
    return {
        "balance_prices": ("Ausgleichsenergiepreis 2026", prices),
        "control_area_balance": ("Swissgrid Regelzonenbilanz 2026", cab),
        "weather_forecasts": ("ECMWF-Wetterprognosen, Lauf D-2 18 UTC", weather),
        "holidays_ch": ("Kantonale Feiertage", holidays),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    for name, (title, frame) in tables().items():
        # Pairwise interaction plots grow quadratically with the columns and add little to the overview.
        report = ProfileReport(frame, title=title, interactions={"continuous": False}, progress_bar=False)
        path = args.out / f"profile_{name}.html"
        report.to_file(path)
        print(f"{path}: {len(frame)} rows, {frame.shape[1]} columns")


if __name__ == "__main__":
    main()
