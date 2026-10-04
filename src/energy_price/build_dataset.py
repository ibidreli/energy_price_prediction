"""Build ``data/processed/balance_prices.parquet`` from the raw Swissgrid XML files.

Usage: ``python -m energy_price.build_dataset --raw data/ausgleichpreis --out data/processed/balance_prices.parquet``
"""

from __future__ import annotations

import argparse
from pathlib import Path

from energy_price.swissgrid import load_prices


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", type=Path, required=True, help="directory with the Swissgrid XML files")
    parser.add_argument("--out", type=Path, required=True, help="parquet file to write")
    args = parser.parse_args(argv)

    prices = load_prices(args.raw)

    # Write to a temporary file first so an aborted run never leaves a half-written dataset.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".parquet.tmp")
    try:
        prices.to_parquet(tmp, index=False)
        tmp.replace(args.out)
    finally:
        tmp.unlink(missing_ok=True)

    first = prices["timestamp_local"].iloc[0]
    last = prices["timestamp_local"].iloc[-1]
    print(f"wrote {args.out}: {len(prices)} quarter hours, {first} to {last}")
    for regime, count in prices["regime"].value_counts(sort=False).items():
        print(f"  {regime}: {count}")
    print(f"  with BG-AEP: {prices['aep_ct_kwh'].notna().sum()}")


if __name__ == "__main__":
    main()
