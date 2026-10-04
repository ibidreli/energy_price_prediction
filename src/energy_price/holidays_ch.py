"""Write the legal public holidays of all 26 Swiss cantons to a CSV file.

Holidays are fixed by cantonal law and known years in advance, so they are available at any
forecast time. Only the category "public" of the python-holidays package is used; customary
days off without legal status (for example Berchtoldstag in Zurich) are not included.

Usage: ``python -m energy_price.holidays_ch --years 2026 2027 --out data/meta/holidays_ch.csv``
"""

from __future__ import annotations

import argparse
from pathlib import Path

import holidays
import pandas as pd

CANTONS = [
    "AG", "AI", "AR", "BE", "BL", "BS", "FR", "GE", "GL", "GR", "JU", "LU", "NE",
    "NW", "OW", "SG", "SH", "SO", "SZ", "TG", "TI", "UR", "VD", "VS", "ZG", "ZH",
]  # fmt: skip


def cantonal_holidays(years: list[int]) -> pd.DataFrame:
    """One row per canton and legal holiday."""
    source = f"python-holidays {holidays.__version__}, category public"
    rows = [
        {"date": day, "canton": canton, "name": name, "source": source}
        for canton in CANTONS
        for day, name in holidays.country_holidays("CH", subdiv=canton, years=years, language="de").items()
    ]
    return pd.DataFrame(rows).sort_values(["date", "canton"], ignore_index=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--years", type=int, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    table = cantonal_holidays(args.years)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(table)} rows, {table['date'].nunique()} distinct days")


if __name__ == "__main__":
    main()
