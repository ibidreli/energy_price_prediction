# energy_price_prediction
CDS1 Balance Energy Price Prediction Challenge

## Documentation

| Path | Content |
|---|---|
| [`docs/overview.md`](docs/overview.md) | Domain wiki: actors, how balance energy prices are formed, open questions |
| [`docs/data.md`](docs/data.md) | Data sources, leakage rules, choice of the six weather sites, figures |
| [`docs/project_management.md`](docs/project_management.md) | Kanban workflow, WIP limits, Definition of Ready/Done, labels, rituals |
| [`docs/meetings/`](docs/meetings/) | Meeting notes, one file per meeting (`YYYY-MM-DD_<who>.md`) |

## Setup

Requires Python 3.14 and `make`.

```bash
make venv        # create .venv with the exact versions from requirements.lock
make data        # build all tables in data/processed/ from the raw files
make fetch       # download new Swissgrid and weather data (network)
make test        # run the unit tests
make help        # list all targets
```

`make data` only rebuilds when a raw file was added, removed or changed.

**Dependencies.** `requirements.lock` pins every package version and records the Python version it was created with. `make venv` installs exactly these versions and stops with a hint if your Python version differs, for example `make venv PYTHON=python3.14`. To add or update a package: change `pyproject.toml`, install it into `.venv`, then run `make lock` and commit the new lock file. The code itself also runs on Python 3.11 with pandas 2.2.

## Data

| Folder | Content |
|---|---|
| `data/ausgleichpreis/<year>/` | Raw Swissgrid balance energy prices, monthly XML and XLSX, unchanged as downloaded |
| `data/control_area_balance/snapshots/` | Swissgrid control area balance, every downloaded version plus `manifest.csv` (`make fetch-cab`) |
| `data/weather/ecmwf_ifs/` | Archived ECMWF weather forecasts for six sites, one JSON per model run (`make fetch-weather`) |
| `data/meta/` | Weather sites, PV capacity per canton, cantonal holidays (`make sites`, `make holidays`) |
| `data/external/` | Large third-party downloads (BFE plant register), not in git |
| `data/processed/` | Generated, not in git. Rebuild with `make data` |

See [`docs/data.md`](docs/data.md) for what each source means and why it is available at forecast time.

`balance_prices.parquet` has one row per quarter hour:

| Column | Meaning |
|---|---|
| `timestamp_utc`, `timestamp_local` | Start of the quarter hour, UTC and Europe/Zurich |
| `aep_ct_kwh` | Single balance energy price `BG-AEP` in ct/kWh. Valid since 2026, published in parallel from 2025-07 |
| `long_ct_kwh`, `short_ct_kwh` | Two-price system `BG-long` / `BG-short` until 2025-12-31 |
| `regime` | `two_price` before 2026-01-01, `single_price` from then on |
