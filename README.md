# energy_price_prediction
CDS1 Balance Energy Price Prediction Challenge

## Documentation

| Path | Content |
|---|---|
| [`docs/overview.md`](docs/overview.md) | Domain wiki: actors, how balance energy prices are formed, open questions |
| [`docs/data.md`](docs/data.md) | Data sources, leakage rules, choice of the six weather sites, figures |
| [`docs/metric.md`](docs/metric.md) | Evaluation metric (MAE, pinball loss, coverage) and baseline |
| [`docs/project_management.md`](docs/project_management.md) | Kanban workflow, WIP limits, Definition of Ready/Done, labels, rituals |
| [`docs/experiment_tracking.md`](docs/experiment_tracking.md) | W&B setup per person, run conventions, what never goes online |
| [`docs/meetings/`](docs/meetings/) | Meeting notes, one file per meeting (`YYYY-MM-DD_<who>.md`) |

## Setup

The pinned setup uses Python 3.14 and GNU Make. Windows can also use an existing
Python >=3.11 environment as described below.

```bash
make venv        # create .venv with the exact versions from requirements.lock
make data        # build all tables in data/processed/ from the raw files
make fetch       # download new Swissgrid, weather and reservoir data (network)
make test        # run the unit tests
make tracking-check  # check the W&B login, after the setup in docs/experiment_tracking.md
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
| `data/reservoirs/snapshots/` | Swiss reservoir filling levels directly from the BFE CSV, versioned JSON snapshots with `manifest.csv` and `fetch_log.csv` (`make fetch-reservoirs`) |
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

`reservoir_filling.parquet` contains weekly energy content (`stored_twh`), capacity
(`capacity_twh`) and filling level (`filling_pct`) for four Swiss regions and the
Swiss total, retaining every snapshot version. Use `latest_reservoir_report` to select
the latest report actually observed by D-1 11:00. Historical publication times
are unknown; a snapshot downloaded today cannot establish availability earlier
in 2026. See [reservoir data and availability](docs/data.md#6-stauseen-bfe).

```bash
make fetch-reservoirs                  # fetch BFE weekly reports for the current year (network)
make data                              # rebuild all four processed tables
# Optional: explicitly choose years
make fetch-reservoirs RESERVOIR_YEARS="2025 2026"
```

### Windows / PowerShell

Install Git for Windows (its Bash shell supplies the Unix tools used by Make).
Set up the environment and install GNU Make once:

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[dev]" gnumake "matplotlib>=3.9"
```

Activate the environment in each new terminal, then use the same Make targets:

```powershell
.venv/Scripts/Activate.ps1
make fetch-reservoirs
make data
make figures-reservoirs
make test-all
```

Run from the repository root. Windows run targets check the existing Python
>=3.11 environment. The explicit `make venv` command retains the pinned Python
3.14 workflow. Git Bash is expected at `C:/Program Files/Git/bin/bash.exe`;
use `GIT_BASH="<path>"` if Git is installed elsewhere.

The reservoir download defaults to the current calendar year in Europe/Zurich
and `data/reservoirs/snapshots/`. Override with
`make fetch-reservoirs RESERVOIR_YEARS="2025 2026"` when needed.
The direct Python command remains `python -m energy_price.reservoir`, with
optional `--years` and `--out` arguments.

The download reads the official BFE CSV directly and compares its weekly values
with the last snapshot. Output distinguishes new weeks, revisions and unchanged
values, and reports the age of the latest reference Sunday. Historical
Energy-Charts snapshots remain readable. Raw BFE CSV text is stored inside the
JSON envelope with `source_format="bfe_csv"`; the CSV covers all years, while
processing selects the requested year. Failed or invalid downloads do not
replace existing snapshots.
