.DEFAULT_GOAL := help

VENV    := .venv
ifeq ($(OS),Windows_NT)
PYTHON  ?= python
PY      := $(VENV)/Scripts/python.exe
GIT_BASH ?= $(subst \,/,$(ProgramFiles))/Git/bin/bash.exe
SHELL   := $(GIT_BASH)
.SHELLFLAGS := -c
export PATH := $(subst /bin/bash.exe,/usr/bin,$(GIT_BASH));$(PATH)
ENV_READY := check-venv
else
PYTHON  ?= python3
PY      := $(VENV)/bin/python
ENV_READY := $(VENV)/.installed
endif
LOCK    := requirements.lock
RAW_DIR := data/ausgleichpreis
OUT_DIR := data/processed
DATASET := $(OUT_DIR)/balance_prices.parquet

RAW_STAMP := $(OUT_DIR)/.raw_checksums
SOURCES   := $(shell find src -name '*.py' 2>/dev/null)

# Additional sources (see docs/data.md)
CAB_DIR      := data/control_area_balance/snapshots
WEATHER_DIR  := data/weather/ecmwf_ifs
WEATHER_FROM := 2026-01-01
META_DIR     := data/meta
SITES        := $(META_DIR)/weather_sites.csv
HOLIDAYS     := $(META_DIR)/holidays_ch.csv
REGISTER     := data/external/bfe_anlagen.zip
REGISTER_URL := https://data.geo.admin.ch/ch.bfe.elektrizitaetsproduktionsanlagen/csv/2056/ch.bfe.elektrizitaetsproduktionsanlagen.zip
CAB_DATA     := $(OUT_DIR)/control_area_balance.parquet
WEATHER_DATA := $(OUT_DIR)/weather_forecasts.parquet
RESERVOIR_DIR := data/reservoirs/snapshots
RESERVOIR_YEARS ?=
RESERVOIR_DATA := $(OUT_DIR)/reservoir_filling.parquet
# Python version the lock file was created with, read from its first line "# python X.Y".
LOCK_PY   := $(shell sed -n '1s/^\# python //p' $(LOCK) 2>/dev/null)

.PHONY: help venv check-venv lock data fetch fetch-cab fetch-weather fetch-reservoirs sites holidays figures figures-reservoirs test test-all tracking-check baseline baseline-tune clean-data

help: ## Show available targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "  make %-14s %s\n", $$1, $$2}'

venv: $(VENV)/.installed ## Create .venv with the exact versions from requirements.lock

# Windows run targets use the existing supported Python environment.
# The explicit `make venv` target retains the Python 3.14 lock workflow.
check-venv:
	@test -x $(PY) || { echo "no $(VENV), create it as described in README.md"; exit 1; }
	@$(PY) -c 'import energy_price, pandas, pyarrow, holidays'

# requirements.lock pins exact versions as constraints. Packages it does not list,
# e.g. platform-specific ones or a dependency newly added to pyproject.toml, are resolved normally.
$(VENV)/.installed: pyproject.toml $(wildcard $(LOCK))
	test -d $(VENV) || $(PYTHON) -m venv $(VENV)
	@$(PY) -c 'import sys; have = "%d.%d" % sys.version_info[:2]; want = "$(LOCK_PY)"; \
		sys.exit(f"$(VENV) uses Python {have}, but $(LOCK) was created with Python {want}.\nDelete $(VENV) and run: make venv PYTHON=python{want}") if want and have != want else None'
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install $(if $(wildcard $(LOCK)),-c $(LOCK)) -e ".[notebooks,dev,tracking]"
	touch $@

lock: ## Write the package versions installed in .venv to requirements.lock
	@test -x $(PY) || { echo "no $(VENV), run make venv first"; exit 1; }
	@$(PY) -c 'import sys; print("# python %d.%d" % sys.version_info[:2])' > $(LOCK).tmp
	$(PY) -m pip freeze --exclude-editable >> $(LOCK).tmp
	@mv $(LOCK).tmp $(LOCK)

data: $(DATASET) $(CAB_DATA) $(WEATHER_DATA) $(RESERVOIR_DATA) ## Build all processed tables (only if inputs changed)

fetch: fetch-cab fetch-weather fetch-reservoirs ## Download new Swissgrid, weather and reservoir data (network)

fetch-reservoirs: $(ENV_READY) ## Fetch official BFE reservoir data and report new weeks or revisions
	$(PY) -m energy_price.reservoir --out $(RESERVOIR_DIR) $(if $(strip $(RESERVOIR_YEARS)),--years $(RESERVOIR_YEARS))

fetch-cab: $(ENV_READY) ## Store a new Swissgrid control area balance snapshot if it changed
	$(PY) -m energy_price.fetch_swissgrid --out $(CAB_DIR)

fetch-weather: $(ENV_READY) ## Download weather forecast runs that are not stored yet
	$(PY) -m energy_price.fetch_weather --sites $(SITES) --out $(WEATHER_DIR) --start $(WEATHER_FROM)

$(REGISTER):
	mkdir -p $(dir $@)
	curl -sSfL -o $@.tmp $(REGISTER_URL) && mv $@.tmp $@

sites: $(REGISTER) $(ENV_READY) ## Choose the weather sites from the BFE photovoltaic register
	$(PY) -m energy_price.pv_sites --register $(REGISTER) --out $(META_DIR)

holidays: $(ENV_READY) ## Write the legal holidays of all cantons
	$(PY) -m energy_price.holidays_ch --years 2026 2027 --out $(HOLIDAYS)

# Checksum stamps as for the prices: rebuild when a file is added, removed or its content changes.
# A changed stamp also deletes the derived table: make 3.81 compares whole seconds, so a change in
# the same second as the last build would otherwise go unnoticed and leave the table stale for good.
CAB_STAMP     := $(OUT_DIR)/.cab_checksums
WEATHER_STAMP := $(OUT_DIR)/.weather_checksums
RESERVOIR_STAMP := $(OUT_DIR)/.reservoir_checksums

$(RESERVOIR_STAMP): FORCE | $(ENV_READY)
	@mkdir -p $(OUT_DIR)
	@$(PY) -m energy_price.checksums $(RESERVOIR_DIR) --pattern 'reservoirs-*.json' > $@.tmp
	@if cmp -s $@.tmp $@; then rm $@.tmp; else mv $@.tmp $@; rm -f $(RESERVOIR_DATA); fi

$(RESERVOIR_DATA): $(RESERVOIR_STAMP) $(SOURCES) | $(ENV_READY)
	$(PY) -m energy_price.build_sources reservoirs --src $(RESERVOIR_DIR) --out $@

$(CAB_STAMP): FORCE | $(ENV_READY)
	@mkdir -p $(OUT_DIR)
	@$(PY) -m energy_price.checksums $(CAB_DIR) --pattern 'control-area-balance-*.csv' > $@.tmp
	@if cmp -s $@.tmp $@; then rm $@.tmp; else mv $@.tmp $@; rm -f $(CAB_DATA); fi

$(WEATHER_STAMP): FORCE | $(ENV_READY)
	@mkdir -p $(OUT_DIR)
	@$(PY) -m energy_price.checksums $(WEATHER_DIR) --pattern 'run_*.json' > $@.tmp
	@if cmp -s $@.tmp $@; then rm $@.tmp; else mv $@.tmp $@; rm -f $(WEATHER_DATA); fi

$(CAB_DATA): $(CAB_STAMP) $(SOURCES) | $(ENV_READY)
	$(PY) -m energy_price.build_sources cab --src $(CAB_DIR) --out $@

$(WEATHER_DATA): $(WEATHER_STAMP) $(SOURCES) | $(ENV_READY)
	$(PY) -m energy_price.build_sources weather --src $(WEATHER_DIR) --out $@

figures: data $(REGISTER) ## Render the figures of docs/data.md
	$(PY) -m energy_price.figures_data --out docs/figures

figures-reservoirs: $(RESERVOIR_DATA) ## Render only the reservoir figures
	$(PY) -m energy_price.figures_data --out docs/figures --only-reservoirs

# Checksums of all raw files. Rewritten only when a file is added, removed or its content
# changes, so a rebuild never depends on file modification times (cp -p, unzip, git checkout).
$(RAW_STAMP): FORCE | $(ENV_READY)
	@mkdir -p $(OUT_DIR)
	@$(PY) -m energy_price.checksums $(RAW_DIR) > $@.tmp
	@if cmp -s $@.tmp $@; then rm $@.tmp; else mv $@.tmp $@; rm -f $(DATASET); fi

$(DATASET): $(RAW_STAMP) $(SOURCES) | $(ENV_READY)
	$(PY) -m energy_price.build_dataset --raw $(RAW_DIR) --out $@

test: $(ENV_READY) ## Run unit tests
	$(PY) -m pytest -m "not realdata"

test-all: $(ENV_READY) ## Run unit tests and checks against the real data
	$(PY) -m pytest

tracking-check: $(ENV_READY) ## Check the W&B login and team entity (network, see docs/experiment_tracking.md)
	$(PY) -m energy_price.tracking

baseline-tune: data ## Choose the baseline settings on April and May (docs/baseline.md)
	$(PY) -m energy_price.model.backtest tune

baseline: data ## Backtest reference and baseline on June to August (docs/baseline.md)
	$(PY) -m energy_price.model.backtest test $(if $(TRACK),--track)

clean-data: ## Delete the processed data (raw data stays untouched)
	rm -rf $(OUT_DIR)

FORCE:
