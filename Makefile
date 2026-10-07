.DEFAULT_GOAL := help

PYTHON  ?= python3
VENV    := .venv
PY      := $(VENV)/bin/python
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
# Python version the lock file was created with, read from its first line "# python X.Y".
LOCK_PY   := $(shell sed -n '1s/^\# python //p' $(LOCK) 2>/dev/null)

.PHONY: help venv lock data fetch fetch-cab fetch-weather sites holidays figures profile notebook test test-all clean-data

help: ## Show available targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "  make %-11s %s\n", $$1, $$2}'

venv: $(VENV)/.installed ## Create .venv with the exact versions from requirements.lock

# requirements.lock pins exact versions as constraints. Packages it does not list,
# e.g. platform-specific ones or a dependency newly added to pyproject.toml, are resolved normally.
$(VENV)/.installed: pyproject.toml $(wildcard $(LOCK))
	test -d $(VENV) || $(PYTHON) -m venv $(VENV)
	@$(PY) -c 'import sys; have = "%d.%d" % sys.version_info[:2]; want = "$(LOCK_PY)"; \
		sys.exit(f"$(VENV) uses Python {have}, but $(LOCK) was created with Python {want}.\nDelete $(VENV) and run: make venv PYTHON=python{want}") if want and have != want else None'
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install $(if $(wildcard $(LOCK)),-c $(LOCK)) -e ".[notebooks,dev]"
	touch $@

lock: ## Write the package versions installed in .venv to requirements.lock
	@test -x $(PY) || { echo "no $(VENV), run make venv first"; exit 1; }
	@$(PY) -c 'import sys; print("# python %d.%d" % sys.version_info[:2])' > $(LOCK).tmp
	$(PY) -m pip freeze --exclude-editable >> $(LOCK).tmp
	@mv $(LOCK).tmp $(LOCK)

data: $(DATASET) $(CAB_DATA) $(WEATHER_DATA) ## Build all processed tables (only if inputs changed)

fetch: fetch-cab fetch-weather ## Download new Swissgrid and weather data (network)

fetch-cab: venv ## Store a new Swissgrid control area balance snapshot if it changed
	$(PY) -m energy_price.fetch_swissgrid --out $(CAB_DIR)

fetch-weather: venv ## Download weather forecast runs that are not stored yet
	$(PY) -m energy_price.fetch_weather --sites $(SITES) --out $(WEATHER_DIR) --start $(WEATHER_FROM)

$(REGISTER):
	mkdir -p $(dir $@)
	curl -sSfL -o $@.tmp $(REGISTER_URL) && mv $@.tmp $@

sites: $(REGISTER) venv ## Choose the weather sites from the BFE photovoltaic register
	$(PY) -m energy_price.pv_sites --register $(REGISTER) --out $(META_DIR)

holidays: venv ## Write the legal holidays of all cantons
	$(PY) -m energy_price.holidays_ch --years 2026 2027 --out $(HOLIDAYS)

# Checksum stamps as for the prices: rebuild when a file is added, removed or its content changes.
# A changed stamp also deletes the derived table: make 3.81 compares whole seconds, so a change in
# the same second as the last build would otherwise go unnoticed and leave the table stale for good.
CAB_STAMP     := $(OUT_DIR)/.cab_checksums
WEATHER_STAMP := $(OUT_DIR)/.weather_checksums

$(CAB_STAMP): FORCE | $(VENV)/.installed
	@mkdir -p $(OUT_DIR)
	@$(PY) -m energy_price.checksums $(CAB_DIR) --pattern 'control-area-balance-*.csv' > $@.tmp
	@if cmp -s $@.tmp $@; then rm $@.tmp; else mv $@.tmp $@; rm -f $(CAB_DATA); fi

$(WEATHER_STAMP): FORCE | $(VENV)/.installed
	@mkdir -p $(OUT_DIR)
	@$(PY) -m energy_price.checksums $(WEATHER_DIR) --pattern 'run_*.json' > $@.tmp
	@if cmp -s $@.tmp $@; then rm $@.tmp; else mv $@.tmp $@; rm -f $(WEATHER_DATA); fi

$(CAB_DATA): $(CAB_STAMP) $(SOURCES) | $(VENV)/.installed
	$(PY) -m energy_price.build_sources cab --src $(CAB_DIR) --out $@

$(WEATHER_DATA): $(WEATHER_STAMP) $(SOURCES) | $(VENV)/.installed
	$(PY) -m energy_price.build_sources weather --src $(WEATHER_DIR) --out $@

figures: data $(REGISTER) ## Render the figures of docs/data.md
	$(PY) -m energy_price.figures_data --out docs/figures

# Checksums of all raw files. Rewritten only when a file is added, removed or its content
# changes, so a rebuild never depends on file modification times (cp -p, unzip, git checkout).
$(RAW_STAMP): FORCE | $(VENV)/.installed
	@mkdir -p $(OUT_DIR)
	@$(PY) -m energy_price.checksums $(RAW_DIR) > $@.tmp
	@if cmp -s $@.tmp $@; then rm $@.tmp; else mv $@.tmp $@; rm -f $(DATASET); fi

$(DATASET): $(RAW_STAMP) $(SOURCES) | $(VENV)/.installed
	$(PY) -m energy_price.build_dataset --raw $(RAW_DIR) --out $@

# fg-data-profiling requires pandas < 3, the project pins pandas 3. The profile therefore runs in its own
# environment, pinned by requirements-profile.lock (same first-line format as requirements.lock).
PROFILE_VENV := .venv-profile
PROFILE_PY   := $(PROFILE_VENV)/bin/python
PROFILE_LOCK := requirements-profile.lock

PROFILE_LOCK_PY := $(shell sed -n '1s/^\# python //p' $(PROFILE_LOCK) 2>/dev/null)

$(PROFILE_VENV)/.installed: $(PROFILE_LOCK)
	test -d $(PROFILE_VENV) || $(PYTHON) -m venv $(PROFILE_VENV)
	@$(PROFILE_PY) -c 'import sys; have = "%d.%d" % sys.version_info[:2]; want = "$(PROFILE_LOCK_PY)"; \
		sys.exit(f"$(PROFILE_VENV) uses Python {have}, but $(PROFILE_LOCK) was created with Python {want}.\nDelete $(PROFILE_VENV) and run: make profile PYTHON=python{want}") if want and have != want else None'
	$(PROFILE_PY) -m pip install --upgrade pip
	$(PROFILE_PY) -m pip install -r $(PROFILE_LOCK)
	touch $@

profile: data $(PROFILE_VENV)/.installed ## Automated HTML profile per table in reports/ (fg-data-profiling)
	PYTHONPATH=src $(PROFILE_PY) -m energy_price.profile_data --out reports

# The default kernel starts a bare `python`; .venv/bin comes first on PATH so that it is the one from .venv.
# JUPYTER_CONFIG_DIR skips the personal ~/.jupyter, so the run is the same on every machine.
notebook: data ## Run notebooks/01_eda.ipynb top to bottom and store the outputs in place
	PATH="$(abspath $(VENV))/bin:$$PATH" JUPYTER_CONFIG_DIR="$(abspath $(VENV))/etc/jupyter" $(PY) -m jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb

test: venv ## Run unit tests
	$(PY) -m pytest -m "not realdata"

test-all: venv ## Run unit tests and checks against the real data
	$(PY) -m pytest

clean-data: ## Delete the processed data (raw data stays untouched)
	rm -rf $(OUT_DIR)

FORCE:
