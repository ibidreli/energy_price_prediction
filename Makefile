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
# Python version the lock file was created with, read from its first line "# python X.Y".
LOCK_PY   := $(shell sed -n '1s/^\# python //p' $(LOCK) 2>/dev/null)

.PHONY: help venv lock data test test-all clean-data

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

data: $(DATASET) ## Build the processed dataset (only if raw files changed)

# Checksums of all raw files. Rewritten only when a file is added, removed or its content
# changes, so a rebuild never depends on file modification times (cp -p, unzip, git checkout).
$(RAW_STAMP): FORCE | $(VENV)/.installed
	@mkdir -p $(OUT_DIR)
	@$(PY) -m energy_price.checksums $(RAW_DIR) > $@.tmp
	@if cmp -s $@.tmp $@; then rm $@.tmp; else mv $@.tmp $@; fi

$(DATASET): $(RAW_STAMP) $(SOURCES) | $(VENV)/.installed
	$(PY) -m energy_price.build_dataset --raw $(RAW_DIR) --out $@

test: venv ## Run unit tests
	$(PY) -m pytest -m "not realdata"

test-all: venv ## Run unit tests and checks against the real data
	$(PY) -m pytest

clean-data: ## Delete the processed data (raw data stays untouched)
	rm -rf $(OUT_DIR)

FORCE:
