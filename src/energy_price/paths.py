"""Paths of the repository, defined once so that modules do not count parent folders."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
META_DIR = DATA_DIR / "meta"
EXTERNAL_DIR = DATA_DIR / "external"

REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = ROOT / "docs" / "figures"
ENV_FILE = ROOT / ".env"
