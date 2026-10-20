"""Read a Swissgrid control area balance snapshot into one row per quarter hour.

Values are quarter-hour averages in MW as published by Swissgrid ("indicative estimates
calculated from real-time systems"). Sign convention of the total system imbalance (TSI):
positive = long (more generation than consumption), negative = short.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

TZ = "Europe/Zurich"
QUARTER_HOUR = pd.Timedelta(minutes=15)

COLUMNS = {
    "Abgedeckte Bedarf der aFRR+": "afrr_pos_mw",
    "Abgedeckte Bedarf der aFRR-": "afrr_neg_mw",
    "Abgedeckte Bedarf der SA mFRR+": "mfrr_sa_pos_mw",
    "Abgedeckte Bedarf der SA mFRR-": "mfrr_sa_neg_mw",
    "Abgedeckte Bedarf der DA mFRR+": "mfrr_da_pos_mw",
    "Abgedeckte Bedarf der DA mFRR-": "mfrr_da_neg_mw",
    "NRV+ (Import)": "igcc_import_mw",
    "NRV- (Export)": "igcc_export_mw",
    "FRCE+ (Import)": "frce_pos_mw",
    "FRCE- (Export)": "frce_neg_mw",
    "Total System Imbalance": "tsi_mw",
    "AE-Preis": "price_ct_kwh",
}
OUTPUT_COLUMNS = ["timestamp_utc", "timestamp_local", *COLUMNS.values()]


class ControlAreaBalanceFormatError(ValueError):
    """The snapshot does not match the expected Swissgrid format."""


def read_snapshot(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    raw = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    missing = [c for c in ["Date Time [UTC]", *COLUMNS] if c not in raw.columns]
    if missing:
        raise ControlAreaBalanceFormatError(f"{path.name}: columns missing {missing}")

    try:
        stamps = pd.to_datetime(raw["Date Time [UTC]"], format="%d.%m.%Y %H:%M", utc=True)
        values = raw[list(COLUMNS)].apply(pd.to_numeric, errors="raise").rename(columns=COLUMNS)
    except (ValueError, TypeError) as exc:
        raise ControlAreaBalanceFormatError(f"{path.name}: unreadable row: {exc}") from exc

    out = values.assign(timestamp_utc=stamps).sort_values("timestamp_utc", ignore_index=True)
    if out["timestamp_utc"].duplicated().any():
        raise ControlAreaBalanceFormatError(f"{path.name}: duplicate quarter hours")
    expected = pd.date_range(out["timestamp_utc"].iloc[0], out["timestamp_utc"].iloc[-1], freq=QUARTER_HOUR)
    missing_qh = expected.difference(out["timestamp_utc"])
    if len(missing_qh) > 0:
        raise ControlAreaBalanceFormatError(f"{path.name}: {len(missing_qh)} quarter hours missing, first {missing_qh[0]}")
    if out[["tsi_mw", "price_ct_kwh"]].isna().any().any():
        raise ControlAreaBalanceFormatError(f"{path.name}: quarter hours without TSI or price")

    out["timestamp_local"] = out["timestamp_utc"].dt.tz_convert(TZ)
    return out[OUTPUT_COLUMNS]


def latest_snapshot(snapshot_dir: str | Path) -> Path:
    files = sorted(Path(snapshot_dir).glob("control-area-balance-*.csv"))
    if not files:
        raise FileNotFoundError(f"no control area balance snapshot in {snapshot_dir}")
    return files[-1]
