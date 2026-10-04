from pathlib import Path

import pandas as pd
import pytest

from energy_price import fetch_swissgrid
from energy_price.control_area_balance import COLUMNS, ControlAreaBalanceFormatError, latest_snapshot, read_snapshot

HEADER = ";".join(["Date Time [UTC]", *COLUMNS])


def write_csv(path: Path, stamps: list[str], tsi: float = -100.0) -> Path:
    rows = [";".join([s, *["1.5"] * (len(COLUMNS) - 2), str(tsi), "16.9"]) for s in stamps]
    path.write_text("\n".join([HEADER, *rows]) + "\n", encoding="utf-8")
    return path


def test_read_snapshot_renames_columns_and_converts_time(tmp_path):
    f = write_csv(tmp_path / "control-area-balance-2026_a.csv", ["31.12.2025 23:00", "31.12.2025 23:15"])

    df = read_snapshot(f)

    assert df["timestamp_local"].iloc[0] == pd.Timestamp("2026-01-01 00:00", tz="Europe/Zurich")
    assert df["tsi_mw"].tolist() == [-100.0, -100.0]
    assert df["price_ct_kwh"].tolist() == [16.9, 16.9]
    assert {"afrr_pos_mw", "mfrr_sa_neg_mw", "igcc_import_mw"} <= set(df.columns)


def test_read_snapshot_rejects_gaps(tmp_path):
    f = write_csv(tmp_path / "control-area-balance-2026_a.csv", ["31.12.2025 23:00", "31.12.2025 23:30"])

    with pytest.raises(ControlAreaBalanceFormatError, match="1 quarter hours missing"):
        read_snapshot(f)


def test_read_snapshot_rejects_a_changed_format(tmp_path):
    f = tmp_path / "control-area-balance-2026_a.csv"
    f.write_text("Date Time [UTC];Total System Imbalance\n31.12.2025 23:00;1\n")

    with pytest.raises(ControlAreaBalanceFormatError, match="columns missing"):
        read_snapshot(f)


def test_latest_snapshot_is_the_newest_download(tmp_path):
    for name in ["2026-10-04T13-47-09Z", "2026-10-05T02-00-00Z", "2026-09-30T08-00-00Z"]:
        write_csv(tmp_path / f"control-area-balance-2026_{name}.csv", ["31.12.2025 23:00"])

    assert latest_snapshot(tmp_path).name == "control-area-balance-2026_2026-10-05T02-00-00Z.csv"


def test_fetch_stores_a_snapshot_only_when_the_content_changed(tmp_path, monkeypatch):
    first = write_csv(tmp_path / "a.csv", ["31.12.2025 23:00"]).read_bytes()
    second = write_csv(tmp_path / "b.csv", ["31.12.2025 23:00", "31.12.2025 23:15"]).read_bytes()
    downloads = [(first, "Sun, 04 Oct 2026 01:21:00 GMT"), (first, "same"), (second, "Mon, 05 Oct 2026 01:21:00 GMT")]
    monkeypatch.setattr(fetch_swissgrid, "download", lambda: downloads.pop(0))
    out = tmp_path / "snapshots"
    clock = iter(pd.Timestamp(f"2026-10-0{d} 13:47:0{d}", tz="UTC") for d in (4, 4, 5))
    monkeypatch.setattr(fetch_swissgrid.pd.Timestamp, "now", lambda tz=None: next(clock))

    for _ in range(3):
        fetch_swissgrid.main(["--out", str(out)])

    manifest = pd.read_csv(out / "manifest.csv")
    assert len(list(out.glob("control-area-balance-*.csv"))) == 2
    assert manifest["rows"].tolist() == [1, 2]
    assert manifest["last_modified"].tolist() == ["Sun, 04 Oct 2026 01:21:00 GMT", "Mon, 05 Oct 2026 01:21:00 GMT"]

    log = pd.read_csv(out / "fetch_log.csv")
    assert log["changed"].tolist() == [True, False, True]
    assert log["snapshot"].isna().tolist() == [False, True, False]


@pytest.mark.parametrize(
    ("fetched_at", "last_timestamp_utc", "expected"),
    [
        ("2026-10-05 08:30", "2026-10-03T21:45:00+00:00", 2),  # Monday 10:30: Sunday missing
        ("2026-10-04 08:30", "2026-10-03T21:45:00+00:00", 1),  # Sunday 10:30: Saturday complete
        ("2026-10-04 08:30", "2026-10-03T10:00:00+00:00", 2),  # Saturday only half there
        ("2026-10-26 09:30", "2026-10-24T21:45:00+00:00", 2),  # after the switch to winter time
    ],
)
def test_days_behind_counts_from_the_last_complete_local_day(fetched_at, last_timestamp_utc, expected):
    assert fetch_swissgrid.days_behind(pd.Timestamp(fetched_at, tz="UTC"), last_timestamp_utc) == expected
