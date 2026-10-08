import datetime as dt
import json

import pandas as pd
import pytest

from energy_price import build_sources, fetch_reservoirs
from energy_price.reservoir import (
    SERIES, ReservoirFormatError, load_reservoirs, parse_payload, read_snapshot, latest_reservoir_report,
)


def payload():
    series = [
        {"name": [{"en": name}], "data": [
            8.0 if region == "switzerland" else 2.0 if column == "capacity_twh" else 1.0
        ] * 2}
        for name, (region, column) in SERIES.items()
    ]
    series[0].update(
        format="Highcharts", timeZone="UTC", y0AxisLabel=[{"en": "Energy (TWh)"}],
        xAxisValues=[int(pd.Timestamp(day, tz="UTC").timestamp() * 1000)
                     for day in ["2026-01-05", "2026-01-12"]],
    )
    return series


def snapshot(tmp_path, fetched="2026-01-14T08:00:00Z", data=None, name="a", year=2026):
    path = tmp_path / f"filling-level-{year}_{name}.json"
    path.write_text(json.dumps(dict(year=year, fetched_at_utc=fetched, data=payload() if data is None else data)))
    return path


def test_totals_use_only_energy_series_and_percent_uses_capacity():
    frame = parse_payload(payload())
    swiss = frame[frame.region == "switzerland"]
    assert len(frame) == 10
    assert swiss.stored_twh.tolist() == [4.0, 4.0]
    assert swiss.capacity_twh.tolist() == [8.0, 8.0]
    assert swiss.filling_pct.tolist() == [50.0, 50.0]
    assert swiss.reference_date.tolist() == [pd.Timestamp("2026-01-04"), pd.Timestamp("2026-01-11")]
    assert swiss.timestamp_local.iloc[0] == pd.Timestamp("2026-01-05 01:00", tz="Europe/Zurich")


def test_null_energy_remains_missing_in_region_and_national_total():
    data = payload()
    data[0]["data"][1] = None
    frame = parse_payload(data)
    last = frame[frame.timestamp_utc == frame.timestamp_utc.max()].set_index("region")
    assert pd.isna(last.loc["valais", "stored_twh"])
    assert pd.isna(last.loc["switzerland", "stored_twh"])
    assert pd.isna(last.loc["switzerland", "filling_pct"])


@pytest.mark.parametrize("problem, message", [
    ("unit", "unknown unit"), ("short", "different lengths"),
    ("duplicate_series", "duplicate series"), ("missing", "missing series"),
    ("duplicate_time", "duplicate timestamps"), ("negative", "negative"),
    ("overflow", "exceeds capacity"), ("zero_capacity", "positive"),
    ("bad_total", "total capacity"), ("different_axis", "shared axis"),
])
def test_rejects_corrupt_or_changed_source_formats(problem, message):
    data = payload()
    if problem == "unit":
        data[0]["y0AxisLabel"][0]["en"] = "Energy (GWh)"
    elif problem == "short":
        data[1]["data"].pop()
    elif problem == "duplicate_series":
        data.append(data[1])
    elif problem == "missing":
        data.pop()
    elif problem == "duplicate_time":
        data[0]["xAxisValues"][1] = data[0]["xAxisValues"][0]
    elif problem == "negative":
        data[0]["data"][0] = -1
    elif problem == "overflow":
        data[0]["data"][0] = 20
    elif problem == "zero_capacity":
        data[4]["data"][0] = 0
    elif problem == "bad_total":
        data[-1]["data"][0] = 9
    else:
        data[1]["xAxisValues"] = [0, 1]
    with pytest.raises(ReservoirFormatError, match=message):
        parse_payload(data)


def test_no_historical_availability_is_invented_from_measurement_dates(tmp_path):
    frame = read_snapshot(snapshot(tmp_path, fetched="2026-10-08T15:00:00Z"))
    assert latest_reservoir_report(frame, dt.date(2026, 1, 15)).empty
    assert latest_reservoir_report(frame, dt.date(2026, 10, 9)).empty  # fetch after D-1 11:00
    assert len(latest_reservoir_report(frame, dt.date(2026, 10, 10))) == 5


def test_revisions_are_used_only_after_their_fetch_time(tmp_path):
    snapshot(tmp_path, name="a")
    revision = payload()
    revision[0]["data"][1] = 1.5
    snapshot(tmp_path, fetched="2026-01-15T10:30:00Z", data=revision, name="b")
    frame = load_reservoirs(tmp_path)
    before = latest_reservoir_report(frame, dt.date(2026, 1, 16)).set_index("region")  # 10:00 UTC cutoff
    after = latest_reservoir_report(frame, dt.date(2026, 1, 17)).set_index("region")
    assert before.loc["switzerland", "stored_twh"] == 4.0
    assert after.loc["switzerland", "stored_twh"] == 4.5
    assert before.snapshot.unique().tolist() == ["filling-level-2026_a.json"]
    assert after.snapshot.unique().tolist() == ["filling-level-2026_b.json"]


@pytest.mark.parametrize("day, fetched", [
    (dt.date(2026, 3, 30), "2026-03-29T09:00:00Z"),
    (dt.date(2026, 10, 26), "2026-10-25T10:00:00Z"),
])
def test_issue_cutoff_respects_swiss_dst(tmp_path, day, fetched):
    frame = read_snapshot(snapshot(tmp_path, fetched=fetched))
    assert len(latest_reservoir_report(frame, day)) == 5
    frame["fetched_at_utc"] += pd.Timedelta(seconds=1)
    assert latest_reservoir_report(frame, day).empty


def test_naive_fetch_time_is_rejected(tmp_path):
    with pytest.raises(ReservoirFormatError, match="timezone-aware"):
        read_snapshot(snapshot(tmp_path, fetched="2026-01-14T08:00:00"))


def test_wrong_year_is_rejected(tmp_path):
    with pytest.raises(ReservoirFormatError, match="requested year"):
        read_snapshot(snapshot(tmp_path, year=2025))


def test_fetch_archives_changes_and_logs_unchanged_downloads(tmp_path, monkeypatch):
    first = json.dumps(payload()).encode()
    revised = payload()
    revised[0]["data"][1] = 1.5
    second = json.dumps(revised).encode()
    downloads = iter([(first, "first"), (first, "same"), (second, "second")])
    monkeypatch.setattr(fetch_reservoirs, "download", lambda year: next(downloads))
    clock = iter(pd.Timestamp(day, tz="UTC") for day in ["2026-01-14", "2026-01-15", "2026-01-16"])
    monkeypatch.setattr(fetch_reservoirs.pd.Timestamp, "now", lambda tz: next(clock))
    for _ in range(3):
        fetch_reservoirs.main(["--out", str(tmp_path), "--years", "2026"])
    assert len(list(tmp_path.glob("filling-level-*.json"))) == 2
    manifest = pd.read_csv(tmp_path / "manifest.csv")
    assert manifest.weeks.tolist() == [2, 2]
    assert pd.read_csv(tmp_path / "fetch_log.csv").changed.tolist() == [True, False, True]
    # An unchanged download preserves the first observed time for this version.
    assert load_reservoirs(tmp_path).fetched_at_utc.unique().tolist() == [
        pd.Timestamp("2026-01-14", tz="UTC"), pd.Timestamp("2026-01-16", tz="UTC")
    ]
    assert first.decode() in (tmp_path / manifest.file.iloc[0]).read_text()


def test_invalid_download_does_not_write_any_files(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch_reservoirs, "download", lambda year: (b"[]", ""))
    with pytest.raises(ReservoirFormatError):
        fetch_reservoirs.main(["--out", str(tmp_path), "--years", "2026"])
    assert list(tmp_path.iterdir()) == []


def test_pipeline_writes_parquet_with_provenance_and_preserves_it_on_failure(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    path = snapshot(raw)
    out = tmp_path / "reservoirs.parquet"
    args = ["reservoirs", "--src", str(raw), "--out", str(out)]
    build_sources.main(args)
    pd.testing.assert_frame_equal(pd.read_parquet(out), load_reservoirs(raw))
    before = out.read_bytes()
    path.write_text("[]")
    with pytest.raises(ReservoirFormatError):
        build_sources.main(args)
    assert out.read_bytes() == before
    assert not out.with_suffix(".parquet.tmp").exists()


def test_empty_directory_explains_how_to_fetch(tmp_path):
    with pytest.raises(FileNotFoundError, match="fetch-reservoirs"):
        load_reservoirs(tmp_path)
