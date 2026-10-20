import datetime as dt
import hashlib
import io
import json

import pandas as pd
import pytest

from energy_price.data import build_sources
from energy_price.data import reservoir
from energy_price.data.reservoir import (
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
    path = tmp_path / f"reservoirs-{year}_{name}.json"
    data = payload() if data is None else data
    sha = hashlib.sha256(json.dumps(data).encode()).hexdigest()
    path.write_text(json.dumps(dict(year=year, fetched_at_utc=fetched, sha256=sha, data=data)))
    return path


def bfe_csv(data=None):
    frame = parse_payload(payload() if data is None else data)
    records = {"Datum": frame[frame.region == "switzerland"].timestamp_utc.dt.strftime("%Y-%m-%d").tolist()}
    for region, prefix in reservoir.BFE_REGIONS.items():
        selected = frame[frame.region == region]
        records[f"{prefix}_speicherinhalt_gwh"] = (selected.stored_twh * 1000).tolist()
        records[f"{prefix}_max_speicherinhalt_gwh"] = (selected.capacity_twh * 1000).tolist()
    return pd.DataFrame(records).to_csv(index=False, lineterminator="\r\n").encode()


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
    assert before.snapshot.unique().tolist() == ["reservoirs-2026_a.json"]
    assert after.snapshot.unique().tolist() == ["reservoirs-2026_b.json"]


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


def test_fetch_archives_changes_and_logs_unchanged_downloads(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "data/reservoirs/snapshots"
    first = bfe_csv()
    revised = payload()
    revised[0]["data"][1] = 1.5
    second = bfe_csv(revised)
    downloads = iter([(first, "first"), (first, "same"), (second, "second")])
    monkeypatch.setattr(reservoir, "download", lambda: next(downloads))
    clock = iter(pd.Timestamp(day, tz="UTC") for day in ["2026-01-14", "2026-01-15", "2026-01-16"])
    monkeypatch.setattr(reservoir.pd.Timestamp, "now", lambda tz: next(clock))
    for _ in range(3):
        reservoir.main(["--years", "2026"])
    snapshots = sorted(out.glob("reservoirs-*.json"))
    assert len(snapshots) == 2
    manifest = pd.read_csv(out / "manifest.csv")
    assert manifest.file.tolist() == [path.name for path in snapshots]
    assert manifest.weeks.tolist() == [2, 2]
    assert manifest.fetched_at_utc.tolist() == [
        "2026-01-14T00:00:00+00:00", "2026-01-16T00:00:00+00:00",
    ]
    assert manifest.first_timestamp_utc.tolist() == ["2026-01-05T00:00:00+00:00"] * 2
    assert manifest.last_timestamp_utc.tolist() == ["2026-01-12T00:00:00+00:00"] * 2
    for path, checksum in zip(snapshots, manifest.sha256):
        assert json.loads(path.read_text())["sha256"] == checksum
    assert pd.read_csv(out / "fetch_log.csv").changed.tolist() == [True, False, True]
    # An unchanged download preserves the first observed time for this version.
    assert load_reservoirs(out).fetched_at_utc.unique().tolist() == [
        pd.Timestamp("2026-01-14", tz="UTC"), pd.Timestamp("2026-01-16", tz="UTC")
    ]
    record = json.loads(snapshots[0].read_text())
    assert record["data"] == first.decode()
    assert record["source_format"] == "bfe_csv"
    assert record["url"] == reservoir.BFE_URL
    output = capsys.readouterr().out
    assert "weekly values unchanged" in output
    assert "weekly values revised" in output


@pytest.mark.parametrize("calendar_year, args, source_year", [
    (2026, [], 2026),
    (2027, [], 2027),
    (2027, ["--years", "2026"], 2026),
])
def test_fetch_uses_current_swiss_year_unless_overridden(tmp_path, monkeypatch, calendar_year, args, source_year):
    monkeypatch.chdir(tmp_path)
    # At the Swiss New Year, UTC still belongs to the previous calendar year.
    now = pd.Timestamp(f"{calendar_year}-01-01 00:30", tz="Europe/Zurich")
    data = payload()
    data[0]["xAxisValues"] = [int(stamp.timestamp() * 1000)
                             for stamp in pd.date_range(f"{source_year}-01-01", periods=2, freq="W-MON", tz="UTC")]

    def download():
        return bfe_csv(data), ""

    monkeypatch.setattr(reservoir, "download", download)
    monkeypatch.setattr(reservoir.pd.Timestamp, "now", lambda tz: now.tz_convert(tz))
    reservoir.main(args)

    out = tmp_path / "data/reservoirs/snapshots"
    assert pd.read_csv(out / "manifest.csv").year.tolist() == [source_year]
    frame = load_reservoirs(out)
    assert frame.source_year.unique().tolist() == [source_year]
    assert frame.fetched_at_utc.unique().tolist() == [now.tz_convert("UTC")]


def test_invalid_download_does_not_write_any_files(tmp_path, monkeypatch):
    monkeypatch.setattr(reservoir, "download", lambda: (b"[]", ""))
    with pytest.raises(ReservoirFormatError):
        reservoir.main(["--out", str(tmp_path), "--years", "2026"])
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


def test_bfe_csv_matches_historical_json_values_and_converts_gwh():
    expected = parse_payload(payload())
    actual = reservoir.parse_bfe_csv(bfe_csv(), 2026)
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
    assert actual.loc[actual.region == "switzerland", "stored_twh"].tolist() == [4.0, 4.0]


def test_bfe_csv_selects_requested_year_from_full_history():
    current = pd.read_csv(io.BytesIO(bfe_csv()))
    previous = current.copy()
    previous["Datum"] = ["2025-01-06", "2025-01-13"]
    content = pd.concat([previous, current]).to_csv(index=False).encode()
    frame = reservoir.parse_bfe_csv(content, 2026)
    assert frame.timestamp_utc.dt.year.unique().tolist() == [2026]
    assert len(frame) == 10
    with pytest.raises(ReservoirFormatError, match="no BFE reports for requested year 2027"):
        reservoir.parse_bfe_csv(content, 2027)


@pytest.mark.parametrize("problem, message", [
    ("unit", "missing BFE columns"), ("date", "time data"),
    ("duplicate", "duplicate timestamps"), ("weekday", "Monday"),
    ("negative", "negative"), ("overflow", "exceeds capacity"),
    ("total", "total energy"),
])
def test_bfe_csv_rejects_invalid_reports(problem, message):
    raw = pd.read_csv(io.BytesIO(bfe_csv()))
    if problem == "unit":
        raw = raw.rename(columns={"Wallis_speicherinhalt_gwh": "Wallis_speicherinhalt_twh"})
    elif problem == "date":
        raw.loc[0, "Datum"] = "not-a-date"
    elif problem == "duplicate":
        raw.loc[1, "Datum"] = raw.loc[0, "Datum"]
    elif problem == "weekday":
        raw.loc[0, "Datum"] = "2026-01-06"
    elif problem == "negative":
        raw.loc[0, "Wallis_speicherinhalt_gwh"] = -1
    elif problem == "overflow":
        raw.loc[0, "Wallis_speicherinhalt_gwh"] = 20000
    else:
        raw.loc[0, "TotalCH_speicherinhalt_gwh"] = 5000
    with pytest.raises(ReservoirFormatError, match=message):
        reservoir.parse_bfe_csv(raw.to_csv(index=False).encode(), 2026)


def test_bfe_csv_keeps_missing_regional_and_total_values():
    raw = pd.read_csv(io.BytesIO(bfe_csv()))
    raw.loc[1, "Wallis_speicherinhalt_gwh"] = float("nan")
    frame = reservoir.parse_bfe_csv(raw.to_csv(index=False).encode(), 2026)
    last = frame[frame.reference_date == frame.reference_date.max()].set_index("region")
    assert pd.isna(last.loc["valais", "stored_twh"])
    assert pd.isna(last.loc["switzerland", "stored_twh"])


def test_bfe_csv_ignores_future_calendar_rows_without_reports():
    raw = pd.read_csv(io.BytesIO(bfe_csv()))
    future = raw.iloc[[-1]].copy()
    future["Datum"] = "2026-01-19"
    for prefix in reservoir.BFE_REGIONS.values():
        future[f"{prefix}_speicherinhalt_gwh"] = float("nan")
    content = pd.concat([raw, future]).to_csv(index=False).encode()
    frame = reservoir.parse_bfe_csv(content, 2026)
    assert frame.timestamp_utc.nunique() == 2
    assert frame.reference_date.max() == pd.Timestamp("2026-01-11")


def test_new_bfe_version_preserves_historical_json_and_forecast_cutoff(tmp_path):
    snapshot(tmp_path)
    revised = payload()
    revised[0]["data"][1] = 1.5
    record = dict(year=2026, source_format="bfe_csv", data=bfe_csv(revised).decode(),
                  fetched_at_utc="2026-01-14T10:30:00Z")
    (tmp_path / "reservoirs-2026_b.json").write_text(json.dumps(record))
    frame = load_reservoirs(tmp_path)
    assert len(frame) == 20
    before = latest_reservoir_report(frame, dt.date(2026, 1, 15)).set_index("region")
    after = latest_reservoir_report(frame, dt.date(2026, 1, 16)).set_index("region")
    assert before.loc["switzerland", "stored_twh"] == 4.0
    assert after.loc["switzerland", "stored_twh"] == 4.5


def test_fetch_reports_new_week_when_migrating_from_energy_charts(tmp_path, monkeypatch, capsys):
    snapshot(tmp_path)
    raw = pd.read_csv(io.BytesIO(bfe_csv()))
    next_week = raw.iloc[[-1]].copy()
    next_week["Datum"] = "2026-01-19"
    content = pd.concat([raw, next_week]).to_csv(index=False).encode()
    monkeypatch.setattr(reservoir, "download", lambda: (content, ""))
    reservoir.fetch_year(2026, tmp_path)
    assert "1 new weekly report(s)" in capsys.readouterr().out
    assert len(list(tmp_path.glob("reservoirs-*.json"))) == 2


def test_raw_format_change_is_not_reported_as_revised_values(tmp_path, monkeypatch, capsys):
    first = bfe_csv()
    responses = iter([first, first.replace(b"\r\n", b"\n")])
    clock = iter(pd.Timestamp(day, tz="UTC") for day in ["2026-01-14", "2026-01-15"])
    monkeypatch.setattr(reservoir, "download", lambda: (next(responses), ""))
    monkeypatch.setattr(reservoir.pd.Timestamp, "now", lambda tz: next(clock))
    reservoir.fetch_year(2026, tmp_path)
    reservoir.fetch_year(2026, tmp_path)
    output = capsys.readouterr().out
    assert "new snapshot (BFE; weekly values unchanged)" in output
    assert "weekly values revised" not in output
