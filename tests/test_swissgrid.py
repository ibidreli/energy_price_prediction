from pathlib import Path

import pandas as pd
import pytest

from energy_price.swissgrid import TZ, SwissgridFormatError, load_prices, read_price_file
from xml_fixtures import month_rows, write_xml

REAL_DATA = Path(__file__).resolve().parents[1] / "data" / "ausgleichpreis"


def local_day(df: pd.DataFrame, day: str) -> pd.DataFrame:
    return df[df["timestamp_local"].dt.strftime("%Y-%m-%d") == day]


# read_price_file


def test_read_price_file_converts_to_utc(tmp_path):
    f = write_xml(tmp_path / "a.xml", {"BG-AEP": [("2026-01-01T00:00:00+01:00", "12.50"), ("2026-01-01T00:15:00+01:00", "-3.25")]})

    df = read_price_file(f)

    assert list(df["timestamp_utc"]) == [pd.Timestamp("2025-12-31 23:00", tz="UTC"), pd.Timestamp("2025-12-31 23:15", tz="UTC")]
    assert list(df["value_ct_kwh"]) == [12.5, -3.25]
    assert set(df["series"]) == {"BG-AEP"}


@pytest.mark.parametrize(
    ("series", "unit", "message"),
    [
        ({"BG-XYZ": [("2026-01-01T00:00:00+01:00", "1")]}, "ct/kWh", "unknown series"),
        ({"BG-AEP": [("2026-01-01T00:00:00+01:00", "1")]}, "EUR/MWh", "unit"),
        ({"BG-AEP": [("2026-01-01T00:00:00+01:00", "abc")]}, "ct/kWh", "unreadable"),
        ({"BG-AEP": [("2026-01-01T00:00:00+01:00", "")]}, "ct/kWh", "empty"),
        ({"BG-AEP": [("2026-01-01T00:00:00+01:00", "inf")]}, "ct/kWh", "non-finite"),
        ({"BG-AEP": [("2026-01-01T00:00:00", "1")]}, "ct/kWh", "without UTC offset"),
        ({"BG-AEP": [("2026-01-01T00:05:00+01:00", "1")]}, "ct/kWh", "15-minute grid"),
    ],
)
def test_read_price_file_rejects_bad_input(tmp_path, series, unit, message):
    f = write_xml(tmp_path / "bad.xml", series, unit=unit)

    with pytest.raises(SwissgridFormatError, match=message):
        read_price_file(f)


# load_prices: daylight saving time


def test_load_prices_handles_switch_to_summer_time(tmp_path):
    write_xml(tmp_path / "2603.xml", {"BG-AEP": month_rows(2026, 3)})

    df = load_prices(tmp_path)
    day = local_day(df, "2026-03-29")

    assert len(df) == 2972
    assert len(day) == 92
    assert df["timestamp_utc"].is_unique and df["timestamp_utc"].is_monotonic_increasing
    times = day["timestamp_local"].dt.strftime("%H:%M%z").tolist()
    assert times[7:9] == ["01:45+0100", "03:00+0200"]


def test_load_prices_handles_switch_to_winter_time(tmp_path):
    write_xml(tmp_path / "2610.xml", {"BG-AEP": month_rows(2026, 10)})

    df = load_prices(tmp_path)
    day = local_day(df, "2026-10-25")

    assert len(df) == 2980
    assert len(day) == 100
    assert df["timestamp_utc"].is_unique and df["timestamp_utc"].is_monotonic_increasing
    two_oclock = day.loc[day["timestamp_local"].dt.strftime("%H:%M") == "02:00", "timestamp_local"]
    assert two_oclock.dt.strftime("%z").tolist() == ["+0200", "+0100"]


# load_prices: regimes and series


def test_load_prices_assigns_regime_and_keeps_parallel_aep(tmp_path):
    write_xml(
        tmp_path / "2512.xml",
        {"BG-long": month_rows(2025, 12, "1.0"), "BG-short": month_rows(2025, 12, "9.0"), "BG-AEP": month_rows(2025, 12, "4.0")},
    )
    write_xml(tmp_path / "2601.xml", {"BG-AEP": month_rows(2026, 1, "5.0")})

    df = load_prices(tmp_path)
    boundary = df[df["timestamp_local"].between(pd.Timestamp("2025-12-31 23:45", tz=TZ), pd.Timestamp("2026-01-01 00:00", tz=TZ))]

    assert boundary["regime"].tolist() == ["two_price", "single_price"]
    assert boundary["aep_ct_kwh"].tolist() == [4.0, 5.0]
    assert df.loc[df["regime"] == "single_price", ["long_ct_kwh", "short_ct_kwh"]].isna().all().all()


def test_load_prices_requires_aep_in_single_price_regime(tmp_path):
    write_xml(tmp_path / "2601.xml", {"BG-long": month_rows(2026, 1), "BG-short": month_rows(2026, 1)})

    with pytest.raises(SwissgridFormatError, match="single_price quarter hours without aep_ct_kwh"):
        load_prices(tmp_path)


def test_load_prices_requires_long_and_short_in_two_price_regime(tmp_path):
    write_xml(tmp_path / "2512.xml", {"BG-long": month_rows(2025, 12), "BG-AEP": month_rows(2025, 12)})

    with pytest.raises(SwissgridFormatError, match="two_price quarter hours without short_ct_kwh"):
        load_prices(tmp_path)


def test_load_prices_rejects_holes_in_aep_after_it_started(tmp_path):
    write_xml(tmp_path / "2511.xml", {"BG-long": month_rows(2025, 11), "BG-short": month_rows(2025, 11), "BG-AEP": month_rows(2025, 11)})
    write_xml(tmp_path / "2512.xml", {"BG-long": month_rows(2025, 12), "BG-short": month_rows(2025, 12)})
    write_xml(tmp_path / "2601.xml", {"BG-AEP": month_rows(2026, 1)})

    with pytest.raises(SwissgridFormatError, match="without aep_ct_kwh after it started"):
        load_prices(tmp_path)


# load_prices: completeness


def test_load_prices_rejects_duplicates_across_files(tmp_path):
    write_xml(tmp_path / "a.xml", {"BG-AEP": month_rows(2026, 1)})
    write_xml(tmp_path / "b.xml", {"BG-AEP": month_rows(2026, 1)})

    with pytest.raises(SwissgridFormatError, match="duplicate.*a.xml, b.xml"):
        load_prices(tmp_path)


def test_load_prices_rejects_gaps(tmp_path):
    rows = month_rows(2026, 1)
    write_xml(tmp_path / "2601.xml", {"BG-AEP": rows[:100] + rows[101:]})

    with pytest.raises(SwissgridFormatError, match="1 quarter hours missing"):
        load_prices(tmp_path)


@pytest.mark.parametrize(
    ("cut", "message"),
    [
        (slice(1, None), "starts at"),
        (slice(None, -1), "ends at"),
        (slice(None, 96 * 15), "ends at"),
    ],
    ids=["missing first quarter hour", "missing last quarter hour", "file ends mid-month"],
)
def test_load_prices_requires_complete_months(tmp_path, cut, message):
    write_xml(tmp_path / "2601.xml", {"BG-AEP": month_rows(2026, 1)[cut]})

    with pytest.raises(SwissgridFormatError, match=message):
        load_prices(tmp_path)


def test_load_prices_without_files_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_prices(tmp_path)


@pytest.mark.realdata
@pytest.mark.skipif(not REAL_DATA.exists(), reason="raw Swissgrid data not downloaded")
def test_real_data_is_complete():
    df = load_prices(REAL_DATA)

    single = df[df["regime"] == "single_price"]
    assert single["aep_ct_kwh"].notna().all()
    assert df["timestamp_local"].iloc[0] == pd.Timestamp("2023-01-01 00:00", tz=TZ)
    # January to August 2026: 243 days, minus 4 quarter hours for the switch to summer time.
    assert len(single[single["timestamp_local"] < pd.Timestamp("2026-09-01", tz=TZ)]) == 243 * 96 - 4
    # BG-AEP is published from July 2025 onwards without holes.
    assert df.loc[df["aep_ct_kwh"].notna(), "timestamp_local"].iloc[0] == pd.Timestamp("2025-07-01 00:00", tz=TZ)
