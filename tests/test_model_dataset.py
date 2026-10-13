import numpy as np
import pandas as pd
import pytest

from energy_price.model.dataset import (
    Inputs,
    build_table,
    calendar_features,
    holiday_pv_share,
    quarter_hours,
    radiation_by_hour,
)

TZ = "Europe/Zurich"
SITES = pd.DataFrame({"site": ["A", "B"], "pv_share": [0.75, 0.25]})


def weather_for(day: str, value_of_stamp=lambda stamp: float(stamp.hour), sites=("A", "B")) -> pd.DataFrame:
    """Hourly stamps D 00:00 to D+1 00:00 local, as in weather_forecasts.parquet."""
    start = pd.Timestamp(day).tz_localize(TZ)
    end = (pd.Timestamp(day) + pd.Timedelta(days=1)).tz_localize(TZ)
    stamps = pd.date_range(start, end, freq="h").tz_convert("UTC")
    rows = [
        {"delivery_day": pd.Timestamp(day), "site": s, "valid_time_utc": t, "shortwave_radiation": value_of_stamp(t)}
        for s in sites for t in stamps
    ]
    return pd.DataFrame(rows)


def inputs(weather: pd.DataFrame, prices: pd.DataFrame | None = None, holidays: pd.DataFrame | None = None) -> Inputs:
    if prices is None:
        prices = pd.DataFrame({"timestamp_utc": pd.Series([], dtype="datetime64[us, UTC]"),
                               "timestamp_local": pd.Series([], dtype=f"datetime64[us, {TZ}]"),
                               "aep_ct_kwh": pd.Series([], dtype=float)})
    if holidays is None:
        holidays = pd.DataFrame({"date": [], "canton": []})
    return Inputs(prices=prices, weather=weather, sites=SITES, holidays=holidays,
                  pv_capacity=pd.DataFrame({"canton": ["ZH", "BE"], "share": [0.6, 0.4]}))


@pytest.mark.parametrize(("day", "rows"), [("2026-03-29", 92), ("2026-07-16", 96), ("2026-10-25", 100)])
def test_quarter_hours_follow_local_time_on_switch_days(day, rows):
    assert len(quarter_hours([pd.Timestamp(day)])) == rows


def test_calendar_position_and_weekend():
    stamps = pd.Series(pd.date_range(pd.Timestamp("2026-07-18 12:00", tz=TZ), periods=4, freq="15min"))

    cal = calendar_features(stamps)

    assert cal["position"].tolist() == [-1.5, -0.5, 0.5, 1.5]
    assert (cal["hour"] == 12).all()
    assert (cal["weekend"] == 1.0).all()  # Saturday


def test_holiday_share_sums_the_pv_share_of_cantons_with_a_holiday():
    holidays = pd.DataFrame({"date": ["2026-08-01", "2026-08-01", "2026-01-02"], "canton": ["ZH", "BE", "ZH"]})
    pv = pd.DataFrame({"canton": ["ZH", "BE"], "share": [0.6, 0.4]})

    share = holiday_pv_share(holidays, pv)

    assert share.loc[pd.Timestamp("2026-08-01")] == pytest.approx(1.0)
    assert share.loc[pd.Timestamp("2026-01-02")] == pytest.approx(0.6)


def test_quarter_hours_take_the_radiation_of_the_hour_they_lie_in():
    # Stamp h:00 is the mean of the hour before it, so 12:00 to 12:45 local take the stamp 13:00.
    weather = weather_for("2026-07-16", lambda t: float(t.tz_convert(TZ).hour))

    table = build_table(inputs(weather), days=[pd.Timestamp("2026-07-16")])

    noon = table[table["hour"] == 12]
    assert (noon["radiation"] == 13.0).all()
    last = table[table["hour"] == 23]
    assert (last["radiation"] == 0.0).all()  # stamp D+1 00:00 of the own run


def test_the_midnight_stamp_of_the_next_run_never_reaches_the_previous_day():
    # The run for D+1 also contains the stamp D+1 00:00. Its value must not reach D 23:00 to 23:45.
    own = weather_for("2026-07-16", lambda t: 1.0)
    later_run = weather_for("2026-07-17", lambda t: 999.0)

    table = build_table(inputs(pd.concat([own, later_run], ignore_index=True)), days=[pd.Timestamp("2026-07-16")])

    assert (table["radiation"] == 1.0).all()


def test_sites_are_weighted_by_pv_share_and_missing_sites_renormalised():
    a = weather_for("2026-07-16", lambda t: 100.0, sites=("A",))
    b = weather_for("2026-07-16", lambda t: 200.0, sites=("B",))
    b.loc[b["valid_time_utc"].dt.tz_convert(TZ).dt.hour == 13, "shortwave_radiation"] = np.nan

    hourly = radiation_by_hour(pd.concat([a, b], ignore_index=True), SITES)
    local_hour = hourly["hour_stamp_utc"].dt.tz_convert(TZ).dt.hour

    assert hourly.loc[local_hour == 12, "radiation"].iloc[0] == pytest.approx(0.75 * 100 + 0.25 * 200)
    assert hourly.loc[local_hour == 13, "radiation"].iloc[0] == pytest.approx(100.0)


def test_radiation_change_is_the_centred_difference_over_the_hour():
    weather = weather_for("2026-07-16", lambda t: 10.0 * t.tz_convert(TZ).hour)

    hourly = radiation_by_hour(weather, SITES)
    local_hour = hourly["hour_stamp_utc"].dt.tz_convert(TZ).dt.hour

    assert hourly.loc[local_hour == 12, "radiation_change"].iloc[0] == pytest.approx(10.0)


def test_target_comes_only_from_the_own_quarter_hour():
    day = pd.Timestamp("2026-07-16")
    grid = quarter_hours([day])
    prices = pd.DataFrame({"timestamp_utc": grid["timestamp_utc"], "timestamp_local": grid["timestamp_local"],
                           "aep_ct_kwh": np.arange(len(grid), dtype=float)})

    table = build_table(inputs(weather_for("2026-07-16"), prices), days=[day])

    assert table["aep_ct_kwh"].tolist() == list(range(96))
