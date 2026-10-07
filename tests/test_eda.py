import datetime as dt

import numpy as np
import pandas as pd
import pytest

from energy_price.eda import (
    BRIDGE_DAY,
    HOLIDAY,
    SATURDAY,
    SUNDAY,
    WORKDAY,
    clock_slot,
    day_types,
    direction_matches,
    extreme_events,
    holiday_pv_share,
    weather_to_quarter_hours,
    within_hour_trend,
)

TZ = "Europe/Zurich"


def quarter_hours(start: str, end: str, tz: str = TZ) -> pd.Series:
    return pd.Series(pd.date_range(pd.Timestamp(start, tz=tz), pd.Timestamp(end, tz=tz), freq="15min", inclusive="left"))


def test_clock_slot_counts_quarter_hours_of_the_local_clock():
    slots = clock_slot(quarter_hours("2026-07-16 00:00", "2026-07-17 00:00"))

    assert slots.tolist() == list(range(96))


def test_clock_slot_skips_the_lost_hour_on_the_switch_to_summer_time():
    slots = clock_slot(quarter_hours("2026-03-29 00:00", "2026-03-30 00:00"))

    assert len(slots) == 92
    assert not set(range(8, 12)) & set(slots)


def test_clock_slot_rejects_naive_timestamps():
    with pytest.raises(TypeError, match="timezone-aware"):
        clock_slot(pd.Series(pd.date_range("2026-07-16", periods=2, freq="15min")))


def test_holiday_pv_share_adds_the_shares_of_the_cantons_with_a_holiday():
    holidays = pd.DataFrame({"date": ["2026-01-01", "2026-01-01", "2026-01-02"], "canton": ["ZH", "BE", "BE"]})
    pv = pd.DataFrame({"canton": ["ZH", "BE", "TI"], "share": [0.6, 0.3, 0.1]})

    share = holiday_pv_share(holidays, pv)

    assert share[dt.date(2026, 1, 1)] == pytest.approx(0.9)
    assert share[dt.date(2026, 1, 2)] == pytest.approx(0.3)


def test_holiday_pv_share_refuses_cantons_without_pv_share():
    holidays = pd.DataFrame({"date": ["2026-01-01"], "canton": ["XX"]})
    pv = pd.DataFrame({"canton": ["ZH"], "share": [1.0]})

    with pytest.raises(ValueError, match="XX"):
        holiday_pv_share(holidays, pv)


def test_day_types_around_ascension_2026():
    # Ascension is Thursday 14.05.2026, so Friday 15.05. is a bridge day.
    share = pd.Series({dt.date(2026, 5, 14): 1.0})
    days = pd.Series(pd.date_range("2026-05-13", "2026-05-18").date)

    types = day_types(days, share)

    assert types.tolist() == [WORKDAY, HOLIDAY, BRIDGE_DAY, SATURDAY, SUNDAY, WORKDAY]


def test_day_types_ignore_holidays_of_a_minority_and_keep_weekends():
    share = pd.Series({dt.date(2026, 1, 2): 0.2, dt.date(2026, 8, 1): 1.0})  # Friday, Saturday
    days = pd.Series([dt.date(2026, 1, 2), dt.date(2026, 8, 1)])

    assert day_types(days, share).tolist() == [WORKDAY, SATURDAY]


def test_direction_matches_compares_the_same_clock_time_lag_days_earlier():
    stamps = quarter_hours("2026-07-01 00:00", "2026-07-05 00:00")
    day = stamps.dt.day
    # 1st and 3rd short in every quarter hour, 2nd and 4th long
    frame = pd.DataFrame({"timestamp_local": stamps, "tsi_mw": np.where(day % 2 == 1, -100.0, 100.0)})

    matches = direction_matches(frame, [1, 2])

    by_lag = matches.groupby("lag_days")["match"]
    assert by_lag.mean().to_dict() == {1: 0.0, 2: 1.0}
    assert by_lag.size().to_dict() == {1: 3 * 96, 2: 2 * 96}


def test_direction_matches_leaves_out_missing_values_and_gaps():
    stamps = quarter_hours("2026-07-01 00:00", "2026-07-04 00:00")
    frame = pd.DataFrame({"timestamp_local": stamps, "tsi_mw": -1.0})
    frame.loc[0, "tsi_mw"] = np.nan  # 01.07. 00:00
    frame = frame[frame["timestamp_local"].dt.day != 2]  # whole day missing

    matches = direction_matches(frame, [2])

    assert len(matches) == 95
    assert matches["match"].all()


def test_direction_matches_merges_the_repeated_hour_and_drops_a_tie():
    stamps = quarter_hours("2026-10-24 00:00", "2026-10-28 00:00")  # 25.10. has 100 quarter hours
    frame = pd.DataFrame({"timestamp_local": stamps, "tsi_mw": -1.0})
    # The second 02:00 to 02:45 on 25.10. (UTC 01:00 to 01:45) is long: a 1:1 tie with the first one.
    repeated = (stamps.dt.tz_convert("UTC") >= pd.Timestamp("2026-10-25 01:00", tz="UTC")) & (
        stamps.dt.tz_convert("UTC") < pd.Timestamp("2026-10-25 02:00", tz="UTC"))
    frame.loc[repeated, "tsi_mw"] = 1.0

    matches = direction_matches(frame, [1])

    assert len(matches) == 92 + 92 + 96  # 25.10. against 24.10., 26.10. against 25.10., 27.10. against 26.10.
    assert matches["match"].all()


def test_extreme_events_groups_consecutive_quarter_hours_of_the_same_kind():
    stamps = quarter_hours("2026-07-16 00:00", "2026-07-16 02:00", tz="UTC")
    prices = [10, -150, -200, 10, 60, 70, -120, 10]
    frame = pd.DataFrame({"timestamp_utc": stamps, "aep_ct_kwh": prices})

    events = extreme_events(frame, low=-100, high=50)

    assert events["kind"].tolist() == ["low", "high", "low"]
    assert events["quarter_hours"].tolist() == [2, 2, 1]
    assert events["extreme"].tolist() == [-200, 70, -120]
    assert events.loc[0, "end"] == stamps[2]


def test_extreme_events_splits_on_gaps_in_time():
    stamps = pd.Series(pd.to_datetime(["2026-07-16 00:00", "2026-07-16 01:00"], utc=True))
    frame = pd.DataFrame({"timestamp_utc": stamps, "aep_ct_kwh": [-150.0, -150.0]})

    assert len(extreme_events(frame, low=-100, high=50)) == 2


def test_extreme_events_without_any_event_is_empty():
    frame = pd.DataFrame({"timestamp_utc": quarter_hours("2026-07-16", "2026-07-17", tz="UTC"), "aep_ct_kwh": 10.0})

    assert extreme_events(frame, low=-100, high=50).empty


def weather_frame(delivery_days: list[str], valid_utc: list[str], sites: list[str], values: list[float]) -> pd.DataFrame:
    valid = pd.Series(pd.to_datetime(valid_utc, utc=True))
    return pd.DataFrame({
        "delivery_day": pd.to_datetime(delivery_days),
        "site": sites,
        "valid_time_utc": valid,
        "valid_time_local": valid.dt.tz_convert(TZ),
        "radiation": values,
    })


def weather_rows() -> pd.DataFrame:
    return weather_frame(["2026-07-16"] * 4, ["2026-07-16 10:00"] * 2 + ["2026-07-16 11:00"] * 2,
                         ["A", "B", "A", "B"], [100.0, 300.0, 400.0, np.nan])


def test_weather_is_pv_weighted_and_renormalised_over_present_sites():
    out = weather_to_quarter_hours(weather_rows(), pd.Series({"A": 0.75, "B": 0.25}), ["radiation"])

    assert out.loc[pd.Timestamp("2026-07-16 09:00", tz="UTC"), "radiation"] == pytest.approx(150.0)
    assert out.loc[pd.Timestamp("2026-07-16 10:00", tz="UTC"), "radiation"] == pytest.approx(400.0)


def test_weather_value_of_hour_h_covers_the_four_quarter_hours_before_h():
    out = weather_to_quarter_hours(weather_rows(), pd.Series({"A": 0.5, "B": 0.5}), ["radiation"])

    assert out.index[0] == pd.Timestamp("2026-07-16 09:00", tz="UTC")
    assert out.index[-1] == pd.Timestamp("2026-07-16 10:45", tz="UTC")
    assert out["radiation"].tolist() == [200.0] * 4 + [400.0] * 4


def test_weather_stays_missing_when_every_site_misses_it():
    rows = weather_rows().assign(radiation=np.nan)

    out = weather_to_quarter_hours(rows, pd.Series({"A": 0.5, "B": 0.5}), ["radiation"])

    assert out["radiation"].isna().all()


def test_weather_refuses_sites_without_weight():
    with pytest.raises(ValueError, match="B"):
        weather_to_quarter_hours(weather_rows(), pd.Series({"A": 1.0}), ["radiation"])


def test_weather_at_midnight_comes_from_the_run_of_the_day_it_describes():
    # 22:00 UTC is 00:00 local: the last hour of 16.07. in the run for 16.07., the first row of the run for 17.07.
    rows = weather_frame(["2026-07-16", "2026-07-17"], ["2026-07-16 22:00"] * 2, ["A", "A"], [1.0, 2.0])

    out = weather_to_quarter_hours(rows, pd.Series({"A": 1.0}), ["radiation"])

    assert out["radiation"].tolist() == [1.0] * 4
    assert out.index[0] == pd.Timestamp("2026-07-16 23:00", tz=TZ)


def test_weather_of_two_delivery_days_gives_a_gap_free_quarter_hour_index():
    frames = []
    for day, value in [("2026-07-16", 16.0), ("2026-07-17", 17.0)]:
        local = pd.date_range(pd.Timestamp(day, tz=TZ), periods=25, freq="h")
        frames.append(weather_frame([day] * 25, list(local.tz_convert("UTC")), ["A"] * 25, [value] * 25))

    out = weather_to_quarter_hours(pd.concat(frames, ignore_index=True), pd.Series({"A": 1.0}), ["radiation"])

    assert len(out) == 2 * 96
    assert (out.index.to_series().diff().dropna() == pd.Timedelta(minutes=15)).all()
    assert out["radiation"].tolist() == [16.0] * 96 + [17.0] * 96


def test_day_types_refuse_a_timestamp_index():
    share = pd.Series({pd.Timestamp("2026-05-14"): 1.0})

    with pytest.raises(TypeError, match="datetime.date"):
        day_types(pd.Series([dt.date(2026, 5, 14)]), share)


def test_holiday_pv_share_refuses_a_canton_listed_twice_on_one_date():
    holidays = pd.DataFrame({"date": ["2026-01-01", "2026-01-01"], "canton": ["ZH", "ZH"]})
    pv = pd.DataFrame({"canton": ["ZH"], "share": [1.0]})

    with pytest.raises(ValueError, match="twice"):
        holiday_pv_share(holidays, pv)


def test_within_hour_trend_gives_the_change_from_the_first_to_the_last_quarter_hour():
    stamps = quarter_hours("2026-07-16 10:00", "2026-07-16 12:00")
    frame = pd.DataFrame({"timestamp_local": stamps, "tsi_mw": [-40.0, -10.0, 5.0, 30.0, 20.0, 0.0, -5.0, -20.0]})

    out = within_hour_trend(frame)

    assert out["hour"].tolist() == [10, 11]
    assert out[["q00", "q15", "q30", "q45"]].iloc[0].tolist() == [-40.0, -10.0, 5.0, 30.0]
    assert out["trend"].tolist() == [70.0, -40.0]


def test_within_hour_trend_drops_incomplete_and_repeated_hours():
    stamps = quarter_hours("2026-10-25 00:00", "2026-10-25 04:00")  # 02:00 to 02:45 occurs twice
    frame = pd.DataFrame({"timestamp_local": stamps, "tsi_mw": 1.0})
    frame.loc[frame.index[1], "tsi_mw"] = np.nan  # 00:15 missing

    out = within_hour_trend(frame)

    assert out["hour"].tolist() == [1, 3]
