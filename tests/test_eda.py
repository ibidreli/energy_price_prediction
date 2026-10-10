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
    daily_mean_abs_change,
    extreme_events,
    holiday_pv_share,
    ks_distance,
    past_slot_quantiles,
    same_slot_pairs,
    site_spread,
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


def test_site_spread_is_the_standard_deviation_over_the_sites():
    rows = weather_frame(["2026-07-16"] * 3, ["2026-07-16 10:00"] * 3, ["A", "B", "C"], [100.0, 300.0, np.nan])

    out = site_spread(rows, ["radiation"])

    assert out["radiation"].tolist() == [100.0] * 4  # std of 100 and 300; C has no value


def test_site_spread_needs_two_sites_and_keeps_the_own_delivery_day_at_midnight():
    # 22:00 UTC is 00:00 local, in the runs for 16.07. and 17.07.; only the run for 16.07. may describe 16.07. 23:xx
    rows = weather_frame(["2026-07-16", "2026-07-16", "2026-07-17", "2026-07-17"], ["2026-07-16 22:00"] * 4,
                         ["A", "B", "A", "B"], [0.0, 10.0, 500.0, 900.0])

    assert site_spread(rows, ["radiation"])["radiation"].tolist() == [5.0] * 4
    assert site_spread(rows.iloc[[0]], ["radiation"])["radiation"].isna().all()


def test_daily_mean_abs_change_stays_within_the_local_day():
    stamps = pd.date_range("2026-07-15 22:00", "2026-07-16 01:00", freq="15min", inclusive="left", tz="UTC")
    # 15.07. 22:00 UTC = 16.07. 00:00 local; hourly values 0, 100, 300; 21:00 UTC belongs to 15.07. local
    values = pd.Series(np.repeat([0.0, 100.0, 300.0], 4), index=stamps)
    before = pd.Series([50.0] * 4, index=pd.date_range("2026-07-15 21:00", periods=4, freq="15min", tz="UTC"))

    out = daily_mean_abs_change(pd.concat([before, values]))

    assert out[dt.date(2026, 7, 16)] == pytest.approx(150.0)  # mean of 100 and 200, not the step from 50
    assert dt.date(2026, 7, 15) not in out.index  # a single hour has no change


def test_same_slot_pairs_match_the_clock_time_lag_days_earlier():
    stamps = quarter_hours("2026-07-01 00:00", "2026-07-04 00:00")
    frame = pd.DataFrame({"timestamp_local": stamps, "aep_ct_kwh": stamps.dt.day * 10.0})

    pairs = same_slot_pairs(frame, [2])

    assert len(pairs) == 96  # only 03.07. has a value two days earlier
    assert (pairs["value"] == 30.0).all() and (pairs["earlier"] == 10.0).all()


def test_past_slot_quantiles_use_only_the_window_up_to_d_minus_2():
    stamps = quarter_hours("2026-07-01 00:00", "2026-07-17 00:00")
    frame = pd.DataFrame({"timestamp_local": stamps, "aep_ct_kwh": stamps.dt.day.astype(float)})

    out = past_slot_quantiles(frame, dt.date(2026, 7, 16), window_days=7, quantiles=(0.0, 1.0))

    # days 8 to 14 July: nothing from 15 or 16 July (D-1 and D), nothing before the window
    assert out.loc[0, 0.0] == 8.0
    assert out.loc[95, 1.0] == 14.0
    assert list(out.index) == list(range(96))


@pytest.mark.parametrize(("start", "end", "hours"), [
    ("2026-10-25 00:00", "2026-10-26 00:00", 25),  # switch to winter time: 25 hours
    ("2026-03-29 00:00", "2026-03-30 00:00", 23),  # switch to summer time: 23 hours
])
def test_daily_mean_abs_change_works_on_switch_days_with_a_local_index(start, end, hours):
    stamps = pd.date_range(pd.Timestamp(start, tz=TZ), pd.Timestamp(end, tz=TZ), freq="15min", inclusive="left")
    values = pd.Series(np.repeat(np.arange(hours) * 10.0, 4), index=stamps)  # +10 per hour

    out = daily_mean_abs_change(values)

    assert out[pd.Timestamp(start).date()] == pytest.approx(10.0)


def test_daily_mean_abs_change_does_not_bridge_a_missing_hour():
    stamps = pd.date_range("2026-07-16 06:00", periods=4 * 4, freq="15min", tz=TZ)
    values = pd.Series(np.repeat([0.0, 10.0, 20.0, 30.0], 4), index=stamps)
    values = values[values.index.hour != 8]  # 08:00 to 08:45 missing

    out = daily_mean_abs_change(values)

    assert out[dt.date(2026, 7, 16)] == pytest.approx(10.0)  # only 06->07; 07->09 is not one hour


def test_same_slot_pairs_average_the_repeated_hour_and_refuse_bad_lags():
    stamps = quarter_hours("2026-10-24 00:00", "2026-10-27 00:00")
    frame = pd.DataFrame({"timestamp_local": stamps, "aep_ct_kwh": 1.0})

    pairs = same_slot_pairs(frame, [1])

    assert pairs.groupby("date").size().to_dict() == {dt.date(2026, 10, 25): 96, dt.date(2026, 10, 26): 96}
    for bad in ([], [0], [-1]):
        with pytest.raises(ValueError, match="lags_days"):
            same_slot_pairs(frame, bad)


def test_past_slot_quantiles_across_the_switch_to_winter_time():
    stamps = quarter_hours("2026-10-23 00:00", "2026-10-28 00:00")
    frame = pd.DataFrame({"timestamp_local": stamps, "aep_ct_kwh": stamps.dt.day.astype(float)})

    out = past_slot_quantiles(frame, dt.date(2026, 10, 27), window_days=2, quantiles=(0.0, 1.0))

    assert list(out.index) == list(range(96))
    assert out[0.0].min() == 24.0 and out[1.0].max() == 25.0  # 24. and 25.10. only, nothing from 26.10. (D-1)


def test_past_slot_quantiles_refuse_an_empty_or_invalid_window():
    frame = pd.DataFrame({"timestamp_local": quarter_hours("2026-07-01", "2026-07-02"), "aep_ct_kwh": 1.0})

    with pytest.raises(ValueError, match="no values known"):
        past_slot_quantiles(frame, dt.date(2026, 9, 1), window_days=7)
    with pytest.raises(ValueError, match="at least 1"):
        past_slot_quantiles(frame, dt.date(2026, 7, 5), window_days=0)


def test_ks_distance_drops_missing_values_and_refuses_empty_samples():
    assert ks_distance(pd.Series([1.0, np.nan]), pd.Series([1.0])) == 0.0
    with pytest.raises(ValueError, match="at least one value"):
        ks_distance(pd.Series([np.nan]), pd.Series([1.0]))


def test_ks_distance_is_zero_for_equal_and_one_for_disjoint_samples():
    assert ks_distance(pd.Series([1.0, 2.0, 3.0]), pd.Series([3.0, 2.0, 1.0])) == 0.0
    assert ks_distance(pd.Series([1.0, 2.0]), pd.Series([5.0, 6.0])) == 1.0
