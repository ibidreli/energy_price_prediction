import numpy as np
import pandas as pd
import pytest

from energy_price.evaluation.backtest import PERIODS
from energy_price.features.dataset import (
    HISTORY_COLUMNS,
    Inputs,
    build_table,
    calendar_features,
    holiday_pv_share,
    load_inputs,
    quarter_hours,
    radiation_by_hour,
)
from energy_price.features.design import design_matrix
from energy_price.models.shared.reference import reference_forecast
from energy_price.paths import ROOT

TZ = "Europe/Zurich"
SITES = pd.DataFrame({"site": ["A", "B"], "pv_share": [0.75, 0.25]})
PROCESSED = ROOT / "data" / "processed"
SOURCE_TABLES = ["balance_prices", "control_area_balance", "weather_forecasts"]


def weather_for(day: str, value_of_stamp=lambda stamp: float(stamp.hour), sites=("A", "B"),
                cloud_of_stamp=lambda stamp: 50.0) -> pd.DataFrame:
    """Hourly stamps D 00:00 to D+1 00:00 local, as in weather_forecasts.parquet."""
    start = pd.Timestamp(day).tz_localize(TZ)
    end = (pd.Timestamp(day) + pd.Timedelta(days=1)).tz_localize(TZ)
    stamps = pd.date_range(start, end, freq="h").tz_convert("UTC")
    rows = [
        {"delivery_day": pd.Timestamp(day), "site": s, "valid_time_utc": t, "shortwave_radiation": value_of_stamp(t),
         "cloud_cover": cloud_of_stamp(t)}
        for s in sites for t in stamps
    ]
    return pd.DataFrame(rows)


def empty_series(column: str) -> pd.DataFrame:
    return pd.DataFrame({"timestamp_utc": pd.Series([], dtype="datetime64[us, UTC]"),
                         "timestamp_local": pd.Series([], dtype=f"datetime64[us, {TZ}]"),
                         column: pd.Series([], dtype=float)})


def inputs(weather: pd.DataFrame, prices: pd.DataFrame | None = None, holidays: pd.DataFrame | None = None,
           balance: pd.DataFrame | None = None) -> Inputs:
    if prices is None:
        prices = empty_series("aep_ct_kwh")
    if balance is None:
        balance = empty_series("tsi_mw")
    if holidays is None:
        holidays = pd.DataFrame({"date": [], "canton": []})
    return Inputs(prices=prices, weather=weather, sites=SITES, holidays=holidays,
                  pv_capacity=pd.DataFrame({"canton": ["ZH", "BE"], "share": [0.6, 0.4]}), balance=balance)


def series(first: str, last: str, column: str, value=lambda stamp: 1.0) -> pd.DataFrame:
    """A Swissgrid table with one value per quarter hour of the local days ``first`` to ``last``."""
    grid = quarter_hours(list(pd.date_range(first, last, freq="D")))
    return pd.DataFrame({"timestamp_utc": grid["timestamp_utc"], "timestamp_local": grid["timestamp_local"],
                         column: [float(value(stamp)) for stamp in grid["timestamp_local"]]})


def prices_of(first: str = "2026-01-01", last: str = "2026-11-05", value=lambda stamp: 1.0) -> pd.DataFrame:
    return series(first, last, "aep_ct_kwh", value)


def balance_of(first: str = "2026-01-01", last: str = "2026-11-05", value=lambda stamp: 1.0) -> pd.DataFrame:
    return series(first, last, "tsi_mw", value)


def on_day(frame: pd.DataFrame, day: str) -> pd.Series:
    return frame["timestamp_local"].dt.tz_localize(None).dt.normalize() == pd.Timestamp(day)


def table_for(day: str, prices: pd.DataFrame | None = None, balance: pd.DataFrame | None = None) -> pd.DataFrame:
    return build_table(inputs(weather_for(day), prices, balance=balance), days=[pd.Timestamp(day)])


def at_clock(table: pd.DataFrame, hour: int, minute: int = 0) -> pd.DataFrame:
    stamps = table["timestamp_local"]
    return table[(stamps.dt.hour == hour) & (stamps.dt.minute == minute)]


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


# --- B3: price level until D-2 -------------------------------------------------------------------


def test_daily_price_mean_uses_only_d_minus_2():
    prices = prices_of(value=lambda stamp: stamp.day)  # the day of the month as price

    table = table_for("2026-07-16", prices)

    assert (table["price_mean_d2"] == 14.0).all()


def test_seven_day_mean_covers_d_minus_8_to_d_minus_2():
    prices = prices_of(value=lambda stamp: stamp.day)

    table = table_for("2026-07-16", prices)

    assert table["price_mean_7d"].iloc[0] == pytest.approx(np.mean(range(8, 15)))


@pytest.mark.parametrize("day", ["2026-02-27", "2026-04-05", "2026-07-16", "2026-11-01"])
def test_price_quantiles_equal_the_reference(day):
    # Also after the short day (no 02:xx) and the long day (02:xx twice) inside the window.
    rng = np.random.default_rng(0)
    prices = prices_of()
    prices["aep_ct_kwh"] = rng.normal(5, 10, len(prices))

    table = table_for(day, prices)
    expected = reference_forecast(prices, table, pd.Timestamp(day).date())

    assert table["price_q10_56d"].to_numpy() == pytest.approx(expected["q10"].to_numpy())
    assert table["price_q50_56d"].to_numpy() == pytest.approx(expected["q50"].to_numpy())
    assert table["price_q90_56d"].to_numpy() == pytest.approx(expected["q90"].to_numpy())


def test_price_quantiles_leave_out_an_incomplete_day_inside_the_window():
    # Unlike the reference, which also uses the quarter hours of the incomplete day.
    rng = np.random.default_rng(3)
    prices = prices_of()
    prices["aep_ct_kwh"] = rng.normal(5, 10, len(prices))
    gap_day = on_day(prices, "2026-07-01")
    partial = prices[~(gap_day & (prices["timestamp_local"].dt.hour == 12))]

    table = table_for("2026-07-16", partial)
    without_day = reference_forecast(prices[~gap_day], table, pd.Timestamp("2026-07-16").date())
    with_partial_day = reference_forecast(partial, table, pd.Timestamp("2026-07-16").date())

    assert table["price_q50_56d"].to_numpy() == pytest.approx(without_day["q50"].to_numpy())
    assert table["price_q50_56d"].to_numpy() != pytest.approx(with_partial_day["q50"].to_numpy())


def test_price_quantiles_are_per_clock_time():
    prices = prices_of(value=lambda stamp: stamp.hour)

    table = table_for("2026-07-16", prices)

    assert (table["price_q50_56d"].to_numpy() == table["hour"].to_numpy()).all()


def test_quantiles_need_28_days_of_history():
    prices = prices_of()

    assert table_for("2026-01-29", prices)["price_q50_56d"].isna().all()  # D-2 is 01-27, 27 days
    assert table_for("2026-01-30", prices)["price_q50_56d"].notna().all()  # D-2 is 01-28, 28 days


def test_means_start_as_soon_as_their_days_exist():
    prices = prices_of()

    assert table_for("2026-01-02", prices)["price_mean_d2"].isna().all()
    assert table_for("2026-01-03", prices)["price_mean_d2"].notna().all()
    assert table_for("2026-01-08", prices)["price_mean_7d"].isna().all()
    assert table_for("2026-01-09", prices)["price_mean_7d"].notna().all()


def test_prices_and_balance_before_2026_are_never_used():
    # BG-AEP exists from 2025-07 but the single price system only applies since 2026-01-01.
    prices = prices_of("2025-07-01", "2026-03-31")
    balance = balance_of("2025-07-01", "2026-03-31")

    table = table_for("2026-01-02", prices, balance)

    assert table[HISTORY_COLUMNS].isna().all().all()


def test_incomplete_d_minus_2_gives_no_price_features_at_all():
    # Quantiles of an old window would otherwise continue silently after the data ends.
    prices = prices_of(last="2026-07-14")
    prices = prices[prices["timestamp_local"] < pd.Timestamp("2026-07-14 12:00", tz=TZ)]

    table = table_for("2026-07-16", prices)

    columns = ["price_mean_d2", "price_mean_7d", "price_q10_56d", "price_q50_56d", "price_q90_56d"]
    assert table[columns].isna().all().all()


def test_seven_day_mean_needs_all_seven_days():
    prices = prices_of()
    prices = prices[~on_day(prices, "2026-07-11")]

    table = table_for("2026-07-16", prices)

    assert table["price_mean_7d"].isna().all()
    assert table["price_mean_d2"].notna().all()
    assert table["price_q50_56d"].notna().all()


# --- B4: system balance until D-2 ----------------------------------------------------------------


def test_mean_tsi_and_short_share_use_only_d_minus_2():
    balance = balance_of(value=lambda stamp: -100.0 if stamp.day == 14 and stamp.hour < 6 else 100.0)

    table = table_for("2026-07-16", balance=balance)

    assert (table["tsi_mean_d2"] == 50.0).all()  # 6 hours at -100, 18 hours at +100
    assert (table["short_share_d2"] == 0.25).all()


def test_direction_is_the_sign_of_the_tsi_at_the_same_clock_time():
    def tsi(stamp):
        if stamp.day == 14:
            return -5.0 if stamp.hour == 12 else 5.0
        if stamp.day == 9:
            return -5.0 if stamp.hour == 13 else 5.0
        return 5.0

    table = table_for("2026-07-16", balance=balance_of(value=tsi))

    assert (at_clock(table, 12)[["direction_d2"]] == -1.0).all().all()
    assert (at_clock(table, 12)[["direction_d7"]] == 1.0).all().all()
    assert (at_clock(table, 13)[["direction_d2"]] == 1.0).all().all()
    assert (at_clock(table, 13)[["direction_d7"]] == -1.0).all().all()


def test_zero_tsi_counts_as_long_like_the_evaluation():
    table = table_for("2026-07-16", balance=balance_of(value=lambda stamp: 0.0))

    assert (table["direction_d2"] == 1.0).all()
    assert (table["short_share_d2"] == 0.0).all()


def test_incomplete_d_minus_2_gives_no_balance_features_at_all():
    balance = balance_of(last="2026-07-14")
    balance = balance[balance["timestamp_local"] < pd.Timestamp("2026-07-14 12:00", tz=TZ)]

    table = table_for("2026-07-16", balance=balance)

    assert table[["tsi_mean_d2", "short_share_d2", "direction_d2", "direction_d7"]].isna().all().all()


# --- Leakage: the D-2 boundary, also around the clock changes --------------------------------------

BOUNDARY_DAYS = ["2026-03-29", "2026-03-30", "2026-03-31", "2026-04-05",
                 "2026-07-16", "2026-10-25", "2026-10-26", "2026-10-27", "2026-11-01"]


@pytest.mark.parametrize("day", BOUNDARY_DAYS)
def test_changes_from_d_minus_1_midnight_change_no_feature(day):
    rng = np.random.default_rng(1)
    prices, balance = prices_of(), balance_of()
    prices["aep_ct_kwh"] = rng.normal(5, 10, len(prices))
    balance["tsi_mw"] = rng.normal(0, 100, len(balance))
    cutoff = (pd.Timestamp(day) - pd.Timedelta(days=1)).tz_localize(TZ)
    before = table_for(day, prices, balance)

    prices.loc[prices["timestamp_local"] >= cutoff, "aep_ct_kwh"] = 9999.0
    balance.loc[balance["timestamp_local"] >= cutoff, "tsi_mw"] = -9999.0
    after = table_for(day, prices, balance)

    pd.testing.assert_frame_equal(before[HISTORY_COLUMNS], after[HISTORY_COLUMNS])
    assert before[HISTORY_COLUMNS].notna().all().all()


@pytest.mark.parametrize("day", BOUNDARY_DAYS)
def test_the_last_known_quarter_hour_d_minus_2_23_45_is_used(day):
    # Guards against a boundary that is stricter than necessary or off by one quarter hour.
    prices, balance = prices_of(), balance_of()
    last = (pd.Timestamp(day) - pd.Timedelta(days=1)).tz_localize(TZ) - pd.Timedelta(minutes=15)
    assert last.strftime("%H:%M") == "23:45"
    before = table_for(day, prices, balance)

    prices.loc[prices["timestamp_local"] == last, "aep_ct_kwh"] = 9999.0
    balance.loc[balance["timestamp_local"] == last, "tsi_mw"] = -9999.0
    after = table_for(day, prices, balance)

    assert (after["price_mean_d2"] > before["price_mean_d2"]).all()
    assert (after["tsi_mean_d2"] < before["tsi_mean_d2"]).all()
    assert (after["short_share_d2"] > before["short_share_d2"]).all()


# --- Clock changes: 92 and 100 quarter hours -------------------------------------------------------


def test_daily_features_use_all_92_quarter_hours_of_the_short_day():
    prices = prices_of(value=lambda stamp: 92.0 if stamp.hour == 1 and stamp.minute == 0 else 0.0)
    balance = balance_of(value=lambda stamp: -1.0 if stamp.hour == 1 else 1.0)  # 01:xx short, 4 quarter hours

    table = table_for("2026-03-31", prices, balance)  # D-2 is the short day 2026-03-29

    assert table["price_mean_d2"].iloc[0] == pytest.approx(92.0 / 92)
    assert table["short_share_d2"].iloc[0] == pytest.approx(4 / 92)


def test_daily_features_use_all_100_quarter_hours_of_the_long_day():
    prices = prices_of(value=lambda stamp: 100.0 if stamp.hour == 1 and stamp.minute == 0 else 0.0)
    balance = balance_of(value=lambda stamp: -1.0 if stamp.hour == 2 else 1.0)  # both passes of 02:xx short

    table = table_for("2026-10-27", prices, balance)  # D-2 is the long day 2026-10-25

    assert table["price_mean_d2"].iloc[0] == pytest.approx(100.0 / 100)
    assert table["short_share_d2"].iloc[0] == pytest.approx(8 / 100)


@pytest.mark.parametrize(("day", "column", "other"), [("2026-03-31", "direction_d2", "direction_d7"),
                                                      ("2026-04-05", "direction_d7", "direction_d2")])
def test_direction_is_0_for_the_hour_the_short_day_does_not_have(day, column, other):
    # 2026-03-29 has no 02:xx. A NaN would remove the delivery day for every model in the comparison.
    table = table_for(day, balance=balance_of(value=lambda stamp: -1.0))

    night = table["hour"] == 2
    assert night.sum() == 4
    assert (table.loc[night, column] == 0.0).all()
    assert (table.loc[~night, column] == -1.0).all()
    assert (table[other] == -1.0).all()


@pytest.mark.parametrize(("first", "second", "expected"), [(-1.0, -1.0, -1.0), (1.0, 1.0, 1.0),
                                                           (-1.0, 1.0, 0.0), (1.0, -1.0, 0.0)])
@pytest.mark.parametrize(("day", "column"), [("2026-10-27", "direction_d2"), ("2026-11-01", "direction_d7")])
def test_direction_averages_both_passes_of_the_repeated_hour(day, column, first, second, expected):
    # 2026-10-25 has 02:xx twice, first at +02:00 (summer time) and then at +01:00 (winter time).
    def tsi(stamp):
        if stamp.strftime("%Y-%m-%d %H") != "2026-10-25 02":
            return 7.0
        return first if stamp.utcoffset() == pd.Timedelta(hours=2) else second

    table = table_for(day, balance=balance_of(value=tsi))

    night = table["hour"] == 2
    assert (table.loc[night, column] == expected).all()
    assert (table.loc[~night, column] == 1.0).all()


def test_both_passes_of_a_repeated_target_hour_get_the_same_history():
    # They share a clock time, so they share the slot of D-2, D-7 and the 56-day quantiles.
    rng = np.random.default_rng(2)
    prices, balance = prices_of(), balance_of()
    prices["aep_ct_kwh"] = rng.normal(5, 10, len(prices))
    balance["tsi_mw"] = rng.normal(0, 100, len(balance))

    table = table_for("2026-10-25", prices, balance)

    assert len(table) == 100
    for minute in (0, 15, 30, 45):
        passes = at_clock(table, 2, minute)
        assert len(passes) == 2
        assert passes[HISTORY_COLUMNS].nunique().eq(1).all()


# --- Cloud cover ----------------------------------------------------------------------------------


def test_cloud_cover_is_pv_weighted_and_uses_the_stamp_at_the_end_of_the_hour():
    a = weather_for("2026-07-16", sites=("A",), cloud_of_stamp=lambda t: float(t.tz_convert(TZ).hour))
    b = weather_for("2026-07-16", sites=("B",), cloud_of_stamp=lambda t: float(t.tz_convert(TZ).hour) + 40.0)

    table = build_table(inputs(pd.concat([a, b], ignore_index=True)), days=[pd.Timestamp("2026-07-16")])

    assert (table.loc[table["hour"] == 12, "cloud_cover"] == 0.75 * 13 + 0.25 * 53).all()
    assert (table.loc[table["hour"] == 23, "cloud_cover"] == 0.25 * 40).all()  # stamp D+1 00:00 of the own run


def test_cloud_cover_never_comes_from_the_next_run():
    own = weather_for("2026-07-16", cloud_of_stamp=lambda t: 10.0)
    later_run = weather_for("2026-07-17", cloud_of_stamp=lambda t: 99.0)

    table = build_table(inputs(pd.concat([own, later_run], ignore_index=True)), days=[pd.Timestamp("2026-07-16")])

    assert (table["cloud_cover"] == 10.0).all()


# --- Real data ------------------------------------------------------------------------------------


@pytest.mark.realdata
@pytest.mark.skipif(not all((PROCESSED / f"{name}.parquet").exists() for name in SOURCE_TABLES),
                    reason="processed tables not built, run make data")
@pytest.mark.parametrize("period", ["tune", "test"])
def test_b3_and_b4_lose_no_delivery_day_that_b2_has(period):
    # A missing feature removes a day for every model, so B3 and B4 must not cost any further days.
    days = list(pd.date_range(*PERIODS[period], freq="D"))
    table = build_table(load_inputs(), days=days)

    def complete(feature_set):
        ok = design_matrix(table, feature_set).notna().all(axis=1)
        return set(ok.groupby(table["delivery_day"]).all().loc[lambda s: s].index)

    assert complete("B3") == complete("B2")
    assert complete("B4") == complete("B2")
    assert len(complete("B2")) >= len(days) - 1  # 2026-06-24 has no radiation
