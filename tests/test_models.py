"""Shared model building blocks: reference, feature sets and linear quantile regression."""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from energy_price.features.dataset import calendar_features, quarter_hours
from energy_price.features.design import design_matrix
from energy_price.models.shared.linear import LinearQuantileModel
from energy_price.models.shared.reference import reference_forecast

TZ = "Europe/Zurich"


def price_history(first: str, last: str, value) -> pd.DataFrame:
    days = list(pd.date_range(first, last, freq="D"))
    grid = quarter_hours(days)
    grid["aep_ct_kwh"] = [value(row) for row in grid.itertuples()]
    return grid


def synthetic_table(days: int = 28, seed: int = 0) -> pd.DataFrame:
    """Price = 5 + 2 per 100 W/m² of radiation + symmetric noise."""
    rng = np.random.default_rng(seed)
    table = quarter_hours(list(pd.date_range("2026-06-01", periods=days, freq="D")))
    table = pd.concat([table, calendar_features(table["timestamp_local"])], axis=1)
    table["holiday_pv_share"] = 0.0
    table["radiation"] = rng.uniform(0, 800, len(table))
    table["radiation_change"] = rng.normal(0, 50, len(table))
    table["aep_ct_kwh"] = 5 + 2 * table["radiation"] / 100 + rng.normal(0, 3, len(table))
    return table


def test_reference_uses_only_the_window_up_to_d_minus_2():
    day = dt.date(2026, 7, 16)
    # 1 inside the window D-57 to D-2, 1000 on D-1 (unknown at 11:00) and on D-58 (outside the window)
    history = price_history("2026-05-01", "2026-07-16", lambda r: 1.0)
    d = history["timestamp_local"].dt.tz_localize(None).dt.normalize()
    history.loc[d == pd.Timestamp("2026-07-15"), "aep_ct_kwh"] = 1000.0
    history.loc[d == pd.Timestamp("2026-05-19"), "aep_ct_kwh"] = 1000.0
    target = quarter_hours([pd.Timestamp(day)])

    forecast = reference_forecast(history, target, day, window_days=56)

    assert (forecast[["q10", "q50", "q90"]] == 1.0).all().all()


def test_reference_window_includes_d_minus_57_and_d_minus_2():
    day = dt.date(2026, 7, 16)
    history = price_history("2026-05-20", "2026-07-14", lambda r: 0.0)  # D-57 to D-2, exactly 56 days
    d = history["timestamp_local"].dt.tz_localize(None).dt.normalize()
    history.loc[d == pd.Timestamp("2026-05-20"), "aep_ct_kwh"] = 100.0
    history.loc[d == pd.Timestamp("2026-07-14"), "aep_ct_kwh"] = 100.0
    target = quarter_hours([pd.Timestamp(day)])

    forecast = reference_forecast(history, target, day, window_days=56)

    assert forecast["q90"].iloc[0] == pytest.approx(np.quantile([0.0] * 54 + [100.0] * 2, 0.9))


def test_reference_quantiles_are_per_clock_time():
    day = dt.date(2026, 7, 16)
    history = price_history("2026-05-01", "2026-07-14", lambda r: float(r.timestamp_local.hour))
    target = quarter_hours([pd.Timestamp(day)])

    forecast = reference_forecast(history, target, day)

    assert (forecast["q50"].to_numpy() == target["timestamp_local"].dt.hour.to_numpy()).all()


def test_feature_sets_differ_only_by_the_weather_columns():
    table = synthetic_table(days=2)

    b1, b2 = design_matrix(table, "B1"), design_matrix(table, "B2")

    assert set(b2.columns) - set(b1.columns) == {"radiation_100", "radiation_change_100_x_position"}
    assert "hour_00" not in b1.columns  # reference category, absorbed by the intercept


def with_history_columns(table: pd.DataFrame) -> pd.DataFrame:
    return table.assign(price_mean_d2=1.0, price_mean_7d=2.0, price_q10_56d=3.0, price_q50_56d=4.0, price_q90_56d=5.0,
                        tsi_mean_d2=-250.0, short_share_d2=0.5, direction_d2=-1.0, direction_d7=1.0, cloud_cover=40.0)


def test_b3_adds_exactly_the_price_level_to_b2():
    table = with_history_columns(synthetic_table(days=2))

    b2, b3 = design_matrix(table, "B2"), design_matrix(table, "B3")

    assert list(b3.columns[:len(b2.columns)]) == list(b2.columns)
    assert list(b3.columns[len(b2.columns):]) == ["price_mean_d2", "price_mean_7d", "price_q10_56d", "price_q50_56d",
                                                  "price_q90_56d"]
    assert b3.shape[1] == 56


def test_b4_adds_exactly_system_balance_and_cloud_cover_to_b3():
    table = with_history_columns(synthetic_table(days=2))

    b3, b4 = design_matrix(table, "B3"), design_matrix(table, "B4")

    assert list(b4.columns[:len(b3.columns)]) == list(b3.columns)
    assert list(b4.columns[len(b3.columns):]) == ["tsi_mean_d2_100", "short_share_d2", "direction_d2", "direction_d7",
                                                  "cloud_cover_100"]
    assert (b4["tsi_mean_d2_100"] == -2.5).all()
    assert (b4["cloud_cover_100"] == 0.4).all()


def test_b1_and_b2_do_not_use_the_history_columns():
    table = synthetic_table(days=2)

    plain = {fs: design_matrix(table, fs) for fs in ("B1", "B2")}
    with_history = with_history_columns(table)

    for fs, expected in plain.items():
        pd.testing.assert_frame_equal(design_matrix(with_history, fs), expected)


def test_rows_without_minimum_history_are_left_out_of_training_for_b3_only():
    table = with_history_columns(synthetic_table(days=14))
    early = table["delivery_day"] < pd.Timestamp("2026-06-08")
    table.loc[early, "price_q50_56d"] = np.nan

    b2 = LinearQuantileModel("B2", alpha=0.01).fit(table)
    b3 = LinearQuantileModel("B3", alpha=0.01).fit(table)

    assert b2.n_train_ == len(table)
    assert b3.n_train_ == (~early).sum()
    assert b3.predict(table).loc[early].isna().all().all()
    assert b3.predict(table).loc[~early].notna().all().all()


def test_linear_model_recovers_the_radiation_effect_and_keeps_quantiles_ordered():
    table = synthetic_table()

    model = LinearQuantileModel("B2", alpha=0.0).fit(table)
    forecast = model.predict(table)
    coef = model.coefficients()

    assert coef.loc["radiation_100", "q50"] == pytest.approx(2.0, abs=0.15)
    assert (forecast["q10"] <= forecast["q50"]).all() and (forecast["q50"] <= forecast["q90"]).all()
    inside = ((table["aep_ct_kwh"] >= forecast["q10"]) & (table["aep_ct_kwh"] <= forecast["q90"])).mean()
    assert inside == pytest.approx(0.8, abs=0.03)


def test_rows_with_missing_features_get_no_forecast():
    table = synthetic_table(days=7)
    model = LinearQuantileModel("B2").fit(table)
    table.loc[table.index[:4], "radiation"] = np.nan

    forecast = model.predict(table)

    assert forecast.iloc[:4].isna().all().all()
    assert forecast.iloc[4:].notna().all().all()


def test_unknown_feature_set_is_rejected():
    with pytest.raises(ValueError, match="feature set"):
        LinearQuantileModel("B9")
