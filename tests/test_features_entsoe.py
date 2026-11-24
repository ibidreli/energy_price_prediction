"""Features of the issue #37 data in the modelling table: values and the D-1 11:00 limit."""

from dataclasses import replace
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from energy_price.data import entsoe
from energy_price.data.entsoe import load_entsoe
from energy_price.features.dataset import ENTSOE_COLUMNS, ENTSOE_PROVIDER, build_table
from energy_price.features.design import FEATURE_SETS, design_matrix
from entsoe_store import local, store
from test_features_dataset import inputs, weather_for

D = date(2026, 11, 25)
BEFORE = local(D - timedelta(days=1), 10, 37)
ZONES = ["CH", "DE_LU", "FR", "IT_NORTH", "AT"]


def table(root, day: date = D) -> pd.DataFrame:
    data = replace(inputs(weather_for(day.isoformat())), entsoe=load_entsoe(root))
    return build_table(data, days=[pd.Timestamp(day)])


def at(frame: pd.DataFrame, hour: int, minute: int = 0) -> pd.Series:
    stamps = frame["timestamp_local"]
    return frame[(stamps.dt.hour == hour) & (stamps.dt.minute == minute)].iloc[0]


def store_prices(root, day: date, fetched=BEFORE, values=None, last: date | None = None):
    """Day-ahead prices of all five zones; ``values`` maps zone to a value per hour index."""
    for zone in ZONES:
        store(root, "price_day_ahead", zone, fetched, day, last, value=(values or {}).get(zone, lambda h: 100.0))


def test_the_features_come_from_energy_charts():
    assert ENTSOE_PROVIDER == "energy_charts"


def test_without_data_every_column_exists_and_is_missing(tmp_path):
    frame = table(tmp_path)
    assert frame[ENTSOE_COLUMNS].isna().all().all()
    for feature_set in ["B5", "B6"]:
        assert design_matrix(frame, feature_set).shape[0] == 96


def test_load_forecast_is_the_live_version_fetched_before_11(tmp_path):
    store(tmp_path, "load_day_ahead", "CH", BEFORE, D, value=lambda h: 7000.0 + h)
    store(tmp_path, "load_day_ahead", "CH", local(D - timedelta(days=1), 11, 30), D, value=lambda h: 1.0)
    frame = table(tmp_path)
    assert at(frame, 0)["load_forecast_mw"] == 7000.0
    assert at(frame, 23, 45)["load_forecast_mw"] == 7023.0


def test_values_of_the_other_source_never_mix_in(tmp_path):
    store(tmp_path, "load_day_ahead", "CH", BEFORE, D, value=lambda h: 7000.0)
    store(tmp_path, "load_day_ahead", "CH", BEFORE, D, value=lambda h: 1.0, provider="entsoe_api")
    assert (table(tmp_path)["load_forecast_mw"] == 7000.0).all()


def test_load_change_is_the_centred_difference_over_the_hour(tmp_path):
    store(tmp_path, "load_day_ahead", "CH", BEFORE, D, value=lambda h: 100.0 * h ** 2)
    frame = table(tmp_path)
    # hour 5: (36 - 16) * 100 / 2 = 1000 MW per hour; one-sided at the first and last hour
    assert at(frame, 5, 30)["load_forecast_change_mw"] == 1000.0
    assert at(frame, 0)["load_forecast_change_mw"] == 100.0
    assert at(frame, 23)["load_forecast_change_mw"] == 100.0 * (23 ** 2 - 22 ** 2)
    columns = design_matrix(frame, "B5")
    row = frame.index[(frame["timestamp_local"].dt.hour == 5) & (frame["timestamp_local"].dt.minute == 45)][0]
    assert columns.loc[row, "load_forecast_change_gw_x_position"] == pytest.approx(1.0 * 1.5)


def test_actual_load_is_the_mean_of_a_whole_d_minus_2(tmp_path):
    store(tmp_path, "load_actual", "CH", BEFORE, D - timedelta(days=2), D - timedelta(days=1),
          value=lambda h: 6000.0 + h if h < 24 else 1e6)
    assert (table(tmp_path)["load_actual_mean_d2_mw"] == 6011.5).all()


def test_actual_load_with_a_gap_on_d_minus_2_gives_no_mean(tmp_path):
    store(tmp_path, "load_actual", "CH", BEFORE, D - timedelta(days=2), value=lambda h: None if h == 10 else 6000.0)
    assert table(tmp_path)["load_actual_mean_d2_mw"].isna().all()


def test_price_features_come_from_d_minus_1(tmp_path):
    store_prices(tmp_path, D - timedelta(days=1), values={
        "CH": lambda h: 80.0 + h, "DE_LU": lambda h: 70.0, "FR": lambda h: 60.0, "IT_NORTH": lambda h: 120.0,
        "AT": lambda h: 90.0})
    frame = table(tmp_path)
    assert at(frame, 7, 15)["da_price_ch_d1"] == 87.0
    assert (frame["da_price_ch_mean_d1"] == 91.5).all()
    assert at(frame, 0)["da_spread_de_lu_d1"] == 21.5
    assert at(frame, 0)["da_spread_it_north_d1"] == -28.5


def test_the_prices_of_the_delivery_day_change_no_feature(tmp_path):
    store_prices(tmp_path, D - timedelta(days=1), values={"CH": lambda h: 80.0 + h})
    store(tmp_path, "load_day_ahead", "CH", BEFORE, D, value=lambda h: 7000.0)
    before = table(tmp_path)
    early = local(D - timedelta(days=1), 9, 17)
    store_prices(tmp_path, D, fetched=early, last=D + timedelta(days=1), values={zone: lambda h: 1e6 for zone in ZONES})
    after = table(tmp_path)
    pd.testing.assert_frame_equal(before[ENTSOE_COLUMNS], after[ENTSOE_COLUMNS])
    assert before[["da_price_ch_d1", "load_forecast_mw"]].notna().all().all()


def test_a_gap_in_the_live_period_is_not_filled(tmp_path):
    store_prices(tmp_path, D - timedelta(days=4), fetched=local(D - timedelta(days=5), 10, 17))
    store(tmp_path, "load_day_ahead", "CH", local(D + timedelta(days=2), 12, 0), date(2026, 11, 1),
          D + timedelta(days=1), source="backfill")
    assert table(tmp_path)[ENTSOE_COLUMNS].isna().all().all()


def test_history_load_forecast_waits_for_the_rule(tmp_path, monkeypatch):
    day, first, last = date(2026, 2, 10), date(2026, 2, 1), date(2026, 2, 28)
    fetched = local(date(2026, 11, 19), 12, 0)
    store(tmp_path, "load_day_ahead", "CH", fetched, first, last, value=lambda h: 7000.0, source="backfill")
    store(tmp_path, "price_day_ahead", "CH", fetched, first, last, value=lambda h: 50.0, source="backfill")
    frame = table(tmp_path, day)
    assert frame["load_forecast_mw"].isna().all()
    assert (frame["da_price_ch_mean_d1"] == 50.0).all()

    monkeypatch.setitem(entsoe.HISTORY_RULES["energy_charts"], "load_day_ahead", True)
    assert (table(tmp_path, day)["load_forecast_mw"] == 7000.0).all()


def test_after_the_short_day_02_xx_takes_the_daily_mean(tmp_path):
    first, last = date(2026, 3, 28), date(2026, 3, 30)
    store(tmp_path, "price_day_ahead", "CH", local(date(2026, 11, 19), 12, 0), first, last, value=float,
          source="backfill")
    frame = table(tmp_path, date(2026, 3, 30))
    # 29 March runs from hour index 24 (00:00) to 46 (23:00), 23 hours without 02:00
    assert at(frame, 1)["da_price_ch_d1"] == 25.0
    assert at(frame, 2, 30)["da_price_ch_d1"] == pytest.approx(np.mean(range(24, 47)))
    assert at(frame, 3)["da_price_ch_d1"] == 26.0


def test_after_the_long_day_02_xx_averages_both_passes(tmp_path):
    day = date(2026, 10, 25)
    store(tmp_path, "price_day_ahead", "CH", local(date(2026, 11, 19), 12, 0), day, value=float, source="backfill")
    frame = table(tmp_path, date(2026, 10, 26))
    # hour index 2 is 02:00 summer time, 3 is 02:00 winter time
    assert at(frame, 2, 15)["da_price_ch_d1"] == 2.5
    assert at(frame, 4)["da_price_ch_d1"] == 5.0


def test_residual_load_subtracts_solar_and_wind(tmp_path):
    store(tmp_path, "solar_day_ahead", "CH", BEFORE, D, value=lambda h: 300.0)
    store(tmp_path, "wind_day_ahead", "CH", BEFORE, D, value=lambda h: 20.0)
    store(tmp_path, "load_day_ahead", "CH", BEFORE, D, value=lambda h: 7000.0)
    frame = table(tmp_path)
    assert (frame["solar_forecast_mw"] == 300.0).all() and (frame["wind_forecast_mw"] == 20.0).all()
    assert (frame["residual_load_mw"] == 6680.0).all()
    assert (frame["residual_load_change_mw"] == 0.0).all()


def test_b5_and_b6_add_the_blocks_to_b4():
    assert FEATURE_SETS["B5"] == [*FEATURE_SETS["B4"], "load_forecast"]
    assert FEATURE_SETS["B6"] == [*FEATURE_SETS["B5"], "day_ahead_prices"]
