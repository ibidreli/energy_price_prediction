"""Tests of the quantile LASSO (issue #27): settings, interface, leakage and the comparison with other models."""

import numpy as np
import pandas as pd
import pytest

from energy_price.evaluation.backtest import REFERENCE, evaluate, run_reference
from energy_price.evaluation.experiment import label, load, run_variant
from energy_price.features.dataset import Inputs, build_table, calendar_features, quarter_hours
from energy_price.models.quantile_lasso.diagnostics import (
    BASELINE,
    BOOSTING,
    analyse,
    best_per_feature_set,
    comparison,
    first_complete_day,
    surviving_features,
)
from energy_price.models.shared.base import QuantileModel
from energy_price.models.shared.linear import LinearQuantileModel

NAME = "quantile_lasso"
TZ = "Europe/Zurich"
SITES = pd.DataFrame({"site": ["A"], "pv_share": [1.0]})
LASSO = {"window": "expanding", "alpha": 0.03}


def synthetic_table(first: str = "2026-03-01", days: int = 40, seed: int = 0) -> pd.DataFrame:
    """Price depends on the hour and on the level of D-2; all columns of B4 are filled."""
    rng = np.random.default_rng(seed)
    table = quarter_hours(list(pd.date_range(first, periods=days, freq="D")))
    table = pd.concat([table, calendar_features(table["timestamp_local"])], axis=1)
    table["holiday_pv_share"] = 0.0
    table["radiation"] = rng.uniform(0, 800, len(table))
    table["radiation_change"] = rng.normal(0, 50, len(table))
    table["cloud_cover"] = rng.uniform(0, 100, len(table))
    level = table["delivery_day"].map({d: rng.normal(0, 5) for d in table["delivery_day"].unique()})
    table["price_mean_d2"] = level
    table["price_mean_7d"] = level / 2
    table["price_q10_56d"], table["price_q50_56d"], table["price_q90_56d"] = -5.0, 5.0, 15.0
    table["tsi_mean_d2"] = rng.normal(0, 100, len(table))
    table["short_share_d2"] = rng.uniform(0, 1, len(table))
    table["direction_d2"] = rng.choice([-1.0, 1.0], len(table))
    table["direction_d7"] = rng.choice([-1.0, 1.0], len(table))
    table["aep_ct_kwh"] = 10 + table["hour"] + level + rng.normal(0, 2, len(table))
    return table


def weather_of(days: pd.DatetimeIndex) -> pd.DataFrame:
    frames = []
    for day in days:
        stamps = pd.date_range(day.tz_localize(TZ), (day + pd.Timedelta(days=1)).tz_localize(TZ), freq="h")
        frames.append(pd.DataFrame({"delivery_day": day, "site": "A", "valid_time_utc": stamps.tz_convert("UTC"),
                                    "shortwave_radiation": 100.0 * stamps.hour, "cloud_cover": 50.0}))
    return pd.concat(frames, ignore_index=True)


def swissgrid_table(days: pd.DatetimeIndex, column: str, seed: int) -> pd.DataFrame:
    grid = quarter_hours(list(days))
    values = np.random.default_rng(seed).normal(5, 10, len(grid))
    return pd.DataFrame({"timestamp_utc": grid["timestamp_utc"], "timestamp_local": grid["timestamp_local"],
                         column: values})


def inputs_of(prices: pd.DataFrame, balance: pd.DataFrame, days: pd.DatetimeIndex) -> Inputs:
    return Inputs(prices=prices, weather=weather_of(days), sites=SITES, holidays=pd.DataFrame({"date": [], "canton": []}),
                  pv_capacity=pd.DataFrame({"canton": ["ZH"], "share": [1.0]}), balance=balance)


def forecast_for(day: pd.Timestamp, prices: pd.DataFrame, balance: pd.DataFrame, days: pd.DatetimeIndex) -> pd.DataFrame:
    table = build_table(inputs_of(prices, balance, days), days=list(days))
    settings = {"feature_set": "B4", **LASSO}
    forecast, _ = run_variant(load(NAME), table, [day], "lasso", settings, jobs=1)
    return forecast[["q10", "q50", "q90"]]


def test_grid_is_the_one_of_the_issue():
    grid = load(NAME).GRID

    assert len(grid) == 12
    assert {s["feature_set"] for s in grid} == {"B2", "B3", "B4"}
    assert {s["alpha"] for s in grid} == {0.001, 0.003, 0.01, 0.03}
    assert {s["window"] for s in grid} == {"expanding"}


def test_chosen_only_holds_settings_of_the_grid():
    module = load(NAME)

    assert all(settings in module.GRID for settings in module.CHOSEN.values())


def test_model_fulfils_the_interface_with_sorted_quantiles():
    table = synthetic_table()

    model = load(NAME).MODEL(feature_set="B4", alpha=0.03).fit(table)
    out = model.predict(table)

    assert isinstance(model, QuantileModel)
    assert list(out.columns) == ["q10", "q50", "q90"]
    assert out.index.equals(table.index)
    assert ((out["q10"] <= out["q50"]) & (out["q50"] <= out["q90"])).all()


def test_no_forecast_for_a_row_without_history():
    table = synthetic_table()
    model = load(NAME).MODEL(feature_set="B3", alpha=0.03).fit(table)

    out = model.predict(table.assign(price_mean_d2=np.nan))

    assert out.isna().all().all()


def test_backtest_runs_end_to_end_on_the_same_quarter_hours_as_the_reference():
    table = synthetic_table()
    days = list(pd.date_range("2026-04-01", "2026-04-08", freq="D"))

    forecast, coefficients = run_variant(load(NAME), table, days, "lasso", {"feature_set": "B3", **LASSO}, jobs=1)
    scores, _ = evaluate(pd.concat([run_reference(table, days), forecast], ignore_index=True))

    assert set(scores.index) == {REFERENCE, "lasso"}
    assert scores.loc["lasso", "n_days"] == len(days)
    assert "price_mean_d2" in set(coefficients["feature"])


class TestLeakage:
    """Through the whole pipeline: raw tables to forecast, unlike the feature test of #28."""

    DAY = pd.Timestamp("2026-03-10")
    DAYS = pd.date_range("2026-01-01", DAY, freq="D")

    def tables(self):
        return swissgrid_table(self.DAYS, "aep_ct_kwh", 1), swissgrid_table(self.DAYS, "tsi_mw", 2)

    def test_changes_from_d_minus_1_do_not_change_the_forecast_of_d(self):
        prices, balance = self.tables()
        start = (self.DAY - pd.Timedelta(days=1)).tz_localize(TZ)
        changed_prices = prices.assign(aep_ct_kwh=prices["aep_ct_kwh"].mask(prices["timestamp_local"] >= start, 1e6))
        changed_balance = balance.assign(tsi_mw=balance["tsi_mw"].mask(balance["timestamp_local"] >= start, 1e6))

        original = forecast_for(self.DAY, prices, balance, self.DAYS)
        changed = forecast_for(self.DAY, changed_prices, changed_balance, self.DAYS)

        pd.testing.assert_frame_equal(original, changed, check_exact=True)

    def test_a_change_at_the_end_of_d_minus_2_does_change_the_forecast(self):
        prices, balance = self.tables()
        end = (self.DAY - pd.Timedelta(days=1)).tz_localize(TZ) - pd.Timedelta(minutes=15)
        changed_prices = prices.assign(aep_ct_kwh=prices["aep_ct_kwh"].mask(prices["timestamp_local"] == end, 1e6))

        original = forecast_for(self.DAY, prices, balance, self.DAYS)
        changed = forecast_for(self.DAY, changed_prices, balance, self.DAYS)

        assert not original.equals(changed)


def test_best_per_feature_set_takes_the_lowest_pinball_of_each_set():
    grid = [{"feature_set": fs, "window": "expanding", "alpha": a} for fs in ("B2", "B3") for a in (0.01, 0.03)]
    scores = pd.DataFrame({"pinball": [3.0, 2.0, 1.0, 4.0]}, index=[label(s) for s in grid])

    best = best_per_feature_set(scores, grid)

    assert best == {"B2|expanding|0.03": grid[1], "B3|expanding|0.01": grid[2]}


def test_first_complete_day_of_a_feature_set():
    table = synthetic_table(days=5)
    table.loc[table["delivery_day"] < "2026-03-03", "price_q50_56d"] = np.nan

    assert first_complete_day(table, "B3") == pd.Timestamp("2026-03-03")
    assert first_complete_day(table, "B2") == pd.Timestamp("2026-03-01")


def test_comparison_reports_the_daily_difference_and_its_interval():
    days = pd.date_range("2026-06-01", periods=28, freq="D")
    daily = pd.DataFrame({"a": np.full(28, 1.0), "b": np.full(28, 2.0)}, index=days)

    row = comparison(daily, [("a", "b")]).iloc[0]

    assert row["model"] == "a" and row["against"] == "b"
    assert (row["pinball_model"], row["pinball_against"]) == (1.0, 2.0)
    assert row["mean_difference"] == pytest.approx(-1.0)
    assert row["low"] == pytest.approx(-1.0) and row["high"] == pytest.approx(-1.0)
    assert bool(row["entirely_below_zero"]) is True
    assert row["n_days"] == 28


def test_surviving_features_count_the_weekly_fits_with_a_coefficient():
    fits = pd.to_datetime(["2026-04-01", "2026-04-08"])
    coefficients = pd.DataFrame({
        "feature": ["price_mean_d2", "hour_01", "price_mean_d2", "hour_01"],
        "q10": [0.5, 0.0, 0.0, 0.0], "q50": [0.5, 1.0, 0.7, 1.0], "q90": [0.0, 0.0, 0.0, 0.0],
        "fit_day": [fits[0], fits[0], fits[1], fits[1]], "model": "lasso",
    })

    out = surviving_features(coefficients, synthetic_table(days=1)).set_index(["feature", "quantile"])

    assert out.loc[("price_mean_d2", "q10"), "share_of_fits"] == 0.5
    assert out.loc[("price_mean_d2", "q50"), "share_of_fits"] == 1.0
    assert out.loc[("price_mean_d2", "q50"), "last_coefficient"] == 0.7
    assert out.loc[("price_mean_d2", "q90"), "share_of_fits"] == 0.0
    assert out.loc[("price_mean_d2", "q50"), "block"] == "price_level"
    assert out.loc[("hour_01", "q50"), "block"] == "hour"


def test_surviving_features_leave_out_the_intercept():
    coefficients = pd.DataFrame({"feature": ["intercept"], "q10": [1.0], "q50": [1.0], "q90": [1.0],
                                 "fit_day": [pd.Timestamp("2026-04-01")], "model": "lasso"})

    assert surviving_features(coefficients, synthetic_table(days=1)).empty


class TestAnalyse:
    DAYS = list(pd.date_range("2026-04-01", "2026-04-08", freq="D"))
    CANDIDATES = {"lasso_b3": {"feature_set": "B3", **LASSO}, "lasso_b2": {"feature_set": "B2", **LASSO}}

    def saved(self, table: pd.DataFrame, with_baseline: bool = True) -> pd.DataFrame:
        module = load(NAME)
        frames = [run_reference(table, self.DAYS)]
        frames += [run_variant(module, table, self.DAYS, n, s, jobs=1)[0] for n, s in self.CANDIDATES.items()]
        if with_baseline:
            frames.append(run_variant(module, table, self.DAYS, BASELINE, {"feature_set": "B2", **LASSO}, jobs=1)[0])
        return pd.concat(frames, ignore_index=True)

    def test_compares_on_the_same_days_with_all_models_of_the_issue(self):
        table = synthetic_table()
        table.loc[table["delivery_day"] < "2026-03-05", "price_q50_56d"] = np.nan

        scores, compared, features = analyse(table, self.DAYS, self.saved(table), self.CANDIDATES, jobs=1)

        same_rows_b2 = "lasso_b2_from_03-05"
        assert set(scores.index) == {REFERENCE, BASELINE, BOOSTING, "lasso_b3", "lasso_b2", same_rows_b2}
        assert (scores["n_days"] == len(self.DAYS)).all()
        pairs = set(zip(compared["model"], compared["against"]))
        assert {("lasso_b3", "lasso_b2"), ("lasso_b3", same_rows_b2), ("lasso_b3", REFERENCE), ("lasso_b3", BASELINE),
                ("lasso_b3", BOOSTING), ("lasso_b2", REFERENCE), ("lasso_b2", BOOSTING)} <= pairs
        assert set(features["model"]) == {"lasso_b3", "lasso_b2"}

    def test_stops_if_the_baseline_is_missing_instead_of_leaving_it_out(self):
        table = synthetic_table()

        with pytest.raises(ValueError, match="baseline"):
            analyse(table, self.DAYS, self.saved(table, with_baseline=False), self.CANDIDATES, jobs=1)

    def test_stops_if_a_fresh_fit_differs_from_the_saved_forecast(self):
        table = synthetic_table()
        saved = self.saved(table)
        rows = saved["model"] == "lasso_b3"
        saved.loc[rows, "q50"] += 1.0

        with pytest.raises(ValueError, match="lasso_b3"):
            analyse(table, self.DAYS, saved, self.CANDIDATES, jobs=1)


def test_linear_model_is_the_one_of_the_baseline():
    assert load(NAME).MODEL is LinearQuantileModel
