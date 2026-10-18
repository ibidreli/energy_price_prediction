import numpy as np
import pandas as pd
import pytest

from energy_price.availability import swissgrid_cutoff
from energy_price.evaluation.backtest import (
    HoldoutLockedError,
    common_rows,
    evaluate,
    folds,
    period_days,
    run_model,
    run_reference,
    training_rows,
)
from energy_price.features.dataset import calendar_features, quarter_hours
from energy_price.evaluation.experiment import load, run_variant


def synthetic_table(first: str = "2026-04-01", days: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    table = quarter_hours(list(pd.date_range(first, periods=days, freq="D")))
    table = pd.concat([table, calendar_features(table["timestamp_local"])], axis=1)
    table["holiday_pv_share"] = 0.0
    table["radiation"] = rng.uniform(0, 800, len(table))
    table["radiation_change"] = rng.normal(0, 50, len(table))
    table["aep_ct_kwh"] = 10 - 2 * table["radiation"] / 100 + rng.normal(0, 4, len(table))
    return table


def test_folds_are_weekly_blocks_in_order():
    days = list(pd.date_range("2026-06-01", "2026-06-17", freq="D"))

    blocks = folds(days)

    assert [len(b) for b in blocks] == [7, 7, 3]
    assert sum(blocks, []) == days


@pytest.mark.parametrize("window", [None, 56])
def test_training_ends_before_d_minus_1_for_every_day_of_the_fold(window):
    table = synthetic_table()
    for block in folds(list(pd.date_range("2026-05-10", "2026-05-30", freq="D"))):
        train = training_rows(table, block[0], window)
        # The fit day is the earliest day of the block, so its cutoff holds for all later days too.
        assert all(train["timestamp_local"].max() < swissgrid_cutoff(day.date()) for day in block)
        assert train["delivery_day"].max() == block[0] - pd.Timedelta(days=2)


def test_rolling_window_starts_at_d_minus_57():
    table = synthetic_table()
    fit_day = pd.Timestamp("2026-05-30")

    train = training_rows(table, fit_day, 56)

    assert train["delivery_day"].min() == fit_day - pd.Timedelta(days=57)
    assert train["delivery_day"].nunique() == 56


def test_holdout_is_locked_unless_released():
    with pytest.raises(HoldoutLockedError):
        period_days("holdout")
    assert period_days("holdout", unlock_holdout=True)[0] == pd.Timestamp("2026-09-01")


def test_backtest_beats_the_reference_when_radiation_drives_the_price():
    table = synthetic_table()
    days = list(pd.date_range("2026-05-20", "2026-05-29", freq="D"))

    settings = {"feature_set": "B2", "window": "expanding", "alpha": 0.0}
    linear, coefficients = run_variant(load("baseline_model"), table, days, "B2", settings, jobs=1)
    scores, daily = evaluate(pd.concat([run_reference(table, days), linear], ignore_index=True))

    assert scores.loc["B2", "skill_vs_reference"] > 0.2
    assert set(coefficients["fit_day"]) == {pd.Timestamp("2026-05-20"), pd.Timestamp("2026-05-27")}
    assert list(daily.index) == days


def test_models_are_compared_on_the_same_days_only():
    forecasts = pd.DataFrame({
        "delivery_day": pd.to_datetime(["2026-06-01", "2026-06-02"] * 2),
        "model": ["reference", "reference", "B2", "B2"],
        "aep_ct_kwh": [1.0, 2.0, 1.0, 2.0],
        "q10": [0.0, 0.0, 0.0, np.nan],  # B2 has no forecast on 06-02, e.g. missing weather
        "q50": [1.0, 1.0, 1.0, np.nan],
        "q90": [2.0, 2.0, 2.0, np.nan],
    })

    rows = common_rows(forecasts)

    assert set(rows["delivery_day"]) == {pd.Timestamp("2026-06-01")}


class MeanModel:
    """Smallest model that fulfils the interface: the training quantiles for every row."""

    def fit(self, table):
        self.q_ = table["aep_ct_kwh"].quantile([0.1, 0.5, 0.9]).to_numpy()
        return self

    def predict(self, table):
        return pd.DataFrame([self.q_] * len(table), index=table.index, columns=["q10", "q50", "q90"])


def test_any_model_with_fit_and_predict_runs_through_the_backtest():
    table = synthetic_table()
    days = list(pd.date_range("2026-05-20", "2026-05-29", freq="D"))

    forecasts, coefficients = run_model(table, days, MeanModel, "mean", jobs=1)

    assert coefficients is None
    assert set(forecasts["model"]) == {"mean"}
    assert forecasts[["q10", "q50", "q90"]].notna().all().all()
    assert sorted(forecasts["delivery_day"].unique()) == days
