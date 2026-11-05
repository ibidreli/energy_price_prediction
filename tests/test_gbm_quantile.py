"""Boosting integration: deterministic quantiles and protection from future target leakage."""

import numpy as np
import pandas as pd

from energy_price.evaluation.backtest import training_rows
from energy_price.evaluation.experiment import load, run_variant
from energy_price.models.shared.boosting import GradientBoostingQuantileModel
from test_evaluation_backtest import synthetic_table


def test_boosting_handles_negative_prices_missing_features_and_is_reproducible():
    table = synthetic_table(days=12)
    table["aep_ct_kwh"] -= 20
    train = table.iloc[:800]
    validation = table.iloc[800:].copy()
    first = GradientBoostingQuantileModel(max_iter=25).fit(train)
    second = GradientBoostingQuantileModel(max_iter=25).fit(train)
    expected = first.predict(validation)
    pd.testing.assert_frame_equal(expected, second.predict(validation))
    assert expected.index.equals(validation.index)
    assert (expected["q10"] <= expected["q50"]).all()
    assert (expected["q50"] <= expected["q90"]).all()
    assert expected["q50"].max() < 0
    validation.iloc[0, validation.columns.get_loc("radiation")] = np.nan
    assert first.predict(validation).iloc[0].isna().all()
    assert all(not estimator.do_early_stopping_ for estimator in first.models.values())


def test_backtest_never_trains_on_unknown_prices_or_uses_actual_direction():
    table = synthetic_table(days=12)
    day = pd.Timestamp("2026-04-10")
    changed = table.copy()
    changed.loc[changed["delivery_day"] >= day - pd.Timedelta(days=1), "aep_ct_kwh"] = 1e9
    changed["tsi_mw"] = 1e9  # Unavailable actual direction must not enter B2.
    settings = {"feature_set": "B2", "window": "expanding", "max_iter": 25}
    original, _ = run_variant(load("gbm_quantile"), table, [day], "gbm", settings, jobs=1)
    perturbed, _ = run_variant(load("gbm_quantile"), changed, [day], "gbm", settings, jobs=1)
    pd.testing.assert_frame_equal(original[["q10", "q50", "q90"]], perturbed[["q10", "q50", "q90"]])
    assert training_rows(table, day, None)["delivery_day"].max() == day - pd.Timedelta(days=2)
